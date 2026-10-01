#!/usr/bin/env bash
#
# Put the indexer on an Oracle Cloud Always Free ARM instance.
#
# Why the indexer and not the whole application: indexing is the only
# CPU-bound part. The API is I/O against Postgres and a model provider, and
# is perfectly happy on a small free instance. Moving one service buys the
# CPU where it is actually spent, and leaves the API, the frontend and the
# database exactly where they are.
#
# It also fixes a limit that has nothing to do with speed. A Render
# workspace gets 750 free instance hours a month shared across every free
# service, and a calendar month is about 730 hours, so two always-on
# services cannot both stay awake -- they suspend around the middle of the
# month. With the indexer moved off, the API alone fits inside the
# allowance and can be kept warm.
#
# Run this on a fresh Ubuntu 22.04 or 24.04 ARM instance (VM.Standard.A1.Flex):
#
#     curl -fsSL https://raw.githubusercontent.com/ayushhh1010/clarix/main/deploy/oracle-indexer.sh \
#       | sudo bash -s -- --domain clarix-indexer.duckdns.org \
#                         --database-url 'postgresql+asyncpg://...' \
#                         --api-key 'the EMBEDDING_API_KEY from Render'
#
# Safe to re-run: it updates the checkout, reinstalls, and restarts.
#
# Two things this script cannot do for you, because they live in a web
# console rather than on the machine:
#
#   1. Open ports 80 and 443 in the VCN Security List for the instance's
#      subnet. Oracle blocks them by default, and a closed Security List
#      looks exactly like a broken web server.
#   2. Point the domain's A record at this instance's public IP. Caddy
#      cannot obtain a certificate until that resolves.
#
set -euo pipefail

DOMAIN=""
DATABASE_URL=""
API_KEY=""
REPO_URL="https://github.com/ayushhh1010/clarix.git"
BRANCH="main"
APP_DIR="/opt/clarix"
ENV_FILE="/etc/clarix-indexer.env"
SERVICE_USER="clarix"
PORT="8081"

while [ $# -gt 0 ]; do
  case "$1" in
    --domain)       DOMAIN="$2"; shift 2 ;;
    --database-url) DATABASE_URL="$2"; shift 2 ;;
    --api-key)      API_KEY="$2"; shift 2 ;;
    --repo)         REPO_URL="$2"; shift 2 ;;
    --branch)       BRANCH="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

for required in DOMAIN DATABASE_URL API_KEY; do
  if [ -z "${!required}" ]; then
    echo "missing --${required,,}" | tr '_' '-' >&2
    exit 2
  fi
done

if [ "$(id -u)" -ne 0 ]; then
  echo "run with sudo" >&2
  exit 2
fi

echo "==> packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl ca-certificates debian-keyring debian-archive-keyring \
                       apt-transport-https python3-venv python3-dev build-essential \
                       iptables-persistent >/dev/null

# Caddy terminates TLS. The API sends EMBEDDING_API_KEY on every request, so
# this leg cannot be plain HTTP. Caddy obtains and renews a Let's Encrypt
# certificate on its own, which is the whole reason to prefer it here.
if ! command -v caddy >/dev/null 2>&1; then
  echo "==> caddy"
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq
  apt-get install -y -qq caddy >/dev/null
fi

# Oracle's Ubuntu images ship an INPUT chain that REJECTs everything except
# SSH. The Security List in the console is a second, independent filter --
# both must allow 80 and 443 or the certificate request never arrives.
echo "==> firewall"
for p in 80 443; do
  if ! iptables -C INPUT -p tcp --dport "$p" -j ACCEPT 2>/dev/null; then
    iptables -I INPUT 1 -p tcp --dport "$p" -j ACCEPT
  fi
done
netfilter-persistent save >/dev/null 2>&1 || true

id -u "$SERVICE_USER" >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin "$SERVICE_USER"

echo "==> source"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" fetch --quiet origin "$BRANCH"
  git -C "$APP_DIR" reset --hard --quiet "origin/$BRANCH"
else
  rm -rf "$APP_DIR"
  git clone --quiet --branch "$BRANCH" --depth 1 "$REPO_URL" "$APP_DIR"
fi
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"

echo "==> python environment"
if [ ! -x "$APP_DIR/.venv/bin/python" ]; then
  python3 -m venv "$APP_DIR/.venv"
fi
"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
# onnxruntime publishes manylinux aarch64 wheels, so this compiles nothing.
( cd "$APP_DIR/backend" && "$APP_DIR/.venv/bin/pip" install --quiet '.[indexer]' )
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"

# One thread per core. The shipped default is 1, which is correct for a
# fractional-CPU instance and leaves most of this machine idle.
THREADS="$(nproc)"

echo "==> configuration ($THREADS threads)"
umask 077
cat > "$ENV_FILE" <<ENV
APP_ENV=production
LOG_LEVEL=INFO
DATABASE_URL=$DATABASE_URL
EMBEDDING_API_KEY=$API_KEY
EMBEDDING_THREADS=$THREADS
# These three compose the embedding identity. The API builds the same string
# and refuses vectors from a different one, so they must match the API's
# environment exactly or dense retrieval silently switches off.
EMBEDDING_MODEL_ID=jinaai/jina-embeddings-v2-base-code
EMBEDDING_ONNX_FILE=onnx/model_quantized.onnx
EMBEDDING_MAX_TOKENS=384
INDEXER_EMBED_BATCH=1
WORKER_DIR=/var/lib/clarix/work
HF_HOME=/var/lib/clarix/huggingface
ENV
chmod 640 "$ENV_FILE"
chown root:"$SERVICE_USER" "$ENV_FILE"

install -d -o "$SERVICE_USER" -g "$SERVICE_USER" /var/lib/clarix/work /var/lib/clarix/huggingface

cat > /etc/systemd/system/clarix-indexer.service <<UNIT
[Unit]
Description=Clarix indexer (embedding endpoint and indexing worker)
After=network-online.target
Wants=network-online.target

[Service]
User=$SERVICE_USER
Group=$SERVICE_USER
EnvironmentFile=$ENV_FILE
WorkingDirectory=$APP_DIR/backend
ExecStart=$APP_DIR/.venv/bin/python -m app.indexing --port $PORT
Restart=always
RestartSec=5
# First boot downloads ~306 MB of model before the port opens.
TimeoutStartSec=600
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true
ReadWritePaths=/var/lib/clarix

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/caddy/Caddyfile <<CADDY
$DOMAIN {
	reverse_proxy 127.0.0.1:$PORT
}
CADDY

echo "==> starting"
systemctl daemon-reload
systemctl enable --quiet --now clarix-indexer
systemctl restart caddy

echo
echo "indexer:  $(systemctl is-active clarix-indexer)"
echo "caddy:    $(systemctl is-active caddy)"
echo
echo "The first start downloads the model; /health answers once it is loaded."
echo "Watch it with:  journalctl -u clarix-indexer -f"
echo
echo "When https://$DOMAIN/health answers, set on the Render API service:"
echo "    EMBEDDING_ENDPOINT = https://$DOMAIN"
echo "    EMBEDDING_API_KEY  = (the same value passed to --api-key)"
