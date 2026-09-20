"""
Indexer entrypoint: `python -m app.indexing`.

Before this existed there was no way to start the worker. `run_worker` and
its signal handlers were written and tested, but nothing called them and no
deployment target referenced them, so the job queue had a producer and no
consumer: enqueued repositories stayed `pending` forever.

Serves the embedding endpoint and runs the worker in one process. See
`service.py` for why those two belong together and why the worker gets its
own thread.

    python -m app.indexing                 # serve and index
    python -m app.indexing --no-worker     # serve only
    python -m app.indexing --port 8081
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import get_settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.indexing")
    ap.add_argument("--host", default="0.0.0.0")  # noqa: S104 - a container binds all interfaces
    ap.add_argument("--port", type=int, default=None,
                    help="default: $PORT, else 8081")
    ap.add_argument("--no-worker", action="store_true",
                    help="serve /embed without claiming jobs")
    ap.add_argument("--log-level", default=None)
    args = ap.parse_args(argv)

    settings = get_settings()
    level = (args.log_level or settings.log_level).upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    if args.no_worker:
        # Read by service._start_worker_thread. Set on the settings object
        # rather than passed down so the flag and the env var (which
        # pydantic-settings already binds) reach the same place.
        settings.indexer_run_worker = False

    port = args.port if args.port is not None else settings.indexer_port

    import uvicorn

    from app.indexing.service import build_app

    logging.getLogger("clarix.indexer").info(
        "starting indexer on %s:%d (worker=%s)",
        args.host, port, not args.no_worker,
    )
    uvicorn.run(build_app(settings), host=args.host, port=port, log_level=level.lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())
