"""
Authentication routes: register, login, OAuth (GitHub / Google), user profile.
"""

import logging
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models import User
from app.schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from app.security import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["Auth"])
logger = logging.getLogger("copilot.auth")
settings = get_settings()


# ── Register ────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Create a new account with email + password."""
    # Check if email already exists
    result = await db.execute(select(User).where(User.email == req.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(
        email=req.email,
        name=req.name,
        hashed_password=hash_password(req.password),
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)

    token = create_access_token(user.id, user.email)
    logger.info("User registered: %s", user.email)
    return TokenResponse(access_token=token, user=UserResponse.model_validate(user))


# ── Login ───────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate with email + password."""
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()

    if not user or not user.hashed_password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    if not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    token = create_access_token(user.id, user.email)
    logger.info("User logged in: %s", user.email)
    return TokenResponse(access_token=token, user=UserResponse.model_validate(user))


# ── Current User ────────────────────────────────────────────

@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)):
    """Get the currently authenticated user."""
    return UserResponse.model_validate(user)


# ═════════════════════════════════════════════════════════════
#  GitHub OAuth
# ═════════════════════════════════════════════════════════════

GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


@router.get("/github")
async def github_login():
    """Redirect user to GitHub OAuth consent screen."""
    if not settings.github_client_id:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured")

    params = {
        "client_id": settings.github_client_id,
        "redirect_uri": f"{settings.backend_url}/api/auth/github/callback",
        "scope": "user:email",
    }
    return RedirectResponse(f"{GITHUB_AUTH_URL}?{urlencode(params)}")


@router.get("/github/callback")
async def github_callback(code: str, db: AsyncSession = Depends(get_db)):
    """Exchange GitHub auth code for access token, find/create user, return JWT."""
    async with httpx.AsyncClient() as client:
        # Exchange code for token
        token_res = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
        token_data = token_res.json()
        gh_access_token = token_data.get("access_token")
        if not gh_access_token:
            raise HTTPException(status_code=400, detail="Failed to get GitHub access token")

        # Get user info
        headers = {"Authorization": f"Bearer {gh_access_token}", "Accept": "application/json"}
        user_res = await client.get(GITHUB_USER_URL, headers=headers)
        gh_user = user_res.json()

        # Get primary email
        emails_res = await client.get(GITHUB_EMAILS_URL, headers=headers)
        emails = emails_res.json()
        primary_email = next((e["email"] for e in emails if e.get("primary")), None)
        if not primary_email:
            primary_email = gh_user.get("email") or f"{gh_user['login']}@github.com"

    # Find or create user
    user = await _find_or_create_oauth_user(
        db,
        provider="github",
        oauth_id=str(gh_user["id"]),
        email=primary_email,
        name=gh_user.get("name") or gh_user.get("login", ""),
        avatar_url=gh_user.get("avatar_url"),
    )

    token = create_access_token(user.id, user.email)
    # Redirect to frontend callback with token
    frontend_url = settings.frontend_url or "http://localhost:3000"
    return RedirectResponse(f"{frontend_url}/auth/callback?token={token}")


# ═════════════════════════════════════════════════════════════
#  Google OAuth
# ═════════════════════════════════════════════════════════════

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


@router.get("/google")
async def google_login():
    """Redirect user to Google OAuth consent screen."""
    if not settings.google_client_id:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": f"{settings.backend_url}/api/auth/google/callback",
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
    }
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}")


@router.get("/google/callback")
async def google_callback(code: str, db: AsyncSession = Depends(get_db)):
    """Exchange Google auth code for access token, find/create user, return JWT."""
    async with httpx.AsyncClient() as client:
        # Exchange code for token
        token_res = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "code": code,
                "redirect_uri": f"{settings.backend_url}/api/auth/google/callback",
                "grant_type": "authorization_code",
            },
        )
        token_data = token_res.json()
        g_access_token = token_data.get("access_token")
        if not g_access_token:
            raise HTTPException(status_code=400, detail="Failed to get Google access token")

        # Get user info
        user_res = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {g_access_token}"},
        )
        g_user = user_res.json()

    # Find or create user
    user = await _find_or_create_oauth_user(
        db,
        provider="google",
        oauth_id=str(g_user["id"]),
        email=g_user["email"],
        name=g_user.get("name", ""),
        avatar_url=g_user.get("picture"),
    )

    token = create_access_token(user.id, user.email)
    frontend_url = settings.frontend_url or "http://localhost:3000"
    return RedirectResponse(f"{frontend_url}/auth/callback?token={token}")


# ═════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════

async def _find_or_create_oauth_user(
    db: AsyncSession,
    provider: str,
    oauth_id: str,
    email: str,
    name: str,
    avatar_url: str | None,
) -> User:
    """Find existing user by OAuth ID or email, or create a new one."""
    # Try by OAuth provider + ID
    result = await db.execute(
        select(User).where(User.oauth_provider == provider, User.oauth_id == oauth_id)
    )
    user = result.scalar_one_or_none()
    if user:
        return user

    # Try by email (link accounts)
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user:
        # Link OAuth to existing account
        user.oauth_provider = provider
        user.oauth_id = oauth_id
        if avatar_url:
            user.avatar_url = avatar_url
        await db.flush()
        return user

    # Create new user
    user = User(
        email=email,
        name=name,
        oauth_provider=provider,
        oauth_id=oauth_id,
        avatar_url=avatar_url,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    logger.info("OAuth user created: %s (%s)", email, provider)
    return user
