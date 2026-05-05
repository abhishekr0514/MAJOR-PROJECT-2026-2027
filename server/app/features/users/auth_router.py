"""Authentication routes — signup, login, refresh, and profile."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError

from app.core.security import decode_token
from app.features.users.dependencies import get_current_active_user
from app.features.users.models import User
from app.features.users.repository import UserRepository, get_user_repository
from app.features.users.schema import (
    RefreshTokenRequest,
    TokenPair,
    UserCreate,
    UserResponse,
)
from app.features.users.service import authenticate_user, create_user, generate_tokens

auth_router = APIRouter()


# ---------------------------------------------------------------------------
# POST /auth/signup
# ---------------------------------------------------------------------------
@auth_router.post(
    "/signup",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def signup(
    body: UserCreate,
    repo: UserRepository = Depends(get_user_repository),
):
    try:
        user = await create_user(repo, body)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    return user


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------
@auth_router.post(
    "/login",
    response_model=TokenPair,
    summary="Authenticate and receive tokens",
)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    repo: UserRepository = Depends(get_user_repository),
):
    user = await authenticate_user(repo, form.username, form.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive account",
        )
    return generate_tokens(user)


# ---------------------------------------------------------------------------
# POST /auth/refresh
# ---------------------------------------------------------------------------
@auth_router.post(
    "/refresh",
    response_model=TokenPair,
    summary="Exchange a refresh token for a new token pair",
)
async def refresh(
    body: RefreshTokenRequest,
    repo: UserRepository = Depends(get_user_repository),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_token(body.refresh_token)
    except JWTError:
        raise credentials_exception

    if payload.get("type") != "refresh":
        raise credentials_exception

    email: str | None = payload.get("sub")
    if email is None:
        raise credentials_exception

    user = await repo.get_by_email(email)
    if user is None or not user.is_active:
        raise credentials_exception

    return generate_tokens(user)


# ---------------------------------------------------------------------------
# GET /auth/me
# ---------------------------------------------------------------------------
@auth_router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current authenticated user profile",
)
async def me(user: User = Depends(get_current_active_user)):
    return user
