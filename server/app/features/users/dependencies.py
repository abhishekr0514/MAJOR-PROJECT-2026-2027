"""FastAPI dependencies for user authentication."""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

from app.core.security import decode_token
from app.features.users.models import User
from app.features.users.repository import UserRepository, get_user_repository

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


async def get_current_user(
    repo: UserRepository = Depends(get_user_repository),
    token: str = Depends(oauth2_scheme),
) -> User:
    """Decode the JWT and return the corresponding ``User``.

    Validates that the token is an *access* token (not a refresh token).
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_token(token)
    except JWTError:
        raise credentials_exception

    # Prevent refresh tokens from being used as access tokens
    if payload.get("type") != "access":
        raise credentials_exception

    email: str | None = payload.get("sub")
    if email is None:
        raise credentials_exception

    user = await repo.get_by_email(email)

    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    user: User = Depends(get_current_user),
) -> User:
    """Wraps ``get_current_user`` and rejects inactive accounts."""
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive account",
        )
    return user
