"""Business-logic layer for user authentication."""

from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.features.users.models import User
from app.features.users.repository import UserRepository
from app.features.users.schema import TokenPair, UserCreate


async def authenticate_user(
    repo: UserRepository, email: str, password: str
) -> User | None:
    """Return the user if credentials are valid, else ``None``."""
    user = await repo.get_by_email(email)
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user


async def create_user(repo: UserRepository, data: UserCreate) -> User:
    """Create a new user.  Raises ``ValueError`` if email already taken."""
    if await repo.email_exists(data.email):
        raise ValueError("A user with this email already exists")

    user = User(
        email=data.email,
        full_name=data.full_name,
        hashed_password=hash_password(data.password),
    )
    return await repo.create(user)


def generate_tokens(user: User) -> TokenPair:
    """Generate an access + refresh token pair for *user*."""
    payload = {"sub": user.email}
    return TokenPair(
        access_token=create_access_token(payload),
        refresh_token=create_refresh_token(payload),
    )
