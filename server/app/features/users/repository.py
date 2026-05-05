"""Data-access layer for the User model."""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.repository import BaseRepository
from app.features.users.models import User


class UserRepository(BaseRepository[User]):
    """User-specific queries — extends the generic base."""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, User)

    async def get_by_email(self, email: str) -> User | None:
        return await self.get_one_by(email=email)

    async def email_exists(self, email: str) -> bool:
        return await self.get_by_email(email) is not None


async def get_user_repository(
    db: AsyncSession = Depends(get_db),
) -> UserRepository:
    """FastAPI dependency — yields a ``UserRepository`` bound to the current session."""
    return UserRepository(db)
