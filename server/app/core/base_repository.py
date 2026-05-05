"""Generic base repository for common CRUD operations."""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import Base


class BaseRepository[SQLModelType: Base]:
    """Abstract data-access layer — subclass per model.

    Provides common operations so feature repositories only need
    to implement domain-specific queries.
    """

    def __init__(self, db: AsyncSession, model: type[SQLModelType]) -> None:
        self.db = db
        self.model = model

    async def get_by_id(self, id: uuid.UUID) -> SQLModelType | None:
        return await self.db.get(self.model, id)

    async def get_all(self, *, skip: int = 0, limit: int = 100) -> list[SQLModelType]:
        result = await self.db.execute(select(self.model).offset(skip).limit(limit))
        return list(result.scalars().all())

    async def get_one_by(self, **filters: Any) -> SQLModelType | None:
        """Return a single row matching the given column filters, or ``None``.

        Raises ``MultipleResultsFound`` if more than one row matches.
        Use ``get_first_by`` if duplicates are acceptable.
        """
        stmt = select(self.model).filter_by(**filters)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_first_by(self, **filters: Any) -> SQLModelType | None:
        """Return the first row matching the given filters, or ``None``.

        Unlike ``get_one_by``, this does not raise if multiple rows exist.
        """
        stmt = select(self.model).filter_by(**filters).limit(1)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, obj: SQLModelType) -> SQLModelType:
        self.db.add(obj)
        await self.db.commit()
        await self.db.refresh(obj)
        return obj

    async def update(self, obj: SQLModelType) -> SQLModelType:
        await self.db.commit()
        await self.db.refresh(obj)
        return obj

    async def delete(self, obj: SQLModelType) -> None:
        await self.db.delete(obj)
        await self.db.commit()
