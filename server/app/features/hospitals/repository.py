"""Data-access layer for the Hospital model."""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.base_repository import BaseRepository
from app.features.hospitals.models import Hospital


class HospitalRepository(BaseRepository[Hospital]):
    """Hospital-specific queries."""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, Hospital)

    async def get_by_name(self, name: str) -> Hospital | None:
        return await self.get_one_by(name=name)

    async def get_by_license_code(self, license_code: str) -> Hospital | None:
        return await self.get_one_by(license_code=license_code)


async def get_hospital_repository(
    db: AsyncSession = Depends(get_db),
) -> HospitalRepository:
    """FastAPI dependency for injecting the HospitalRepository."""
    return HospitalRepository(db)
