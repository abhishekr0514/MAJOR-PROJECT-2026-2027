"""Repository layer for Patient entity persistence."""

from app.core.base_repository import BaseRepository
from app.features.patients.models import Patient
from sqlalchemy.ext.asyncio import AsyncSession


class PatientRepository(BaseRepository[Patient]):
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, Patient)
