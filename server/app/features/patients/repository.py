"""Repository layer for Patient entity persistence."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.base_repository import BaseRepository
from app.features.patients.models import Patient


class PatientRepository(BaseRepository[Patient]):
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, Patient)
