"""Repository layer for prediction persistence."""

import uuid
from collections.abc import Sequence

from app.core.base_repository import BaseRepository
from app.features.prediction.models import Prediction
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


class PredictionRepository(BaseRepository[Prediction]):
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, Prediction)

    async def get_history_by_patient(
        self, patient_id: uuid.UUID
    ) -> Sequence[Prediction]:
        stmt = (
            select(self.model)
            .filter_by(patient_id=patient_id)
            .order_by(self.model.created_at.desc())
        )
        result = await self.db.execute(stmt)
        return result.scalars().all()
