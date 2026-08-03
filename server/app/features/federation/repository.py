"""Repository layer for Federated Learning rounds persistence."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.base_repository import BaseRepository
from app.features.federation.models import FLRound


class FLRoundRepository(BaseRepository[FLRound]):
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, FLRound)

    async def get_latest_round(self) -> FLRound | None:
        stmt = select(self.model).order_by(self.model.round_number.desc()).limit(1)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
