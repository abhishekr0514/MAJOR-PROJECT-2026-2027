"""Service layer for Federated Learning orchestration."""

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.features.federation.models import FLRound
from app.features.federation.repository import FLRoundRepository
from app.features.federation.schema import (
    FLStartRoundRequest,
    FLStartRoundResponse,
    FLStatusResponse,
)


async def get_fl_status(db: AsyncSession) -> FLStatusResponse:
    repo = FLRoundRepository(db)
    latest = await repo.get_latest_round()

    if not latest:
        return FLStatusResponse(
            current_round=0,
            total_rounds=10,
            status="IDLE",
            global_accuracy=0.0,
            global_loss=0.0,
            participating_hospitals_count=0,
        )

    return FLStatusResponse(
        current_round=latest.round_number,
        total_rounds=10,
        status=latest.status,
        global_accuracy=latest.global_accuracy or 0.0,
        global_loss=latest.global_loss or 0.0,
        participating_hospitals_count=3,
    )


async def start_fl_round(
    db: AsyncSession, data: FLStartRoundRequest
) -> FLStartRoundResponse:
    repo = FLRoundRepository(db)
    latest = await repo.get_latest_round()

    next_round_number = (latest.round_number + 1) if latest else 1
    new_round = FLRound(
        round_number=next_round_number,
        status="IN_PROGRESS",
        global_accuracy=0.85,
        global_loss=0.32,
    )
    created_round = await repo.create(new_round)

    return FLStartRoundResponse(
        message=f"FL Round {created_round.round_number} triggered successfully.",
        round_id=created_round.id,
        started_at=datetime.now(timezone.utc),
    )
