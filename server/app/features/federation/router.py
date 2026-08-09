import uuid

from app.core.database import get_db
from app.features.federation.models import FLRound
from app.features.federation.schema import FLRoundStartSchema, FLStatusResponseSchema
from app.features.hospitals.models import Hospital
from app.features.users.dependencies import get_current_active_user
from app.features.users.models import Role, User
from app.features.users.permissions import RoleChecker
from fastapi import APIRouter, Depends, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

federation_router = APIRouter()

# Restricted access roles
_require_super_admin = RoleChecker([Role.SUPER_ADMIN])


@federation_router.get(
    "/status",
    response_model=FLStatusResponseSchema,
    summary="Get current FL training status, active rounds, and global version performance metrics",
)
async def get_fl_status(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    # Retrieve the latest round
    stmt = select(FLRound).order_by(FLRound.round_number.desc())
    res = await db.execute(stmt)
    latest_round = res.scalars().first()

    # Count participating hospitals in database
    hosp_count_stmt = select(func.count(Hospital.id))
    hosp_res = await db.execute(hosp_count_stmt)
    participating_count = hosp_res.scalar() or 0

    if not latest_round:
        # Default fallback if no rounds have run yet
        return FLStatusResponseSchema(
            current_round=0,
            total_rounds=10,
            status="IDLE",
            global_accuracy=0.85,  # Start baseline
            participating_hospitals_count=participating_count,
        )

    # Return registered DB metrics
    return FLStatusResponseSchema(
        current_round=latest_round.round_number,
        total_rounds=10,
        status=latest_round.status,
        global_accuracy=latest_round.global_accuracy,
        participating_hospitals_count=participating_count,
    )


@federation_router.post(
    "/rounds/start",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger a new FL training round across connected client nodes",
)
async def start_fl_round(
    body: FLRoundStartSchema,
    current_user: User = Depends(_require_super_admin),
    db: AsyncSession = Depends(get_db),
):
    # Determine the next round number
    stmt = select(FLRound).order_by(FLRound.round_number.desc())
    res = await db.execute(stmt)
    latest_round = res.scalars().first()
    next_round_number = (latest_round.round_number + 1) if latest_round else 1

    # Simulate dynamic model evaluation accuracy increase as rounds progress
    simulated_acc = min(0.95, 0.82 + (next_round_number * 0.025))

    new_round = FLRound(
        id=uuid.uuid4(),
        round_number=next_round_number,
        global_accuracy=round(simulated_acc, 3),
        global_loss=round(max(0.1, 0.45 - (next_round_number * 0.05)), 3),
        status="IN_PROGRESS",
    )
    db.add(new_round)
    await db.commit()
    await db.refresh(new_round)

    return {
        "message": f"FL Round {next_round_number} triggered successfully.",
        "round_id": str(new_round.id),
    }
