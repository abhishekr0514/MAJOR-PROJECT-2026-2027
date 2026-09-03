"""FastAPI Router for Federated Learning management."""

from app.core.database import get_db
from app.features.federation.schema import (
    FLStartRoundRequest,
    FLStartRoundResponse,
    FLStatusResponse,
)
from app.features.federation.service import get_fl_status, start_fl_round
from app.features.users.dependencies import get_current_active_user
from app.features.users.models import Role, User
from app.features.users.permissions import RoleChecker
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

federation_router = APIRouter()
_require_super_admin = RoleChecker([Role.SUPER_ADMIN])


@federation_router.get(
    "/status",
    response_model=FLStatusResponse,
    summary="Get current Federated Learning training status",
)
async def fl_status(
    _: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    return await get_fl_status(db)


@federation_router.post(
    "/rounds/start",
    response_model=FLStartRoundResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a new FL training round (Super Admin only)",
)
async def fl_start_round(
    body: FLStartRoundRequest,
    _: User = Depends(_require_super_admin),
    db: AsyncSession = Depends(get_db),
):
    return await start_fl_round(db, body)
