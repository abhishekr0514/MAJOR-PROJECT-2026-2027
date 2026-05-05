"""Hospital management routes."""

from fastapi import APIRouter, Depends, status

from app.features.hospitals.repository import (
    HospitalRepository,
    get_hospital_repository,
)
from app.features.hospitals.schema import HospitalCreate, HospitalResponse
from app.features.hospitals.service import create_hospital
from app.features.users.models import Role, User
from app.features.users.permissions import RoleChecker

hospital_router = APIRouter()

# Only SUPER_ADMIN can manage hospitals
_require_super_admin = RoleChecker([Role.SUPER_ADMIN])


@hospital_router.post(
    "/",
    response_model=HospitalResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new hospital (Super Admin only)",
)
async def register_hospital(
    body: HospitalCreate,
    _: User = Depends(_require_super_admin),
    repo: HospitalRepository = Depends(get_hospital_repository),
):
    return await create_hospital(repo, body)


@hospital_router.get(
    "/",
    response_model=list[HospitalResponse],
    summary="List all hospitals (Super Admin only)",
)
async def list_hospitals(
    skip: int = 0,
    limit: int = 50,
    _: User = Depends(_require_super_admin),
    repo: HospitalRepository = Depends(get_hospital_repository),
):
    return await repo.get_all(skip=skip, limit=limit)
