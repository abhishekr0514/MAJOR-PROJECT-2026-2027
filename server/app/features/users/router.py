"""Admin-protected user management routes."""

from fastapi import APIRouter, Depends, status

from app.features.users.dependencies import get_current_active_user
from app.features.users.models import Role, User
from app.features.users.permissions import RoleChecker
from app.features.users.repository import UserRepository, get_user_repository
from app.features.users.schema import AdminUserCreate, UserResponse
from app.features.users.service import admin_create_user

user_router = APIRouter()

# Only SUPER_ADMIN and HOSPITAL_ADMIN can reach any endpoint on this router.
_require_admin = RoleChecker([Role.SUPER_ADMIN, Role.HOSPITAL_ADMIN])


# ---------------------------------------------------------------------------
# POST /users/
# ---------------------------------------------------------------------------
@user_router.post(
    "/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a user (admin only)",
    description=(
        "**SUPER_ADMIN** can create any role. "
        "**HOSPITAL_ADMIN** can only create **CLINICIAN** accounts."
    ),
)
async def create_user_as_admin(
    body: AdminUserCreate,
    actor: User = Depends(_require_admin),
    repo: UserRepository = Depends(get_user_repository),
):
    return await admin_create_user(repo, body, actor)


# ---------------------------------------------------------------------------
# GET /users/  (SUPER_ADMIN only)
# ---------------------------------------------------------------------------
@user_router.get(
    "/",
    response_model=list[UserResponse],
    summary="List all users (Super Admin only)",
)
async def list_users(
    skip: int = 0,
    limit: int = 50,
    _: User = Depends(RoleChecker([Role.SUPER_ADMIN])),
    repo: UserRepository = Depends(get_user_repository),
):
    return await repo.get_all(skip=skip, limit=limit)
