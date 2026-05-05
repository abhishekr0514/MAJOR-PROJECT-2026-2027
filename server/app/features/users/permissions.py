from fastapi import Depends, HTTPException, status

from app.features.users.dependencies import get_current_active_user
from app.features.users.models import Role, User


class RoleChecker:
    """Callable dependency that enforces role-based access control.

    Usage::

        allow_admins = RoleChecker([Role.SUPER_ADMIN, Role.HOSPITAL_ADMIN])

        @router.get("/admin-only")
        async def admin_view(user: User = Depends(allow_admins)):
            ...
    """

    def __init__(self, allowed_roles: list[Role]) -> None:
        self.allowed_roles = allowed_roles

    async def __call__(self, user: User = Depends(get_current_active_user)) -> User:
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user
