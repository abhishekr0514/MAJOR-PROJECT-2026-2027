from fastapi import Depends, HTTPException, status
from app.features.users.models import User, Role
from app.features.users.dependencies import get_current_user


class RoleChecker:
    def __init__(self, allowed_roles: list[Role]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: User = Depends(get_current_user)):
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized"
            )
        return user
