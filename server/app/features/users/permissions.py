import enum
from fastapi import Depends, HTTPException, status

class Role(str, enum):
    SUPER_ADMIN = "super_admin"
    HOSPITAL_ADMIN = "hospital_admin"
    CLINICIAN = "clinician"

class RoleChecker:
    def __init__(self, allowed_roles = list[Role]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, user: User = Depends(get_current_user)):
        if user.role not in self.allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized")
        return user    
        
