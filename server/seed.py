import asyncio
import sys

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.models import Hospital, User  # noqa: F401 — registers all models & relationships
from app.core.security import hash_password
from app.features.users.models import Role
from app.features.users.repository import UserRepository


async def create_first_super_admin():
    """Create the first super admin user if it doesn't exist."""
    print("Checking for initial super admin...")
    
    async with SessionLocal() as db:
        repo = UserRepository(db)
        
        # Check if any super admin exists
        exists = await repo.get_first_by(role=Role.SUPER_ADMIN)
        if exists:
            print(f"Super admin already exists: {exists.email}")
            return

        print(f"Creating super admin: {settings.FIRST_SUPER_ADMIN_EMAIL}")
        
        admin = User(
            email=settings.FIRST_SUPER_ADMIN_EMAIL,
            full_name="System Administrator",
            hashed_password=hash_password(settings.FIRST_SUPER_ADMIN_PASSWORD),
            role=Role.SUPER_ADMIN,
            is_active=True,
        )
        
        try:
            await repo.create(admin)
            print("Successfully created the first super admin!")
        except Exception as e:
            print(f"Error creating super admin: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(create_first_super_admin())
