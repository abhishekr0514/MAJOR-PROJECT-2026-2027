"""Database Seeding Script for MedShield FL."""

import asyncio
import sys

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.models import Hospital, User
from app.core.security import hash_password
from app.features.hospitals.repository import HospitalRepository
from app.features.users.models import Role
from app.features.users.repository import UserRepository


async def seed_database() -> None:
    """Seed initial hospitals and role-based accounts (Super Admin, Hospital Admin, Clinician)."""
    print("[Database Seed] Starting database seeding...")

    async with SessionLocal() as db:
        user_repo = UserRepository(db)
        hosp_repo = HospitalRepository(db)

        # 1. Seed Base Hospitals
        hospitals_to_seed = [
            {"name": "Central General Hospital", "license_code": "MSFL-CENTRAL-001"},
            {"name": "Hospital Alpha", "license_code": "MSFL-ALPHA-002"},
            {"name": "Hospital Beta", "license_code": "MSFL-BETA-003"},
        ]

        created_hospitals = {}
        for h_info in hospitals_to_seed:
            existing = await hosp_repo.get_one_by(license_code=h_info["license_code"])
            if not existing:
                hosp = Hospital(name=h_info["name"], license_code=h_info["license_code"])
                hosp = await hosp_repo.create(hosp)
                created_hospitals[h_info["name"]] = hosp.id
                print(f"✅ Created Hospital: {h_info['name']} (ID: {hosp.id})")
            else:
                created_hospitals[h_info["name"]] = existing.id
                print(f"ℹ️ Hospital exists: {h_info['name']}")

        # 2. Seed Initial Super Admin
        admin_email = getattr(settings, "FIRST_SUPER_ADMIN_EMAIL", "admin@medshield.org")
        admin_pass = getattr(settings, "FIRST_SUPER_ADMIN_PASSWORD", "AdminPass123!")

        super_admin = await user_repo.get_first_by(email=admin_email)
        if not super_admin:
            admin_user = User(
                email=admin_email,
                full_name="System Administrator",
                hashed_password=hash_password(admin_pass),
                role=Role.SUPER_ADMIN,
                is_active=True,
            )
            await user_repo.create(admin_user)
            print(f"✅ Created Super Admin: {admin_email}")
        else:
            print(f"ℹ️ Super Admin exists: {admin_email}")

        # 3. Seed Hospital Admin & Clinician Accounts
        demo_users = [
            {
                "email": "admin@alpha.hospital.org",
                "full_name": "Dr. Sarah Alpha",
                "role": Role.HOSPITAL_ADMIN,
                "hospital": "Hospital Alpha",
            },
            {
                "email": "clinician@medshield.org",
                "full_name": "Dr. John Watson",
                "role": Role.CLINICIAN,
                "hospital": "Central General Hospital",
            },
        ]

        for u_info in demo_users:
            existing_user = await user_repo.get_first_by(email=u_info["email"])
            if not existing_user:
                h_id = created_hospitals.get(u_info["hospital"])
                user_obj = User(
                    email=u_info["email"],
                    full_name=u_info["full_name"],
                    hashed_password=hash_password("ClinicianPass123!"),
                    role=u_info["role"],
                    hospital_id=h_id,
                    is_active=True,
                )
                await user_repo.create(user_obj)
                print(f"✅ Created User: {u_info['email']} ({u_info['role']})")
            else:
                print(f"ℹ️ User exists: {u_info['email']}")

    print("✅ Database seeding completed successfully!")


if __name__ == "__main__":
    try:
        asyncio.run(seed_database())
    except Exception as e:
        print(f"Seeding error: {e}")
        sys.exit(1)
