"""Business logic for hospitals."""

from fastapi import HTTPException, status

from app.features.hospitals.models import Hospital
from app.features.hospitals.repository import HospitalRepository
from app.features.hospitals.schema import HospitalCreate


async def create_hospital(repo: HospitalRepository, data: HospitalCreate) -> Hospital:
    """Create a new hospital.

    Ensures that the name and license code are unique.
    """
    if await repo.get_by_name(data.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A hospital with this name already exists",
        )

    if await repo.get_by_license_code(data.license_code):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A hospital with this license code already exists",
        )

    hospital = Hospital(
        name=data.name,
        license_code=data.license_code,
        address=data.address,
    )
    return await repo.create(hospital)
