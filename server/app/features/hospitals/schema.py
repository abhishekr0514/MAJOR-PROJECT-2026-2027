"""Pydantic v2 schemas for the hospitals feature."""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class HospitalCreate(BaseModel):
    """Payload for creating a new hospital (Super Admin only)."""

    name: str = Field(min_length=2, max_length=200)
    license_code: str = Field(min_length=3, max_length=50)
    address: str | None = Field(default=None, max_length=500)


class HospitalResponse(BaseModel):
    """Safe hospital representation."""

    id: uuid.UUID
    name: str
    license_code: str
    address: str | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}
