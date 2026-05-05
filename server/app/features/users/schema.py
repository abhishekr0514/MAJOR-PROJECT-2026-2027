"""Pydantic v2 schemas for the users feature."""

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from app.features.users.models import Role


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------
class UserCreate(BaseModel):
    """Payload for ``POST /auth/signup`` (public — always creates a Clinician)."""

    email: EmailStr
    full_name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=128)


class AdminUserCreate(BaseModel):
    """Payload for ``POST /users/`` (admin-only — role is explicitly chosen).

    Role constraints are enforced in the service layer:
    - ``SUPER_ADMIN``   → can create any role
    - ``HOSPITAL_ADMIN`` → can only create ``CLINICIAN``
    """

    email: EmailStr
    full_name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=128)
    role: Role


class RefreshTokenRequest(BaseModel):
    """Payload for ``POST /auth/refresh``."""

    refresh_token: str


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class UserResponse(BaseModel):
    """Safe representation of a user (no password)."""

    id: uuid.UUID
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenPair(BaseModel):
    """Access + refresh token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
