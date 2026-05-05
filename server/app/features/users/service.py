"""Business-logic layer for user authentication."""

from fastapi import HTTPException, status

from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.features.users.models import Role, User
from app.features.users.repository import UserRepository
from app.features.users.schema import AdminUserCreate, TokenPair, UserCreate

# Defines which roles each actor is permitted to create.
_CREATABLE_ROLES: dict[Role, set[Role]] = {
    Role.SUPER_ADMIN: {Role.SUPER_ADMIN, Role.HOSPITAL_ADMIN, Role.CLINICIAN},
    Role.HOSPITAL_ADMIN: {Role.CLINICIAN},
}


async def authenticate_user(
    repo: UserRepository, email: str, password: str
) -> User | None:
    """Return the user if credentials are valid, else ``None``."""
    user = await repo.get_by_email(email)
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user


async def create_user(repo: UserRepository, data: UserCreate) -> User:
    """Create a new public user (always a Clinician, no hospital).

    Raises ``ValueError`` if the email is already taken.
    """
    if await repo.email_exists(data.email):
        raise ValueError("A user with this email already exists")

    user = User(
        email=data.email,
        full_name=data.full_name,
        hashed_password=hash_password(data.password),
        role=Role.CLINICIAN,
    )
    return await repo.create(user)


async def admin_create_user(
    repo: UserRepository,
    data: AdminUserCreate,
    actor: User,
) -> User:
    """Create a user on behalf of an admin actor.

    Business rules:
    - ``SUPER_ADMIN`` can create any role, must provide ``hospital_id`` for
      non-super-admin roles.
    - ``HOSPITAL_ADMIN`` can only create ``CLINICIAN``s; ``hospital_id`` is
      automatically inherited from the actor — the request value is ignored.

    Raises:
        ``HTTP 403`` if the actor cannot assign the requested role.
        ``HTTP 422`` if SUPER_ADMIN omits ``hospital_id`` for a non-super role.
        ``HTTP 409`` if the email is already taken.
    """
    allowed = _CREATABLE_ROLES.get(actor.role, set())
    if data.role not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"A {actor.role.value} cannot create a user with role '{data.role.value}'",
        )

    # Determine the hospital_id for the new user
    if actor.role == Role.HOSPITAL_ADMIN:
        # Always inherit from the acting admin — cannot be overridden
        hospital_id = actor.hospital_id
    elif data.role == Role.SUPER_ADMIN:
        # Super admins belong to no hospital
        hospital_id = None
    else:
        # SUPER_ADMIN creating a HOSPITAL_ADMIN or CLINICIAN must specify a hospital
        if data.hospital_id is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"hospital_id is required when creating a '{data.role.value}'",
            )
        hospital_id = data.hospital_id

    if await repo.email_exists(data.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists",
        )

    user = User(
        email=data.email,
        full_name=data.full_name,
        hashed_password=hash_password(data.password),
        role=data.role,
        hospital_id=hospital_id,
    )
    return await repo.create(user)


def generate_tokens(user: User) -> TokenPair:
    """Generate an access + refresh token pair for *user*."""
    payload = {"sub": user.email}
    return TokenPair(
        access_token=create_access_token(payload),
        refresh_token=create_refresh_token(payload),
    )
