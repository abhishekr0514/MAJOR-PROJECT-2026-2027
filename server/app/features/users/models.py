from enum import Enum
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class Role(str, Enum):
    SUPER_ADMIN = "super_admin"
    HOSPITAL_ADMIN = "hospital_admin"
    CLINICIAN = "clinician"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column()
    role: Mapped[Role] = mapped_column(default=Role.CLINICIAN)
    is_active: Mapped[bool] = mapped_column(default=True)
