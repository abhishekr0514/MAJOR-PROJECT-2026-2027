"""Hospital model — a federated node."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.features.users.models import User


class Hospital(Base):
    __tablename__ = "hospitals"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(unique=True, index=True)
    license_code: Mapped[str] = mapped_column(unique=True, index=True)
    address: Mapped[str | None] = mapped_column(default=None)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Reverse relation — users that belong to this hospital
    users: Mapped[list["User"]] = relationship(back_populates="hospital")  # type: ignore[name-defined]

    def __repr__(self) -> str:
        return f"<Hospital {self.name}>"
