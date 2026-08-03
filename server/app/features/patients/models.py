import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.features.hospitals.models import Hospital


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    patient_code: Mapped[str] = mapped_column(unique=True, index=True)
    age: Mapped[int] = mapped_column()
    gender: Mapped[str] = mapped_column()

    hospital_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("hospitals.id", ondelete="CASCADE")
    )
    hospital: Mapped["Hospital"] = relationship()

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
