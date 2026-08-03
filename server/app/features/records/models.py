import uuid
from datetime import datetime

from sqlalchemy import JSON, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ClinicalRecord(Base):
    __tablename__ = "clinical_records"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), index=True
    )
    masked_text: Mapped[str] = mapped_column(Text)
    symptoms: Mapped[dict | None] = mapped_column(JSON, default=None)
    blood_pressure_sys: Mapped[int | None] = mapped_column(default=None)
    blood_pressure_dia: Mapped[int | None] = mapped_column(default=None)
    cholesterol_mg_dl: Mapped[float | None] = mapped_column(default=None)
    fasting_bs_mg_dl: Mapped[float | None] = mapped_column(default=None)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ECGRecord(Base):
    __tablename__ = "ecg_records"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), index=True
    )
    signal_file_path: Mapped[str] = mapped_column()
    sampling_rate_hz: Mapped[int] = mapped_column(default=500)
    lead_count: Mapped[int] = mapped_column(default=12)
    duration_seconds: Mapped[float] = mapped_column(default=10.0)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
