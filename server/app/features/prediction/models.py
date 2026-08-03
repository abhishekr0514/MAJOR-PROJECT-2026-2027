import uuid
from datetime import datetime

from sqlalchemy import JSON, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), index=True
    )
    risk_score: Mapped[float] = mapped_column()
    diagnosis: Mapped[str] = mapped_column()
    xai_counterfactuals: Mapped[dict | None] = mapped_column(JSON, default=None)
    causal_impact: Mapped[dict | None] = mapped_column(JSON, default=None)
    model_version: Mapped[str] = mapped_column(default="1.0.0")

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
