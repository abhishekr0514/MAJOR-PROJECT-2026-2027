import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class FLRound(Base):
    __tablename__ = "fl_rounds"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    round_number: Mapped[int] = mapped_column(unique=True, index=True)
    global_accuracy: Mapped[float | None] = mapped_column(default=None)
    global_loss: Mapped[float | None] = mapped_column(default=None)
    weights_path: Mapped[str | None] = mapped_column(default=None)
    status: Mapped[str] = mapped_column(default="IN_PROGRESS")

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class FLModelUpdate(Base):
    __tablename__ = "fl_model_updates"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    fl_round_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("fl_rounds.id", ondelete="CASCADE")
    )
    hospital_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("hospitals.id", ondelete="CASCADE")
    )
    samples_count: Mapped[int] = mapped_column()
    local_loss: Mapped[float] = mapped_column()

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
