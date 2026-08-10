"""Pydantic v2 schemas for Federated Learning orchestration."""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class FLStatusResponse(BaseModel):
    current_round: int
    total_rounds: int = 10
    status: str = "IN_PROGRESS"
    global_accuracy: float | None = None
    global_loss: float | None = None
    participating_hospitals_count: int = 0


class FLStartRoundRequest(BaseModel):
    num_rounds: int = Field(default=5, ge=1, le=100)
    min_clients: int = Field(default=2, ge=1, le=50)


class FLStartRoundResponse(BaseModel):
    message: str
    round_id: uuid.UUID
    started_at: datetime
