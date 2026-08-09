
from pydantic import BaseModel, Field


class FLRoundStartSchema(BaseModel):
    num_rounds: int = Field(default=5, ge=1, le=100)
    min_clients: int = Field(default=2, ge=1, le=10)


class FLStatusResponseSchema(BaseModel):
    current_round: int
    total_rounds: int
    status: str
    global_accuracy: float | None = None
    participating_hospitals_count: int
