from pydantic import BaseModel
from typing import List, Optional

class Observation(BaseModel):
    ticket_id: str
    customer_message: str
    conversation_history: List[str]
    status: str

class Action(BaseModel):
    action_type: str  # reply, escalate, close, ask_info
    message: Optional[str] = None

class Reward(BaseModel):
    score: float
    reason: str