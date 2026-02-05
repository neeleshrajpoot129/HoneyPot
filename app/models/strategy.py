"""Strategy decision models."""
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ConversationGoal(str, Enum):
    """Conversation goals determined by strategy agent."""
    CLARIFY = "clarify"  # Ask for clarification
    DELAY = "delay"  # Delay and create confusion
    ESCALATE = "escalate"  # Escalate concern
    WRAP_UP = "wrap_up"  # End conversation gracefully
    CONTINUE = "continue"  # Continue normal engagement


class StrategyDecision(BaseModel):
    """Decision made by strategy agent."""
    should_engage: bool = Field(..., description="Whether to engage at all")
    goal: ConversationGoal = Field(..., description="Conversation goal")
    reasoning: str = Field(..., description="Reason for this decision")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for persona agent")
