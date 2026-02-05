"""API request/response schemas."""
from typing import List, Optional
from pydantic import BaseModel, Field
from app.models.session_state import Message, Metadata


class HoneypotRequest(BaseModel):
    """Request model for the honeypot API."""
    sessionId: str = Field(..., description="Unique session identifier")
    message: Message = Field(..., description="Latest incoming message")
    conversationHistory: List[Message] = Field(
        default_factory=list,
        description="Previous messages in the conversation"
    )
    metadata: Optional[Metadata] = Field(None, description="Conversation metadata")


class HoneypotResponse(BaseModel):
    """Response model for the honeypot API."""
    status: str = Field(..., description="Response status: 'success' or 'error'")
    reply: Optional[str] = Field(None, description="Agent's reply message")
    error: Optional[str] = Field(None, description="Error message if status is 'error'")
