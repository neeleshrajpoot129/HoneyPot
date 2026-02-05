"""Session state models."""
from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from app.models.intelligence import ExtractedIntelligence


class Message(BaseModel):
    """Represents a single message in the conversation."""
    sender: str = Field(..., description="Message sender: 'scammer' or 'user'")
    text: str = Field(..., description="Message content")
    timestamp: Union[int, str] = Field(..., description="Timestamp as Unix timestamp (int) or ISO-8601 string")
    
    @field_validator('timestamp', mode='before')
    @classmethod
    def normalize_timestamp(cls, v):
        """Normalize timestamp to string format for internal use."""
        if isinstance(v, int):
            # Convert Unix timestamp (milliseconds) to ISO-8601 string
            try:
                dt = datetime.fromtimestamp(v / 1000.0)
                return dt.isoformat() + 'Z'
            except (ValueError, OSError):
                # Fallback: use current time if conversion fails
                return datetime.now().isoformat() + 'Z'
        elif isinstance(v, str):
            # Already a string, return as-is
            return v
        else:
            # Unknown type, convert to string
            return str(v)


class Metadata(BaseModel):
    """Metadata about the conversation."""
    channel: Optional[str] = Field(None, description="SMS / WhatsApp / Email / Chat")
    language: Optional[str] = Field(None, description="Language used")
    locale: Optional[str] = Field(None, description="Country or region")


class SessionState(BaseModel):
    """Internal session state management."""
    sessionId: str
    conversationHistory: List[Message] = Field(default_factory=list)
    extractedIntelligence: ExtractedIntelligence = Field(default_factory=ExtractedIntelligence)
    scamDetected: bool = False
    scamConfidence: float = 0.0
    totalMessagesExchanged: int = 0
    agentNotes: List[str] = Field(default_factory=list)
    conversationEnded: bool = Field(default=False, description="True when conversation is completed")
    callbackSent: bool = Field(default=False, description="True when callback has been sent")
    createdAt: datetime = Field(default_factory=datetime.now)
    lastUpdated: datetime = Field(default_factory=datetime.now)
    
    # Audit fields for hybrid scam detection
    scamRuleScore: Optional[float] = Field(default=None, description="Rule-based confidence score")
    scamRuleEvidence: List[str] = Field(default_factory=list, description="List of rules that matched")
    llmFallbackUsed: bool = Field(default=False, description="Whether LLM fallback was used")
    llmScamResult: Optional[Dict] = Field(default=None, description="LLM fallback result (JSON)")
    finalDecisionReason: Optional[str] = Field(default=None, description="Explanation for final scam decision")