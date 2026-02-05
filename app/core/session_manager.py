"""Session management for conversation state."""
from typing import Dict, Optional
from datetime import datetime
from app.models.session_state import SessionState, Message
from app.models.intelligence import ExtractedIntelligence


class SessionManager:
    """Manages conversation sessions and state."""
    
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
    
    def get_or_create_session(self, session_id: str) -> SessionState:
        """Get existing session or create a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(
                sessionId=session_id,
                conversationHistory=[],
                extractedIntelligence=ExtractedIntelligence(),
                scamDetected=False,
                scamConfidence=0.0,
                totalMessagesExchanged=0
            )
        return self.sessions[session_id]
    
    def update_session(
        self,
        session_id: str,
        new_message: Message,
        scam_detected: Optional[bool] = None,
        scam_confidence: Optional[float] = None,
        intelligence: Optional[ExtractedIntelligence] = None
    ) -> SessionState:
        """Update session with new message and state."""
        session = self.get_or_create_session(session_id)
        
        # Add new message to history
        session.conversationHistory.append(new_message)
        session.totalMessagesExchanged += 1
        session.lastUpdated = datetime.now()
        
        # Update scam detection status
        if scam_detected is not None:
            session.scamDetected = scam_detected
        if scam_confidence is not None:
            session.scamConfidence = scam_confidence
        
        # Merge extracted intelligence
        if intelligence:
            session.extractedIntelligence.bankAccounts.extend(
                intelligence.bankAccounts
            )
            session.extractedIntelligence.upiIds.extend(intelligence.upiIds)
            session.extractedIntelligence.phishingLinks.extend(
                intelligence.phishingLinks
            )
            session.extractedIntelligence.phoneNumbers.extend(
                intelligence.phoneNumbers
            )
            session.extractedIntelligence.suspiciousKeywords.extend(
                intelligence.suspiciousKeywords
            )
            # Remove duplicates
            session.extractedIntelligence.bankAccounts = list(
                set(session.extractedIntelligence.bankAccounts)
            )
            session.extractedIntelligence.upiIds = list(
                set(session.extractedIntelligence.upiIds)
            )
            session.extractedIntelligence.phishingLinks = list(
                set(session.extractedIntelligence.phishingLinks)
            )
            session.extractedIntelligence.phoneNumbers = list(
                set(session.extractedIntelligence.phoneNumbers)
            )
            session.extractedIntelligence.suspiciousKeywords = list(
                set(session.extractedIntelligence.suspiciousKeywords)
            )
        
        return session
    
    def add_agent_note(self, session_id: str, note: str):
        """Add a note about agent observations."""
        session = self.get_or_create_session(session_id)
        session.agentNotes.append(note)
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = [
            sid for sid, session in self.sessions.items()
            if session.lastUpdated < cutoff
        ]
        
        for sid in to_remove:
            del self.sessions[sid]


# Global session manager instance
session_manager = SessionManager()
