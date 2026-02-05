"""Service for sending callbacks to GUVI evaluation endpoint."""
import requests
from app.models.session_state import SessionState
from app.models.intelligence import GuviCallbackPayload
from app.config import config
from app.utils.logger import logger


class CallbackService:
    """Handles callbacks to GUVI evaluation endpoint."""
    
    def __init__(self):
        self.callback_url = config.GUVI_CALLBACK_URL
        self.sent_callbacks = set()  # Track sent callbacks to avoid duplicates
    
    def send_callback(self, session: SessionState) -> bool:
        """
        Send intelligence callback to GUVI endpoint.
        
        Returns:
            True if successful, False otherwise
        """
        # Avoid duplicate callbacks
        if session.sessionId in self.sent_callbacks:
            return True
        
        # Prepare payload
            # Format agent notes
            if session.agentNotes:
                agent_notes_text = "; ".join(session.agentNotes)
            else:
                # Generate basic notes if none exist
                notes_parts = []
                if session.scamDetected:
                    notes_parts.append(f"Scam detected (confidence: {session.scamConfidence:.2f})")
                if session.extractedIntelligence.bankAccounts:
                    notes_parts.append(f"Bank accounts extracted: {len(session.extractedIntelligence.bankAccounts)}")
                if session.extractedIntelligence.phoneNumbers:
                    notes_parts.append(f"Phone numbers extracted: {len(session.extractedIntelligence.phoneNumbers)}")
                if session.extractedIntelligence.upiIds:
                    notes_parts.append(f"UPI IDs extracted: {len(session.extractedIntelligence.upiIds)}")
                agent_notes_text = "; ".join(notes_parts) if notes_parts else "Scam detected but no specific notes recorded"
            
            payload = GuviCallbackPayload(
                sessionId=session.sessionId,
                scamDetected=session.scamDetected,
                totalMessagesExchanged=session.totalMessagesExchanged,
                extractedIntelligence=session.extractedIntelligence,
                agentNotes=agent_notes_text
            )
        
        # Print the request that will be sent to GUVI
        import json
        payload_dict = payload.model_dump()
        
        print("\n" + "="*80)
        print("📤 GUVI CALLBACK REQUEST")
        print("="*80)
        print(f"URL: {self.callback_url}")
        print(f"Method: POST")
        print(f"Headers: {{'Content-Type': 'application/json'}}")
        print("\nPayload:")
        print(json.dumps(payload_dict, indent=2, ensure_ascii=False))
        print("="*80 + "\n")
        
        logger.info("="*80)
        logger.info("📤 GUVI CALLBACK REQUEST")
        logger.info("="*80)
        logger.info(f"URL: {self.callback_url}")
        logger.info("Method: POST")
        logger.info("Headers: {'Content-Type': 'application/json'}")
        logger.info("Payload:\n" + json.dumps(payload_dict, indent=2, ensure_ascii=False))
        logger.info("="*80)
        
        # Send callback to GUVI endpoint
        try:
            response = requests.post(
                self.callback_url,
                json=payload_dict,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            # Print response
            print(f"Response Status: {response.status_code}")
            print(f"Response Body: {response.text[:500]}")
            print("="*80 + "\n")
            
            if response.status_code in [200, 201]:
                self.sent_callbacks.add(session.sessionId)
                session.callbackSent = True
                logger.info(f"✅ Callback sent successfully for session {session.sessionId}")
                logger.info(f"Response Status: {response.status_code}")
                logger.info(f"Response Body: {response.text[:500]}")
                return True
            else:
                logger.error(f"❌ Callback failed with status {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error sending callback: {e}", exc_info=True)
            print(f"❌ Error sending callback: {e}")
            print("="*80 + "\n")
            return False
    
    def should_send_callback(self, session: SessionState) -> bool:
        """
        Check if callback should be sent for this session.
        
        Callback should ONLY be sent when:
        1. Conversation has ended (conversationEnded = True)
        2. Scam was detected
        3. Minimum messages exchanged
        4. We have meaningful intelligence
        5. Callback hasn't been sent yet
        """
        # Don't send if already sent
        if session.sessionId in self.sent_callbacks or session.callbackSent:
            return False
        
        # Only send when conversation has ended
        if not session.conversationEnded:
            return False
        
        if not session.scamDetected:
            return False
        
        if session.totalMessagesExchanged < config.MIN_MESSAGES_FOR_CALLBACK:
            return False
        
        # Check if we have meaningful intelligence
        intelligence = session.extractedIntelligence
        has_intelligence = (
            len(intelligence.bankAccounts) > 0 or
            len(intelligence.upiIds) > 0 or
            len(intelligence.phishingLinks) > 0 or
            len(intelligence.phoneNumbers) > 0 or
            len(intelligence.suspiciousKeywords) > 0
        )
        
        return has_intelligence


# Global callback service instance
callback_service = CallbackService()
