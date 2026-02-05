"""Service for sending callbacks to GUVI evaluation endpoint."""
import requests
from typing import List
from app.models.session_state import SessionState, Message
from app.models.intelligence import GuviCallbackPayload, ExtractedIntelligence
from app.config import config
from app.utils.logger import logger


class CallbackService:
    """Handles callbacks to GUVI evaluation endpoint."""
    
    def __init__(self):
        self.callback_url = config.GUVI_CALLBACK_URL
        self.sent_callbacks = set()  # Track sent callbacks to avoid duplicates
    
    def generate_agent_notes_summary(
        self,
        session: SessionState,
        conversation_history: List[Message],
        intelligence: ExtractedIntelligence
    ) -> str:
        """
        Generate comprehensive summary explaining why the agent thought it was a scam.
        
        This summary includes:
        - Conversation overview
        - Scam detection reasoning
        - Extracted intelligence (especially phone numbers)
        - Key indicators that led to scam classification
        """
        summary_parts = []
        
        # 1. Conversation Overview
        summary_parts.append("CONVERSATION SUMMARY:")
        summary_parts.append(f"Total messages exchanged: {session.totalMessagesExchanged}")
        summary_parts.append(f"Scam detected with confidence: {session.scamConfidence:.2f}")
        
        # 2. Detection Reasoning
        if session.finalDecisionReason:
            summary_parts.append(f"\nDETECTION REASONING:")
            summary_parts.append(session.finalDecisionReason)
        
        # 3. Key Conversation Points
        if conversation_history:
            summary_parts.append("\nKEY CONVERSATION POINTS:")
            # Show first and last few messages to understand context
            scammer_messages = [msg for msg in conversation_history if msg.sender == "scammer"]
            if scammer_messages:
                if len(scammer_messages) <= 3:
                    # Show all if few messages
                    for i, msg in enumerate(scammer_messages, 1):
                        summary_parts.append(f"  Message {i}: {msg.text[:150]}{'...' if len(msg.text) > 150 else ''}")
                else:
                    # Show first and last messages
                    summary_parts.append(f"  Initial message: {scammer_messages[0].text[:150]}{'...' if len(scammer_messages[0].text) > 150 else ''}")
                    summary_parts.append(f"  Latest message: {scammer_messages[-1].text[:150]}{'...' if len(scammer_messages[-1].text) > 150 else ''}")
        
        # 4. Extracted Intelligence (with emphasis on phone numbers)
        summary_parts.append("\nEXTRACTED INTELLIGENCE:")
        
        # Phone numbers - emphasized
        if intelligence.phoneNumbers:
            summary_parts.append(f"  Phone Numbers ({len(intelligence.phoneNumbers)}): {', '.join(intelligence.phoneNumbers)}")
        else:
            summary_parts.append("  Phone Numbers: None detected")
        
        # Other intelligence
        if intelligence.upiIds:
            summary_parts.append(f"  UPI IDs ({len(intelligence.upiIds)}): {', '.join(intelligence.upiIds[:5])}{'...' if len(intelligence.upiIds) > 5 else ''}")
        
        if intelligence.bankAccounts:
            summary_parts.append(f"  Bank Accounts ({len(intelligence.bankAccounts)}): {len(intelligence.bankAccounts)} account(s) detected")
        
        if intelligence.phishingLinks:
            summary_parts.append(f"  Phishing Links ({len(intelligence.phishingLinks)}): {', '.join(intelligence.phishingLinks[:3])}{'...' if len(intelligence.phishingLinks) > 3 else ''}")
        
        if intelligence.suspiciousKeywords:
            summary_parts.append(f"  Suspicious Keywords: {', '.join(intelligence.suspiciousKeywords[:10])}{'...' if len(intelligence.suspiciousKeywords) > 10 else ''}")
        
        # 5. Why it's a scam - synthesize from all available information
        summary_parts.append("\nSCAM INDICATORS IDENTIFIED:")
        scam_indicators = []
        
        if intelligence.phoneNumbers:
            scam_indicators.append(f"Phone numbers shared ({len(intelligence.phoneNumbers)})")
        
        if intelligence.upiIds:
            scam_indicators.append(f"UPI IDs requested/shared ({len(intelligence.upiIds)})")
        
        if intelligence.phishingLinks:
            scam_indicators.append(f"Suspicious links provided ({len(intelligence.phishingLinks)})")
        
        if intelligence.bankAccounts:
            scam_indicators.append("Bank account information requested")
        
        # Check for urgency/threats in conversation
        conversation_text = " ".join([msg.text.lower() for msg in conversation_history])
        if any(keyword in conversation_text for keyword in ['urgent', 'immediately', 'blocked', 'suspended', 'frozen']):
            scam_indicators.append("Urgency/threat language detected")
        
        if any(keyword in conversation_text for keyword in ['verify', 'confirm', 'share', 'send', 'provide']):
            scam_indicators.append("Requests for verification/sensitive information")
        
        if scam_indicators:
            summary_parts.append("  " + "; ".join(scam_indicators))
        else:
            summary_parts.append("  General scam patterns detected based on conversation analysis")
        
        # 6. Additional agent notes if available
        if session.agentNotes:
            summary_parts.append("\nADDITIONAL AGENT OBSERVATIONS:")
            for note in session.agentNotes[-5:]:  # Last 5 notes
                summary_parts.append(f"  - {note}")
        
        return "\n".join(summary_parts)
    
    def send_callback(self, session: SessionState) -> bool:
        """
        Send intelligence callback to GUVI endpoint.
        
        Returns:
            True if successful, False otherwise
        """
        # Avoid duplicate callbacks
        if session.sessionId in self.sent_callbacks:
            return True
        
        # Generate comprehensive agent notes summary
        agent_notes_summary = self.generate_agent_notes_summary(
            session,
            session.conversationHistory,
            session.extractedIntelligence
        )
        
        # Prepare payload
        payload = GuviCallbackPayload(
            sessionId=session.sessionId,
            scamDetected=session.scamDetected,
            totalMessagesExchanged=session.totalMessagesExchanged,
            extractedIntelligence=session.extractedIntelligence,
            agentNotes=agent_notes_summary
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
