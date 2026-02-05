"""Strategy agent - decides conversation strategy and when to engage/terminate."""
from typing import Optional
from groq import Groq
from app.models.session_state import Message, SessionState
from app.models.strategy import StrategyDecision, ConversationGoal
from app.config import config
from app.utils.logger import logger
from app.utils.prompts import StrategyAgentPrompts


class StrategyAgent:
    """Decides conversation strategy and when to engage/terminate."""
    
    def __init__(self):
        # Lazy initialization of Groq client for LLM-based conversation end detection
        self._groq_client: Optional[Groq] = None
    
    def decide_strategy(
        self,
        session: SessionState,
        message: Message
    ) -> StrategyDecision:
        """
        Decide conversation strategy and goal.
        
        This is the PLANNER - determines what the agent should do next.
        """
        # Don't continue if max messages reached
        if session.totalMessagesExchanged >= config.MAX_MESSAGES_PER_SESSION:
            return StrategyDecision(
                should_engage=False,
                goal=ConversationGoal.WRAP_UP,
                reasoning="Maximum messages per session reached"
            )
        
        # If not a scam, don't engage
        if not session.scamDetected:
            return StrategyDecision(
                should_engage=False,
                goal=ConversationGoal.WRAP_UP,
                reasoning="No scam detected"
            )
        
        # Analyze intelligence to determine strategy
        intelligence = session.extractedIntelligence
        has_intelligence = (
            len(intelligence.bankAccounts) > 0 or
            len(intelligence.upiIds) > 0 or
            len(intelligence.phishingLinks) > 0 or
            len(intelligence.phoneNumbers) > 0
        )
        
        # Determine conversation goal based on context
        goal = self._determine_goal(message, session, intelligence, has_intelligence)
        
        # Check if conversation should end (only AFTER we know it's a scam and we're engaged)
        # Only check for end signals if we have enough messages and intelligence, OR if goal is already WRAP_UP
        if goal == ConversationGoal.WRAP_UP:
            # Already decided to wrap up based on intelligence/context - check for explicit end signals
            if config.USE_LLM_FOR_CONVERSATION_END and session.totalMessagesExchanged > 1:
                should_end = self._llm_detect_conversation_end(message, session)
                if should_end:
                    return StrategyDecision(
                        should_engage=False,
                        goal=ConversationGoal.WRAP_UP,
                        reasoning="LLM detected conversation should end"
                    )
            elif session.totalMessagesExchanged > 1:
                # Use static keyword check
                should_end = self._static_keyword_check(message)
                if should_end:
                    return StrategyDecision(
                        should_engage=False,
                        goal=ConversationGoal.WRAP_UP,
                        reasoning="Static keywords detected conversation should end"
                    )
        
        # Build minimal context for persona agent (behavior hint only, no intelligence)
        # Persona agent should NOT know about intelligence - it just acts
        context = {
            "behavior_hint": goal.value  # Just the goal, persona will get behavior hint from prompts
        }
        
        return StrategyDecision(
            should_engage=True,
            goal=goal,
            reasoning=self._get_reasoning(goal, message, has_intelligence),
            context=context
        )
    
    def _determine_goal(
        self,
        message: Message,
        session: SessionState,
        intelligence,
        has_intelligence: bool
    ) -> ConversationGoal:
        """Determine the conversation goal based on current state."""
        text_lower = message.text.lower()
        
        # If we need more intelligence, try to extract it
        if not has_intelligence or session.totalMessagesExchanged < config.MIN_MESSAGES_FOR_CALLBACK:
            # Check what the scammer is asking for
            if "upi" in text_lower or "upi id" in text_lower:
                return ConversationGoal.CLARIFY  # Ask questions to delay
            elif "link" in text_lower or "click" in text_lower or "verify" in text_lower:
                return ConversationGoal.CLARIFY  # Ask for clarification
            elif "urgent" in text_lower or "immediately" in text_lower:
                return ConversationGoal.DELAY  # Create delay/confusion
            else:
                return ConversationGoal.CONTINUE  # Normal engagement
        
        # If intelligence already extracted, escalate concern (more emotionally realistic)
        if has_intelligence and session.totalMessagesExchanged >= 2:
            if "upi" in text_lower or "send" in text_lower:
                return ConversationGoal.ESCALATE
        
        # If we have intelligence but need more engagement
        if session.totalMessagesExchanged < config.MIN_MESSAGES_FOR_CALLBACK:
            if "upi" in text_lower or "account" in text_lower:
                return ConversationGoal.ESCALATE  # Show more concern
            else:
                return ConversationGoal.CONTINUE
        
        # We have enough intelligence, can wrap up
        return ConversationGoal.WRAP_UP
    
    def _get_reasoning(
        self,
        goal: ConversationGoal,
        message: Message,
        has_intelligence: bool
    ) -> str:
        """Get human-readable reasoning for the decision."""
        text_lower = message.text.lower()
        
        if goal == ConversationGoal.CLARIFY:
            if "upi" in text_lower:
                return "Need to extract UPI ID - asking clarifying questions to delay"
            elif "link" in text_lower:
                return "Phishing link detected - asking for more information"
            else:
                return "Need more intelligence - asking clarifying questions"
        
        elif goal == ConversationGoal.DELAY:
            return "Creating delay to extract more intelligence while maintaining engagement"
        
        elif goal == ConversationGoal.ESCALATE:
            return "Showing increased concern to maintain engagement and extract more intelligence"
        
        elif goal == ConversationGoal.CONTINUE:
            return "Continuing normal engagement to extract intelligence"
        
        else:  # WRAP_UP
            return "Sufficient intelligence gathered or conversation should end"
    
    def should_continue_engagement(
        self,
        session: SessionState,
        message: Message
    ) -> bool:
        """Legacy method for backward compatibility."""
        decision = self.decide_strategy(session, message)
        return decision.should_engage
    
    def _get_groq_client(self) -> Optional[Groq]:
        """Get or initialize Groq client for conversation end detection."""
        if self._groq_client is None and config.GROQ_API_KEY:
            try:
                self._groq_client = Groq(api_key=config.GROQ_API_KEY)
            except Exception as e:
                logger.warning(f"StrategyAgent: Failed to initialize Groq: {e}")
                return None
        return self._groq_client
    
    def _llm_detect_conversation_end(
        self,
        message: Message,
        session: SessionState
    ) -> bool:
        """
        Use LLM to detect if conversation should end.
        
        This replaces static keyword matching with intelligent LLM-based detection.
        Falls back to static keywords if LLM is not available.
        """
        # Get Groq client
        groq_client = self._get_groq_client()
        
        # If Groq is not available, fall back to static keyword matching
        if not groq_client:
            return self._static_keyword_check(message)
        
        # Build context for Groq
        conversation_summary = self._build_conversation_summary(session)
        
        # Create prompt for conversation end detection
        # CRITICAL: Only end if scammer explicitly ends, NOT on active scam attempts
        message_lower = message.text.lower()
        
        # Hard rule: Never end on active scam keywords (asking for action/info)
        # If message contains active scam keywords, NEVER end
        if any(keyword in message_lower for keyword in StrategyAgentPrompts.ACTIVE_SCAM_KEYWORDS):
            logger.info(f"Active scam keywords detected in '{message.text}' - NOT ending conversation")
            return False
        
        # Get prompt from centralized prompts module
        prompt = StrategyAgentPrompts.get_conversation_end_detection_prompt(
            message_text=message.text,
            message_count=session.totalMessagesExchanged,
            upi_count=len(session.extractedIntelligence.upiIds),
            link_count=len(session.extractedIntelligence.phishingLinks)
        )
        
        try:
            response = groq_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Very low temperature for deterministic thinking
                max_tokens=10,  # Binary output only
                top_p=0.5  # Focused output
            )
            
            decision = response.choices[0].message.content.strip().upper()
            should_end = decision == "YES"
            
            logger.info(
                f"Groq conversation end detection for session {session.sessionId}: "
                f"{decision} (should_end={should_end})"
            )
            
            return should_end
            
        except Exception as e:
            logger.warning(f"Groq conversation end detection failed, using fallback: {e}")
            return self._static_keyword_check(message)
    
    def _static_keyword_check(self, message: Message) -> bool:
        """Fallback to static keyword matching if LLM is not available."""
        text_lower = message.text.lower()
        end_keywords = config.CONVERSATION_END_KEYWORDS
        
        return any(keyword in text_lower for keyword in end_keywords)
    
    def _build_conversation_summary(self, session: SessionState) -> str:
        """Build a summary of the conversation for LLM context."""
        if not session.conversationHistory:
            return "No previous messages"
        
        # Get last few messages for context
        recent_messages = session.conversationHistory[-5:]
        summary = "Recent messages:\n"
        
        for msg in recent_messages:
            summary += f"- {msg.sender}: {msg.text}\n"
        
        return summary
    
    def should_send_callback(self, session: SessionState) -> bool:
        """Determine if we should send callback to GUVI."""
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
