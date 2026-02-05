"""
Agent orchestrator - coordinates multiple AI agents with proper execution model.

Execution Model:
1. Intelligence Extraction Agent - PARALLEL (non-blocking, read-only)
2. Strategy Agent - SEQUENTIAL (decides intent/goal)
3. Persona Agent - SEQUENTIAL (expresses intent)
4. Safety Guard - SEQUENTIAL (final gatekeeper)
"""
import asyncio
import json
from typing import Optional
from groq import Groq
from app.models.session_state import Message, SessionState
from app.models.strategy import StrategyDecision
from app.core.session_manager import session_manager
from app.core.intelligence_aggregator import intelligence_aggregator
from app.agents.persona_agent import PersonaAgent
from app.agents.strategy_agent import StrategyAgent
from app.agents.safety_guard import safety_guard
from app.utils.prompts import AgentNotesPrompts
from app.config import config
from app.utils.logger import logger


class Orchestrator:
    """
    Orchestrates all agents to handle scam engagement.
    
    Implements agent-by-agent execution model:
    - Intelligence extraction runs in parallel (non-blocking)
    - Strategy, Persona, and Safety run sequentially
    """
    
    def __init__(self):
        self.persona_agent = PersonaAgent()
        self.strategy_agent = StrategyAgent()
        self._groq_client = None
        
        # Initialize Groq for LLM-based agent notes
        if config.GROQ_API_KEY:
            try:
                self._groq_client = Groq(api_key=config.GROQ_API_KEY)
            except Exception as e:
                logger.warning(f"Orchestrator: Failed to initialize Groq for agent notes: {e}")
                self._groq_client = None
    
    def process_message(
        self,
        message: Message,
        session: SessionState
    ) -> Optional[str]:
        """
        Process incoming message and generate response.
        
        Execution flow:
        1. PARALLEL: Extract intelligence (non-blocking)
        2. SEQUENTIAL: Strategy agent decides goal
        3. SEQUENTIAL: Persona agent expresses goal
        4. SEQUENTIAL: Safety guard validates response
        
        Returns:
            Response message if agent should engage, None otherwise
        """
        # ============================================================
        # STEP 1: INTELLIGENCE EXTRACTION (PARALLEL/NON-BLOCKING)
        # ============================================================
        # This runs early and conceptually in parallel
        # It's read-only and doesn't affect response generation
        intelligence = self._extract_intelligence_parallel(message, session)
        
        # Update session with extracted intelligence (non-blocking operation)
        session_manager.update_session(
            session.sessionId,
            message,
            intelligence=intelligence
        )
        
        # ============================================================
        # STEP 2: STRATEGY AGENT (SEQUENTIAL - PLANNER)
        # ============================================================
        # This agent DECIDES the conversation goal/intent
        # Must run before Persona Agent to avoid conflicting outputs
        strategy_decision = self.strategy_agent.decide_strategy(session, message)
        
        logger.info(
            f"Strategy decision for session {session.sessionId}: "
            f"goal={strategy_decision.goal.value}, "
            f"reasoning={strategy_decision.reasoning}"
        )
        
        # If strategy says don't engage (conversation ending), mark for callback
        if not strategy_decision.should_engage:
            # Conversation is ending - mark session for callback
            from app.models.strategy import ConversationGoal
            if strategy_decision.goal == ConversationGoal.WRAP_UP:
                session.conversationEnded = True
            return None
        
        # ============================================================
        # STEP 3: PERSONA AGENT (SEQUENTIAL - EXECUTOR)
        # ============================================================
        # This agent EXPRESSES the strategy, doesn't decide it
        # Runs after Strategy Agent to ensure consistent output
        response = self.persona_agent.generate_response(
            message,
            session.conversationHistory,
            strategy_decision
        )
        
        if not response:
            return None
        
        # ============================================================
        # STEP 4: SAFETY GUARD (SEQUENTIAL - GATEKEEPER)
        # ============================================================
        # This is the FINAL gate before output
        # Must run last to inspect final text
        is_valid, error = safety_guard.validate_response(response)
        
        if not is_valid:
            logger.warning(
                f"Safety guard blocked response for session {session.sessionId}: {error}"
            )
            # Use safe fallback
            response = "I'm not sure how to respond to that. Can you clarify?"
            session_manager.add_agent_note(
                session.sessionId,
                f"Safety guard triggered: {error}"
            )
        
        # Add agent notes about extracted intelligence
        self._add_intelligence_notes(session, intelligence)
        
        # Check if strategy indicates conversation should end
        # If goal is WRAP_UP, mark conversation as ended
        from app.models.strategy import ConversationGoal
        if strategy_decision.goal == ConversationGoal.WRAP_UP:
            session.conversationEnded = True
            logger.info(f"Conversation marked as ended for session {session.sessionId}")
        
        return response
    
    def _extract_intelligence_parallel(
        self,
        message: Message,
        session: SessionState
    ):
        """
        Extract intelligence in parallel (non-blocking).
        
        This is read-only and observational, so it:
        - Never blocks response generation
        - Runs on every message, including failures
        - Does not affect conversational flow
        """
        try:
            # Extract intelligence (fast, non-blocking operation)
            intelligence = intelligence_aggregator.extract_intelligence(
                message,
                session.conversationHistory
            )
            return intelligence
        except Exception as e:
            # Even if extraction fails, don't block the conversation
            logger.error(f"Intelligence extraction failed (non-blocking): {e}")
            from app.models.intelligence import ExtractedIntelligence
            return ExtractedIntelligence()  # Return empty intelligence
    
    def _add_intelligence_notes(
        self,
        session: SessionState,
        intelligence
    ):
        """
        Generate reasoning-based agent notes using LLM.
        
        Notes explain WHY it was detected as scam, not just what was extracted.
        This is a summary/reasoning of the entire conversation.
        """
        # Prepare intelligence summary
        intelligence_dict = {
            "bankAccounts": intelligence.bankAccounts,
            "phoneNumbers": intelligence.phoneNumbers,
            "upiIds": intelligence.upiIds,
            "phishingLinks": intelligence.phishingLinks,
            "suspiciousKeywords": intelligence.suspiciousKeywords
        }
        
        # Try LLM-based reasoning notes
        if self._groq_client:
            try:
                reasoning_note = self._generate_reasoning_notes(
                    session,
                    intelligence_dict
                )
                if reasoning_note:
                    session_manager.add_agent_note(session.sessionId, reasoning_note)
                    return
            except Exception as e:
                logger.warning(f"LLM agent notes generation failed: {e}. Using fallback.")
        
        # Fallback to rule-based notes if LLM fails
        self._generate_fallback_notes(session, intelligence_dict)
    
    def _generate_reasoning_notes(
        self,
        session: SessionState,
        intelligence_dict: dict
    ) -> Optional[str]:
        """Generate reasoning-based notes using LLM."""
        try:
            prompt = AgentNotesPrompts.get_agent_notes_prompt(
                conversation_history=session.conversationHistory,
                extracted_intelligence=intelligence_dict,
                scam_detection_reason=session.finalDecisionReason or "Scam detected",
                scam_confidence=session.scamConfidence
            )
            
            response = self._groq_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Low temperature for consistent reasoning
                max_tokens=200
            )
            
            reasoning = response.choices[0].message.content.strip()
            
            # Clean up the response
            if reasoning.startswith('"') and reasoning.endswith('"'):
                reasoning = reasoning[1:-1]
            
            logger.debug(f"Generated reasoning notes: {reasoning[:100]}...")
            return reasoning
            
        except Exception as e:
            logger.error(f"Error generating reasoning notes: {e}", exc_info=True)
            return None
    
    def _generate_fallback_notes(
        self,
        session: SessionState,
        intelligence_dict: dict
    ):
        """Fallback rule-based notes generation."""
        notes = []
        
        if session.finalDecisionReason:
            notes.append(f"Scam detected: {session.finalDecisionReason}")
        
        if intelligence_dict.get("bankAccounts"):
            notes.append(f"Extracted bank account(s): {', '.join(intelligence_dict['bankAccounts'])}")
        
        if intelligence_dict.get("phoneNumbers"):
            notes.append(f"Extracted phone number(s): {', '.join(intelligence_dict['phoneNumbers'])}")
        
        if intelligence_dict.get("upiIds"):
            notes.append(f"Extracted UPI ID(s): {', '.join(intelligence_dict['upiIds'])}")
        
        if intelligence_dict.get("phishingLinks"):
            notes.append(f"Extracted phishing link(s): {', '.join(intelligence_dict['phishingLinks'])}")
        
        # Add summary note
        if notes:
            summary = "; ".join(notes)
            session_manager.add_agent_note(session.sessionId, summary)


# Global orchestrator instance
orchestrator = Orchestrator()
