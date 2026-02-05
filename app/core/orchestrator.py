"""
Agent orchestrator - coordinates multiple AI agents with proper execution model.

Execution Model:
1. Intelligence Extraction Agent - PARALLEL (non-blocking, read-only)
2. Strategy Agent - SEQUENTIAL (decides intent/goal)
3. Persona Agent - SEQUENTIAL (expresses intent)
4. Safety Guard - SEQUENTIAL (final gatekeeper)
"""
import asyncio
from typing import Optional
from app.models.session_state import Message, SessionState
from app.models.strategy import StrategyDecision
from app.core.session_manager import session_manager
from app.core.intelligence_aggregator import intelligence_aggregator
from app.agents.persona_agent import PersonaAgent
from app.agents.strategy_agent import StrategyAgent
from app.agents.safety_guard import safety_guard
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
        """Add notes about extracted intelligence."""
        if intelligence.phishingLinks:
            session_manager.add_agent_note(
                session.sessionId,
                f"Extracted phishing link: {intelligence.phishingLinks[-1]}"
            )
        
        if intelligence.upiIds:
            session_manager.add_agent_note(
                session.sessionId,
                f"Extracted UPI ID: {intelligence.upiIds[-1]}"
            )
        
        if intelligence.phoneNumbers:
            session_manager.add_agent_note(
                session.sessionId,
                f"Extracted phone number: {intelligence.phoneNumbers[-1]}"
            )


# Global orchestrator instance
orchestrator = Orchestrator()
