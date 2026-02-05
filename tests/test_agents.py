"""Tests for agents."""
import pytest
from app.models.session_state import Message, SessionState
from app.models.intelligence import ExtractedIntelligence
from app.agents.persona_agent import PersonaAgent
from app.agents.strategy_agent import StrategyAgent
from app.agents.safety_guard import safety_guard


def test_persona_agent_fallback():
    """Test persona agent fallback responses."""
    agent = PersonaAgent()
    
    message = Message(
        sender="scammer",
        text="Your account will be blocked.",
        timestamp="2026-01-21T10:15:30Z"
    )
    
    response = agent._fallback_response(message, [])
    
    assert response is not None
    assert len(response) > 0
    assert "blocked" in response.lower() or "why" in response.lower()


def test_strategy_agent_should_continue():
    """Test strategy agent engagement decision."""
    agent = StrategyAgent()
    
    session = SessionState(
        sessionId="test-123",
        scamDetected=True,
        totalMessagesExchanged=3
    )
    
    message = Message(
        sender="scammer",
        text="Share your UPI ID.",
        timestamp="2026-01-21T10:15:30Z"
    )
    
    should_continue = agent.should_continue_engagement(session, message)
    
    assert should_continue is True


def test_safety_guard_validation():
    """Test safety guard response validation."""
    # Valid response
    is_valid, error = safety_guard.validate_response("Why is my account blocked?")
    assert is_valid is True
    assert error is None
    
    # Invalid response (reveals detection)
    is_valid, error = safety_guard.validate_response("I am an AI detection system")
    assert is_valid is False
    assert error is not None
    
    # Too long response
    long_response = "a" * 501
    is_valid, error = safety_guard.validate_response(long_response)
    assert is_valid is False
    assert error is not None
