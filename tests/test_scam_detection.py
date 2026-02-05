"""Tests for scam detection."""
import pytest
from app.models.session_state import Message
from app.core.scam_detector import scam_detector


def test_scam_detection_urgent_message():
    """Test detection of urgent scam message."""
    message = Message(
        sender="scammer",
        text="Your bank account will be blocked today. Verify immediately.",
        timestamp="2026-01-21T10:15:30Z"
    )
    
    is_scam, confidence = scam_detector.detect_scam(message, [])
    
    assert is_scam is True
    assert confidence > 0.5


def test_scam_detection_normal_message():
    """Test that normal messages are not detected as scams."""
    message = Message(
        sender="user",
        text="Hello, how are you?",
        timestamp="2026-01-21T10:15:30Z"
    )
    
    is_scam, confidence = scam_detector.detect_scam(message, [])
    
    assert is_scam is False
    assert confidence < 0.5


def test_scam_detection_with_upi_request():
    """Test detection of UPI ID request."""
    message = Message(
        sender="scammer",
        text="Share your UPI ID to avoid account suspension.",
        timestamp="2026-01-21T10:15:30Z"
    )
    
    is_scam, confidence = scam_detector.detect_scam(message, [])
    
    assert is_scam is True
    assert confidence > 0.5
