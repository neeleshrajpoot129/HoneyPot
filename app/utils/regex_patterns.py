"""Regex patterns for intelligence extraction and scam detection."""
import re


class RegexPatterns:
    """Centralized regex patterns."""
    
    # Bank account patterns
    BANK_ACCOUNT = re.compile(r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b')
    
    # UPI ID patterns
    UPI_ID = re.compile(
        r'\b[\w\.-]+@(?:paytm|gpay|phonepe|ybl|axl|okicici|okaxis|okhdfcbank|oksbi|payzapp|upi)\b',
        re.IGNORECASE
    )
    
    # Phone number patterns (Indian) - handles dashes, spaces, dots
    # Matches: +91-9876543210, +91 9876543210, +919876543210, 9876543210, etc.
    # Pattern handles: +91 followed by optional separator, then 10 digits (with optional separators)
    PHONE_NUMBER = re.compile(r'\+91[-.\s]?\d{10}|\+91\d{10}|91[-.\s]?\d{10}|0[-.\s]?\d{10}|[6-9]\d{9}')
    
    # URL patterns
    URL = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    # Urgency patterns (true urgency only - removed "verify" and "account" to reduce false inflation)
    URGENCY_PATTERNS = [
        re.compile(r'\b(urgent|immediately|asap|right now|hurry|quickly)\b', re.IGNORECASE),
        re.compile(r'\b(blocked|suspended|frozen|locked|closed)\b', re.IGNORECASE),
        re.compile(r'\b(within|today|hours left|final notice)\b', re.IGNORECASE),
        re.compile(r'\b(will be|going to|about to)\b', re.IGNORECASE),
    ]
    
    # Contextual banking terms (low weight, separate from urgency)
    CONTEXTUAL_BANKING_TERMS = [
        re.compile(r'\b(verify|confirm|validate|authenticate)\b', re.IGNORECASE),
        re.compile(r'\b(account|bank|upi|payment|transaction)\b', re.IGNORECASE),
    ]
    
    # Phishing indicators
    PHISHING_INDICATORS = [
        re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', re.IGNORECASE),
        re.compile(r'bit\.ly|tinyurl|short\.link', re.IGNORECASE),
        re.compile(r'verify.*link|click.*here|visit.*url', re.IGNORECASE),
    ]
    
    # Sensitive information patterns
    SENSITIVE_INFO_PATTERNS = [
        re.compile(r'\b(upi id|upi|account number|bank account|card number|pin|otp|cvv)\b', re.IGNORECASE),
        re.compile(r'\b(share|send|provide|give|tell).*(upi|account|otp|pin)\b', re.IGNORECASE),
    ]
