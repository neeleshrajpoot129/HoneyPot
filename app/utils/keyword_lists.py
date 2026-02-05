"""Keyword lists for scam detection and intelligence extraction."""


class ScamKeywords:
    """Scam-related keywords."""
    
    SCAM_KEYWORDS = [
        'verify immediately', 'account blocked', 'suspended',
        'click here', 'verify now', 'urgent action required',
        'your account', 'will be blocked', 'avoid suspension',
        'share your', 'send otp', 'verify your identity',
        'winning prize', 'congratulations', 'claim now',
        'free money', 'lottery winner', 'inheritance',
        'tax refund', 'government benefit'
    ]
    
    SUSPICIOUS_KEYWORDS = [
        'urgent', 'verify', 'blocked', 'suspended', 'immediately',
        'click here', 'verify now', 'account', 'upi', 'otp',
        'share', 'send', 'provide', 'winning', 'prize', 'free'
    ]
    
    CONVERSATION_END_KEYWORDS = [
        'bye', 'goodbye', 'thank you', 'thanks', 'done', 'finished'
    ]
    
    # Reward / lottery scam keywords
    REWARD_SCAM_KEYWORDS = [
        "won a prize",
        "won cash",
        "cash prize",  # Added: matches "won a cash prize"
        "you have won",
        "you won",  # Added: matches "you won a cash prize"
        "lottery",
        "congratulations",
        "claim your prize",
        "free money",
        "cash reward",
        "reward amount"
    ]


class ProhibitedPhrases:
    """Phrases that should not appear in agent responses."""
    
    PROHIBITED_RESPONSES = [
        "I am an AI",
        "I'm a bot",
        "detection system",
        "honeypot",
        "I'm detecting"
    ]
    
    PROHIBITED_ACTIONS = [
        "impersonate",
        "pretend to be",
        "act as",
        "illegal",
        "harass",
        "threaten"
    ]
