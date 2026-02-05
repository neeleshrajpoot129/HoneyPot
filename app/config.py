"""Configuration management for the honeypot system."""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""
    
    # API Configuration
    API_KEY = os.getenv("API_KEY", "your-secret-api-key-here")
    GUVI_CALLBACK_URL = os.getenv(
        "GUVI_CALLBACK_URL",
        "https://hackathon.guvi.in/api/updateHoneyPotFinalResult"
    )
    
    # Groq Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")  # Default: llama-3.1-70b-versatile
    
    # Agent Configuration
    MIN_MESSAGES_FOR_CALLBACK = int(os.getenv("MIN_MESSAGES_FOR_CALLBACK", "5"))
    MAX_MESSAGES_PER_SESSION = int(os.getenv("MAX_MESSAGES_PER_SESSION", "50"))
    SCAM_DETECTION_CONFIDENCE_THRESHOLD = float(
        os.getenv("SCAM_DETECTION_CONFIDENCE_THRESHOLD", "0.7")
    )
    
    # Hybrid Scam Detection Thresholds (Rule-First Approach)
    SCAM_HIGH_CONFIDENCE_THRESHOLD = float(
        os.getenv("SCAM_HIGH_CONFIDENCE_THRESHOLD", "0.75")
    )  # Rule score >= this → is_scam = True (no LLM)
    SCAM_LOW_CONFIDENCE_THRESHOLD = float(
        os.getenv("SCAM_LOW_CONFIDENCE_THRESHOLD", "0.25")
    )  # Rule score <= this → is_scam = False (no LLM)
    # Ambiguous range: (LOW_THRESHOLD, HIGH_THRESHOLD) → use LLM fallback
    SCAM_LLM_FALLBACK_ENABLED = os.getenv("SCAM_LLM_FALLBACK_ENABLED", "true").lower() == "true"
    SCAM_LLM_CONFIDENCE_THRESHOLD = float(
        os.getenv("SCAM_LLM_CONFIDENCE_THRESHOLD", "0.7")
    )  # LLM confidence >= this → accept LLM decision (fixed: changed from 0.75 to 0.7)
    
    # Conversation Configuration
    CONVERSATION_END_KEYWORDS = [
        "bye", "goodbye", "thank you", "thanks", "done", "finished"
    ]
    
    # LLM-based conversation end detection (replaces static keyword matching)
    USE_LLM_FOR_CONVERSATION_END = os.getenv("USE_LLM_FOR_CONVERSATION_END", "true").lower() == "true"


config = Config()
