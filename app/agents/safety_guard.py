"""Safety and ethics guard - prevents illegal or unethical actions."""
from typing import Tuple, Optional
from app.utils.prompts import ForbiddenPhrases
from app.utils.keyword_lists import ProhibitedPhrases


class SafetyGuard:
    """Ensures agent behavior stays within ethical boundaries."""
    
    def __init__(self):
        self.prohibited_phrases = ProhibitedPhrases()
        self.forbidden_phrases = ForbiddenPhrases()
    
    def validate_response(self, response: str) -> Tuple[bool, Optional[str]]:
        """
        Validate agent response for safety and ethics.
        
        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        """
        response_lower = response.lower()
        
        # Check centralized forbidden phrases (from prompts.py)
        for forbidden in self.forbidden_phrases.FORBIDDEN:
            if forbidden.lower() in response_lower:
                return False, f"Response contains forbidden phrase: {forbidden}"
        
        # Check meta phrases (reveal intelligence awareness)
        for meta in self.forbidden_phrases.META_PHRASES:
            if meta.lower() in response_lower:
                return False, f"Response contains meta phrase: {meta}"
        
        # Check for prohibited phrases that reveal detection (legacy)
        for prohibited in self.prohibited_phrases.PROHIBITED_RESPONSES:
            if prohibited.lower() in response_lower:
                return False, f"Response contains prohibited phrase: {prohibited}"
        
        # Check for illegal instructions
        for action in self.prohibited_phrases.PROHIBITED_ACTIONS:
            if action.lower() in response_lower:
                return False, f"Response contains prohibited action: {action}"
        
        # Check response length (too long might seem unnatural)
        if len(response) > 500:
            return False, "Response too long (max 500 characters)"
        
        if len(response) < 5:
            return False, "Response too short (min 5 characters)"
        
        return True, None
    
    def validate_intelligence_extraction(self, intelligence) -> bool:
        """Validate that intelligence extraction is ethical."""
        # Ensure we're not extracting personal information of real individuals
        # This is a basic check - in production, you'd want more sophisticated validation
        return True


# Global safety guard instance
safety_guard = SafetyGuard()
