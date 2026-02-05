"""Intelligence aggregation and management."""
import re
from typing import List
from app.models.session_state import Message
from app.models.intelligence import ExtractedIntelligence
from app.utils.regex_patterns import RegexPatterns
from app.utils.keyword_lists import ScamKeywords


class IntelligenceAggregator:
    """Aggregates intelligence from multiple sources."""
    
    def __init__(self):
        self.patterns = RegexPatterns()
        self.keywords = ScamKeywords()
    
    def extract_intelligence(
        self,
        message: Message,
        conversation_history: List[Message]
    ) -> ExtractedIntelligence:
        """Extract intelligence from message and conversation."""
        intelligence = ExtractedIntelligence()
        text = message.text
        
        # Extract bank accounts
        bank_accounts = self.patterns.BANK_ACCOUNT.findall(text)
        intelligence.bankAccounts.extend([
            re.sub(r'[-.\s]', '', acc) for acc in bank_accounts
        ])
        
        # Extract UPI IDs
        upi_ids = self.patterns.UPI_ID.findall(text)
        intelligence.upiIds.extend(upi_ids)
        
        # Extract phone numbers
        phone_numbers = self.patterns.PHONE_NUMBER.findall(text)
        intelligence.phoneNumbers.extend([
            f"+91{num}" if not num.startswith('+') else num
            for num in phone_numbers if len(num) >= 10
        ])
        
        # Extract phishing links
        urls = self.patterns.URL.findall(text)
        intelligence.phishingLinks.extend(urls)
        
        # Extract suspicious keywords
        text_lower = text.lower()
        found_keywords = [
            keyword for keyword in self.keywords.SUSPICIOUS_KEYWORDS
            if keyword in text_lower
        ]
        intelligence.suspiciousKeywords.extend(found_keywords)
        
        # Also check conversation history for intelligence
        for hist_msg in conversation_history[-5:]:  # Check last 5 messages
            hist_text = hist_msg.text
            
            # Extract from history
            hist_banks = self.patterns.BANK_ACCOUNT.findall(hist_text)
            intelligence.bankAccounts.extend([
                re.sub(r'[-.\s]', '', acc) for acc in hist_banks
            ])
            
            hist_upi = self.patterns.UPI_ID.findall(hist_text)
            intelligence.upiIds.extend(hist_upi)
            
            hist_phones = self.patterns.PHONE_NUMBER.findall(hist_text)
            intelligence.phoneNumbers.extend([
                f"+91{num}" if not num.startswith('+') else num
                for num in hist_phones if len(num) >= 10
            ])
            
            hist_urls = self.patterns.URL.findall(hist_text)
            intelligence.phishingLinks.extend(hist_urls)
        
        # Remove duplicates
        intelligence.bankAccounts = list(set(intelligence.bankAccounts))
        intelligence.upiIds = list(set(intelligence.upiIds))
        intelligence.phoneNumbers = list(set(intelligence.phoneNumbers))
        intelligence.phishingLinks = list(set(intelligence.phishingLinks))
        intelligence.suspiciousKeywords = list(set(intelligence.suspiciousKeywords))
        
        return intelligence


# Global aggregator instance
intelligence_aggregator = IntelligenceAggregator()
