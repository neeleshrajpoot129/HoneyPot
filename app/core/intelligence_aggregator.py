"""Intelligence aggregation and management."""
import re
import json
from typing import List, Optional
from groq import Groq
from app.models.session_state import Message
from app.models.intelligence import ExtractedIntelligence
from app.utils.regex_patterns import RegexPatterns
from app.utils.keyword_lists import ScamKeywords
from app.utils.prompts import IntelligenceExtractionPrompts
from app.config import config
from app.utils.logger import logger


class IntelligenceAggregator:
    """Aggregates intelligence using LLM as PRIMARY method, regex ONLY as fallback."""
    
    def __init__(self):
        self.patterns = RegexPatterns()
        self.keywords = ScamKeywords()
        self._groq_client = None
        
        # Initialize Groq for LLM-based extraction
        if config.GROQ_API_KEY:
            try:
                self._groq_client = Groq(api_key=config.GROQ_API_KEY)
                logger.info("Intelligence aggregator: LLM mode enabled")
            except Exception as e:
                logger.warning(f"Intelligence aggregator: Failed to initialize Groq: {e}. Using regex fallback.")
                self._groq_client = None
    
    def extract_intelligence(
        self,
        message: Message,
        conversation_history: List[Message]
    ) -> ExtractedIntelligence:
        """
        Extract intelligence from message and ENTIRE conversation.
        
        Uses LLM as PRIMARY method - LLM can handle:
        - International phone numbers (any format)
        - Bank accounts (any format)
        - UPI IDs (any format)
        - URLs/links (any format)
        - Keywords and patterns (context-aware)
        
        Regex is ONLY used as fallback when LLM is unavailable.
        Checks ALL messages in conversation history, not just recent ones.
        """
        # PRIMARY: Try LLM-based extraction first
        if self._groq_client:
            try:
                llm_intelligence = self._llm_extract_intelligence(message, conversation_history)
                if llm_intelligence:
                    logger.info(f"LLM extracted intelligence: {len(llm_intelligence.bankAccounts)} banks, {len(llm_intelligence.phoneNumbers)} phones, {len(llm_intelligence.upiIds)} UPI IDs")
                    return llm_intelligence
                else:
                    logger.warning("LLM returned empty intelligence, falling back to regex")
            except Exception as e:
                logger.warning(f"LLM intelligence extraction failed: {e}. Using regex fallback.")
        
        # FALLBACK: Only use regex if LLM is unavailable or failed
        logger.info("Using regex fallback for intelligence extraction")
        return self._regex_extract_intelligence(message, conversation_history)
    
    def _llm_extract_intelligence(
        self,
        message: Message,
        conversation_history: List[Message]
    ) -> Optional[ExtractedIntelligence]:
        """Extract intelligence using LLM."""
        try:
            prompt = IntelligenceExtractionPrompts.get_intelligence_extraction_prompt(
                conversation_history + [message],
                message.text
            )
            
            response = self._groq_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            # Extract JSON (handle nested structures)
            import re
            # Try to find complete JSON object
            json_match = re.search(r'\{.*"bankAccounts".*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            llm_result = json.loads(response_text)
            
            # Validate and normalize phone numbers
            if llm_result.get("phoneNumbers"):
                normalized_phones = []
                for phone in llm_result["phoneNumbers"]:
                    # Keep as-is - LLM should handle formatting
                    normalized_phones.append(str(phone).strip())
                llm_result["phoneNumbers"] = normalized_phones
            
            # Convert to ExtractedIntelligence
            intelligence = ExtractedIntelligence()
            intelligence.bankAccounts = llm_result.get("bankAccounts", [])
            intelligence.phoneNumbers = llm_result.get("phoneNumbers", [])
            intelligence.upiIds = llm_result.get("upiIds", [])
            intelligence.phishingLinks = llm_result.get("phishingLinks", [])
            intelligence.suspiciousKeywords = llm_result.get("suspiciousKeywords", [])
            
            logger.debug(f"LLM extracted: {len(intelligence.bankAccounts)} banks, {len(intelligence.phoneNumbers)} phones, {len(intelligence.upiIds)} UPI IDs")
            return intelligence
            
        except Exception as e:
            logger.error(f"LLM intelligence extraction error: {e}", exc_info=True)
            return None
    
    def _regex_extract_intelligence(
        self,
        message: Message,
        conversation_history: List[Message]
    ) -> ExtractedIntelligence:
        """Extract intelligence using regex patterns (fallback)."""
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
        phone_matches = self.patterns.PHONE_NUMBER.findall(text)
        for match in phone_matches:
            # Clean the phone number (remove dashes, spaces, dots)
            cleaned = re.sub(r'[-.\s]', '', str(match))
            
            # Format as +91XXXXXXXXXX
            if cleaned.startswith('+91') and len(cleaned) == 13:
                # Already has +91, just ensure format
                intelligence.phoneNumbers.append(cleaned)
            elif cleaned.startswith('91') and len(cleaned) == 12:
                intelligence.phoneNumbers.append('+' + cleaned)
            elif cleaned.startswith('0') and len(cleaned) == 11:
                intelligence.phoneNumbers.append('+91' + cleaned[1:])
            elif not cleaned.startswith('+') and len(cleaned) == 10 and cleaned[0] in '6789':
                intelligence.phoneNumbers.append('+91' + cleaned)
        
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
        
        # Check ALL conversation history for intelligence (not just last 5)
        for hist_msg in conversation_history:  # Check ALL messages
            hist_text = hist_msg.text
            
            # Extract from history
            hist_banks = self.patterns.BANK_ACCOUNT.findall(hist_text)
            intelligence.bankAccounts.extend([
                re.sub(r'[-.\s]', '', acc) for acc in hist_banks
            ])
            
            hist_upi = self.patterns.UPI_ID.findall(hist_text)
            intelligence.upiIds.extend(hist_upi)
            
            hist_phone_matches = self.patterns.PHONE_NUMBER.findall(hist_text)
            for match in hist_phone_matches:
                # Clean the phone number (remove dashes, spaces, dots)
                cleaned = re.sub(r'[-.\s]', '', str(match))
                
                # Format as +91XXXXXXXXXX
                if cleaned.startswith('+91') and len(cleaned) == 13:
                    # Already has +91, just ensure format
                    intelligence.phoneNumbers.append(cleaned)
                elif cleaned.startswith('91') and len(cleaned) == 12:
                    intelligence.phoneNumbers.append('+' + cleaned)
                elif cleaned.startswith('0') and len(cleaned) == 11:
                    intelligence.phoneNumbers.append('+91' + cleaned[1:])
                elif not cleaned.startswith('+') and len(cleaned) == 10 and cleaned[0] in '6789':
                    intelligence.phoneNumbers.append('+91' + cleaned)
            
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
