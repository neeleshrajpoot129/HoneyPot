"""LLM-first scam detection engine with rule-based fallback."""
from typing import List, Dict, Optional
import json
from groq import Groq
from app.models.session_state import Message
from app.utils.regex_patterns import RegexPatterns
from app.utils.keyword_lists import ScamKeywords
from app.utils.prompts import ScamDetectionPrompts
from app.config import config
from app.utils.logger import logger


class ScamDetectionResult:
    """Structured result from scam detection."""
    def __init__(
        self,
        is_scam: bool,
        confidence: float,
        llm_result: Optional[Dict] = None,
        rule_based_fallback: bool = False,
        final_decision_reason: str = ""
    ):
        self.is_scam = is_scam
        self.confidence = confidence
        self.llm_result = llm_result
        self.rule_based_fallback = rule_based_fallback
        self.final_decision_reason = final_decision_reason
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/audit."""
        return {
            "is_scam": self.is_scam,
            "confidence": self.confidence,
            "llm_result": self.llm_result,
            "rule_based_fallback": self.rule_based_fallback,
            "final_decision_reason": self.final_decision_reason
        }


class ScamDetector:
    """LLM-first scam detector with rule-based fallback on LLM errors."""
    
    def __init__(self):
        self.patterns = RegexPatterns()
        self.keywords = ScamKeywords()
        self._groq_client = None
        
        # Initialize Groq - try to enable LLM, but allow fallback
        if config.GROQ_API_KEY:
            try:
                self._groq_client = Groq(api_key=config.GROQ_API_KEY)
                logger.info("Scam detector: LLM mode enabled (Groq) with rule-based fallback")
            except Exception as e:
                logger.warning(f"Scam detector: Failed to initialize Groq: {e}. Will use rule-based fallback.")
                self._groq_client = None
        else:
            logger.warning("Scam detector: GROQ_API_KEY not found - will use rule-based detection")
            self._groq_client = None
    
    def detect_scam(
        self,
        message: Message,
        conversation_history: list[Message]
    ) -> ScamDetectionResult:
        """
        Detect scam intent using LLM-first approach with rule-based fallback.
        
        Returns:
            ScamDetectionResult with is_scam, confidence, and detection method
        """
        # Try LLM first if available
        if self._groq_client:
            try:
                # Extract artifacts for LLM context
                extracted_artifacts = self._extract_artifacts(message.text)
                
                # Call LLM for scam detection
                llm_result = self._llm_detect_scam(
                    message,
                    conversation_history,
                    extracted_artifacts
                )
                
                if llm_result:
                    # LLM succeeded - use LLM result
                    is_scam = llm_result.get("is_scam", False)
                    confidence = llm_result.get("confidence", 0.0)
                    reason = llm_result.get("reason", "LLM analysis")
                    
                    # Constrain confidence to 0-1 range
                    confidence = max(0.0, min(1.0, float(confidence)))
                    
                    # Apply threshold
                    if confidence >= config.SCAM_DETECTION_CONFIDENCE_THRESHOLD:
                        is_scam = True
                    else:
                        is_scam = False
                    
                    final_reason = f"LLM detection (confidence={confidence:.2f}): {reason}"
                    
                    logger.info(
                        f"Scam detection (LLM): is_scam={is_scam}, "
                        f"confidence={confidence:.2f}, reason={reason[:50]}"
                    )
                    
                    return ScamDetectionResult(
                        is_scam=is_scam,
                        confidence=confidence,
                        llm_result=llm_result,
                        rule_based_fallback=False,
                        final_decision_reason=final_reason
                    )
                else:
                    # LLM returned invalid result - fallback to rules
                    logger.warning("Scam detector: LLM returned invalid result, falling back to rule-based detection")
            except Exception as e:
                # LLM API error - fallback to rules
                logger.warning(f"Scam detector: LLM API error ({e}), falling back to rule-based detection")
        
        # Fallback to rule-based detection
        return self._rule_based_detection(message, conversation_history)
    
    def _rule_based_detection(
        self,
        message: Message,
        conversation_history: list[Message]
    ) -> ScamDetectionResult:
        """Rule-based scam detection (fallback when LLM unavailable)."""
        text = message.text.lower()
        rule_score = 0.0
        rule_evidence = []
        
        # Check for urgency patterns
        urgency_score = 0
        for pattern in self.patterns.URGENCY_PATTERNS:
            if pattern.search(text):
                urgency_score += 0.15
        
        rule_score += min(urgency_score, 0.4)
        if urgency_score > 0:
            rule_evidence.append("Urgency patterns detected")
        
        # Check for scam keywords
        keyword_matches = []
        for keyword in self.keywords.SCAM_KEYWORDS:
            if keyword in text:
                keyword_matches.append(keyword)
        
        if keyword_matches:
            rule_evidence.append(f"Scam keywords: {', '.join(keyword_matches[:3])}")
        keyword_score = min(len(keyword_matches) * 0.2, 0.4)
        rule_score += keyword_score
        
        # Reward / lottery scam detection
        reward_keyword_found = False
        for keyword in self.keywords.REWARD_SCAM_KEYWORDS:
            if keyword in text:
                rule_score += 0.4
                rule_evidence.append(f"Reward scam keyword: '{keyword}'")
                reward_keyword_found = True
                break
        
        # Boost score if reward scam combined with UPI/phone
        if reward_keyword_found:
            upi_match = self.patterns.UPI_ID.search(text)
            phone_match = self.patterns.PHONE_NUMBER.search(text)
            if upi_match or phone_match:
                rule_score += 0.3
                if upi_match:
                    rule_evidence.append(f"Reward scam with UPI ID: {upi_match.group()}")
                if phone_match:
                    rule_evidence.append(f"Reward scam with phone number: {phone_match.group()}")
        
        # Contextual banking terms (low weight)
        for pattern in self.patterns.CONTEXTUAL_BANKING_TERMS:
            if pattern.search(text):
                rule_score += 0.1
                rule_evidence.append("Contextual banking terms")
                break
        
        # Check for phishing links
        for pattern in self.patterns.PHISHING_INDICATORS:
            if pattern.search(text):
                rule_score += 0.3
                rule_evidence.append("Phishing indicator: URL/link detected")
                break
        
        # Check for requests for sensitive information
        for pattern in self.patterns.SENSITIVE_INFO_PATTERNS:
            if pattern.search(text):
                rule_score += 0.2
                rule_evidence.append("Sensitive info request detected")
                break
        
        # Context-based detection (check conversation history)
        if conversation_history:
            prev_scam_count = sum(
                1 for msg in conversation_history[-3:]
                if self._quick_check(msg.text)
            )
            if prev_scam_count > 0:
                rule_score += 0.1 * prev_scam_count
                rule_evidence.append(f"Context: {prev_scam_count} previous messages had scam indicators")
        
        # Normalize rule score to 0-1 range
        rule_score = min(rule_score, 1.0)
        
        # Determine if scam using threshold
        is_scam = rule_score >= config.SCAM_DETECTION_CONFIDENCE_THRESHOLD
        
        reason = f"Rule-based fallback (score={rule_score:.2f}): {', '.join(rule_evidence[:3]) if rule_evidence else 'No indicators'}"
        
        logger.info(
            f"Scam detection (rule-based fallback): is_scam={is_scam}, "
            f"score={rule_score:.2f}, evidence={len(rule_evidence)} indicators"
        )
        
        return ScamDetectionResult(
            is_scam=is_scam,
            confidence=rule_score,
            llm_result=None,
            rule_based_fallback=True,
            final_decision_reason=reason
        )
    
    def _llm_detect_scam(
        self,
        message: Message,
        conversation_history: list[Message],
        extracted_artifacts: Dict
    ) -> Optional[Dict]:
        """Call LLM for scam detection."""
        try:
            prompt = ScamDetectionPrompts.get_llm_scam_detection_prompt(
                message.text,
                conversation_history,
                extracted_artifacts
            )
            
            # Use Groq API
            response = self._groq_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low temperature for consistent analysis
                max_tokens=200,
                top_p=0.7,
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            # Try to extract JSON if wrapped in text
            import re
            json_match = re.search(r'\{[^{}]*"is_scam"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            llm_result = json.loads(response_text)
            
            # Validate structure
            if not isinstance(llm_result, dict):
                logger.warning(f"LLM returned invalid structure: {llm_result}")
                return None
            
            # Required fields
            required_fields = ["is_scam", "confidence"]
            missing_fields = [f for f in required_fields if f not in llm_result]
            if missing_fields:
                logger.warning(f"LLM result missing required fields: {missing_fields}")
                return None
            
            # Type validation
            if not isinstance(llm_result["is_scam"], bool):
                logger.warning(f"LLM is_scam must be bool, got: {type(llm_result['is_scam'])}")
                return None
            
            if not isinstance(llm_result["confidence"], (int, float)):
                logger.warning(f"LLM confidence must be number, got: {type(llm_result['confidence'])}")
                return None
            
            # Constrain confidence to 0-1 range
            llm_result["confidence"] = max(0.0, min(1.0, float(llm_result["confidence"])))
            
            # Constrain reason length
            if "reason" in llm_result and len(llm_result["reason"]) > 200:
                llm_result["reason"] = llm_result["reason"][:197] + "..."
            
            logger.debug(f"LLM scam detection result: is_scam={llm_result.get('is_scam')}, confidence={llm_result.get('confidence')}")
            return llm_result
            
        except json.JSONDecodeError as e:
            response_content = response.choices[0].message.content[:100] if 'response' in locals() and response.choices else 'N/A'
            logger.warning(f"Failed to parse LLM JSON response: {e}. Response: {response_content}")
            return None
        except Exception as e:
            logger.error(f"LLM scam detection error: {e}", exc_info=True)
            raise  # Re-raise to trigger fallback
    
    def _extract_artifacts(self, text: str) -> Dict:
        """Extract artifacts (URLs, UPI IDs, phone numbers) for LLM context."""
        artifacts = {
            "urls": [],
            "upi_ids": [],
            "phone_numbers": []
        }
        
        # Extract URLs
        url_matches = self.patterns.URL.findall(text)
        artifacts["urls"] = list(set(url_matches))
        
        # Extract UPI IDs
        upi_matches = self.patterns.UPI_ID.findall(text)
        artifacts["upi_ids"] = list(set(upi_matches))
        
        # Extract phone numbers
        phone_matches = self.patterns.PHONE_NUMBER.findall(text)
        artifacts["phone_numbers"] = list(set(phone_matches))
        
        return artifacts
    
    def _quick_check(self, text: str) -> bool:
        """Quick check for scam indicators."""
        text_lower = text.lower()
        return any(
            keyword in text_lower
            for keyword in ['verify', 'blocked', 'urgent', 'suspended', 'upi']
        )


# Global detection engine instance
scam_detector = ScamDetector()
