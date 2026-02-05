"""
Centralized prompts for all agents.

This module contains all prompts used by different agents in the honeypot system.
Prompts are organized by agent for easy maintenance and updates.
"""

from typing import Dict, List
from app.models.strategy import ConversationGoal


# ============================================================================
# STRATEGY AGENT PROMPTS
# ============================================================================

class StrategyAgentPrompts:
    """Prompts for the Strategy Agent (conversation end detection)."""
    
    # Active scam keywords that should NEVER trigger conversation end
    ACTIVE_SCAM_KEYWORDS = [
        "verify", "verify immediately", "blocked", "suspended", "share", 
        "send", "provide", "click", "link", "upi", "account", "urgent",
        "immediately", "now", "asap", "required", "must", "need to"
    ]
    
    @staticmethod
    def get_conversation_end_detection_prompt(
        message_text: str,
        message_count: int,
        upi_count: int,
        link_count: int
    ) -> str:
        """
        Get the prompt for LLM-based conversation end detection.
        
        Args:
            message_text: Current message from scammer
            message_count: Total messages exchanged
            upi_count: Number of UPI IDs extracted
            link_count: Number of phishing links extracted
            
        Returns:
            Formatted prompt string
        """
        return f"""You are analyzing a conversation with a scammer.

Your ONLY task:
Decide if the scammer is clearly trying to END the conversation.

STRICT RULES:
- Answer "NO" if the scammer is asking for anything (verification, UPI, links, actions).
- Answer "NO" if the scammer is making threats or demands.
- Answer "YES" ONLY if the scammer clearly says goodbye, thanks, or indicates the conversation is finished.

Examples:
- "Thanks, goodbye" → YES
- "Okay, that's all" → YES
- "Send your UPI now" → NO
- "Click this link to verify" → NO

Current message:
"{message_text}"

Conversation length: {message_count}
Extracted UPI IDs: {upi_count}
Extracted links: {link_count}

Is the scammer ending the conversation?

Answer ONLY with "YES" or "NO"."""


# ============================================================================
# PERSONA AGENT PROMPTS
# ============================================================================

class PersonaAgentPrompts:
    """Prompts for the Persona Agent (human-like response generation)."""
    
    # Base persona traits
    PERSONA_TRAITS = {
        "curiosity": "Asks questions when something seems unclear",
        "caution": "Shows concern but not panic",
        "helpfulness": "Wants to resolve issues",
        "natural_language": "Uses casual, everyday language"
    }
    
    # Base system prompt (revised - no intelligence awareness)
    BASE_PROMPT = """You are an ordinary person who received a message about a bank or account issue.

PERSONA:
- You are polite, slightly worried, and cooperative.
- You are not very technical.
- You use simple, everyday English.
- Your replies are short (usually 1–2 sentences).

STRICT RULES:
- Never say you are an AI, system, bot, or detector.
- Never mention analysis, intelligence, rules, or confidence.
- Never share or confirm sensitive information (OTP, PIN, UPI ID, account number).
- Never instruct anyone to make a payment or transfer money.

BEHAVIOR GUIDELINES:
- Sound unsure or confused, not aggressive.
- Prefer questions over statements.
- Avoid repeating the same sentence structure.
- Keep responses natural and human.

STYLE EXAMPLES:
- "I'm not sure why this is needed. Can you explain?"
- "This sounds serious. What exactly should I do?"
- "I don't usually do this online. Can you tell me more?"

Return ONLY the reply text. Do not include explanations."""
    
    @staticmethod
    def get_goal_instruction(goal: ConversationGoal) -> str:
        """
        Get goal-specific behavior hint for the persona agent.
        Returns concise behavior instructions without intelligence awareness.
        
        Args:
            goal: Conversation goal (CLARIFY, DELAY, ESCALATE, CONTINUE, WRAP_UP)
            
        Returns:
            Goal-specific behavior hint string
        """
        goal_instructions: Dict[ConversationGoal, str] = {
            ConversationGoal.CLARIFY: """
Current goal: Ask for clarification.
- Ask one simple question.
- Show mild concern or confusion.
- Do not agree to any action.
""",
            
            ConversationGoal.DELAY: """
Current goal: Delay politely.
- Say you need time or need to check.
- Keep tone calm and reasonable.
""",
            
            ConversationGoal.ESCALATE: """
Current goal: Show increased concern.
- Express worry.
- Ask what needs to be done.
- Do not commit to anything.
""",
            
            ConversationGoal.CONTINUE: """
Current goal: Continue the conversation naturally.
- Respond briefly.
- Ask a relevant follow-up if needed.
""",
            
            ConversationGoal.WRAP_UP: """
Current goal: End the conversation politely.
- Be firm but respectful.
- Example: "I'll check with my bank directly. Thank you."
"""
        }
        
        return goal_instructions.get(goal, goal_instructions[ConversationGoal.CONTINUE])
    
    @staticmethod
    def build_conversation_context(
        system_prompt: str,
        conversation_history: list,
        current_message: str
    ) -> str:
        """
        Build the full conversation context for the LLM.
        Includes few-shot examples for better human-likeness.
        
        Args:
            system_prompt: Base system prompt with goal instructions
            conversation_history: List of previous messages
            current_message: Current message from scammer
            
        Returns:
            Full conversation context string
        """
        context = system_prompt + "\n\n"
        
        # Add few-shot examples for tone & length
        context += """Example replies (tone & length):
- "I'm not sure why this is happening. Can you explain?"
- "This sounds worrying. What exactly do I need to do?"
- "I don't usually click links. Can you tell me more?"
- "I need some time to check this. Please wait."
"""
        context += "\n"
        
        # Add conversation history (last 8 messages to avoid token limits)
        if conversation_history:
            context += "Previous conversation:\n"
            for msg in conversation_history[-8:]:  # Last 8 messages for context
                if msg.sender == "scammer":
                    context += f"Scammer: {msg.text}\n"
                else:
                    context += f"You: {msg.text}\n"
            context += "\n"
        
        # Add current message
        context += f"Current message from scammer: {current_message}\n\n"
        context += "Your response (be natural, varied, and don't repeat previous responses):"
        
        return context


# ============================================================================
# FALLBACK RESPONSES (Rule-based)
# ============================================================================

class FallbackResponses:
    """Rule-based fallback responses when LLM is unavailable."""
    
    @staticmethod
    def get_response(
        goal: ConversationGoal,
        message_text: str
    ) -> str:
        """
        Get rule-based fallback response based on goal and message.
        
        Args:
            goal: Conversation goal
            message_text: Message text from scammer
            
        Returns:
            Fallback response string
        """
        text_lower = message_text.lower()
        
        if goal == ConversationGoal.CLARIFY:
            if "upi" in text_lower or "upi id" in text_lower:
                return "I'm not comfortable sharing my UPI ID. Is there another way to verify?"
            elif "link" in text_lower or "click" in text_lower:
                return "I'm not sure about clicking links. Can you tell me more about this?"
            elif "verify" in text_lower:
                return "How do I verify? Can you explain the process step by step?"
            else:
                return "I see. Can you provide more details about this?"
        
        elif goal == ConversationGoal.DELAY:
            if "urgent" in text_lower or "immediately" in text_lower:
                return "I'm at work right now. Can you explain what I need to do? I need a few minutes to understand this."
            else:
                return "I need to check something first. Can you give me more information about this?"
        
        elif goal == ConversationGoal.ESCALATE:
            if "blocked" in text_lower or "suspended" in text_lower:
                return "This is really worrying. What exactly do I need to do to prevent this? I want to fix this immediately."
            else:
                return "I'm concerned about this. What should I do next?"
        
        elif goal == ConversationGoal.CONTINUE:
            if "blocked" in text_lower or "suspended" in text_lower:
                return "Why is my account being blocked? What did I do wrong?"
            elif "verify" in text_lower:
                return "How do I verify? Can you explain the process?"
            else:
                return "I see. Can you provide more details about this?"
        
        else:  # WRAP_UP
            return "I'll check with my bank directly. Thanks for letting me know."


# ============================================================================
# INTELLIGENCE EXTRACTION PROMPTS
# ============================================================================

class IntelligenceExtractionPrompts:
    """Prompts for intelligence extraction using LLM."""
    
    @staticmethod
    def get_intelligence_extraction_prompt(
        conversation_history: List,
        current_message: str
    ) -> str:
        """
        Get prompt for LLM-based intelligence extraction.
        
        Args:
            conversation_history: All messages in the conversation
            current_message: Current message text
            
        Returns:
            Formatted prompt string
        """
        # Build conversation context
        conversation_text = ""
        for i, msg in enumerate(conversation_history, 1):
            conversation_text += f"Message {i} ({msg.sender}): {msg.text}\n"
        
        conversation_text += f"Current Message (scammer): {current_message}\n"
        
        return f"""You are an intelligence extraction agent analyzing a scam conversation.

Your task: Extract ALL intelligence from the ENTIRE conversation (all messages, not just the latest).

Analyze EVERY message in the conversation and extract:

1. BANK ACCOUNT NUMBERS:
   - Any format: XXXX-XXXX-XXXX-XXXX, XXXX XXXX XXXX XXXX, or 16 consecutive digits
   - Indian bank accounts, international accounts, any format
   - Example: 1234-5678-9012-3456, 1234567890123456, 1234 5678 9012 3456

2. PHONE NUMBERS:
   - Indian numbers: +91-XXXXXXXXXX, +91 XXXXXXXXXX, +91XXXXXXXXXX, 91XXXXXXXXXX, 0XXXXXXXXXX, or 10-digit numbers starting with 6-9
   - International numbers: ANY format with country codes (e.g., +1-XXX-XXX-XXXX, +44-XXXX-XXXXXX, etc.)
   - Handle dashes, spaces, parentheses, dots: +1 (555) 123-4567, +44.20.1234.5678
   - Extract ALL phone numbers regardless of format

3. UPI IDs:
   - Any format: name@paytm, name@gpay, name@phonepe, name@ybl, name@okicici, etc.
   - Handle variations: name@upi, name@payzapp, etc.

4. PHISHING LINKS:
   - Any URLs: http://, https://, www., short links (bit.ly, tinyurl, etc.)
   - Extract ALL links mentioned in conversation

5. SUSPICIOUS KEYWORDS:
   - Urgency: urgent, immediately, asap, hurry, quickly, right now
   - Threats: blocked, suspended, frozen, locked, closed, terminated
   - Sensitive info: OTP, PIN, password, account number, UPI ID, CVV
   - Actions: verify, share, send, provide, click, call
   - Time pressure: minutes, hours, today, deadline, final notice

CRITICAL INSTRUCTIONS:
- Extract from ALL messages in the conversation, not just the latest
- Cumulate everything - don't miss intelligence from earlier messages
- Handle ANY format - don't rely on specific patterns
- International phone numbers are valid - extract them
- Bank accounts can be in any format - extract them
- Be thorough - check every single message
- Normalize phone numbers to include country code (e.g., +91XXXXXXXXXX for Indian, +1XXXXXXXXXX for US)
- Remove duplicates before returning

Conversation:
{conversation_text}

Return ONLY valid JSON in this EXACT format:
{{
  "bankAccounts": ["1234567890123456"],
  "phoneNumbers": ["+919876543210", "+15551234567"],
  "upiIds": ["scammer@paytm"],
  "phishingLinks": ["http://example.com"],
  "suspiciousKeywords": ["urgent", "blocked", "otp"]
}}

If no intelligence found, return empty arrays:
{{
  "bankAccounts": [],
  "phoneNumbers": [],
  "upiIds": [],
  "phishingLinks": [],
  "suspiciousKeywords": []
}}

JSON response:"""


# ============================================================================
# AGENT NOTES PROMPTS (Reasoning-based)
# ============================================================================

class AgentNotesPrompts:
    """Prompts for generating reasoning-based agent notes."""
    
    @staticmethod
    def get_agent_notes_prompt(
        conversation_history: List,
        extracted_intelligence: Dict,
        scam_detection_reason: str,
        scam_confidence: float
    ) -> str:
        """
        Get prompt for generating reasoning-based agent notes.
        
        Args:
            conversation_history: All messages in the conversation
            extracted_intelligence: Extracted intelligence summary
            scam_detection_reason: Why scam was detected
            scam_confidence: Scam detection confidence score
            
        Returns:
            Formatted prompt string
        """
        # Build conversation summary
        conversation_summary = ""
        for i, msg in enumerate(conversation_history[-10:], 1):  # Last 10 messages
            conversation_summary += f"{i}. {msg.sender}: {msg.text}\n"
        
        intelligence_summary = f"""
Extracted Intelligence:
- Bank Accounts: {len(extracted_intelligence.get('bankAccounts', []))} found
- Phone Numbers: {len(extracted_intelligence.get('phoneNumbers', []))} found
- UPI IDs: {len(extracted_intelligence.get('upiIds', []))} found
- Phishing Links: {len(extracted_intelligence.get('phishingLinks', []))} found
- Suspicious Keywords: {len(extracted_intelligence.get('suspiciousKeywords', []))} found
"""
        
        return f"""You are analyzing a scam conversation to generate agent notes.

Your task: Write a SHORT, CLEAR summary explaining WHY this was detected as a scam.

Focus on:
- Scam tactics used (urgency, threats, time pressure)
- What the scammer was trying to extract (OTP, PIN, account numbers, etc.)
- Red flags and suspicious behavior
- Key intelligence gathered

Do NOT just list what was extracted. Explain the REASONING and TACTICS.

Conversation Summary:
{conversation_summary}

Scam Detection:
- Reason: {scam_detection_reason}
- Confidence: {scam_confidence:.2f}
{intelligence_summary}

Write a concise summary (2-3 sentences) explaining why this is a scam and what tactics were used.

Example format:
"Scammer used urgency tactics (threatened account blocking) and repeatedly requested sensitive information (OTP, account number, UPI PIN) to gain unauthorized access. Extracted bank account number and phone number from conversation. Time-pressure tactics escalated from '2 hours' to '2 minutes' to create panic."

Your summary:"""


# ============================================================================
# FORBIDDEN PHRASES & ALLOWED FILLERS
# ============================================================================

class ForbiddenPhrases:
    """Phrases that should NEVER appear in persona responses."""
    
    FORBIDDEN = [
        "I am an AI",
        "I'm a bot",
        "I'm an AI",
        "detection system",
        "honeypot",
        "I'm detecting",
        "I'm analyzing",
        "intelligence",
        "gathered intelligence",
        "extracted",
        "confidence score",
        "rule-based",
        "scam detection",
        "I'm a system",
        "automated",
        "algorithm"
    ]
    
    # Phrases that reveal meta-awareness
    META_PHRASES = [
        "we've already",
        "we have gathered",
        "we extracted",
        "our system",
        "the system",
        "detection",
        "analysis"
    ]


class AllowedFillers:
    """Casual fillers that make responses more human-like."""
    
    HESITATION = [
        "um", "uh", "hmm", "well", "actually", "I mean", "you know"
    ]
    
    POLITE = [
        "please", "sorry", "excuse me", "if you don't mind", "I hope"
    ]
    
    UNCERTAINTY = [
        "I think", "I guess", "maybe", "perhaps", "I'm not sure", 
        "I don't know", "kind of", "sort of"
    ]


# ============================================================================
# SCAM DETECTION LLM FALLBACK PROMPTS
# ============================================================================

class ScamDetectionPrompts:
    """Prompts for LLM-only scam detection."""
    
    @staticmethod
    def get_llm_scam_detection_prompt(
        message_text: str,
        conversation_history: list,
        extracted_artifacts: dict
    ) -> str:
        """
        Get the prompt for LLM-only scam detection.
        
        Args:
            message_text: Current message to analyze
            conversation_history: Last N messages in conversation
            extracted_artifacts: Dict with URLs, UPI IDs, phone numbers found
            
        Returns:
            Formatted prompt string for structured JSON response
        """
        history_context = ""
        if conversation_history:
            history_context = "\n\nRecent conversation history:\n"
            for msg in conversation_history[-3:]:
                history_context += f"- {msg.sender}: {msg.text}\n"
        
        artifacts_context = ""
        if extracted_artifacts:
            artifacts_context = "\n\nExtracted artifacts from message:\n"
            if extracted_artifacts.get("urls"):
                artifacts_context += f"- URLs: {', '.join(extracted_artifacts['urls'])}\n"
            if extracted_artifacts.get("upi_ids"):
                artifacts_context += f"- UPI IDs: {', '.join(extracted_artifacts['upi_ids'])}\n"
            if extracted_artifacts.get("phone_numbers"):
                artifacts_context += f"- Phone numbers: {', '.join(extracted_artifacts['phone_numbers'])}\n"
        
        return f"""You are a security analyst evaluating a suspicious message for scam intent.

Message to analyze:
"{message_text}"
{history_context}{artifacts_context}

Your task: Determine if this message is a SCAM.

SCAM INDICATORS TO LOOK FOR:
- Urgency or threats (blocked account, immediate action required, time-sensitive)
- Requests for sensitive information (UPI ID, account number, OTP, PIN, passwords)
- Phishing links or suspicious URLs
- Reward/lottery scams (won prize, free money, congratulations)
- Payment requests or money transfers
- Impersonation (bank, government, service provider)
- Suspicious phone numbers or contact methods
- Unusual grammar, spelling errors, or formatting

LEGITIMATE MESSAGES:
- Informational messages from verified sources
- Routine notifications (statements, updates)
- Customer support follow-ups
- Casual social messages

Return ONLY valid JSON in this EXACT format:
{{
  "is_scam": true or false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation (max 50 words)"
}}

Guidelines:
- Mark as scam (is_scam=true) if message shows clear scam intent
- Mark as not scam (is_scam=false) if message is legitimate or unclear
- Confidence should reflect certainty:
  * 0.9-1.0: Very high confidence (clear scam indicators)
  * 0.7-0.9: High confidence (strong indicators)
  * 0.5-0.7: Moderate confidence (some indicators)
  * 0.3-0.5: Low confidence (unclear)
  * 0.0-0.3: Very low confidence (likely legitimate)
- Reason should be specific and explainable

JSON response:"""
    
    @staticmethod
    def get_llm_fallback_prompt(
        message_text: str,
        conversation_history: list,
        rule_score: float,
        rule_evidence: list[str],
        extracted_artifacts: dict
    ) -> str:
        """
        Get the prompt for LLM fallback when rule score is ambiguous.
        
        Args:
            message_text: Current message to analyze
            conversation_history: Last N messages in conversation
            rule_score: Rule-based confidence score (0.0-1.0)
            rule_evidence: List of rules that matched
            extracted_artifacts: Dict with URLs, UPI IDs, phone numbers found
            
        Returns:
            Formatted prompt string for structured JSON response
        """
        history_context = ""
        if conversation_history:
            history_context = "\n\nRecent conversation history:\n"
            for msg in conversation_history[-3:]:
                history_context += f"- {msg.sender}: {msg.text}\n"
        
        artifacts_context = ""
        if extracted_artifacts:
            artifacts_context = "\n\nExtracted artifacts from message:\n"
            if extracted_artifacts.get("urls"):
                artifacts_context += f"- URLs: {', '.join(extracted_artifacts['urls'])}\n"
            if extracted_artifacts.get("upi_ids"):
                artifacts_context += f"- UPI IDs: {', '.join(extracted_artifacts['upi_ids'])}\n"
            if extracted_artifacts.get("phone_numbers"):
                artifacts_context += f"- Phone numbers: {', '.join(extracted_artifacts['phone_numbers'])}\n"
        
        return f"""You are a security analyst reviewing a suspicious message.

The rule-based system produced an AMBIGUOUS score: {rule_score:.2f}

Rule indicators found:
{chr(10).join(f'- {e}' for e in rule_evidence)}

Message:
"{message_text}"
{history_context}{artifacts_context}

Task:
Decide if this message is a SCAM.

Return ONLY valid JSON in this EXACT format:
{{
  "is_scam": true or false,
  "confidence": 0.0-1.0,
  "reason": "short explanation (max 30 words)"
}}

Guidelines:
- Mark as scam if there is urgency, threats, sensitive data requests, or phishing behavior.
- Mark as not scam if message is informational or unclear.
- Use confidence ≥ 0.7 only when clearly sure.

JSON response:"""
