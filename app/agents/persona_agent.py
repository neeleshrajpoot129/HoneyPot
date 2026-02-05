"""Persona agent - maintains human-like persona in conversations."""
import random
from groq import Groq
from app.models.session_state import Message
from app.models.strategy import StrategyDecision, ConversationGoal
from app.config import config
from app.utils.logger import logger
from app.utils.prompts import PersonaAgentPrompts, FallbackResponses, ForbiddenPhrases, AllowedFillers


class PersonaAgent:
    """
    Persona agent - EXECUTOR, not planner.
    
    This agent:
    - NEVER decides goals
    - Only expresses the chosen strategy
    - Maintains human-like persona
    - No intelligence awareness (acts, doesn't analyze)
    """
    
    def __init__(self):
        # Initialize Groq client
        if config.GROQ_API_KEY:
            try:
                self.client = Groq(api_key=config.GROQ_API_KEY)
                self.model_name = config.GROQ_MODEL
            except Exception as e:
                logger.warning(f"PersonaAgent: Failed to initialize Groq: {e}")
                self.client = None
                self.model_name = None
        else:
            self.client = None
            self.model_name = None
        
        self.persona_traits = PersonaAgentPrompts.PERSONA_TRAITS
    
    def generate_response(
        self,
        message: Message,
        conversation_history: list[Message],
        strategy_decision: StrategyDecision
    ) -> str:
        """
        Generate human-like response based on strategy decision.
        
        This is the SPEAKER - expresses the strategy, doesn't decide it.
        No intelligence awareness - just acts naturally.
        """
        # If strategy says don't engage, return None
        if not strategy_decision.should_engage:
            return None
        
        # Generate response based on strategy goal
        if not self.client:
            # Fallback to rule-based responses
            return self._fallback_response(message, conversation_history, strategy_decision)
        
        # Build system prompt with behavior hint (not intelligence)
        system_prompt = self._build_system_prompt(strategy_decision)
        
        # Build conversation context using centralized prompt builder
        conversation_context = PersonaAgentPrompts.build_conversation_context(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            current_message=message.text
        )
        
        try:
            # Generate response using Groq with tuned params for persona
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": conversation_context}
                ],
                temperature=0.7,  # Moderate temperature for natural human variety
                max_tokens=100,  # Shorter responses (1-2 sentences)
                top_p=0.9,  # Nucleus sampling for diversity
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            # Clean and humanize the response
            generated_text = self._clean_response(generated_text)
            generated_text = self._humanize_response(generated_text, probability=0.3)
            
            # Validate response (check for forbidden phrases)
            if self._has_forbidden_phrases(generated_text):
                logger.warning(f"PersonaAgent generated forbidden phrase, using fallback")
                return self._fallback_response(message, conversation_history, strategy_decision)
            
            # Log for debugging
            logger.debug(f"PersonaAgent generated response for goal={strategy_decision.goal.value}: {generated_text[:50]}...")
            
            return generated_text
        except Exception as e:
            logger.error(f"Error in PersonaAgent (Groq): {e}")
            return self._fallback_response(message, conversation_history, strategy_decision)
    
    def _build_system_prompt(self, strategy_decision: StrategyDecision) -> str:
        """
        Build system prompt with behavior hint (no intelligence awareness).
        Strategy sends concise behavior_hint, not full intelligence context.
        """
        # Get goal-specific behavior hint (simplified, no intelligence context)
        behavior_hint = PersonaAgentPrompts.get_goal_instruction(strategy_decision.goal)
        
        return PersonaAgentPrompts.BASE_PROMPT + behavior_hint
    
    def _clean_response(self, text: str) -> str:
        """Clean response text - remove quotes, extra whitespace, etc."""
        # Remove surrounding quotes if present
        text = text.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        
        # Remove explanation prefixes
        prefixes_to_remove = [
            "Response:",
            "Reply:",
            "Your response:",
            "Here's your response:"
        ]
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        return text.strip()
    
    def _humanize_response(self, text: str, probability: float = 0.3) -> str:
        """
        Probabilistically add human-like imperfections.
        
        Args:
            text: Response text
            probability: Probability of adding humanizer (0.0-1.0)
            
        Returns:
            Humanized text
        """
        if random.random() > probability:
            return text
        
        # Add hesitation at start (30% chance)
        if random.random() < 0.3 and len(text) > 20:
            hesitation = random.choice(AllowedFillers.HESITATION)
            text = f"{hesitation.capitalize()}, {text.lower()}"
        
        # Add uncertainty modifier (20% chance)
        if random.random() < 0.2:
            uncertainty = random.choice(AllowedFillers.UNCERTAINTY)
            # Insert before first verb or question
            words = text.split()
            if len(words) > 2:
                insert_pos = min(2, len(words) - 1)
                words.insert(insert_pos, uncertainty)
                text = " ".join(words)
        
        return text
    
    def _has_forbidden_phrases(self, text: str) -> bool:
        """Check if response contains forbidden phrases."""
        text_lower = text.lower()
        
        # Check forbidden phrases
        for phrase in ForbiddenPhrases.FORBIDDEN:
            if phrase.lower() in text_lower:
                return True
        
        # Check meta phrases
        for phrase in ForbiddenPhrases.META_PHRASES:
            if phrase.lower() in text_lower:
                return True
        
        return False
    
    def _fallback_response(
        self,
        message: Message,
        conversation_history: list[Message],
        strategy_decision: StrategyDecision
    ) -> str:
        """Fallback rule-based responses based on strategy."""
        return FallbackResponses.get_response(
            goal=strategy_decision.goal,
            message_text=message.text
        )
