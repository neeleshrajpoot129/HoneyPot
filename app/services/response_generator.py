"""Response generator service."""
# Response generation is handled by PersonaAgent
# This service can be extended for additional response processing
from app.agents.persona_agent import PersonaAgent

# Alias for service layer
response_generator = PersonaAgent()
