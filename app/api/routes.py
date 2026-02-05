"""API routes."""
from fastapi import APIRouter, Depends, Request, Query
from typing import Optional
from app.api.schemas import HoneypotRequest, HoneypotResponse
from app.dependencies import verify_api_key
from app.core.orchestrator import orchestrator
from app.core.session_manager import session_manager
from app.core.scam_detector import scam_detector
from app.services.callback_service import callback_service
from app.agents.safety_guard import safety_guard
from app.utils.logger import logger
from app.models.session_state import Message, Metadata

router = APIRouter()


@router.post("/honeypot/message", response_model=HoneypotResponse)
@router.get("/honeypot/message", response_model=HoneypotResponse)
async def process_message(
    http_request: Request,
    request: Optional[HoneypotRequest] = None,
    api_key_valid: bool = Depends(verify_api_key),
    # GET parameters (for GET requests)
    sessionId: Optional[str] = Query(None, alias="sessionId"),
    text: Optional[str] = Query(None, alias="text"),
    sender: Optional[str] = Query("scammer", alias="sender"),
    timestamp: Optional[str] = Query(None, alias="timestamp")
) -> HoneypotResponse:
    """
    Main endpoint for processing scam messages.
    
    Supports both POST (with JSON body) and GET (with query parameters).
    
    This endpoint:
    1. Receives incoming messages
    2. Detects scam intent
    3. Activates AI agent if scam detected
    4. Extracts intelligence
    5. Returns human-like response
    """
    try:
        # Handle GET requests with query parameters
        if http_request and http_request.method == "GET":
            if not sessionId or not text:
                return HoneypotResponse(
                    status="error",
                    error="GET requests require sessionId and text query parameters"
                )
            
            # Build request from query parameters
            request = HoneypotRequest(
                sessionId=sessionId,
                message=Message(
                    sender=sender or "scammer",
                    text=text,
                    timestamp=timestamp or "2026-01-21T10:15:30Z"
                ),
                conversationHistory=[],
                metadata=Metadata(
                    channel="SMS",
                    language="English",
                    locale="IN"
                ) if timestamp else None
            )
        
        # Ensure request is provided
        if not request:
            return HoneypotResponse(
                status="error",
                error="Request body or query parameters required"
            )
        
        # Get or create session
        session = session_manager.get_or_create_session(request.sessionId)
        
        # Update session with conversation history from request
        if request.conversationHistory:
            session.conversationHistory = request.conversationHistory
        
        # Detect scam intent (LLM-first with rule-based fallback)
        detection_result = scam_detector.detect_scam(
            request.message,
            session.conversationHistory
        )
        
        # Update session with scam detection status and audit fields
        session = session_manager.update_session(
            request.sessionId,
            request.message,
            scam_detected=detection_result.is_scam,
            scam_confidence=detection_result.confidence
        )
        
        # Store audit fields for explainability
        session.llmScamResult = detection_result.llm_result
        session.finalDecisionReason = detection_result.final_decision_reason
        
        # Add initial scam detection note
        if is_scam:
            session_manager.add_agent_note(
                request.sessionId,
                f"Scam detected with confidence {detection_result.confidence:.2f}"
            )
        
        # Log detection details for audit trail
        method = "rule-based fallback" if detection_result.rule_based_fallback else "LLM"
        logger.info(
            f"Scam detection for session {session.sessionId} ({method}): "
            f"is_scam={detection_result.is_scam}, "
            f"confidence={detection_result.confidence:.2f}, "
            f"reason={detection_result.final_decision_reason}"
        )
        
        is_scam = detection_result.is_scam
        
        # If scam detected, activate agent
        if is_scam:
            # Process message through orchestrator
            # Orchestrator handles: Intelligence (parallel) -> Strategy -> Persona -> Safety
            agent_response = orchestrator.process_message(
                request.message,
                session
            )
            
            # Safety guard is already called inside orchestrator
            # But we still need to handle the response
            if agent_response:
                # Create user message from agent response for history
                from app.models.session_state import Message
                user_message = Message(
                    sender="user",
                    text=agent_response,
                    timestamp=request.message.timestamp
                )
                session.conversationHistory.append(user_message)
                
                # Check if conversation has ended and callback should be sent
                # Callback is only sent when conversation is completed (not during active engagement)
                if session.conversationEnded and callback_service.should_send_callback(session):
                    # Send callback asynchronously (don't block response)
                    try:
                        callback_service.send_callback(session)
                        logger.info(f"Callback sent for completed conversation: {session.sessionId}")
                    except Exception as e:
                        logger.error(f"Error in callback (non-blocking): {e}")
                
                return HoneypotResponse(
                    status="success",
                    reply=agent_response
                )
            else:
                # Agent decided not to continue engagement (conversation ended)
                # Mark conversation as ended and send callback
                session.conversationEnded = True
                
                if callback_service.should_send_callback(session):
                    try:
                        callback_service.send_callback(session)
                        logger.info(f"Callback sent for ended conversation: {session.sessionId}")
                    except Exception as e:
                        logger.error(f"Error in callback (non-blocking): {e}")
                
                return HoneypotResponse(
                    status="success",
                    reply=None  # No response - conversation ended
                )
        else:
            # Not a scam - return safe response
            return HoneypotResponse(
                status="success",
                reply=None  # Don't engage with non-scam messages
            )
    
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        return HoneypotResponse(
            status="error",
            error=str(e)
        )
