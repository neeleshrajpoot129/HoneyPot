"""FastAPI dependencies."""
from fastapi import HTTPException, Header
from app.config import config


def verify_api_key(x_api_key: str = Header(...)) -> bool:
    """Verify API key from header."""
    if x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True
