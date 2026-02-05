"""Main FastAPI application for the honeypot system."""
from fastapi import FastAPI
from app.api.routes import router
from app.utils.logger import logger

app = FastAPI(
    title="Agentic Honeypot API",
    description="AI-powered honeypot for scam detection and intelligence extraction",
    version="1.0.0"
)

# Include routers
app.include_router(router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Agentic Honeypot API is running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Agentic Honeypot API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
