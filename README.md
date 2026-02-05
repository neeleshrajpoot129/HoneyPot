# Agentic Honeypot for Scam Detection & Intelligence Extraction

An AI-powered honeypot system that detects scam messages, autonomously engages scammers in multi-turn conversations, and extracts actionable intelligence without revealing detection.

## 🎯 Features

- **LLM-First Scam Detection** - Uses Groq (Llama 3.1 70B) with rule-based fallback
- **Multi-Agent System** - Persona, Strategy, Intelligence Extraction, and Safety Guard agents
- **Human-like Engagement** - Maintains believable persona throughout conversations
- **Intelligence Extraction** - Extracts UPI IDs, phone numbers, phishing links, bank accounts
- **Safety & Ethics** - Built-in guardrails to prevent unethical behavior

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```env
API_KEY=your-secret-api-key-here
GUVI_CALLBACK_URL=https://hackathon.guvi.in/api/updateHoneyPotFinalResult
GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL=llama-3.1-70b-versatile
```


### 3. Run Server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API available at: `http://localhost:8000`

## 📡 API Usage

### POST `/honeypot/message`

```bash
curl -X POST "http://localhost:8000/honeypot/message" \
  -H "x-api-key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "test-123",
    "message": {
      "sender": "scammer",
      "text": "Your account will be blocked. Verify immediately.",
      "timestamp": "2026-01-21T10:15:30Z"
    },
    "conversationHistory": [],
    "metadata": {
      "channel": "SMS",
      "language": "English",
      "locale": "IN"
    }
  }'
```

**Response:**
```json
{
  "status": "success",
  "reply": "Why is my account being blocked?"
}
```

## 📁 Project Structure

```
app/
├── main.py              # FastAPI app
├── config.py           # Configuration
├── api/                 # API routes & schemas
├── core/                # Orchestrator, Session Manager, Scam Detector
├── agents/              # Persona, Strategy, Safety Guard
├── services/            # Callback Service
├── utils/               # Prompts, Patterns, Keywords
└── models/              # Data models
```

## 🔄 How It Works

1. **Detect** - Analyzes incoming message for scam indicators
2. **Engage** - If scam detected, activates multi-agent system
3. **Extract** - Passively extracts intelligence (UPI, phone, links, etc.)
4. **Respond** - Generates human-like responses maintaining persona
5. **Validate** - Safety guard ensures ethical compliance
6. **Callback** - Sends final intelligence to your api endpoint(make an api to receive callback of information extracted)

## 🛡️ Safety & Ethics

- Never reveals detection status
- No impersonation of real individuals
- No illegal instructions
- Responsible data handling
- Built-in safety guardrails