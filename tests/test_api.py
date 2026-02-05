"""API integration tests."""
import requests
import json
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000/honeypot/message"
API_KEY = "your-secret-api-key-here"  # Update this


def test_first_message():
    """Test with first message (no conversation history)."""
    payload = {
        "sessionId": "test-session-123",
        "message": {
            "sender": "scammer",
            "text": "Your bank account will be blocked today. Verify immediately.",
            "timestamp": datetime.now().isoformat() + "Z"
        },
        "conversationHistory": [],
        "metadata": {
            "channel": "SMS",
            "language": "English",
            "locale": "IN"
        }
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print("🧪 Testing first message...")
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        print(f"\n✅ Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_follow_up_message():
    """Test with follow-up message (with conversation history)."""
    payload = {
        "sessionId": "test-session-123",
        "message": {
            "sender": "scammer",
            "text": "Share your UPI ID to avoid account suspension.",
            "timestamp": datetime.now().isoformat() + "Z"
        },
        "conversationHistory": [
            {
                "sender": "scammer",
                "text": "Your bank account will be blocked today. Verify immediately.",
                "timestamp": datetime.now().isoformat() + "Z"
            },
            {
                "sender": "user",
                "text": "Why will my account be blocked?",
                "timestamp": datetime.now().isoformat() + "Z"
            }
        ],
        "metadata": {
            "channel": "SMS",
            "language": "English",
            "locale": "IN"
        }
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print("\n🧪 Testing follow-up message...")
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        print(f"\n✅ Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_health_check():
    """Test health check endpoint."""
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"\n🏥 Health Check: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Agentic Honeypot API Test Suite")
    print("=" * 60)
    
    # Test health check
    test_health_check()
    
    # Test first message
    result1 = test_first_message()
    
    # Test follow-up message
    if result1:
        result2 = test_follow_up_message()
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)
