"""
Simple test script to verify the toxicity moderator is working
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_moderate_endpoint():
    """Test the /moderate endpoint with various messages"""
    
    print("=" * 60)
    print("TESTING CHAT TOXICITY MODERATOR")
    print("=" * 60)
    print()
    
    test_messages = [
        "Hello, how are you?",  # Normal message
        "You're an idiot!",  # Toxic
        "This is stupid and you're wrong",  # Toxic
        "I hate this",  # Toxic
        "Great job on the project!",  # Normal
        "You're so dumb it's unbelievable",  # Toxic
        "Can you help me understand this?",  # Normal
    ]
    
    for message in test_messages:
        print(f"\n📝 Testing: '{message}'")
        print("-" * 60)
        
        try:
            response = requests.post(
                f"{BASE_URL}/moderate",
                json={"text": message},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("allowed"):
                    print(f"✅ ALLOWED (Toxicity: {data['score']})")
                    print(f"   Message: {data['text']}")
                else:
                    print(f"⚠️  TOXIC (Toxicity: {data['score']})")
                    print(f"   Original:  {data['original']}")
                    print(f"   Rephrased: {data['rephrased']}")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    print("\n⚠️  Make sure the server is running:")
    print("   uvicorn main:app --reload")
    print()
    input("Press Enter to start testing...")
    
    test_moderate_endpoint()
