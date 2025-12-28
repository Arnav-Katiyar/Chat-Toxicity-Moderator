import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    print("Starting Chat Toxicity Moderator Server...")
    print("API Keys loaded:")
    print(f"Gemini API Key: {'Found' if os.getenv('GEMINI_API_KEY') else 'Not found'}")
    print(f"AIML API Key: {'Found' if os.getenv('AIMLAPI_KEY') else 'Not found'}")
    print("\nServer starting on http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )