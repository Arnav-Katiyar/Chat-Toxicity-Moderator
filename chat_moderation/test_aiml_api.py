import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client for AIML API
aiml_api_key = os.getenv("AIMLAPI_KEY")

if not aiml_api_key:
    print("AIMLAPI_KEY not found in .env file")
    print("Please add your AIML API key to the .env file")
    exit(1)

client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=aiml_api_key,
)

# Test the API
try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Write a one-sentence story about numbers."}],
        temperature=0.7,
        max_tokens=100
    )
    
    print("AIML API Test Successful!")
    print("Response:", response.choices[0].message.content)
    
except Exception as e:
    print(f"Error testing AIML API: {e}")