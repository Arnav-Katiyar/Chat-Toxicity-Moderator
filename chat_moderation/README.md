# Chat Toxicity Moderator with AI Rephrasing

This application moderates chat messages in real-time by detecting toxicity and automatically rephrasing inappropriate messages into polite versions using AI models.

## 🎯 How It Works

1. **Sender** types a message in the sender view
2. **AI Analysis**: The system detects if the message is toxic/inappropriate
3. **Smart Rephrasing**: If toxic, the AI automatically converts it to a polite version
4. **Receiver** only sees the polite, moderated version

## ✨ Features

- **Real-time toxicity detection** using Detoxify model
- **Automatic message rephrasing** using AI (with fallback options):
  - AIML API (GPT-4o) - Primary (requires API key)
  - Google Gemini - Secondary (requires API key)
  - Local T5 model - Fallback (works without API keys)
- **WebSocket-based** real-time communication
- **Clean, modern UI** with separate sender/receiver views
- **Works offline** with local T5 model when API keys are not provided

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)
Create or edit `.env` file in the root directory:
```env
# Optional: For better rephrasing quality
GEMINI_API_KEY=your_gemini_api_key_here
AIMLAPI_KEY=your_aimlapi_key_here
```

**Note**: If you don't provide API keys, the system will automatically use the local T5 model for rephrasing.

- Get Gemini API key: https://aistudio.google.com/
- Get AIML API key: https://aimlapi.com/

### 3. Run the Application
```bash
uvicorn main:app --reload
```

## 📱 Usage

Once the server is running, open these URLs in separate browser windows/tabs:

1. **Sender View** (for sending messages): 
   - `http://localhost:8000/sender`
   - Type any message (including toxic ones)
   - See your original message in your view

2. **Receiver View** (for receiving moderated messages):
   - `http://localhost:8000/receiver`
   - Only sees polite, filtered versions of messages
   - Gets notified when a message was moderated

### Example Flow:

**Sender types**: "You're an idiot and this is stupid"

**What sender sees**: Their original message

**What receiver sees**: "I respectfully disagree with that perspective." *(AI-rephrased polite version)*

## 🔧 API Endpoints

- `GET /` - Main interface
- `GET /sender` - Sender chat view
- `GET /receiver` - Receiver chat view (moderated)
- `POST /moderate` - API endpoint for message moderation
- `WebSocket /ws` - Real-time chat communication

## 🧪 Testing

You can test the moderation API directly:

```python
import requests

response = requests.post(
    "http://localhost:8000/moderate",
    json={"text": "This is a test message"}
)
print(response.json())
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python
- **AI Models**: 
  - Detoxify (toxicity detection)
  - T5-Paws (local paraphrasing)
  - Google Gemini (optional)
  - AIML API/GPT-4o (optional)
- **Frontend**: HTML, CSS, JavaScript, WebSockets
- **Real-time Communication**: WebSockets

## 📝 Notes

- The toxicity threshold is set to 0.5 (messages with toxicity > 0.5 are rephrased)
- The system uses multiple AI models in priority order for best results
- Local T5 model ensures the system works even without internet/API keys
- All messages are processed in real-time with minimal latency

## 🔒 Privacy

- No messages are stored permanently
- All processing happens in real-time
- When using API services, messages are sent to external APIs for rephrasing

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

Run the test script to verify your AIML API key is working:
```bash
python test_aiml_api.py
```

## How It Works

1. A user sends a message through the sender interface
2. The message is checked for toxicity using the Detoxify model
3. If the toxicity score is above 0.5, the message is sent to the AI rephrasing service
4. The rephrased message is sent to the receiver interface
5. The original message remains unchanged for the sender