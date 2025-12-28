# 🚀 Quick Start Guide

## How to Run Your Chat Toxicity Moderator

### Step 1: Start the Server
```bash
cd "c:/Users/Anadi Sharma/2k26 hackathons/Chat-Toxicity-Moderator/chat_moderation"
uvicorn main:app --reload
```

### Step 2: Open Two Browser Windows

#### Window 1 - SENDER (Person who types messages)
- Open: `http://localhost:8000/sender`
- This is where you type messages
- You'll see your original message

#### Window 2 - RECEIVER (Person who receives messages)  
- Open: `http://localhost:8000/receiver`
- This is what your friend sees
- They only see polite versions of toxic messages

### Step 3: Test It Out!

**Try typing toxic messages in the SENDER window:**

❌ "You're an idiot!"  
✅ Receiver sees: "I respectfully disagree with that perspective."

❌ "This is stupid and you're wrong"  
✅ Receiver sees: "I have a different viewpoint on this matter."

❌ "I hate this"  
✅ Receiver sees: "I have concerns about this."

### How It Works

```
SENDER                    AI SYSTEM                    RECEIVER
  │                          │                            │
  │──── "You idiot!" ───────>│                            │
  │                          │                            │
  │                          │ 🤖 Analyzing toxicity...   │
  │                          │ ⚠️  Score: 0.85 (TOXIC!)  │
  │                          │                            │
  │                          │ 🔄 Rephrasing...           │
  │                          │                            │
  │                          │─── "I respectfully ───────>│
  │                          │     disagree"              │
  │                                                       │
  │    Sees: "You idiot!"          Sees: "I respectfully │
  │    (original)                  disagree" (polite)    │
```

## 🎨 Visual Demo

### Sender View:
```
┌──────────────────────────────────┐
│  Chat Sender - Your View         │
├──────────────────────────────────┤
│                                  │
│  You: This is stupid! ➡️         │
│  (Toxicity detected: 75%)        │
│                                  │
│  [Type a message...]        [▶]  │
└──────────────────────────────────┘
```

### Receiver View:
```
┌──────────────────────────────────┐
│  Chat Receiver - Friend's View   │
├──────────────────────────────────┤
│                                  │
│  ⬅️ I have a different view     │
│      on this matter.             │
│  🛡️ (Filtered - Original 75%)   │
│                                  │
│  🛡️ All messages moderated      │
└──────────────────────────────────┘
```

## 💡 Tips

1. **Open both windows side-by-side** to see the difference in real-time
2. **Test with different toxic messages** to see how the AI rephrases them
3. **Normal messages pass through unchanged** (toxicity < 50%)
4. **Works offline!** Uses local T5 model when no API keys provided

## ⚙️ Optional: Add API Keys for Better Results

Edit `.env` file to add API keys:
```env
GEMINI_API_KEY=your_key_here
AIMLAPI_KEY=your_key_here
```

This improves rephrasing quality, but **system works fine without them**!

## 🐛 Troubleshooting

**Server not starting?**
```bash
# Check if port 8000 is already in use
netstat -ano | findstr :8000

# Kill the process if needed
taskkill /F /PID <process_id>
```

**Can't see messages in receiver?**
- Make sure both windows are open
- Check WebSocket connection (green dot in header)
- Try refreshing the receiver window

**Dependencies error?**
```bash
pip install -r requirements.txt --upgrade
```

## 🎯 Expected Behavior

✅ **Normal message** (toxicity < 0.5):
- Sender sees: "Hello, how are you?"
- Receiver sees: "Hello, how are you?" *(unchanged)*

✅ **Toxic message** (toxicity > 0.5):
- Sender sees: "You're an idiot"
- Receiver sees: "I respectfully disagree" *(AI rephrased)*

---

**Ready to test?** Open those two browser windows and start chatting! 💬
