import os
import re
import json
import asyncio
import anyio
from typing import List, Dict, Optional

import google.generativeai as genai
from detoxify import Detoxify
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline
from openai import OpenAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Singletons
tox_model = Detoxify("original")

# Initialize API clients
gemini_model = None
gemini_api_key = os.getenv("GEMINI_API_KEY")

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel(model_name='gemini-1.0-pro')

# Initialize OpenAI client for AIML API
aiml_client = None
aiml_api_key = os.getenv("AIMLAPI_KEY")

if aiml_api_key:
    aiml_client = OpenAI(
        base_url="https://api.aimlapi.com/v1",
        api_key=aiml_api_key,
    )

# Local fallback model: T5-Paws
paraphrase_pipeline = pipeline(
    "text2text-generation",
    model="Vamsi/T5_Paraphrase_Paws",
    tokenizer="Vamsi/T5_Paraphrase_Paws"
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections[:]:
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

# Separate manager for moderator WebSocket
moderator_manager = ConnectionManager()

class Message(BaseModel):
    text: str

def clean_rephrased_text(text: str) -> str:
    """Clean up AI-generated text by removing quotes, explanations, and prefixes."""
    text = text.strip()
    text = re.sub(r'^["\']|["\']$', '', text)
    text = re.sub(r'^\*\*|!\*\*$', '', text)
    
    prefixes_to_remove = [
        "Here's a polite version:",
        "Rewritten text:",
        "Polite version:",
        "Here is the rewritten text:",
        "The rewritten message is:",
    ]
    
    for prefix in prefixes_to_remove:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            text = re.sub(r'^[:;\-—]\s*', '', text)
    
    text = re.sub(r'^["\']|["\']$', '', text)
    return text.strip()

def is_valid_rephrasing(original: str, rephrased: str) -> bool:
    """Check if the rephrased text is valid and meaningful."""
    if not rephrased or len(rephrased) < 3:
        return False
    
    refusal_patterns = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "sorry", "apologize", "inappropriate",
        "i don't feel comfortable", "i cannot assist"
    ]
    
    rephrased_lower = rephrased.lower()
    if any(pattern in rephrased_lower for pattern in refusal_patterns):
        return False
    
    if rephrased.lower() == original.lower():
        return False
    
    return True

def create_generic_polite_message(original_text: str) -> str:
    """Create a context-aware polite message when rephrasing fails."""
    text_lower = original_text.lower()
    
    # Try to maintain context while making it polite
    if any(word in text_lower for word in ["stupid", "dumb", "idiot", "moron"]):
        return "I don't think that's the best approach."
    elif any(word in text_lower for word in ["hate", "awful", "terrible", "worst"]):
        return "I'm not comfortable with this."
    elif any(word in text_lower for word in ["shut up", "shut", "quiet"]):
        return "I'd prefer if we could pause this conversation."
    elif any(word in text_lower for word in ["wrong", "disagree", "no"]):
        return "I have a different perspective on this."
    elif any(word in text_lower for word in ["ugly", "bad", "trash"]):
        return "I don't think this meets our standards."
    elif "?" in original_text:
        return "Could you help me understand this better?"
    else:
        # Try to preserve some context from the original
        return "I'd like to express this more constructively."

def local_paraphrase(text: str, max_length: int = 128) -> str:
    """Enhanced local fallback using T5 with context preservation."""
    try:
        # Better prompt for context preservation
        input_text = f"rephrase politely while keeping the meaning: {text}"
        result = paraphrase_pipeline(
            input_text, 
            max_length=max_length, 
            num_return_sequences=1,
            do_sample=True,
            temperature=0.5  # Lower temperature for more consistent results
        )
        rephrased = result[0]['generated_text']
        rephrased = clean_rephrased_text(rephrased)
        
        if is_valid_rephrasing(text, rephrased):
            return rephrased
        return create_generic_polite_message(text)
    except Exception:
        return create_generic_polite_message(text)

def rephrase_with_gemini(text: str) -> Optional[str]:
    if not gemini_model:
        return None
        
    try:
        # Use delimiters to help the model distinguish instructions from toxic input
        prompt = (
            "TASK: Rewrite the following message to be polite and respectful while PRESERVING the original meaning and context.\n"
            "RULES:\n"
            "- Keep the SAME message intent and context\n"
            "- Only make it polite, respectful, and appropriate\n"
            "- Output ONLY the rewritten text (no quotes, no explanations)\n"
            "- Keep it natural and conversational\n"
            f"ORIGINAL MESSAGE: \"\"\"{text}\"\"\"\n\n"
            "POLITE VERSION:"
        )

        response = gemini_model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.1, # Lowered for consistency
                'max_output_tokens': 100,
            }
        )
        
        # Safe extraction logic
        if not response.candidates:
            return None
            
        candidate = response.candidates[0]
        if candidate.finish_reason == 3: # Safety block
            return None
            
        if candidate.content.parts:
            text_content = candidate.content.parts[0].text
            rephrased = clean_rephrased_text(text_content)
            if is_valid_rephrasing(text, rephrased):
                return rephrased
        
        return None
    except Exception as e:
        print(f"Gemini error: {e}")
        return None


def rephrase_with_aiml(text: str) -> Optional[str]:
    if not aiml_client:
        return None
        
    try:
        # Use delimiters to help the model distinguish instructions from toxic input
        prompt = (
            "TASK: Rewrite the following message to be polite and respectful while PRESERVING the original meaning and context.\n"
            "RULES:\n"
            "- Keep the SAME message intent and context\n"
            "- Only make it polite, respectful, and appropriate\n"
            "- Output ONLY the rewritten text (no quotes, no explanations)\n"
            "- Keep it natural and conversational\n"
            f"ORIGINAL MESSAGE: \"\"\"{text}\"\"\"\n\n"
            "POLITE VERSION:"
        )

        response = aiml_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100,
        )
        
        if response.choices and len(response.choices) > 0:
            rephrased = clean_rephrased_text(response.choices[0].message.content)
            if is_valid_rephrasing(text, rephrased):
                return rephrased
        
        return None
    except Exception as e:
        print(f"AIML API error: {e}")
        return None

async def rephrase_logic(text: str) -> str:
    """Core rephrasing logic with AIML API primary, then Gemini, then local fallback."""
    # Try AIML API first
    result = await anyio.to_thread.run_sync(rephrase_with_aiml, text)
    if result:
        return result
    
    # Then try Gemini
    result = await anyio.to_thread.run_sync(rephrase_with_gemini, text)
    if result:
        return result
    
    # Finally use local fallback
    result = await anyio.to_thread.run_sync(local_paraphrase, text)
    return result

@app.post("/moderate")
async def moderate(msg: Message):
    text = msg.text.strip()
    tox_scores = await anyio.to_thread.run_sync(tox_model.predict, text)
    toxicity = float(tox_scores.get("toxicity", 0))

    if toxicity <= 0.5:
        return {"allowed": True, "score": round(toxicity, 3), "text": text}
    
    rephrased_text = await rephrase_logic(text)
    return {
        "allowed": False, 
        "score": round(toxicity, 3), 
        "original": text,
        "rephrased": rephrased_text
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            text = message_data['text']
            username = message_data.get('username', 'Anonymous')
            
            tox_scores = await anyio.to_thread.run_sync(tox_model.predict, text)
            toxicity = float(tox_scores.get("toxicity", 0))
            
            response_data = {
                "username": username,
                "original_text": text,
                "toxicity": round(toxicity, 3),
                "timestamp": asyncio.get_event_loop().time()
            }
            
            if toxicity <= 0.5:
                response_data.update({"allowed": True, "display_text": text})
            else:
                rephrased_text = await rephrase_logic(text)
                response_data.update({
                    "allowed": False,
                    "display_text": rephrased_text,
                    "original_displayed": False
                })
            
            await manager.broadcast(json.dumps(response_data))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# NEW: Moderator WebSocket endpoint
@app.websocket("/ws/moderator")
async def moderator_websocket(websocket: WebSocket):
    await moderator_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            text = message_data['text']
            
            # Step 1: Compute toxicity
            tox_scores = await anyio.to_thread.run_sync(tox_model.predict, text)
            toxicity = float(tox_scores.get("toxicity", 0))
            
            # Step 2: Decide if rephrasing is needed (threshold 0.70)
            moderated = toxicity >= 0.50
            
            if moderated:
                # Rephrase the message
                final_text = await rephrase_logic(text)
            else:
                # Forward as-is
                final_text = text
            
            # Step 3: Broadcast structured response
            response = {
                "original": text,
                "final": final_text,
                "toxicity": round(toxicity, 3),
                "moderated": moderated
            }
            
            await moderator_manager.broadcast(json.dumps(response))
    except WebSocketDisconnect:
        moderator_manager.disconnect(websocket)

@app.get("/moderator")
async def moderator_page(request: Request):
    return templates.TemplateResponse("moderator.html", {"request": request})

@app.get("/demo")
async def demo_instructions(request: Request):
    return templates.TemplateResponse("demo.html", {"request": request})

@app.get("/sender")
async def sender_view(request: Request):
    return templates.TemplateResponse("sender.html", {"request": request})

@app.get("/receiver")
async def receiver_view(request: Request):
    return templates.TemplateResponse("receiver.html", {"request": request})