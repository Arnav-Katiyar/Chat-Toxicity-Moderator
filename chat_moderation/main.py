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
    gemini_model = genai.GenerativeModel('gemini-pro')

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
    """Create a generic polite message when rephrasing fails."""
    text_lower = original_text.lower()
    
    if any(word in text_lower for word in ["disagree", "wrong", "no", "don't"]):
        return "I respectfully disagree with that perspective."
    elif any(word in text_lower for word in ["****", "****", "****"]):
        return "I have a different viewpoint on this matter."
    elif any(word in text_lower for word in ["hate", "awful", "terrible"]):
        return "I have concerns about this."
    elif "?" in original_text:
        return "Could you please clarify your point?"
    else:
        return "Thank you for sharing your thoughts. I'd like to discuss this further."

def local_paraphrase(text: str, max_length: int = 128) -> str:
    """Enhanced local fallback using T5 with post-processing."""
    try:
        input_text = f"paraphrase: {text}"
        result = paraphrase_pipeline(
            input_text, 
            max_length=max_length, 
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )
        rephrased = result[0]['generated_text']
        rephrased = clean_rephrased_text(rephrased)
        
        if is_valid_rephrasing(text, rephrased):
            return rephrased
        return create_generic_polite_message(text)
    except Exception:
        return create_generic_polite_message(text)

def rephrase_with_gemini(text: str) -> Optional[str]:
    """Enhanced rephrasing using Google Gemini with better prompting."""
    if not gemini_model:
        return None
        
    try:
        prompt = (
            f"You are a professional communication expert. Your task is to transform toxic or rude messages "
            f"into polite, professional equivalents while preserving the core message intent.\n\n"
            f"Rules:\n1. Output ONLY the rewritten text - no explanations, quotes, or prefixes\n"
            f"2. Keep the message concise (1 sentence maximum)\n3. Preserve the original intent and meaning\n"
            f"4. Make it professional and respectful\n5. Do not add apologies or refusals\n\n"
            f"Original message: {text}\n\nPolite version:"
        )

        response = gemini_model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 100,
            }
        )
        
        text_content = None
        if hasattr(response, 'text') and response.text:
            text_content = response.text
        elif response.candidates and response.candidates[0].content.parts:
            text_content = response.candidates[0].content.parts[0].text
        
        if text_content:
            rephrased = clean_rephrased_text(text_content)
            if is_valid_rephrasing(text, rephrased):
                return rephrased
        return None
    except Exception as e:
        print(f"Gemini error: {e}")
        return None

async def rephrase_logic(text: str) -> str:
    """Core rephrasing logic with Gemini primary and local fallback."""
    result = await anyio.to_thread.run_sync(rephrase_with_gemini, text)
    if result:
        return result
    
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

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/sender")
async def sender_view(request: Request):
    return templates.TemplateResponse("sender.html", {"request": request})

@app.get("/receiver")
async def receiver_view(request: Request):
    return templates.TemplateResponse("receiver.html", {"request": request})