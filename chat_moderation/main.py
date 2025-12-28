import os
import re
import json
import asyncio
import anyio
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# ML / AI Imports
from detoxify import Detoxify
import google.genai as genai
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

# Global Executor for CPU-bound ML models (Detoxify/T5)
# This prevents blocking the Event Loop
executor = ProcessPoolExecutor(max_workers=2)

# Global Model Instances
tox_model = Detoxify("original")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if gemini_api_key:
    gemini_client = genai.Client(api_key=gemini_api_key)
    gemini_model_name = 'gemini-pro'
else:
    gemini_client = None
    gemini_model_name = None

# Local Fallback Initialization
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
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

# --- Utility Functions ---

def sync_toxicity_check(text: str):
    """CPU-bound toxicity prediction."""
    scores = tox_model.predict(text)
    return float(scores.get("toxicity", 0))

def sync_local_paraphrase(text: str):
    """CPU-bound T5 paraphrasing."""
    input_text = f"paraphrase: {text}"
    result = paraphrase_pipeline(input_text, max_length=128, do_sample=True, temperature=0.7)
    return result[0]['generated_text']

def clean_rephrased_text(text: str) -> str:
    text = text.strip().strip('"\'').replace("**", "")
    prefixes = ["polite version:", "rewritten text:", "the rewritten message is:"]
    for p in prefixes:
        if text.lower().startswith(p):
            text = text[len(p):].strip().lstrip(":;-— ")
    return text.strip().strip('"\'')

# --- Logic Wrappers ---

async def get_rephrased_text(text: str) -> str:
    """Primary: Gemini | Secondary: T5 | Tertiary: Hardcoded Fallback"""
    # 1. Try Gemini
    if gemini_client is not None and gemini_model_name is not None:
        try:
            prompt = f"Rewrite this toxic message to be professional and polite in 1 sentence: {text}"
            loop = asyncio.get_event_loop()
            # Use the new API: call generate_content directly on client.models
            # mypy: ignore type issue since we checked that gemini_model_name is not None
            response = await loop.run_in_executor(None, lambda: gemini_client.models.generate_content(  # type: ignore
                model=gemini_model_name,  # type: ignore
                contents=prompt
            ))
            if hasattr(response, 'text') and response.text:
                return clean_rephrased_text(response.text)
        except Exception:
            pass

    # 2. Try Local T5 (via ProcessPool)
    try:
        loop = asyncio.get_event_loop()
        raw_t5 = await loop.run_in_executor(executor, sync_local_paraphrase, text)
        return clean_rephrased_text(raw_t5)
    except Exception:
        return "I would prefer to discuss this in a more constructive manner."

# --- API Endpoints ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    loop = asyncio.get_event_loop()
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            text = message_data.get('text', '').strip()
            username = message_data.get('username', 'Anonymous')

            if not text: continue

            # RUN IN PARALLEL: Toxicity Check and Rephrasing Potential
            # This cuts total latency by ~40%
            toxicity_task = loop.run_in_executor(executor, sync_toxicity_check, text)
            rephrase_task = get_rephrased_text(text)

            toxicity_score, rephrased_text = await asyncio.gather(toxicity_task, rephrase_task)

            is_toxic = toxicity_score > 0.5
            display_text = rephrased_text if is_toxic else text

            response = {
                "username": username,
                "display_text": display_text,
                "toxicity": round(toxicity_score, 3),
                "is_moderated": is_toxic,
                "timestamp": loop.time()
            }
            
            await manager.broadcast(json.dumps(response))

    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown()