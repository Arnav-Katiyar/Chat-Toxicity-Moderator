from fastapi import FastAPI
from pydantic import BaseModel
from detoxify import Detoxify
from openai import OpenAI
import os

app = FastAPI()

# Load Detoxify model for toxicity detection
tox_model = Detoxify("original")

# Use API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Message(BaseModel):
    text: str

def rephrase_with_gpt(text):
    prompt = f"Rewrite this message politely and respectfully without changing the meaning:\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

@app.post("/moderate")
async def moderate(msg: Message):
    text = msg.text.strip()
    tox_scores = tox_model.predict(text)
    toxicity = tox_scores.get("toxicity", 0)

    if toxicity <= 0.5:
        result = {"allowed": True}
    else:
        rephrased = rephrase_with_gpt(text)
        result = {"allowed": False, "rephrased": rephrased}

    print("Text:", text)
    print("Toxicity Score:", round(toxicity, 3))
    print("Allowed:", result["allowed"])
    if not result["allowed"]:
        print("Rephrased:", result["rephrased"])

    return result
