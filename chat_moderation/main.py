from fastapi import FastAPI
from pydantic import BaseModel
from detoxify import Detoxify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

# Load Detoxify model for toxicity detection
tox_model = Detoxify("original")

# Load paraphrasing model (T5-based)
para_model_name = "Vamsi/T5_Paraphrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(para_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(para_model_name)

class Message(BaseModel):
    text: str

@app.post("/moderate")
async def moderate(msg: Message):
    text = msg.text.strip()
    tox_scores = tox_model.predict(text)
    toxicity = tox_scores.get("toxicity", 0)

    if toxicity <= 0.5:
        result = {"allowed": True}
    else:
        prompt = "paraphrase: " + text + " </s>"
        input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
        rephrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = {"allowed": False, "rephrased": rephrased}

    print("Text:", text)
    print("Toxicity Score:", round(toxicity, 3))
    print("Allowed:", result["allowed"])
    if not result["allowed"]:
        print("Rephrased:", result["rephrased"])

    return result
