from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import random
import json
import torch
import numpy as np
from model import NeuralNet
from nltk_utils import tokenize, lemmatize_word
from Preprocessing import tfidf_vectorizer
import os

# Initialize FastAPI app
app = FastAPI()


os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)


app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Load intents
with open('Intent.json', 'r') as json_data:
    intents = json.load(json_data)

# Load trained model
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sora"

def preprocess_text(text: str) -> str:
    """Tokenize, lemmatize, and join back to string"""
    tokens = tokenize(text)
    lemmatized = [lemmatize_word(w) for w in tokens]
    return " ".join(lemmatized)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    sentence = request.message

    # Preprocess input
    processed_sentence = preprocess_text(sentence)

    # Transform using TF-IDF
    X = tfidf_vectorizer.transform([processed_sentence])
    X = X.toarray().astype(np.float32)
    X = torch.from_numpy(X)

    # Predict
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.5:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return ChatResponse(response=random.choice(intent['responses']))
    else:
        return ChatResponse(response="I do not understand...")