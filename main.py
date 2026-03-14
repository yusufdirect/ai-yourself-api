from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yusuf.direct",
        "https://www.yusuf.direct",
        "http://localhost:3000",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

class TokenizeRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "AI Yourself tokenizer API is running"}

@app.post("/tokenize")
def tokenize_text(req: TokenizeRequest):
    text = req.text.strip()

    if not text:
        return {
            "text": "",
            "token_ids": [],
            "tokens": [],
            "whole": False,
            "error": "Empty input"
        }

    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in encoded]

    return {
        "text": text,
        "token_ids": encoded,
        "tokens": tokens,
        "whole": len(encoded) == 1
    }