# AI Yourself API

Backend API for **AI Yourself**, a project exploring how GPT-2 represents names through tokenization.

Instead of asking whether a language model understands you, the API exposes one of the first operations it performs: breaking text into tokens.

## Endpoint

### `POST /tokenize`

Tokenizes the provided text using the GPT-2 tokenizer.

#### Request

```json
{
  "text": "Taylor"
}
```

#### Response

```json
{
  "text": "Taylor",
  "token_ids": [29907],
  "tokens": ["Taylor"],
  "whole": true
}
```

Example of a name split across multiple tokens:

```json
{
  "text": "Yusuf",
  "token_ids": [56, 18092],
  "tokens": ["Y", "usuf"],
  "whole": false
}
```

## Running locally

Install dependencies:

```bash
pip install fastapi uvicorn transformers torch
```

Start the server:

```bash
uvicorn main:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

Interactive documentation:

```
http://127.0.0.1:8000/docs
```

## Technology

- FastAPI
- Hugging Face Transformers
- GPT-2 Tokenizer

## About

This API powers **AI Yourself**, a work from the **On Language Models** collection.

Rather than treating language models as black boxes, the project exposes one of their simplest mechanisms: tokenization.
