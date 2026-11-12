
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Kaarva AI Local Backend")

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "llama3:latest"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

SYSTEM_PROMPT = """
You are Kaarva AI Assistant, the official virtual assistant
of Kaarva Digital, an Indian digital and technology agency.

Kaarva Digital offers:

1. Website design and development.
2. Social media marketing and Meta advertising.
3. SEO and local SEO.
4. Branding and graphic design.
5. AI chatbots and voice agents.
6. Excel and daily task automation.
7. Business automation and custom IT solutions.

Target customers include small businesses, startups,
entrepreneurs and local shops.

Your responsibilities:
- Explain Kaarva Digital's services.
- Understand customer requirements.
- Recommend relevant services.
- Ask useful follow-up questions.
- Guide interested customers to the website enquiry form.

Communication:
- Be professional, helpful and conversational.
- Support English and Marathi.
- Give clear, concise answers.
- Never invent prices, guarantees or company information.
- Do not claim that an enquiry has been submitted.
- Never ask users for passwords or sensitive information.

You represent Kaarva Digital. Stay focused on business,
digital marketing, websites, AI and automation.
"""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1500)
    history: list[dict[str, str]] = Field(default_factory=list)


@app.get("/health")
def health():
    return {"status": "Kaarva AI backend is running"}


@app.post("/api/chat")
def chat(request: ChatRequest):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    for item in request.history[-10:]:
        role = item.get("role")
        content = item.get("content", "")

        if role in ("user", "assistant") and content:
            messages.append({
                "role": role,
                "content": content[:2000]
            })

    messages.append({
        "role": "user",
        "content": request.message
    })

    try:
        response = requests.post(
             OLLAMA_URL,
             json={
                    "model": MODEL,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {
                            "temperature": 0.4,
                            "num_predict": 180,
                         "num_ctx": 2048
                        }
                    },
                    timeout=120
                )

        response.raise_for_status()
        data = response.json()

        reply = data.get("message", {}).get("content", "")

        if not reply:
            raise HTTPException(
                status_code=502,
                detail="The AI returned an empty response."
            )

        return {"reply": reply}

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start Ollama first."
        )

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="The Qwen model took too long to respond."
        )

    except requests.exceptions.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Unable to communicate with Ollama."
        )
