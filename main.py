import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from google import genai
from supabase import create_client, Client

app = FastAPI(title="Homework AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

class ChatRequest(BaseModel):
    message: str = None
    prompt: str = None
    model: str = "groq"

SYSTEM_PROMPT = (
    "You are a direct, concise homework tutor. "
    "Always use clear spacing, new lines, and standard Markdown bullet points or tables. "
    "Format math using LaTeX delimiters like $...$ for inline or $$...$$ for standalone equations. "
    "Never repeat sentences or phrases."
)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Homework AI Backend is live!"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_prompt = request.message or request.prompt
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No prompt or message provided.")

    if request.model == "groq":
        async def generate_groq():
            try:
                # Primary model
                stream = groq_client.chat.completions.create(
                    model="qwen/qwen3.6-27b",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=True
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
                        await asyncio.sleep(0.01) # Prevents buffer congestion
            except Exception as e:
                yield f"\n[Error: {str(e)}]"

        return StreamingResponse(generate_groq(), media_type="text/event-stream")

    elif request.model == "gemini":
        async def generate_gemini():
            try:
                response = gemini_client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=user_prompt
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
                        await asyncio.sleep(0.01)
            except Exception as e:
                yield f"\n[Error: {str(e)}]"

        return StreamingResponse(generate_gemini(), media_type="text/event-stream")

    else:
        raise HTTPException(status_code=400, detail="Invalid model selection.")
