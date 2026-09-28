import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from google import genai
from supabase import create_client, Client

app = FastAPI(title="Homework AI Backend")

# Allow requests from your Next.js frontend (Vercel or local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from Vercel & local environments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API clients from environment variables
groq_client = Groq(api_key=os.environ.get("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"))
gemini_client = genai.Client(api_key=os.environ.get("AQ.Ab8RN6JBD0IDojHbgPzBvGm7psykI7YlqJyJsOmm60nYzgMrdA"))

supabase_url = os.environ.get("https://sykibyaqyxsmsfbfqwpg.supabase.co")
supabase_key = os.environ.get("sb_secret__REBtTlKIO7YGima-8_VdQ_no4anlpk")
supabase: Client = create_client(https://sykibyaqyxsmsfbfqwpg.supabase.co, sb_secret__REBtTlKIO7YGima-8_VdQ_no4anlpk) if supabase_url and sb_secret__REBtTlKIO7YGima-8_VdQ_no4anlpk else None

class ChatRequest(BaseModel):
    prompt: str
    model: str = "groq"  # Options: 'groq' or 'gemini'

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Homework AI Backend is live!"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Streams response chunks using Groq or Gemini based on user routing.
    """
    if request.model == "groq":
        def generate_groq():
            stream = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert AI tutor. Explain step-by-step and use standard LaTeX formatting for math."},
                    {"role": "user", "content": request.prompt}
                ],
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        return StreamingResponse(generate_groq(), media_type="text/plain")

    elif request.model == "gemini":
        def generate_gemini():
            response = gemini_client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=request.prompt
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text

        return StreamingResponse(generate_gemini(), media_type="text/plain")

    else:
        raise HTTPException(status_code=400, detail="Invalid model selection. Choose 'groq' or 'gemini'.")
