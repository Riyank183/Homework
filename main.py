import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from google import genai
from supabase import create_client, Client

app = FastAPI(title="Homework AI Backend")

# Allow requests from your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reads keys safely from Render's environment variables (DO NOT paste literal keys here!)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

class ChatRequest(BaseModel):
    prompt: str
    model: str = "groq"

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Homework AI Backend is live!"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if request.model == "groq":
        def generate_groq():
            stream = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a direct, concise homework tutor. Format math in LaTeX."},
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
        raise HTTPException(status_code=400, detail="Invalid model selection.")
