import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from google import genai
from supabase import create_client, Client

app = FastAPI(title="Homework AI Backend")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SDK clients using Render environment variables
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None


class ChatRequest(BaseModel):
    message: str = None
    prompt: str = None
    model: str = "groq"


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Homework AI Backend is live!"}


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_prompt = request.message or request.prompt
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No prompt or message provided.")

    system_instructions = (
        "You are a direct, concise homework tutor. "
        "Always format mathematical formulas using dollar sign delimiters: "
        "use $...$ for inline math and $$...$$ for block math equations. "
        "CRITICAL: Never use square brackets like \\[ ... \\] or [ ... ] for LaTeX equations. "
        "Example block math format: $$\\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$$"
    )

    if request.model == "groq":
        def generate_groq():
            try:
                stream = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=True,
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            except Exception as e:
                try:
                    fallback_stream = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": system_instructions},
                            {"role": "user", "content": user_prompt},
                        ],
                        stream=True,
                    )
                    for chunk in fallback_stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content
                except Exception as fallback_err:
                    yield f"Groq Error: {str(fallback_err)}"

        return StreamingResponse(generate_groq(), media_type="text/plain")

    elif request.model == "gemini":
        def generate_gemini():
            try:
                response = gemini_client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=user_prompt,
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            except Exception as e:
                yield f"Gemini Error: {str(e)}"

        return StreamingResponse(generate_gemini(), media_type="text/plain")

    else:
        raise HTTPException(status_code=400, detail="Invalid model selection specified.")
