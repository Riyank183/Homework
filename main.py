import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from google import genai
from google.genai import types
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


def sanitize_latex(text: str) -> str:
    """Fixes broken LaTeX delimiters before sending to frontend."""
    # Convert \[ ... \] -> $$ ... $$
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'$$\1$$', text, flags=re.DOTALL)
    # Convert \( ... \) -> $ ... $
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'$\1$', text, flags=re.DOTALL)
    return text


SYSTEM_INSTRUCTIONS = (
    "You are an expert AI homework tutor.\n\n"
    "CRITICAL FORMATTING INSTRUCTIONS FOR LATEX MATH:\n"
    "1. Every single mathematical variable, vector, set, symbol, or equation MUST be wrapped in single dollar sign delimiters: $ ... $ for inline math.\n"
    "2. Use double dollar signs $$ ... $$ for standalone block equations.\n"
    "3. NEVER EVER use parentheses like \\( ... \\) or square brackets like \\[ ... \\].\n"
    "4. Examples:\n"
    "   - Correct inline: Let $x$ and $y$ be points in $\\mathbf{R}^n$.\n"
    "   - Incorrect inline: Let (x) and (y) be points in (\\mathbf{R}^n).\n"
    "   - Correct block: $$d(x,y) = \\sqrt{\\sum_{i=1}^n (x_i - y_i)^2}$$\n"
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
        def generate_groq():
            try:
                # Accumulate buffer to ensure split delimiters like \( and \) across chunk boundaries get sanitized cleanly
                buffer = ""
                stream = groq_client.chat.completions.create(
                    model="qwen/qwen3.6-27b",
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    stream=True,
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        buffer += content
                        # Yield sanitized output
                        sanitized = sanitize_latex(buffer)
                        yield sanitized
                        buffer = ""
                if buffer:
                    yield sanitize_latex(buffer)
            except Exception as e:
                yield f"Groq Error: {str(e)}"

        return StreamingResponse(generate_groq(), media_type="text/plain")

    elif request.model == "gemini":
        def generate_gemini():
            try:
                config = types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTIONS,
                    temperature=0.1,
                )
                response = gemini_client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=user_prompt,
                    config=config,
                )
                buffer = ""
                for chunk in response:
                    if chunk.text:
                        buffer += chunk.text
                        yield sanitize_latex(buffer)
                        buffer = ""
                if buffer:
                    yield sanitize_latex(buffer)
            except Exception as e:
                yield f"Gemini Error: {str(e)}"

        return StreamingResponse(generate_gemini(), media_type="text/plain")

    else:
        raise HTTPException(status_code=400, detail="Invalid model selection specified.")
