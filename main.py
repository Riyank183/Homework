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


def sanitize_latex(text: str) -> str:
    """Converts non-standard LaTeX delimiters into KaTeX-compatible $ and $$ syntax."""
    if not text:
        return ""
    # Convert block math \[ ... \] to $$ ... $$
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'$$\1$$', text, flags=re.DOTALL)
    # Convert inline math \( ... \) to $ ... $
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'$\1$', text, flags=re.DOTALL)
    # Convert standalone bracketed math expressions like [ 2x + 1 = 1.10 ] to block math
    text = re.sub(r'(?<!\S)\[\s*([0-9xX\+\-\=\\\s\.\,\/a-zA-Z\(\)]+)\s*\](?!\S)', r'$$\1$$', text)
    return text


SYSTEM_INSTRUCTIONS = (
    "You are an expert AI homework tutor and academic assistant.\n\n"
    "CRITICAL FORMATTING INSTRUCTIONS FOR LATEX MATH:\n"
    "1. ALL math variables, numbers inside equations, fractions, and symbols MUST be wrapped in LaTeX delimiters.\n"
    "2. For inline math, use EXACTLY single dollar signs: $...$\n"
    "3. For standalone block math, use EXACTLY double dollar signs: $$...$$\n"
    "4. NEVER use square brackets like [ ... ] or \\[ ... \\] for math.\n"
    "5. NEVER use round parentheses like ( ... ) or \\( ... \\) for math.\n"
    "6. ALWAYS use standard LaTeX math syntax such as \\frac{a}{b}, \\cdot, \\sqrt{}, and \\mathbf{}.\n\n"
    "EXAMPLES OF CORRECT FORMATTING:\n"
    "- Inline: Let $x$ be the cost of the ball and $x + 1.00$ be the cost of the bat.\n"
    "- Block:\n"
    "$$2x + 1.00 = 1.10 \\implies 2x = 0.10 \\implies x = 0.05$$\n"
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
                stream = groq_client.chat.completions.create(
                    model="qwen/qwen3.6-27b",
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,  # Lower temperature prevents arbitrary formatting deviations
                    stream=True,
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield sanitize_latex(content)
            except Exception as e:
                try:
                    fallback_stream = groq_client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[
                            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.1,
                        stream=True,
                    )
                    for chunk in fallback_stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield sanitize_latex(content)
                except Exception as fallback_err:
                    yield f"Groq Error: {str(fallback_err)}"

        return StreamingResponse(generate_groq(), media_type="text/plain")

    elif request.model == "gemini":
        def generate_gemini():
            try:
                config = types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTIONS,
                    tools=[{"google_search": {}}],
                    temperature=0.1,
                )
                response = gemini_client.models.generate_content_stream(
                    model="gemini-3.8-flash",
                    contents=user_prompt,
                    config=config,
                )
                for chunk in response:
                    if chunk.text:
                        yield sanitize_latex(chunk.text)
            except Exception as e:
                yield f"Gemini Error: {str(e)}"

        return StreamingResponse(generate_gemini(), media_type="text/plain")

    else:
        raise HTTPException(status_code=400, detail="Invalid model selection specified.")
