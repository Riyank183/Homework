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


SYSTEM_INSTRUCTIONS = r"""You are an expert AI homework tutor.

CRITICAL MATHEMATICAL FORMATTING RULES:
1. Every mathematical variable, number in an equation, fraction, or formula MUST be enclosed in dollar sign delimiters.
2. Use single dollar signs $ ... $ for inline math expressions (e.g., $x = 4$ or $3x^2 - 11x - 4 = 0$).
3. Use double dollar signs $$ ... $$ for standalone centered block equations.
4. STRICTLY PROHIBITED: NEVER use \[ ... \] or \( ... \) or raw square brackets [ ... ] or parentheses ( ... ) around equations.
5. Always use standard LaTeX syntax like \frac{a}{b}, \sqrt{}, \pm, and \quad.

EXAMPLE OF CORRECT FORMATTING:
To solve the quadratic equation $3x^2 - 11x - 4 = 0$, use the quadratic formula:
$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$
Here $a = 3$, $b = -11$, and $c = -4$.
"""


def clean_math_delimiters(text: str) -> str:
    """Converts non-standard LaTeX delimiters to KaTeX-compatible $ and $$ syntax."""
    if not text:
        return ""
    # Convert \[ ... \] to $$ ... $$
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'$$\1$$', text, flags=re.DOTALL)
    # Convert \( ... \) to $ ... $
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'$\1$', text, flags=re.DOTALL)
    # Convert standalone bracketed math expressions like [ 3x^2-11x-4=0 ] to $$ ... $$
    text = re.sub(r'(?<!\S)\[\s*([0-9xX\+\-\=\\\s\.\,\/\^\_\{\}\(\)a-zA-Z]+)\s*\](?!\S)', r'$$\1$$', text)
    return text


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
                    temperature=0.1,
                    stream=True,
                )

                # Use a small sliding buffer so opening and closing delimiters in separate chunks get caught
                accumulated_text = ""
                last_sent_len = 0

                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        accumulated_text += content
                        sanitized_full = clean_math_delimiters(accumulated_text)
                        
                        # Only yield newly sanitized delta content
                        new_delta = sanitized_full[last_sent_len:]
                        if new_delta:
                            yield new_delta
                            last_sent_len = len(sanitized_full)

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
                    accumulated_text = ""
                    last_sent_len = 0
                    for chunk in fallback_stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            accumulated_text += content
                            sanitized_full = clean_math_delimiters(accumulated_text)
                            new_delta = sanitized_full[last_sent_len:]
                            if new_delta:
                                yield new_delta
                                last_sent_len = len(sanitized_full)
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
                accumulated_text = ""
                last_sent_len = 0
                for chunk in response:
                    if chunk.text:
                        accumulated_text += chunk.text
                        sanitized_full = clean_math_delimiters(accumulated_text)
                        new_delta = sanitized_full[last_sent_len:]
                        if new_delta:
                            yield new_delta
                            last_sent_len = len(sanitized_full)
            except Exception as e:
                yield f"Gemini Error: {str(e)}"

        return StreamingResponse(generate_gemini(), media_type="text/plain")

    else:
        raise HTTPException(status_code=400, detail="Invalid model selection specified.")
