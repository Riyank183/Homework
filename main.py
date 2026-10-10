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

STRICT MATHEMATICAL FORMATTING RULES:
1. ALWAYS use dollar sign delimiters for all mathematical equations, variables, numbers, percentages, and formulas.
2. Inline math MUST use single dollar signs: $...$ (e.g., $x = 4$, $15\%$, or $\text{price} = 2400$).
3. Standalone block equations MUST use double dollar signs: $$...$$
4. ABSOLUTE PROHIBITION: NEVER use square brackets like \[ ... \] or single [ ... ] or parentheses \( ... \) for equations.
5. ALWAYS format fractions using \frac{a}{b} and ensure text inside math blocks uses \text{...}.
"""


def clean_math_delimiters(text: str) -> str:
    """Universal LaTeX sanitizer that converts ALL non-standard math delimiters to $ and $$ syntax."""
    if not text:
        return ""

    # 1. Convert standard LaTeX block math \[ ... \] to $$ ... $$
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'$$\1$$', text, flags=re.DOTALL)
    
    # 2. Convert inline math \( ... \) to $ ... $
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'$\1$', text, flags=re.DOTALL)

    # 3. Fix unclosed/dangling [ \text{...} or [ math expressions (convert standalone opening [ to $$)
    text = re.sub(r'(?<!\S)\[\s*(\\text\{|\\frac\{|[0-9xX\+\-\=\\\s\.\,\/\^\_\{\}\(\)a-zA-Z]+)', r'$$\1', text)

    # 4. Clean up any leftover trailing raw ] at the end of math statements
    text = re.sub(r'(\\text\{[^\}]+\}|[0-9xX\+\-\=\\\s\.\,\/\^\_\{\}\(\)a-zA-Z]+)\s*\](?!\S)', r'\1$$', text)

    # 5. Fix double-yield dollar sign duplications if any
    text = text.replace("$$$$", "$$")
    
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

                accumulated_text = ""
                last_sent_len = 0

                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        accumulated_text += content
                        sanitized_full = clean_math_delimiters(accumulated_text)
                        
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
