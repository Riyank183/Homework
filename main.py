import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
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

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None


class ChatRequest(BaseModel):
    message: str = None
    prompt: str = None
    model: str = "groq"


# Universal System Prompt - Applies to ALL academic topics
SYSTEM_INSTRUCTIONS = (
    "You are an expert AI homework tutor and up-to-date research assistant.\n\n"
    "MATHEMATICAL FORMATTING RULES:\n"
    "1. Always wrap inline mathematical variables, numbers, and short expressions in single dollar signs: $...$\n"
    "2. Always wrap centered standalone equations and formulas in double dollar signs: $$...$$\n"
    "3. NEVER use square brackets like \\[ ... \\] or [ ... ] for equations.\n"
    "4. NEVER use parentheses like \\( ... \\) for inline math.\n"
    "5. Use standard LaTeX syntax (e.g., \\frac{a}{b}, \\sqrt{}, \\int, \\sum, \\text{})."
)


# Helper function to fetch live web context using Tavily API
async def fetch_tavily_context(query: str) -> str:
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        return ""
    
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "max_results": 4,
        "search_depth": "basic"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=5.0)
            if response.status_code == 200:
                results = response.json().get("results", [])
                if not results:
                    return ""
                
                formatted_snippets = []
                for idx, r in enumerate(results, 1):
                    title = r.get('title', 'Source')
                    content = r.get('content', '')
                    url_source = r.get('url', '')
                    formatted_snippets.append(
                        f"### Source {idx}: {title}\n"
                        f"- **Summary:** {content}\n"
                        f"- **URL:** {url_source}\n"
                    )
                
                return (
                    "\n\n[LIVE WEB SEARCH CONTEXT - CURRENT YEAR 2026]:\n"
                    "Please synthesize the following search results into a clean, well-structured response using bullet points, bold headings, and proper line breaks. Do not output raw pipes or unformatted text blocks.\n\n"
                    + "\n".join(formatted_snippets) + "\n"
                )
    except Exception as e:
        print(f"Tavily search error: {e}")
    
    return ""


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Homework AI Backend is live!"}


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_prompt = request.message or request.prompt
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No prompt or message provided.")

    # Lazy-initialize Groq client at request time to prevent Vercel build-time crashes
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the server.")
    groq_client = Groq(api_key=groq_api_key)

    # INDENTED PROPERLY INSIDE THE ASYNC FUNCTION:
    web_context = ""
    if request.model in ["gemini", "search", "groq-search"] or "search" in str(request.model).lower():
        web_context = await fetch_tavily_context(user_prompt)

    enhanced_prompt = user_prompt + web_context

    def generate_groq():
        try:
            stream = groq_client.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": enhanced_prompt},
                ],
                temperature=0.1,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            try:
                fallback_stream = groq_client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": enhanced_prompt},
                    ],
                    temperature=0.1,
                    stream=True,
                )
                for chunk in fallback_stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            except Exception as fallback_err:
                yield f"Groq Error: {str(fallback_err)}"

    return StreamingResponse(generate_groq(), media_type="text/plain")
