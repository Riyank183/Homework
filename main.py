import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from duckduckgo_search import DDGS

app = FastAPI(title="Homework AI Backend")

# CORS setup for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


class ChatRequest(BaseModel):
    message: str = None
    prompt: str = None
    model: str = "groq"


def get_live_search_context(query: str, max_results: int = 3) -> str:
    """Performs a live DuckDuckGo web search and formats snippets into text context."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return ""
            
            snippets = []
            for r in results:
                snippets.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}")
            
            return "\n\n".join(snippets)
    except Exception:
        return ""


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_prompt = request.message or request.prompt
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No prompt or message provided.")

    system_instructions = (
        "You are a helpful AI homework tutor and academic assistant. "
        "ALWAYS format mathematical equations using dollar sign delimiters: "
        "use $...$ for inline math and $$...$$ for standalone block math equations.\n"
        "CRITICAL RULES FOR LATEX:\n"
        "1. NEVER use square brackets like \\[ ... \\] or [ ... ] for LaTeX.\n"
        "2. NEVER use parentheses like \\( ... \\) for inline LaTeX.\n"
        "3. Use only $ ... $ for inline formulas and $$ ... $$ for block formulas.\n"
        "For general questions outside of homework, provide accurate and clear answers based on provided context."
    )

    # Detect queries needing fresh real-time info
    keywords = ["release", "date", "news", "latest", "today", "when", "current", "price", "who is", "game"]
    needs_search = any(kw in user_prompt.lower() for kw in keywords)

    final_user_prompt = user_prompt
    if needs_search:
        search_context = get_live_search_context(user_prompt)
        if search_context:
            final_user_prompt = (
                f"Use the following real-time web search results to answer the user's question accurately:\n\n"
                f"--- LIVE SEARCH RESULTS ---\n{search_context}\n-----------------------\n\n"
                f"User Question: {user_prompt}"
            )

    def generate_groq():
        try:
            stream = groq_client.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": final_user_prompt},
                ],
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            yield f"Groq Error: {str(e)}"

    return StreamingResponse(generate_groq(), media_type="text/plain")
