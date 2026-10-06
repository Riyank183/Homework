import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from supabase import create_client, Client
from duckduckgo_search import DDGS

app = FastAPI(title="Homework AI Backend")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client using Render environment variables
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None


class ChatRequest(BaseModel):
    message: str = None
    prompt: str = None
    model: str = "groq"


def fetch_duckduckgo_context(query: str, max_results: int = 5) -> str:
    """Executes DuckDuckGo search and extracts clean result text."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return ""
            
            snippets = []
            for item in results:
                title = item.get("title", "")
                body = item.get("body", "")
                if body:
                    snippets.append(f"- {title}: {body}")
            
            return "\n".join(snippets)
    except Exception as e:
        print(f"Search fetch error: {e}")
        return ""


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Homework AI Backend is live!"}


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
        "For general or current events questions, use the provided live search results to give an accurate, up-to-date answer."
    )

    # Perform live web search for context
    live_context = fetch_duckduckgo_context(user_prompt)

    if live_context:
        prompt_with_context = (
            f"Context from real-time web search:\n{live_context}\n\n"
            f"User Question: {user_prompt}\n\n"
            f"Answer the user question accurately using the live search context provided above."
        )
    else:
        prompt_with_context = user_prompt

    def generate_groq():
        try:
            stream = groq_client.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": prompt_with_context},
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
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": prompt_with_context},
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
