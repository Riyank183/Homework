import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from duckduckgo_search import DDGS

app = FastAPI(title="Homework AI Backend")

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


def clean_search_query(query: str) -> str:
    """Strips conversational fluff so DuckDuckGo gets precise search terms."""
    cleaned = re.sub(r'^(what is|who is|tell me about|when is|how does|the story of)\s+', '', query, flags=re.IGNORECASE)
    return cleaned.strip()


def fetch_duckduckgo_context(user_query: str, max_results: int = 5) -> str:
    """Executes live search with cleaned query parameters and robust fallback."""
    search_term = clean_search_query(user_query)
    snippets = []
    
    try:
        with DDGS() as ddgs:
            # 1. Search text snippets
            text_results = list(ddgs.text(search_term, max_results=max_results))
            for item in text_results:
                title = item.get("title", "")
                body = item.get("body", "")
                if body:
                    snippets.append(f"Title: {title}\nSnippet: {body}")
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")

    return "\n\n".join(snippets)


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_prompt = request.message or request.prompt
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No prompt provided.")

    system_instructions = (
        "You are an accurate, factual AI assistant.\n"
        "Do NOT use external tools or call internal function paths.\n"
        "CRITICAL FOR SEARCH CONTEXT:\n"
        "1. If reference search snippets are provided inside <search_results>, you MUST base your answer directly on those facts.\n"
        "2. Do NOT make up fictional plot twists, fake endings, or unverified claims.\n"
        "3. If the search results state specific plot details or news, reflect them accurately.\n"
        "ALWAYS format mathematical equations using $...$ for inline and $$...$$ for block math."
    )

    # Fetch fresh live context from DuckDuckGo
    search_data = fetch_duckduckgo_context(user_prompt)

    if search_data:
        prompt_with_context = (
            f"Here are live search results regarding the user request:\n"
            f"<search_results>\n{search_data}\n</search_results>\n\n"
            f"User Question: {user_prompt}\n"
            f"Provide a accurate response strictly grounded in the provided web context."
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
            yield f"Error: {str(e)}"

    return StreamingResponse(generate_groq(), media_type="text/plain")
