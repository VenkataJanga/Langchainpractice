"""
Simple FastAPI + LCEL + Groq translation service.

Run:
  python app.py
or:
  uvicorn app:app --host 127.0.0.1 --port 8000 --reload

Test (PowerShell):
  $body = @{ language = "Telugu"; text = "Hello, how are you?" } | ConvertTo-Json
  Invoke-RestMethod -Uri 'http://127.0.0.1:8000/translate' -Method POST -ContentType 'application/json' -Body $body
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

# LangChain bits
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------- Models for request/response ----------
class TranslateIn(BaseModel):
    language: str                 # target language (e.g., "Telugu")
    text: str                     # source text to translate

class TranslateOut(BaseModel):
    translation: str              # translated text only
    model: Optional[str] = None   # which model served the request


# ---------- App + startup ----------
app = FastAPI(
    title="Groq LCEL Translator",
    version="1.0",
    description="A minimal FastAPI app using LCEL + Groq for translation."
)

# Disable LangSmith noise (prevents those 405s you saw)
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.pop("LANGCHAIN_TRACING_V2", None)
os.environ.pop("LANGSMITH_ENDPOINT", None)

# Global chain objects (filled in at startup)
_llm = None
_chain = None
_model_name = "llama-3.3-70b-versatile"


@app.on_event("startup")
def startup() -> None:
    """Load env, validate key, and build the LCEL chain once."""
    load_dotenv()  # read .env if present

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Fail early with a clear message
        raise RuntimeError(
            "Missing GROQ_API_KEY. Add it to your environment or a .env file."
        )

    # 1) LLM (Groq)
    global _llm
    _llm = ChatGroq(model=_model_name, temperature=0.2)

    # 2) Prompt → tell model to *only* return translation
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the following into {language}. Return only the translation."),
        ("human", "{text}")
    ])

    # 3) Parse to a plain string
    parser = StrOutputParser()

    # 4) Build the chain (LCEL)
    global _chain
    _chain = prompt | _llm | parser


# ---------- Routes ----------
@app.get("/health")
def health():
    """Basic health check."""
    return {"status": "ok"}

@app.post("/translate", response_model=TranslateOut)
async def translate(payload: TranslateIn):
    """
    Translate `text` into `language` using Groq via LCEL.
    Returns translation only.
    """
    try:
        # run the chain asynchronously (non-blocking for FastAPI)
        out = await _chain.ainvoke({"language": payload.language, "text": payload.text})
        return TranslateOut(translation=out.strip(), model=_model_name)
    except Exception as e:
        # turn any runtime error into a 500 for the client
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Dev server entrypoint ----------
if __name__ == "__main__":
    import uvicorn
    # run uvicorn in-process so `python app.py` just works
    uvicorn.run(app, host="127.0.0.1", port=8000)
