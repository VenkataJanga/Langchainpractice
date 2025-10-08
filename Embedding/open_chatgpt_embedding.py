# pip install -U langchain-openai openai python-dotenv

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

def load_openai_key():
    load_dotenv()  # reads .env
    key = os.getenv("OPENAI_API_KEY")  # <-- correct name
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env file.")
    # Ensure downstream libs see it (optional if already in env)
    os.environ["OPENAI_API_KEY"] = key

def open_ai_embedding(model_name: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model_name)

'''
if __name__ == "__main__":
    load_openai_key()
    emb = open_ai_embedding("text-embedding-3-large")  # or "text-embedding-3-small"

    # quick sanity checks
    v = emb.embed_query("hello world")
    print("Vector dim:", len(v))

    vecs = emb.embed_documents(["first doc", "second doc"])
    print("Batch size:", len(vecs))
'''