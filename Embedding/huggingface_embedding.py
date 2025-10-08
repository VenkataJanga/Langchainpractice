#######
# create the account https://huggingface.co/
# after creating the account
#goto your email account and confirm it
#goto profile--settings--Access token-
# create the new token
# give name and generate the token
# copy the token 

#step1: To get the token
# pip install -U langchain-huggingface sentence-transformers huggingface-hub python-dotenv

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

def huggingfcae_token():
    # Load token if you need private models (public models work without a token)
    load_dotenv()
    # Recommended env var name for HF Hub:
    #hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token  # make sure downstream libs see it

def hugging_face_embedding_model(model: str):
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # good for cosine
    )


# enable below comment if you run this file.
'''
if __name__ == "__main__":
    huggingfcae_token()
    model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = hugging_face_embedding_model(model)
    # quick sanity checks
    vec = embeddings.embed_query("hello world")
    print("Vector dim:", len(vec))
    vecs = embeddings.embed_documents(["first doc", "second doc"])
    print("Batch size:", len(vecs))
'''