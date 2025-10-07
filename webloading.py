from langchain_community.document_loaders import WebBaseLoader
from bs4 import SoupStrainer
import os

def load_webpage(url: str):
    # Option A: set env var (silences the warning globally)
    os.environ.setdefault(
        "USER_AGENT",
        "Langchainpractice/1.0 (+https://github.com/VenkataJanga)"
    )

    loader = WebBaseLoader(
        web_paths=[url],  # <-- correct kwarg
        header_template={"User-Agent": os.environ["USER_AGENT"]},  # or any UA string
        bs_kwargs=dict(parse_only=SoupStrainer(
            class_=("post-title", "post-content", "post-header")
        )),
        raise_for_status=True,
        trust_env=True,  # respect system proxy env vars if you're on a corporate network
    )
    return loader.load()

if __name__ == "__main__":
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    docs = load_webpage(url)
    for i, doc in enumerate(docs):
        print(f"Document {i+1}:\n{doc.page_content[:1000]}\n")
