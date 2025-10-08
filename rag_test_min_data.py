# LLM via Ollama + embeddings via Sentence-Transformers (fast on CPU)
#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 1) Tiny corpus
docs = [
    Document(page_content="The transformer paper 'Attention Is All You Need' was published in 2017."),
    Document(page_content="Mistral 7B is an efficient open-weight language model."),
]


# 2) Chunking (RAG-friendly)
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""], chunk_size=800, chunk_overlap=120
)
chunks = splitter.split_documents(docs)

# 3) CPU-friendly embedding
#emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},                 # optional on CPU
    encode_kwargs={"normalize_embeddings": True},   # good default for cosine
)

store = FAISS.from_documents(chunks, emb)

# 4) Retrieve + answer with a small local LLM
#llm = ChatOllama(model="llama3.2:1b-instruct", temperature=0.2)
llm = ChatOllama(
    model="gemma3:4b",
    base_url="http://127.0.0.1:11434",   # explicit is better
    temperature=0.2,
)
query = "When was the transformer paper published?"

ctx = store.similarity_search(query, k=3)
context_text = "\n\n".join(d.page_content for d in ctx)
prompt = f"Answer concisely using the context.\n\nContext:\n{context_text}\n\nQ: {query}\nA:"
print(llm.invoke(prompt).content)