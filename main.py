from Loading.load_multi_file_format import load_all_documents
from Splitting.TextSplitter import TextSplitter
from Splitting.RecursiveCharacterTextSplitter import RecursiveCharacterTextSplitterWrapper
from Splitting.htmlSplitter import HTMLSplitterWrapper
from Splitting.json_splitter import JSONSplitter 
from Embedding.ollama_default_model import ollamaModel
from Embedding.huggingface_embedding import huggingfcae_token,hugging_face_embedding_model
from Embedding.open_chatgpt_embedding import open_ai_embedding,load_openai_key
import requests
import json 


# Example Usage
if __name__ == "__main__":
    folder_path = "data/"   
    urls = [    
        "https://arxiv.org/abs/1706.03762",
        "https://langchain.readthedocs.io/en/latest/",
        "https://lilianweng.github.io/posts/2023-06-23-agent/"
    ]
    documents = load_all_documents(folder_path=folder_path, urls=urls)
    if not documents:
        print("No documents were loaded.")
        raise SystemExit

    print(f"Example preview: {documents[0].page_content[:300]}")

    # split_docs1.split_documents(documents)

    print("######################## RecursiveCharacterTextSplitter ########################")
    split_docs1 = RecursiveCharacterTextSplitterWrapper(documents, chunk_size=100, chunk_overlap=20).split_recursive_character_text_documents()
    print(f"Total chunks created: {len(split_docs1)}")
    print(f"Example chunk preview: {split_docs1[0].page_content[:300]}")
    print("#########################################################################")  
    print("\n")

    '''
    print("######################## character_text_splitter ########################")
    split_docs = TextSplitter(documents, chunk_size=100, chunk_overlap=20).split_character_documents()
    print(f"Total chunks created: {len(split_docs)}")
    print(f"Example chunk preview: {split_docs[0].page_content[:300]}")
    print("#########################################################################")
    print("\n")


    print("######################## HTMLSplitter ########################")
    url = "https://plato.stanford.edu/entries/goedel/"
    html_splitter = HTMLSplitterWrapper(url).split_html_documents()
    print(f"Total chunks created: {len(html_splitter)}")
    print(f"Example chunk preview: {html_splitter[0].page_content[:300]}")
    print("#########################################################################")
    print("\n")

    print("######################## JSONSplitter ########################")
    json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()
    chunks = JSONSplitter(json_data, chunk_size=1000, chunk_overlap=200).split_json()
    print("Example preview:", json.dumps(json_data)[:300])
    print(f"Total chunks created: {len(chunks)}")
    print("#########################################################################")
    print("\n")
    '''
    print("###################### OLLAMA Emedding#########################################")
    emb  = ollamaModel(model="nomic-embed-text",base_url="http://127.0.0.1:11434")
    print(emb)
    print("######################Emedding#########################################")

    print("#######################HUGGING FACE Embedding###############################")
    huggingfcae_token()  # not needed for public models, safe to keep if you want
    model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = hugging_face_embedding_model(model)

    # quick sanity checks
    #vec = embeddings.embed_query("hello world")

    # after you get split_docs1 (List[Document])
    # A) embed all chunks
    text_chunks = [d.page_content for d in split_docs1 if d.page_content and d.page_content.strip()]
    vecs = embeddings.embed_documents(text_chunks)
    print("Vectors:", len(vecs), "Dim:", len(vecs[0]))
    # B) embed a single query
    qvec = embeddings.embed_query("When was the transformer paper published?")
    print("Query dim:", len(qvec))
    print("#########################OPEN CHATGPAT EMEDDING####################")
    load_openai_key()
    embedding_open_ai = open_ai_embedding("text-embedding-3-large")  # or "text-embedding-3-small"

    # A) embed all chunks
    text_chunks = [d.page_content for d in split_docs1 if d.page_content and d.page_content.strip()]
    vecs_open_ai = embedding_open_ai.embed_documents(text_chunks)
    print("Vectors:", len(vecs_open_ai), "Dim:", len(vecs_open_ai[0]))

    # quick sanity checks
    v = embedding_open_ai.embed_query("hello world")
    print("Vector dim:", len(v))

    vecs = embedding_open_ai.embed_documents(["first doc", "second doc"])
    print("Batch size:", len(vecs))


    