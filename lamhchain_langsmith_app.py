#############################
# first login and create API KEY using https://smith.langchain.com/o/dd9991a2-04d9-43d9-81b8-230c06147dd2/settings/apikeys

# you will get below details
#LANGSMITH_API_KEY = "lsv2_pt_705420bbb1e8test6f5ca99798765474a2d_c9a98e3b9f"
#LANGSMITH_TRACING="true"
#LANGSMITH_WORKSPACE_ID = "workspace 1"
#LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com"
##############################

import os
os.environ["LANGSMITH_TRACING"] = "true"
from dotenv import load_dotenv

def get_env_load():
    load_dotenv()
    os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
    os.environ['LANGSMITH_TRACING'] = os.getenv('LANGSMITH_TRACING')
    os.environ['LANGSMITH_WORKSPACE_ID'] = os.getenv('LANGSMITH_WORKSPACE_ID')
    os.environ['LANGSMITH_ENDPOINT'] = os.getenv('LANGSMITH_ENDPOINT')


# to Read the content from the website/scrape the entire text from webpage
# required to install beautifulsoup4
####################DATA INGESTION#############################
from langchain_community.document_loaders import WebBaseLoader
import bs4

def getWebLoading_scrape_webpage():
    url = os.environ['LANGSMITH_ENDPOINT']
    loader = WebBaseLoader(
        web_paths=[url],  # <-- correct param
        header_template={"User-Agent": "Mozilla/5.0 (LangChain WebBaseLoader)"},
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(["article","main","section","p","h1","h2","h3"]))
    )
    return loader.load()

    return  web_loader.load()
####################DATA into CHUNKS#############################
from langchain_text_splitters import RecursiveCharacterTextSplitter
def getChunks_data_from_documents(web_loader_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunk_docs = text_splitter.split_documents(web_loader_docs)
    return chunk_docs

####################DATA into Embeddings#############################
from langchain_openai import OpenAIEmbeddings
def getOpenAIEmbedding():
     return OpenAIEmbeddings(model="text-embedding-3-small")



####################Embeddings vectors to store in DB#############################
# pip install -U faiss-cpu langchain-community
from langchain_community.vectorstores import FAISS  
def getstore_in_vectordb(chunk_docs,emb):
     vs = FAISS.from_documents(chunk_docs, emb)
     return vs

if __name__ == "__main__":   # <-- fix
    
    #1. loading the Environment varoables
    print('###################loading the Environment varoables#############################')
    get_env_load()
    print("\n")
    
    #2. loading the Data from web page
    web_loader_docs = getWebLoading_scrape_webpage()
    print(type(web_loader_docs))
    print('################### loading the Data from web page##############################')
    print(web_loader_docs[0])
    print("\n")
    
    #3.Divide the documents into small chunks
    print('################Divide the documents into small chunks##########################')
    chunk_docs = getChunks_data_from_documents(web_loader_docs)
    print(chunk_docs)
    print("\n")
   
    #4. convert the text into vectors using vector embedding techniques
    print('################convert the text into vectors using vector embedding techniques##########################')
    emd = getOpenAIEmbedding()
    print(emd)
    print("\n")
  
    #5. vectors  to store in vector DB
    print('################vectors  to store in vector DB##########################')
    vector_store_db = getstore_in_vectordb(chunk_docs,emd)
    print("\n")
    
    #6. Ask the Question
    print('################### query ###########################################')
    query = "what is the Retention Period"
    results = vector_store_db.similarity_search(query, k=3)
    for i, d in enumerate(results, 1):
        print(f"[{i}] {d.metadata.get('source')} :: {d.page_content[:200]}...") 