

####Install Chroma DB-- pip install chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

import os
os.environ["LANGSMITH_TRACING"] = "false"

#1. Loading the data  from source
loader = TextLoader('data/speech.txt')
speech_documents = loader.load()

#2. Split the text data in to chunck
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunk_documents = text_splitter.split_documents(speech_documents)

#3. Embedding
embeddings = OllamaEmbeddings( 
                             model="nomic-embed-text",                 # or "all-minilm", "bge-m3"
                             base_url="http://127.0.0.1:11434",        # if not mentioned the model then it will running Ollama)

                            )

#4 to store the data with embeddings
store_db = Chroma.from_documents(chunk_documents,embeddings)
#print(store_db)

#5. Quering the data
#query = "What does speaker believe is the main reason the united states should enter the war?"
query = "how does the speaker describe the desired outcome of the war?"
#result = store_db.similarity_search(query)
result = store_db.similarity_search(query, k=4) 
#print(result[0].page_content)





