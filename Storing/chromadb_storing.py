########################
# Chroma is a AI- native open source vector data base focussed on developer productivity and happiness.
# Chroma is licenced under Appache 2.0
########################

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

#Retriever
# We can also convert the vectore store into a Retriver class. This allows us easily use it 
# other Langchain methods, which largely work with retrievers
# Retriever is act as a interface when we work with different LLM models and can not directly work with vector DB

retriever = store_db.as_retriever()
result1 = retriever.invoke(query)
#print(result1[0].page_content)

#Similaty Search with scores
#######
#There are some FAISS specific methods. one of them is Similarity Search with score , which allows
# you to return not only the docuemnts but also the distance score of the quality to them. 
# the return distance score is L2 distance . therefore Lower score is better
#documents_with_score = store_db.similarity_search_with_score(query)
documents_with_score = store_db.similarity_search_with_score(query, k=4)
#print(documents_with_score[0])


# Can we directky pass vectors insted of sentences? Yes we can by using 
embedding_vectors = embeddings.embed_query(query)
#print(embedding_vectors)

#documents_with_score_vector = store_db.similarity_search_with_score(embedding_vectors)
documents_with_score_vector = store_db.similarity_search_by_vector(embedding_vectors, k=4)
#print(documents_with_score_vector[0])


############
#SAVING THE RESULT IN LOCAL AND LOADING

# Build & persist
vs = Chroma.from_documents(
    documents=chunk_documents,
    embedding=embeddings,
    persist_directory="chroma_index"  # folder on disk
)
vs.persist()  # <-- flush to disk
# Later: reload
vs2 = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_index"
)
# Query
res = vs2.similarity_search(query, k=3)

print(res[0])



