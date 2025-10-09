
# Facebook AI Similarity Search(FAISS) is a library for efficient similarity search and clustering of dense vectors.
# It contains algorithems that search in sets of vectors of any size, 
# up to ones that possibility do not fit in RAN
# It also contains supporting code for evaluation and parameter tuning.
##################
# Step1: pip install faiss-cpu
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import OllamaEmbeddings # OllamaEmbeddings is depreciated
from langchain_ollama import OllamaEmbeddings      # new package
from langchain_community.vectorstores import FAISS

import os
os.environ["LANGSMITH_TRACING"] = "false"

#1. Loading the data  from source
loader = TextLoader('data/speech.txt')
speech_documents = loader.load()

#2. Split the data into chunks
# split (optional here, useful in real data)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunk_documents = text_splitter.split_documents(speech_documents)
#print(chunk_documents[0])

#3. Chunks data into vectors using Vector Embedding
#  choose an embedding model you actually pulled above
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",                 # or "all-minilm", "bge-m3"
    base_url="http://127.0.0.1:11434",        # if not mentioned the model then it will running Ollama
)

#4 to store the data with embeddings
store_db = FAISS.from_documents(chunk_documents,embeddings)
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
store_db.save_local('faiss_index')

#new_store_db = FAISS.load_local('faiss_index')
new_store_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

new_doc_score = new_store_db.similarity_search(query)
print(new_doc_score[0])
