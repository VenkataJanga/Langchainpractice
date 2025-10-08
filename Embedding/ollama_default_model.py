###
# First install OLLAMA using https://ollama.com/download/windows in your laptop
#in powershel run the below code
'''
        Invoke-RestMethod -Method Post -ContentType 'application/json' `
  -Body '{"name":"nomic-embed-text"}' `
  -Uri http://127.0.0.1:11434/api/pull
'''
###
from langchain_ollama import OllamaEmbeddings, ChatOllama



def ollamaModel(model,base_url):
    # Local embeddings via Ollama
    emb = OllamaEmbeddings(
                            model = model,
                            base_url = base_url
                        )
    return emb


#usage example
if __name__ == "__main__":
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
    emb = ollamaModel(model="nomic-embed-text",base_url="http://127.0.0.1:11434")
    r1 = emb.embed_documents(
        [
            "A is the first letter  of alphabatics"
            "B is the second letter of alphabatics"
        ]
    )
    print("first sentence of the emeddings \n",r1[0])
    print(f"Length of r1[0] is {len(r1[0])}")
    print("##################################################")
    