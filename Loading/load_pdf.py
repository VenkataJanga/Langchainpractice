from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


if __name__ == "__main__":
    # Replace with your PDF file path
    file_path = "C:\\Users\\229164\\OneDrive - NTT DATA Group\\AI\\Projects\\Langchainpractice\\data\\attention.pdf"
    docs = load_pdf(file_path)
    for i, doc in enumerate(docs):
        print(f"Document {i+1}:\n{doc.page_content}\n")