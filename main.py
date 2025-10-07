from Loading.load_multi_file_format import load_all_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  

# Example Usage
if __name__ == "__main__":
    folder_path = "data/"
    
    urls = [
        "https://arxiv.org/abs/1706.03762",
        "https://langchain.readthedocs.io/en/latest/",
        "https://lilianweng.github.io/posts/2023-06-23-agent/"
    ]
    documents = load_all_documents(folder_path=folder_path, urls=urls)
    if documents:
        print(f"Example preview: {documents[0].page_content[:300]}")

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        # Split the documents into smaller chunks
        split_docs = text_splitter.split_documents(documents)
        print("############################################################")
        print(f"\nTotal chunks created: {len(split_docs)}")
        print(f"Example chunk preview: {split_docs[0].page_content[:300]}")
        print("############################################################")
    else:
        print("No documents were loaded.")