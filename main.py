from Loading.load_multi_file_format import load_all_documents
from Splitting.TextSplitter import TextSplitter



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

    split_docs1 = TextSplitter(documents, chunk_size=100, chunk_overlap=20).split_character_documents()
    # split_docs1.split_documents(documents)

    print("######################## character_text_splitter ########################")
    print(f"Total chunks created: {len(split_docs1)}")
    print(f"Example chunk preview: {split_docs1[0].page_content[:300]}")
    print("#########################################################################")
    print("\n")


    print("######################## RecursiveCharacterTextSplitter ########################")
    print(f"Total chunks created: {len(split_docs1)}")
    print(f"Example chunk preview: {split_docs1[0].page_content[:300]}")
    print("#########################################################################")  
    print("\n")