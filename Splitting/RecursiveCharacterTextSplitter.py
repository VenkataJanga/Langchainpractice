from langchain.text_splitter import RecursiveCharacterTextSplitter

class RecursiveCharacterTextSplitterWrapper:
    def __init__(self, documents, chunk_size=100, chunk_overlap=20):
        self.documents = documents
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_recursive_character_text_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],  # <-- list of separators
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        return text_splitter.split_documents(self.documents)