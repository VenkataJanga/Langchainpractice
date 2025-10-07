from langchain.text_splitter import CharacterTextSplitter

class TextSplitter:
    def __init__(self, documents, chunk_size=100, chunk_overlap=20):
        self.documents = documents
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_character_documents(self):
        text_splitter = CharacterTextSplitter(
            separator="\n\n",           # <-- singular
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            is_separator_regex=False
        )
        return text_splitter.split_documents(self.documents)
