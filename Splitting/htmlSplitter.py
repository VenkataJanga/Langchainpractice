from langchain_community.document_loaders import WebBaseLoader
from bs4 import SoupStrainer
from langchain_text_splitters import HTMLHeaderTextSplitter


class HTMLSplitterWrapper:
    def __init__(self, url: str):
        self.url = url
    
    def split_html_documents(self):
        # 1) Load the page
        loader = WebBaseLoader(
            web_paths=[self.url],
            header_template={"User-Agent": "Langchainpractice/1.0"},  # polite UA
            bs_kwargs=dict(parse_only=SoupStrainer(name=["article", "main", "section", "div"]))  # optional: focus main content
        )
        raw_docs = loader.load()   # -> List[Document] with .page_content (HTML)

        # 2) Split by HTML headers (h1..h6)
        headers_to_split_on = [(f"h{i}", f"H{i}") for i in range(1, 7)]
        splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        chunks = []
        for d in raw_docs:
            parts = splitter.split_text(d.page_content)  # returns List[Document]
            # carry over source metadata
            for p in parts:
                p.metadata.setdefault("source", d.metadata.get("source", self.url))
            chunks.extend(parts)

        return chunks

# Example
'''
if __name__ == "__main__":
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    docs = HTMLSplitterWrapper(url).split_html_documents()
    print(len(docs), docs[0].page_content[:300], docs[0].metadata)
'''