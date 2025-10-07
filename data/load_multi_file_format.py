# Load_documents.py
import os, re
from typing import Iterator, Optional, Dict, List
from pathlib import Path
from lxml import etree

from langchain_core.documents import Document
from langchain_core.document_loaders.base import BaseLoader
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader, UnstructuredExcelLoader,
    WebBaseLoader,  # <-- add this
)

# ---- Minimal XML loader (no NLTK/unstructured needed) ----
class XPathXMLLoader(BaseLoader):
    def __init__(self, file_path: str, xpath: str = "//*", namespaces: Optional[Dict[str,str]] = None):
        self.file_path, self.xpath, self.namespaces = file_path, xpath, (namespaces or {})
    def lazy_load(self) -> Iterator[Document]:
        tree = etree.parse(self.file_path)
        for i, node in enumerate(tree.xpath(self.xpath, namespaces=self.namespaces)):
            text = "".join(node.itertext()).strip()
            meta = {"source": str(Path(self.file_path).resolve()), "xpath": self.xpath, "tag": getattr(node, "tag", None), "index": i}
            if hasattr(node, "attrib"):
                meta.update({f"attr_{k}": v for k, v in node.attrib.items()})
            yield Document(page_content=text, metadata=meta)

def _is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s, re.I))

def load_from_urls(urls: List[str]) -> List[Document]:
    """Load web pages; restrict parsing to likely main content (optional)."""
    from bs4 import SoupStrainer
    loader = WebBaseLoader(
        urls,
        bs_kwargs=dict(parse_only=SoupStrainer(name=["article", "main", "section", "div"]))
    )
    return loader.load()

def load_all_documents(folder_path: str = None, urls: List[str] = None) -> List[Document]:
    """Load docs from a folder and/or a list of URLs."""
    all_docs: List[Document] = []

    # 1) URLs (optional)
    if urls:
        print(f"🌐 Loading {len(urls)} URL(s)")
        try:
            all_docs.extend(load_from_urls(urls))
        except Exception as e:
            print(f"❌ Error loading URLs: {e}")

    # 2) Local files (optional)
    if folder_path:
        supported_exts = [".pdf", ".txt", ".csv", ".docx", ".pptx", ".xls", ".xlsx", ".xml"]
        for root, _, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in supported_exts:
                    print(f"⚠️ Skipping unsupported file: {file}")
                    continue
                file_path = os.path.join(root, file)
                print(f"📂 Loading: {file_path}")
                try:
                    if ext == ".pdf":
                        docs = PyPDFLoader(file_path).load()
                    elif ext == ".txt":
                        docs = TextLoader(file_path, encoding="utf-8").load()
                    elif ext == ".csv":
                        docs = CSVLoader(file_path).load()
                    elif ext == ".docx":
                        docs = UnstructuredWordDocumentLoader(file_path).load()
                    elif ext == ".pptx":
                        docs = UnstructuredPowerPointLoader(file_path).load()
                    elif ext in [".xls", ".xlsx"]:
                        # If 'unstructured' is not available, swap to a pandas fallback.
                        docs = UnstructuredExcelLoader(file_path).load()
                    elif ext == ".xml":
                        # One Document per <record>; adjust to your XML schema.
                        docs = list(XPathXMLLoader(file_path=file_path, xpath="//record").lazy_load())
                    else:
                        continue
                    all_docs.extend(docs)
                except Exception as e:
                    print(f" Error loading {file_path}: {e}")

    print(f" Loaded {len(all_docs)} total document(s)")
    return all_docs




# Example Usage
if __name__ == "__main__":
    folder_path = "C:\\Users\\229164\\OneDrive - NTT DATA Group\\AI\\Projects\\LangChain\\1-Langchain"  # e.g., folder containing pdf, txt, xml, etc.
    urls = [
        "https://arxiv.org/abs/1706.03762",
        "https://langchain.readthedocs.io/en/latest/",
        "https://lilianweng.github.io/posts/2023-06-23-agent/"
    ]
    documents = load_all_documents(folder_path=folder_path, urls=urls)
    if documents:
        print(f"Example preview: {documents[0].page_content[:300]}")
    else:
        print("No documents loaded.")