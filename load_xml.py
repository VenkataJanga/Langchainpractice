from pathlib import Path
from typing import Iterator, Optional, Dict, List
from collections import Counter
import re
from lxml import etree

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


def _strip_ns(tag: str) -> str:
    # '{ns}tag' -> 'tag'
    return tag.split('}', 1)[-1] if '}' in tag else tag

def _guess_repeating_tags(tree: etree._ElementTree, top_n: int = 10) -> List[str]:
    tags = [_strip_ns(el.tag) for el in tree.iter() if isinstance(el.tag, str)]
    common = Counter(tags).most_common(top_n)
    return [f"{t} (x{c})" for t, c in common]

class XPathXMLLoader(BaseLoader):
    """Parse XML with lxml and return one Document per XPath match."""
    def __init__(self, file_path: str, xpath: str = "//*", namespaces: Optional[Dict[str, str]] = None):
        self.file_path = file_path
        self.xpath = xpath
        self.namespaces = namespaces or {}

    def lazy_load(self) -> Iterator[Document]:
        file_resolved = Path(self.file_path).resolve()
        if not file_resolved.exists():
            raise FileNotFoundError(f"XML not found: {file_resolved}")

        tree = etree.parse(str(file_resolved))
        root = tree.getroot()

        # 1) Try the user XPath with namespaces (if provided)
        nodes = tree.xpath(self.xpath, namespaces=self.namespaces)

        # 2) If nothing and there is a default namespace, try a local-name() fallback
        if not nodes and (root.nsmap.get(None) is not None) and ":" not in self.xpath:
            # Convert //tag -> //*[local-name()='tag']
            def repl(m): return f"//*[local-name()='{m.group(1)}']"
            ln_xpath = re.sub(r"//(\w+)", repl, self.xpath)
            nodes = tree.xpath(ln_xpath)

        # 3) If still nothing, emit a helpful debug message and stop
        if not nodes:
            print("⚠️ XPath matched 0 nodes.")
            print(f"   File: {file_resolved}")
            print(f"   Tried XPath: {self.xpath}")
            if root.nsmap:
                print(f"   Namespaces: {root.nsmap}")
            print("   Top tags in document:", _guess_repeating_tags(tree))
            return  # yields nothing

        for i, node in enumerate(nodes):
            text = "".join(node.itertext()).strip()
            meta = {
                "source": str(file_resolved),
                "xpath": self.xpath,
                "tag": _strip_ns(getattr(node, "tag", "")),
                "index": i,
            }
            if hasattr(node, "attrib"):
                meta.update({f"attr_{k}": v for k, v in node.attrib.items()})
            yield Document(page_content=text, metadata=meta)


if __name__ == "__main__":
    # 1) Start broad to confirm it prints something:
    file_path = r"C:\Users\229164\OneDrive - NTT DATA Group\AI\Projects\Langchainpractice\data\records.xml"
    loader = XPathXMLLoader(file_path, xpath="/*")   # whole document as one chunk
    docs = list(loader.lazy_load())
    print(f"Found {len(docs)} doc(s) with xpath='/*'")

    # 2) Then narrow down to your record element:
    #    If your repeated node is <record> or <item> (with default xmlns),
    #    these two are robust:
    # loader = XPathXMLLoader(file_path, xpath="//record")
    # loader = XPathXMLLoader(file_path, xpath="//*[local-name()='record']")
    # docs = list(loader.lazy_load())

    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:\n{doc.page_content[:800]}\nMetadata: {doc.metadata}")
