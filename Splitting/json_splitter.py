import json

import requests


class JSONSplitter:
    def __init__(self, json_data, chunk_size, chunk_overlap):
        self.json_data = json_data
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_json(self):
        json_str = json.dumps(self.json_data, indent=2)
        chunks = []
        start = 0
        while start < len(json_str):
            end = min(start + self.chunk_size, len(json_str))
            chunk = json_str[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

# This is a large nested json object and will be loaded as a python dict
#json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()

# Example
'''
if __name__ == "__main__":
    json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()
    splitter = JSONSplitter(json_data, chunk_size=1000, chunk_overlap=200)
    print("Example preview:", json.dumps(json_data)[:300])
    chunks = splitter.split_json()
    print(f"Total chunks created: {len(chunks)}")      
'''