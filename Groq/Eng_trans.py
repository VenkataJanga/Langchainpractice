# pip install -U langchain-groq groq python-dotenv

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def configure_env():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Missing GROQ_API_KEY")
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ.pop("LANGCHAIN_TRACING_V2", None)
    os.environ.pop("LANGSMITH_ENDPOINT", None)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following from English to {input_language}. Return only the translation."),
    ("human", "{text}")
])

if __name__ == "__main__":
    configure_env()
    input_language = "German" #"Telugu"
    chain = prompt | llm |StrOutputParser()
    result = chain.invoke({"text": "Hello, how are you?", "input_language": input_language})
    print(result)
