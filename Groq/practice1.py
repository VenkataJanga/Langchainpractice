import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

def configure_env():
    load_dotenv()  # loads GROQ_API_KEY from .env
    # Disable LangSmith tracing for this run (prevents 405 spam)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ.pop("LANGCHAIN_TRACING_V2", None)   # old var name, just in case
    os.environ.pop("LANGSMITH_ENDPOINT", None)     # clear any bad endpoint lingering in env



def get_llm():
    # requires GROQ_API_KEY in your env / .env
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a concise, helpful assistant."),
        ("human", "{question}")
    ])





if __name__ == "__main__":
    configure_env()
    llm = get_llm()
    promts = get_prompt()
    chain = promts | llm
    print("####################################")
    print(chain.invoke({'question':"why groq LPUs is good comare to others?"}).content)
    print("####################################")
