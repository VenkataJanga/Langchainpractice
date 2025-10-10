#########################
# Prompt Template:- Prompt Templates helps to turn raw user information into a format that the LLM can work with.
# In this case, the raw user input is a just message.
# Which we are passing this message to LLM. Lets now make that a bit more complicated.
# First, lets add in a system message with some custom instructions(but still taking a message as input).
# Next, we will add more input besides just a message

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Disable LangSmith noise (prevents those 405s you saw)
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.pop("LANGCHAIN_TRACING_V2", None)
os.environ.pop("LANGSMITH_ENDPOINT", None)

load_dotenv()  # read .env if present

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    # Fail early with a clear message
    raise RuntimeError(
        "Missing GROQ_API_KEY. Add it to your environment or a .env file."
    )




llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
promt = ChatPromptTemplate.from_messages(
        [
            ('system','you are helpful assistant. Answer all the questions to the nest of your ability in {language}.'),
            MessagesPlaceholder(variable_name = 'messages')
        ]
)   

chain = promt|llm 

store={}
def get_session_history(session_id:str)-> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

result = chain.invoke({'messages':[HumanMessage(content='Hi My name is Venkata')],'language':'HINDI'})
print(result.content)

#print(result)
with_message_history = RunnableWithMessageHistory(
                                    chain, 
                                    get_session_history,
                                    input_messages_key='messages')
config = {"configurable":{'session_id':'chat123'}}
response = with_message_history.invoke(
                                {'messages':[HumanMessage(content='Hi My name is Venkata')],'language':'Telugu'},
                                    config = config
                                    )

print(response.content)