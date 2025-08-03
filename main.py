import os
from re import L 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

os.environ["OPENAI_API_KEY"]  = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

llm = ChatOpenAI(model="gpt-4o")
result = llm.invoke("Tell me about bangladesh")
print(result)
