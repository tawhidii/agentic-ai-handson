## using lcel and groq
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

llm_model = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")

generic_template = """Translate the following text to {language}
"""
prompt = ChatPromptTemplate.from_messages(
    [("system", generic_template),
    ("user", "{text}")]
)
parser = StrOutputParser()
chain = prompt | llm_model | parser
response = chain.invoke({"language": "french", "text": "Hello, I am a student."})
print(response)
