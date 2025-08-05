import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm_model = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")

generic_template = """Translate the following text to {language}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", generic_template),
    ("user", "{text}")
])
parser = StrOutputParser()
chain = prompt | llm_model | parser

# App
app = FastAPI()

# Simplified route addition
add_routes(
    app, 
    chain, 
    path="/translate",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)