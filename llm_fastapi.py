import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# Set environment variables for LangChain tracing
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
if LANGCHAIN_TRACING_V2:
    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
if LANGCHAIN_PROJECT:
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# Initialize the LLM and chain
llm_model = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")

generic_template = """Translate the following text to {language}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", generic_template),
    ("user", "{text}")
])
parser = StrOutputParser()
chain = prompt | llm_model | parser

# Pydantic models for API
class TranslationRequest(BaseModel):
    language: str
    text: str

class TranslationResponse(BaseModel):
    result: str
    language: str
    original_text: str

# FastAPI app
app = FastAPI(
    title="Translation API",
    description="A simple translation API using Groq and LangChain",
    version="1.0.0"
)

# Manual endpoint instead of langserve
@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Translate text to the specified language"""
    try:
        result = await chain.ainvoke({
            "language": request.language,
            "text": request.text
        })
        
        return TranslationResponse(
            result=result,
            language=request.language,
            original_text=request.text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "translation-api"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Translation API",
        "endpoints": {
            "translate": "/translate",
            "docs": "/docs",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "lang_serve_example:app",
        host="0.0.0.0", 
        port=8000,
        reload=True
    )