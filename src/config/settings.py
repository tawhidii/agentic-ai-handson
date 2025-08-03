import os
from dotenv import load_dotenv

class Settings:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
        self.langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
        self.langchain_project = os.getenv("LANGCHAIN_PROJECT")
        
        # Set environment variables
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
        if self.langchain_tracing_v2:
            os.environ["LANGCHAIN_TRACING_V2"] = self.langchain_tracing_v2
        if self.langchain_project:
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project

settings = Settings()