from langchain_openai import OpenAIEmbeddings
from typing import Optional

class EmbeddingManager:
    def __init__(self, model: str = "text-embedding-3-large", dimensions: Optional[int] = 1024):
        self.embeddings = OpenAIEmbeddings(model=model, dimensions=dimensions)
    
    def get_embeddings(self):
        """Get the embeddings instance."""
        return self.embeddings