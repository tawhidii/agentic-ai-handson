from langchain_community.vectorstores import FAISS
from typing import List

class VectorStoreManager:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vectorstore = None
    
    def create_vectorstore(self, documents):
        """Create FAISS vectorstore from documents."""
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        return self.vectorstore
    
    def get_retriever(self):
        """Get retriever from vectorstore."""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not created. Call create_vectorstore first.")
        return self.vectorstore.as_retriever()
    
    def similarity_search(self, query: str, k: int = 4):
        """Perform similarity search."""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not created. Call create_vectorstore first.")
        return self.vectorstore.similarity_search(query, k=k)