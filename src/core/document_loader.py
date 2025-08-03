from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

class DocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def load_from_url(self, url: str):
        """Load documents from a web URL."""
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs
    
    def split_documents(self, docs):
        """Split documents into chunks."""
        return self.text_splitter.split_documents(docs)
    
    def load_and_split(self, url: str):
        """Load documents from URL and split them into chunks."""
        docs = self.load_from_url(url)
        split_docs = self.split_documents(docs)
        return split_docs