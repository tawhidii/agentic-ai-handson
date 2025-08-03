import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.settings import settings
from core.document_loader import DocumentLoader
from core.embeddings import EmbeddingManager
from core.vector_store import VectorStoreManager
from core.rag_chain import RAGChain

def run_basic_demo():
    """Run the basic RAG demo - equivalent to the original demo_llm.py"""
    # Initialize components
    doc_loader = DocumentLoader()
    embedding_manager = EmbeddingManager()
    vector_store_manager = VectorStoreManager(embedding_manager.get_embeddings())
    rag_chain = RAGChain()
    
    # Load and process documents
    url = "https://kubernetes.io/docs/concepts/architecture/"
    split_docs = doc_loader.load_and_split(url)
    
    # Create vectorstore
    vectorstore = vector_store_manager.create_vectorstore(split_docs)
    print(f"Created FAISS vectorstore with {vectorstore.index.ntotal} vectors")
    
    # Create RAG chain
    retriever = vector_store_manager.get_retriever()
    chain = rag_chain.create_chain(retriever)
    
    # Query the system (using the same query as original demo)
    query = "What is Iron man?"
    response = rag_chain.invoke(query)
    
    print(f"Question: {query}")
    print(f"Answer: {response['answer']}")
    print(f"Retrieved {len(response['context'])} documents")

if __name__ == "__main__":
    run_basic_demo()