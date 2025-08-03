from config.settings import settings
from core.document_loader import DocumentLoader
from core.embeddings import EmbeddingManager
from core.vector_store import VectorStoreManager
from core.rag_chain import RAGChain

def main():
    # Initialize components
    doc_loader = DocumentLoader()
    embedding_manager = EmbeddingManager()
    vector_store_manager = VectorStoreManager(embedding_manager.get_embeddings())
    rag_chain = RAGChain()
    
    # Load and process documents
    url = "https://kubernetes.io/docs/concepts/architecture/"
    split_docs = doc_loader.load_and_split(url)
    print(f"Split into {len(split_docs)} chunks")
    
    # Create vectorstore
    vectorstore = vector_store_manager.create_vectorstore(split_docs)
    print(f"Created FAISS vectorstore with {vectorstore.index.ntotal} vectors")
    
    # Create RAG chain
    retriever = vector_store_manager.get_retriever()
    chain = rag_chain.create_chain(retriever)
    
    # Query the system
    query = "What is Kubernetes?"
    response = rag_chain.invoke(query)
    
    print(f"Question: {query}")
    print(f"Answer: {response['answer']}")
    print(f"Retrieved {len(response['context'])} documents")

if __name__ == "__main__":
    main()