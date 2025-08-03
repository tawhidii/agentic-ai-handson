from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class RAGChain:
    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model)
        self.prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {input}

Answer:
"""
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)
        self.retrieval_chain = None
    
    def create_chain(self, retriever):
        """Create the RAG chain with retriever."""
        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retrieval_chain = create_retrieval_chain(retriever, document_chain)
        return self.retrieval_chain
    
    def invoke(self, query: str):
        """Invoke the RAG chain with a query."""
        if self.retrieval_chain is None:
            raise ValueError("Chain not created. Call create_chain first.")
        return self.retrieval_chain.invoke({"input": query})