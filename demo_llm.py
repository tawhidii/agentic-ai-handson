import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
url = "https://kubernetes.io/docs/concepts/architecture/"

loader = WebBaseLoader(url)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

split_docs = text_splitter.split_documents(docs)
# print(split_docs)
# print(f"Split into {len(split_docs)} chunks")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
vectorstore = FAISS.from_documents(split_docs, embeddings)
print(f"Created FAISS vectorstore with {vectorstore.index.ntotal} vectors")

query = "What is Iron man?"
llm = ChatOpenAI(model="gpt-4o")

prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {input}

Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": query})
print(f"Question: {query}")
print(f"Answer: {response['answer']}")
print(f"Retrieved {len(response['context'])} documents")

