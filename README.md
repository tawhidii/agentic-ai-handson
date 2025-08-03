# RAG Implementation with LangChain

A modular Retrieval-Augmented Generation (RAG) implementation using LangChain, OpenAI, and FAISS for intelligent document querying and question-answering.

## 🌟 Features

- 🔍 **Document Loading**: Web-based document loading with URL support
- ✂️ **Text Splitting**: Intelligent document chunking with overlap
- 🧠 **Embeddings**: OpenAI embeddings with configurable models
- 🗄️ **Vector Storage**: FAISS vector store for efficient similarity search
- 🤖 **RAG Chain**: Complete retrieval-augmented generation pipeline
- 🏗️ **Modular Design**: Clean, maintainable code structure

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- LangChain API key (optional, for tracing)

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run Basic Demo**
   ```bash
   python examples/basic_rag_demo.py
   ```

4. **Use Main Application**
   ```bash
   python src/main.py
   ```

## 📁 Project Structure

```
agentic-ai/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Environment configuration
│   ├── core/
│   │   ├── __init__.py
│   │   ├── document_loader.py   # Document loading and splitting
│   │   ├── embeddings.py        # OpenAI embeddings management
│   │   ├── vector_store.py      # FAISS vector store operations
│   │   └── rag_chain.py         # RAG pipeline implementation
│   ├── utils/
│   │   └── __init__.py
│   └── main.py                  # Main application entry point
├── examples/
│   ├── __init__.py
│   └── basic_rag_demo.py        # Basic usage example
├── tests/
│   └── __init__.py              # Unit tests (ready for implementation)
├── docs/                        # Documentation
└── data/
    ├── documents/               # Document storage
    └── vector_stores/           # Vector store persistence
```

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

- **DocumentLoader**: Handles web document loading and text splitting
- **EmbeddingManager**: Manages OpenAI embeddings configuration
- **VectorStoreManager**: FAISS vector store operations and retrieval
- **RAGChain**: Complete RAG pipeline with LLM integration
- **Settings**: Centralized configuration management

## ⚙️ Configuration

All settings are managed through environment variables in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=your_project_name
```

## 📚 Usage Examples

### Basic RAG Implementation

```python
from src.config.settings import settings
from src.core.document_loader import DocumentLoader
from src.core.embeddings import EmbeddingManager
from src.core.vector_store import VectorStoreManager
from src.core.rag_chain import RAGChain

# Initialize components
doc_loader = DocumentLoader()
embedding_manager = EmbeddingManager()
vector_store_manager = VectorStoreManager(embedding_manager.get_embeddings())
rag_chain = RAGChain()

# Load and process documents
url = "https://kubernetes.io/docs/concepts/architecture/"
split_docs = doc_loader.load_and_split(url)

# Create vectorstore and RAG chain
vectorstore = vector_store_manager.create_vectorstore(split_docs)
retriever = vector_store_manager.get_retriever()
chain = rag_chain.create_chain(retriever)

# Query the system
response = rag_chain.invoke("What is Kubernetes?")
print(response['answer'])
```

### Custom Configuration

```python
# Custom chunk size and overlap
doc_loader = DocumentLoader(chunk_size=500, chunk_overlap=100)

# Different embedding model
embedding_manager = EmbeddingManager(
    model="text-embedding-3-small", 
    dimensions=512
)

# Different LLM model
rag_chain = RAGChain(model="gpt-3.5-turbo")
```

## 🧪 Testing

Run the examples to test functionality:

```bash
# Basic demo (equivalent to original demo_llm.py)
python examples/basic_rag_demo.py

# Main application
python src/main.py
```

## 🔧 Key Components

### DocumentLoader
- Web document loading with `WebBaseLoader`
- Configurable text splitting with `RecursiveCharacterTextSplitter`
- Support for various document sources

### EmbeddingManager
- OpenAI embeddings with model selection
- Configurable dimensions (512, 1024, 3072)
- Support for different embedding models

### VectorStoreManager
- FAISS vector store creation and management
- Similarity search functionality
- Retriever configuration

### RAGChain
- Complete RAG pipeline implementation
- Customizable prompt templates
- LLM integration with various models

## 📈 Performance

- **Default Settings**: `text-embedding-3-large` with 1024 dimensions
- **Chunk Strategy**: 1000 characters with 200-character overlap
- **Vector Search**: FAISS for efficient similarity search
- **LLM**: GPT-4o for high-quality responses

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes in the appropriate module
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 📚 Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [FAISS Documentation](https://faiss.ai/)

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root or adjust Python path
2. **API Key Issues**: Verify `.env` file exists and contains valid keys
3. **Module Not Found**: Check that all `__init__.py` files are present

### Migration from Original Code

The original `demo_llm.py` functionality is preserved in `examples/basic_rag_demo.py`. The refactored version provides the same results with improved code organization.