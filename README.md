# SONiCTrace: RAG-based Network Documentation Assistant

A Retrieval-Augmented Generation (RAG) system designed to help network engineers interact with network device documentation through natural language queries.

## Features

- 📚 **Document Processing**
  - PDF document ingestion and processing
  - Semantic-aware document chunking
  - Support for multiple vendor documentations (Broadcom SONiC, Arista EOS, Cisco NX-OS)

- 🔍 **Intelligent Search**
  - Vector-based semantic search
  - Context-aware document retrieval
  - PGVector integration for efficient storage and retrieval

- 💬 **Interactive Chat**
  - Natural language query interface
  - Context-aware responses
  - Conversation history management
  - Session-based interactions

## Architecture

The project is organized into two main modules:

### Vector Store Module
- `chunking.py`: Semantic-aware document chunking
- `embedding.py`: Document embedding generation
- `vector_store.py`: PGVector integration and management

### RAG Agent Module
- `rag_agent.py`: Core RAG implementation
- `rag_pipeline.py`: RAG pipeline configuration
- `llm.py`: Language model integration
- `retriever.py`: Document retrieval system
- `prompting.py`: Prompt templates and management

## Setup

1. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Variables**
Create a .env file in the project root with the following content (fill in your actual values):

```
# PostgreSQL connection settings

PG_DBNAME=your_dbname
PG_USER=your_pg_user
PG_PASSWORD=your_pg_password
PG_HOST=your_pg_host
PG_PORT=your_pg_port

# LLM setting

OPENROUTER_API_KEY = your_openrouter_api_key

# Supabase setting

SUPABASE_URL = your_supabase_url
SUPABASE_KEY = your_supabase_key
```

1. **Run the Application**

   **Option 1: Run Locally**
   ```bash
   # Development mode with auto-reload
   uvicorn app:app --reload --host 0.0.0.0 --port 8000

   # Production mode
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

   **Option 2: Run with Docker**
   ```bash
   # Build and run with Docker Compose
   docker-compose up --build

   # Or run with Docker directly
   docker build -t sonic-rag-api .
   docker run -p 8000:8000 --env-file .env sonic-rag-api
   ```

   After starting the server, you can:
   - Access the API documentation at `http://localhost:8000/docs`
   - Use the interactive Swagger UI at `http://localhost:8000/redoc`
   - Make API calls to `http://localhost:8000/`

## API Endpoints

- `POST /query`: Query the RAG system
- `POST /upload`: Upload and process PDF documents
- `POST /clear`: Clear vector store tables
- `POST /chat`: Interactive chat endpoint

## Configuration

The `config.py` file contains settings for:
- Document chunking parameters
- Embedding model configuration
- Vector store table names
- Language model settings

## Development

### Project Structure
```
.
├── app.py                 # FastAPI application
├── config.py             # Configuration settings
├── requirements.txt      # Project dependencies
├── vector_store/         # Vector store module
│   ├── chunking.py
│   ├── embedding.py
│   └── vector_store.py
└── rag_agent/           # RAG agent module
    ├── rag_agent.py
    ├── rag_pipeline.py
    ├── llm.py
    ├── retriever.py
    └── prompting.py
```

### Adding New Vendor Support
1. Add vendor configuration in `config.py`
2. Implement vendor-specific chunking rules
3. Update document processing pipeline


