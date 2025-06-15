"""
FastAPI application for RAG (Retrieval-Augmented Generation) system with PGVector integration.

This module provides REST API endpoints for:
- Querying the RAG system
- Uploading and processing PDF documents
- Managing vector store tables
- Interactive chat functionality

The API uses FastAPI framework and integrates with:
- Vector store for document storage and retrieval
- RAG agent for intelligent question answering
- PGVector for vector database operations
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vector_store.embedding import get_embedding_model         
from vector_store.chunking import chunks_app  
from vector_store.vector_store import insert_chunks_to_pg, clear_pgvector_table
from rag_agent.rag_pipeline import qa_chain   
from rag_agent.rag_agent import RAGAgent
from config import AVGO_TABLE_NAME
from logger import setup_logger

# Setup logger
logger = setup_logger("app", "app.log")

app = FastAPI(title="RAG + PGVector API")
agent = RAGAgent(vb_table_name=AVGO_TABLE_NAME, history_limit=10)

# Request Schemas
class QueryRequest(BaseModel):
    question: str
    vb_table: str

class UploadRequest(BaseModel):
    pdf_path: str
    vb_table: str

class ClearRequest(BaseModel):
    vb_table: str

class ChatRequest(BaseModel):
    query: str
    session_id: str = None

class ChatResponse(BaseModel):
    session_id: str
    result: str


# API Endpoints

@app.post("/query")
def query_api(req: QueryRequest):
    """
    Query the RAG chain using a specified table and question.
    """
    try:
        logger.info(f"Processing query request: {req.question} for table: {req.vb_table}")
        chain = qa_chain(req.vb_table)
        answer = chain.run(req.question)
        logger.info(f"Successfully generated answer for query: {req.question}")
        return {"question": req.question, "answer": answer}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
def upload_api(req: UploadRequest):
    """
    Upload a PDF, chunk it, embed it, and insert into the specified vector table.
    """
    try:
        logger.info(f"Processing upload request for PDF: {req.pdf_path} to table: {req.vb_table}")
        chunks = chunks_app(req.pdf_path)
        embed_fn = get_embedding_model()
        insert_chunks_to_pg(chunks, embed_fn, req.vb_table)
        logger.info(f"Successfully uploaded {len(chunks)} chunks to table: {req.vb_table}")
        return {"status": "✅ uploaded", "count": len(chunks)}
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
def clear_api(req: ClearRequest):
    """
    Clear all embeddings from a vector table by name.
    """
    try:
        logger.info(f"Processing clear request for table: {req.vb_table}")
        clear_pgvector_table(req.vb_table)
        logger.info(f"Successfully cleared table: {req.vb_table}")
        return {"status": f"✅ cleared {req.vb_table}"}
    except Exception as e:
        logger.error(f"Error clearing table: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        logger.info(f"Processing chat request: {request.query} with session: {request.session_id}")
        result = agent.run(request.query, session_id=request.session_id)
        logger.info(f"Successfully generated chat response for session: {request.session_id}")
        return ChatResponse(session_id=request.session_id or "new-session", result=result)
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))