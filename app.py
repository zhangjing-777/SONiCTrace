from fastapi import FastAPI
from pydantic import BaseModel
from embedding import get_embedding_model         
from chunking import chunks_app  
from vector_store import insert_chunks_to_pg, clear_pgvector_table
from rag_pipeline import qa_chain   
from rag_agent import RAGAgent
from config import AVGO_TABLE_NAME
       

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
    chain = qa_chain(req.vb_table)
    answer = chain.run(req.question)
    return {"question": req.question, "answer": answer}

@app.post("/upload")
def upload_api(req: UploadRequest):
    """
    Upload a PDF, chunk it, embed it, and insert into the specified vector table.
    """
    chunks = chunks_app(req.pdf_path)
    embed_fn = get_embedding_model()
    insert_chunks_to_pg(chunks, embed_fn, req.vb_table)
    return {"status": "✅ uploaded", "count": len(chunks)}

@app.post("/clear")
def clear_api(req: ClearRequest):
    """
    Clear all embeddings from a vector table by name.
    """
    clear_pgvector_table(req.vb_table)
    return {"status": f"✅ cleared {req.vb_table}"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = agent.run(request.query, session_id=request.session_id)
    return ChatResponse(session_id=request.session_id or "new-session", result=result)