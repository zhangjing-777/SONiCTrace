"""
Document Retriever implementation for the RAG system.

This module provides the document retrieval functionality:
- Uses PGVector for vector similarity search
- Integrates with HuggingFace embeddings
- Configures connection to the vector database
- Provides methods for retrieving relevant documents

The retriever is responsible for finding the most relevant documents
based on semantic similarity to the query.
"""

from langchain_community.vectorstores.pgvector import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from ..config import EMBEDDING_MODEL_NAME
from ..logger import setup_logger

load_dotenv()

# Setup logger
logger = setup_logger("retriever", "retriever.log")

dbname = os.getenv("PG_DBNAME")
user = os.getenv("PG_USER")
password = os.getenv("PG_PASSWORD")
host = os.getenv("PG_HOST")
port = os.getenv("PG_PORT")

CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True}
)

def retriever(collection_name, k=3):
    try:
        logger.info(f"Initializing retriever for collection: {collection_name}")
        vectorstore = PGVector(
            collection_name=collection_name,
            connection_string=CONNECTION_STRING,
            embedding_function=embedding_model,
        )
        logger.info(f"Successfully initialized retriever for collection: {collection_name}")
        return vectorstore.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        logger.error(f"Error initializing retriever: {str(e)}")
        raise

