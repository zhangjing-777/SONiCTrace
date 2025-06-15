"""
Vector Store operations and management.

This module provides functionality for:
- Storing document chunks and embeddings in PGVector
- Clearing vector store tables

The vector store is used to persist document embeddings and enable
efficient similarity search.
"""

import os
import psycopg2
from dotenv import load_dotenv
from ..logger import setup_logger


load_dotenv()

# Setup logger
logger = setup_logger("vector_store", "vector_store.log")

PG_CONFIG = {
    "dbname": os.getenv("PG_DBNAME"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
    "host": os.getenv("PG_HOST"),
    "port": os.getenv("PG_PORT"),
}


def insert_chunks_to_pg(chunks, embed_fn, vb_table_name):
    """
    Insert document chunks into PGVector table.
    """
    try:
        logger.info(f"Inserting {len(chunks)} chunks into table: {vb_table_name}")
        conn = psycopg2.connect(**PG_CONFIG)
        cur = conn.cursor()

        cur.execute(f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        
        CREATE TABLE IF NOT EXISTS {vb_table_name} (
            id SERIAL PRIMARY KEY,
            section TEXT,
            content TEXT,
            embedding VECTOR(384),
            page_range INT[],
            source TEXT
        );
        """)

        for chunk in chunks:
            emb = embed_fn(chunk['content'])
            cur.execute(
                f"""INSERT INTO {vb_table_name} (section, content, embedding, page_range, source)
                    VALUES (%s, %s, %s, %s, %s);""",
                (chunk['section'], chunk['content'], emb, chunk['page_range'], chunk['source'])
            )

        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Successfully inserted chunks into table: {vb_table_name}")
    except Exception as e:
        logger.error(f"Error inserting chunks into PGVector: {str(e)}")
        raise


def clear_pgvector_table(vb_table_name):
    """
    Clear all embeddings from a PGVector table.
    """
    try:
        logger.info(f"Clearing table: {vb_table_name}")
        conn = psycopg2.connect(**PG_CONFIG)
        cur = conn.cursor()
        cur.execute(f"DELETE FROM {vb_table_name};")
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Successfully cleared table: {vb_table_name}")
        print(f"✅ Table '{vb_table_name}' cleared successfully.")
    except Exception as e:
        logger.error(f"Error clearing PGVector table: {str(e)}")
        raise