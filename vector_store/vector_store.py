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


load_dotenv()

PG_CONFIG = {
    "dbname": os.getenv("PG_DBNAME"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
    "host": os.getenv("PG_HOST"),
    "port": os.getenv("PG_PORT"),
}


def insert_chunks_to_pg(chunks, embed_fn, vb_table_name):
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


def clear_pgvector_table(vb_table_name):
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {vb_table_name};")
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Table '{vb_table_name}' cleared successfully.")