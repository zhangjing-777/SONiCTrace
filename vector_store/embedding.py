"""
Document Embedding Model configuration.

This module provides functionality for:
- Loading and configuring the embedding model
- Generating embeddings for text chunks
- Managing embedding model parameters

The embedding model is used to convert text into vector representations
for similarity search in the vector store.
"""

from sentence_transformers import SentenceTransformer
from ..config import EMBEDDING_MODEL_NAME


def get_embedding_model():

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return lambda text: model.encode(
        "Represent this sentence for searching relevant passages: " + text,
        normalize_embeddings=True
    ).tolist()
