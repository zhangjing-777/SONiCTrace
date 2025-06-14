from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME


def get_embedding_model():

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return lambda text: model.encode(
        "Represent this sentence for searching relevant passages: " + text,
        normalize_embeddings=True
    ).tolist()
