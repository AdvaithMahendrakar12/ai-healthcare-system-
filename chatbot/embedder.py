"""
BioBERT Sentence Embedder
Singleton loader so model loads only once at startup.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

_model = None
MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"

def get_embedder():
    global _model
    if _model is None:
        print(f"⏳ Loading BioBERT embedder: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        print("✅ BioBERT ready!")
    return _model

def embed_query(text: str) -> np.ndarray:
    """Encode a single query string → normalized numpy vector."""
    model = get_embedder()
    vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return vec  # shape (1, 768)
