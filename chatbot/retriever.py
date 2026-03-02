"""
FAISS Retriever
Loads index + Q&A store, performs semantic search.
"""

import os
import pickle
import faiss
import numpy as np

# Use absolute import so it works when imported from app.py at the project root
from chatbot.embedder import embed_query

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "model", "faiss_index.bin")
STORE_PATH = os.path.join(BASE_DIR, "model", "qa_store.pkl")

_index = None
_qa_store = None


def load_index():
    global _index, _qa_store
    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                "Run: python model/build_faiss_index.py"
            )
        _index = faiss.read_index(INDEX_PATH)
        with open(STORE_PATH, 'rb') as f:
            _qa_store = pickle.load(f)
        print(f"✅ FAISS index loaded: {_index.ntotal} vectors")
    return _index, _qa_store


def retrieve(query: str, top_k: int = 5, threshold: float = 0.30):
    """
    Retrieve top-k most relevant Q&A pairs for a query.
    Returns list of dicts with keys: question, answer, score, category.
    """
    index, qa_store = load_index()

    query_vec = embed_query(query)   # (1, 768)

    # Search FAISS
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        if float(score) < threshold:
            continue
        qa = qa_store[idx]
        results.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "category": qa.get("category", "general"),
            "source": qa.get("source", ""),
            "score": float(score)
        })

    return results
