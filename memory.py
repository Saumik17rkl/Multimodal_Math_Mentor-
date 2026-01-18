import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from datetime import datetime
import os

# ---------- LAZY GLOBALS ----------
_embedder = None
_memory_col = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_collection():
    global _memory_col
    if _memory_col is None:
        mongo = MongoClient(os.getenv("MONGODB_URI"))
        db = mongo["jee_solver"]
        _memory_col = db["jee_memory"]
    return _memory_col


def embed(text: str):
    model = _get_embedder()
    return model.encode(text).tolist()


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return np.dot(a, b) / denom


def store_memory(question, solution, concept, approved, source="llm"):
    col = _get_collection()
    col.insert_one({
        "question": question,
        "solution": solution,
        "concept": concept,
        "embedding": embed(question),
        "approved": approved,
        "confidence": 1.0 if approved else 0.3,
        "source": source,
        "timestamp": datetime.utcnow()
    })


def retrieve_memory(question, threshold=0.85, limit=50):
    col = _get_collection()
    q_emb = embed(question)

    docs = col.find(
        {"approved": True},
        limit=limit
    )

    best = None
    best_score = 0.0

    for d in docs:
        if "embedding" not in d:
            continue

        score = cosine(q_emb, d["embedding"])
        if score > best_score and score >= threshold:
            best = d
            best_score = score

    return best
