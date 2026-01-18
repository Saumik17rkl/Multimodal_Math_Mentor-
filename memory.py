import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from datetime import datetime
import os

embedder = SentenceTransformer("all-MiniLM-L6-v2")

mongo = MongoClient(os.getenv("MONGODB_URI"))
db = mongo["jee_solver"]
memory_col = db["jee_memory"]


def embed(text: str):
    return embedder.encode(text).tolist()


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def store_memory(question, solution, concept, approved, source="llm"):
    memory_col.insert_one({
        "question": question,
        "solution": solution,
        "concept": concept,
        "embedding": embed(question),
        "approved": approved,
        "confidence": 1.0 if approved else 0.3,
        "source": source,
        "timestamp": datetime.utcnow()
    })


def retrieve_memory(question, threshold=0.85):
    q_emb = embed(question)
    docs = list(memory_col.find({"approved": True}))

    best = None
    best_score = 0

    for d in docs:
        score = cosine(q_emb, d["embedding"])
        if score > best_score and score >= threshold:
            best = d
            best_score = score

    return best
