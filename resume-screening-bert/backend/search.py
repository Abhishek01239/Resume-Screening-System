from pathlib import Path

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "resume_index.faiss"
NAMES_PATH = BASE_DIR / "resume_names.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(str(INDEX_PATH))

with open(NAMES_PATH, "rb") as f:
    resume_names = pickle.load(f)


def search_candidates(job_description, top_k=5):
    query_embedding = model.encode(job_description)
    query_embedding = np.array([query_embedding], dtype="float32")

    distance, indices = index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        results.append({
            "rank": rank,
            "candidate": resume_names[idx],
            "distance": float(distance[0][rank - 1]),
        })
    return results


if __name__ == "__main__":
    job_description = """
    Looking for AI Engineer

    Skills:
    Python
    PyTorch
    Deep Learning
    NLP
    Transformers
    """

    results = search_candidates(job_description)
    for r in results:
        print(r)
