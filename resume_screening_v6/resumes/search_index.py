from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

index = faiss.read_index(
    "resume_index.faiss"
)

with open(
    "resume_names.pkl",
    "rb"
) as f:
    resume_names  = pickle.load(f)

job_description = """
Looking for AI Engineer.
Skills:
Python
PyTorch
NLP
Deep Learning
Transformers
"""

query_embedding = model.encode(
    job_description
)

query_embedding = np.array(
    [query_embedding]
).astype("float32")

k = 3
distances, indices = index.search(query_embedding,k)

print("Top Candidates")

for rank, idx in enumerate(
    indices[0],
    start=1
):
    print(f"Rank {rank}")

    print(resume_names[idx])

    print(
        f"Distance: "
        f"{distances[0][rank-1]}"
    )
    