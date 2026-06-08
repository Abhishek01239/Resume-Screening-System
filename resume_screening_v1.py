from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

job_description = """
Looking for a Machine Learning Engineer.
Skills:
Python, TensorFlow, Deep Learning,
NLP, PyTorch
"""

resume = """
Experienced Software Engineer.
Worked on Python, NLP,
Tensorflow and Deep Learning projects.
"""

job_embedding = model.encode(
    job_description
)

resume_embedding = model.encode(
    resume
)

score = cosine_similarity(
    [job_embedding],
    [resume_embedding]
)

print("Match Score: ", round(score[0][0]*100,2), "%")
