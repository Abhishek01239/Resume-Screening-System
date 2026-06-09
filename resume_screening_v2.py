from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# Model
# -----------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------
# Job Description
# -----------------------

job_description = """
Looking for a Machine Learning Engineer.

Required Skills:
Python
TensorFlow
PyTorch
Deep Learning
NLP
"""

required_skills = {
    "python",
    "tensorflow",
    "pytorch",
    "deep learning",
    "nlp"
}

# -----------------------
# Resumes
# -----------------------

resumes = {

    "Candidate A":
    """
    Python developer with experience
    in TensorFlow, NLP and Deep Learning.
    """,

    "Candidate B":
    """
    Java developer with Spring Boot,
    SQL and Microservices.
    """,

    "Candidate C":
    """
    Machine Learning Engineer with
    Python, PyTorch, NLP and
    Deep Learning experience.
    """
}

# -----------------------
# Job Embedding
# -----------------------

job_embedding = model.encode(
    job_description
)

results = []

# -----------------------
# Process Each Resume
# -----------------------

for name, resume in resumes.items():

    resume_embedding = model.encode(
        resume
    )

    similarity = cosine_similarity(
        [job_embedding],
        [resume_embedding]
    )[0][0]

    # Skill Matching

    resume_lower = resume.lower()

    matched_skills = []

    for skill in required_skills:

        if skill in resume_lower:
            matched_skills.append(skill)

    missing_skills = list(
        required_skills -
        set(matched_skills)
    )

    results.append({

        "candidate": name,

        "score":
            round(similarity * 100, 2),

        "matched_skills":
            matched_skills,

        "missing_skills":
            missing_skills
    })

# -----------------------
# Sort By Score
# -----------------------

results.sort(
    key=lambda x: x["score"],
    reverse=True
)

# -----------------------
# Display Results
# -----------------------

for rank, candidate in enumerate(
        results,
        start=1):

    print(
        f"\nRank {rank}"
    )

    print(
        f"Candidate: "
        f"{candidate['candidate']}"
    )

    print(
        f"Match Score: "
        f"{candidate['score']}%"
    )

    print(
        "Matched Skills:",
        candidate["matched_skills"]
    )

    print(
        "Missing Skills:",
        candidate["missing_skills"]
    )