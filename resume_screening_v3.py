from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

required_skills = [
    "Python",
    "TensorFlow",
    "PyTorch",
    "Machine Learning",
    "Natural Language Processing"
]

resume_skills = [
    "Python Programming",
    "Torch",
    "ML Engineer",
    "NLP"
]

threshold = 0.70

matched_skills = []

for job_skill in required_skills:

    job_embedding = model.encode(
        job_skill
    )

    best_score = 0
    best_match = None

    for resume_skill in resume_skills:

        resume_embedding = model.encode(
            resume_skill
        )

        similarity = cosine_similarity(
            [job_embedding],
            [resume_embedding]
        )[0][0]

        if similarity > best_score:

            best_score = similarity
            best_match = resume_skill

    if best_score >= threshold:

        matched_skills.append({

            "required_skill":
                job_skill,

            "matched_with":
                best_match,

            "score":
                round(best_score, 2)
        })

print("\nMatched Skills:\n")

for skill in matched_skills:

    print(skill)