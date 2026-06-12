from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import os
import re

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

required_skills = [
    "Python",
    "TensorFlow",
    "PyTorch",
    "Deep Learning",
    "NLP"
]

required_experience = 3

preferred_education = [
    "B.Tech",
    "M.Tech",
    "Computer Science",
    "Artificial Intelligence"
]

preferred_certifications = [
    "AWS",
    "Machine Learning"
]

def extract_text_from_pdf(path):

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + " "

    return text

def extract_experience(text):

    pattern = r'(\d+)\s+years'

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return 0

def education_score(text):

    score = 0

    for edu in preferred_education:

        if edu.lower() in text.lower():

            score += 25

    return min(score, 100)

def certification_score(text):

    score = 0

    for cert in preferred_certifications:

        if cert.lower() in text.lower():

            score += 50

    return min(score, 100)

def skill_score(text):

    matched = 0

    text_lower = text.lower()

    for skill in required_skills:

        if skill.lower() in text_lower:

            matched += 1

    return (
        matched /
        len(required_skills)
    ) * 100

def experience_score(years):

    if years >= required_experience:
        return 100

    return (
        years /
        required_experience
    ) * 100

results = []

# Determine the resumes folder relative to this script file
script_dir = os.path.dirname(os.path.abspath(__file__))
resume_folder = os.path.join(script_dir, "resumes")

if not os.path.isdir(resume_folder):
    raise FileNotFoundError(
        f"Resumes folder not found at: {resume_folder}"
    )

for file in os.listdir(resume_folder):

    if not file.endswith(".pdf"):
        continue

    path = os.path.join(
        resume_folder,
        file
    )

    text = extract_text_from_pdf(
        path
    )

    skills = skill_score(text)

    years = extract_experience(
        text
    )

    experience = experience_score(
        years
    )

    education = education_score(
        text
    )

    certification = (
        certification_score(text)
    )

    final_score = (

        skills * 0.4 +

        experience * 0.3 +

        education * 0.2 +

        certification * 0.1
    )

    results.append({

        "candidate":
            file,

        "skills":
            round(skills, 2),

        "experience":
            round(experience, 2),

        "education":
            round(education, 2),

        "certification":
            round(certification, 2),

        "ats":
            round(final_score, 2)
    })

results.sort(
    key=lambda x: x["ats"],
    reverse=True
)

print("\nATS RANKING\n")

for rank, candidate in enumerate(
        results,
        start=1):

    print(
        f"Rank #{rank}"
    )

    print(
        f"Resume: "
        f"{candidate['candidate']}"
    )

    print(
        f"Skill Score: "
        f"{candidate['skills']}"
    )

    print(
        f"Experience Score: "
        f"{candidate['experience']}"
    )

    print(
        f"Education Score: "
        f"{candidate['education']}"
    )

    print(
        f"Certification Score: "
        f"{candidate['certification']}"
    )

    print(
        f"ATS Score: "
        f"{candidate['ats']}"
    )

    print("-" * 50)