from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import os

print("Loading Transformer Model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model Loaded Successfully!\n")

# ==========================================
# Job Description
# ==========================================

job_description = """
Looking for a Machine Learning Engineer.

Required Skills:
Python
TensorFlow
PyTorch
Deep Learning
Natural Language Processing

Responsibilities:
Build machine learning models.
Work on NLP systems.
Deploy AI solutions.
"""

job_embedding = model.encode(
    job_description
)

def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + " "

    return text

resume_folder = os.path.join(
    os.path.dirname(__file__),
    "resumes"
)

if not os.path.isdir(resume_folder):
    raise FileNotFoundError(
        f"Resume folder not found: {resume_folder}"
    )

results = []

for file_name in os.listdir(
        resume_folder):

    if file_name.endswith(".pdf"):

        file_path = os.path.join(
            resume_folder,
            file_name
        )

        print(
            f"Processing {file_name}"
        )

        resume_text = extract_text_from_pdf(
            file_path
        )

        resume_embedding = model.encode(
            resume_text
        )

        similarity = cosine_similarity(
            [job_embedding],
            [resume_embedding]
        )[0][0]

        results.append({

            "candidate":
                file_name,

            "score":
                round(
                    similarity * 100,
                    2
                )
        })

results.sort(
    key=lambda x: x["score"],
    reverse=True
)


print("\n")
print("=" * 50)
print("FINAL RANKING")
print("=" * 50)

for rank, candidate in enumerate(
        results,
        start=1):

    print(
        f"\nRank #{rank}"
    )

    print(
        f"Resume : "
        f"{candidate['candidate']}"
    )

    print(
        f"Score  : "
        f"{candidate['score']}%"
    )