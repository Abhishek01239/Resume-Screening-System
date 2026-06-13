from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import os
import numpy as np
import pickle

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def extract_text(path):
    reader = PdfReader(path)

    text = "" 

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text+=page_text
    return text

resume_folder = os.path.join(
    os.path.dirname(__file__),
    "resumes"
)

resume_names = []

embeddings = []

for file in os.listdir(
    resume_folder
):
    if not file.endswith(".pdf"):
        continue

    path = os.path.join(
        resume_folder,
        file
    )

    text = extract_text(path)

    embedding = model.encode(
        text
    )

    embeddings.append(embedding)

    resume_names.append(file)

embeddings = np.array(
    embeddings
).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(embeddings)

output_dir = os.path.dirname(__file__)

faiss.write_index(
    index,
    os.path.join(output_dir, "resume_index.faiss")
)

with open(
    os.path.join(output_dir, "resume_names.pkl"),
    "wb"
) as f:
    pickle.dump(resume_names, f)

print("Index Created Successfully")
