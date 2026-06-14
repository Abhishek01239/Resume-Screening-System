from pathlib import Path

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

from parser import extract_word_from_pdf

BASE_DIR = Path(__file__).resolve().parent
RESUME_FOLDER = BASE_DIR / "uploads"
INDEX_PATH = BASE_DIR / "resume_index.faiss"
NAMES_PATH = BASE_DIR / "resume_names.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")

if not RESUME_FOLDER.exists():
    RESUME_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Created upload folder: {RESUME_FOLDER}")
    print("Add PDF resumes to this folder and rerun the script.")
    raise SystemExit(1)

resume_names = []
embeddings = []

for file in sorted(RESUME_FOLDER.iterdir()):
    if file.suffix.lower() != ".pdf":
        continue

    print(f"Processing: {file.name}")
    text = extract_word_from_pdf(file)
    embedding = model.encode(text)
    embeddings.append(np.asarray(embedding, dtype="float32"))
    resume_names.append(file.name)

if not embeddings:
    print(f"No PDF resumes found in {RESUME_FOLDER}. Add files and rerun.")
    raise SystemExit(1)

embeddings = np.vstack(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, str(INDEX_PATH))
with open(NAMES_PATH, "wb") as f:
    pickle.dump(resume_names, f)

print("\nIndex created successfully")