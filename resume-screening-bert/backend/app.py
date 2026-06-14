from fastapi import FastAPI, UploadFile, File
import shutil
import os
from search import search_candidates

app = FastAPI(
    title = "Resume Screening ATS"
)

UPLOAD_FOLDER ="uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

@app.get('/')
def home():
    return {
        "message":"Resume Screening ATS Running"
    }

@app.post('/upload_resume')
async def upload_resume(
    file: UploadFile = File(...)
):
    save_path  =os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(
        save_path,
        "wb"
    ) as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    return{
        "status":"success",
        "file": file.filename
    }

@app.post('/search')
def search(job_description:str):
    results = search_candidates(
        job_description,
        top_k=5
    )

    return {
        "results": results
    }



