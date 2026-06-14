import  streamlit as st
import requests

st.title(
    "AI Resume Screening ATS"
)

st.write("Upload resumes and research candidates")

st.header("Upload Resume")

uploaded_file = st.file_uploader(
    "Choose Resume PDF",
    type = ["pdf"]
)

if uploaded_file:
    files = {
        "file":(
            uploaded_file.name,
            uploaded_file,
            "application/pdf"
        )
    }

    response = requests.post(
        "http://127.0.0.1:8000/upload_resume",
        files = files
    )

    st.success(
        "Resume Uploaded Successfully"
    )

st.header(
    "Job Description"
)

job_description = st.text_area(
    "Paste Job Description"
)

if st.button(
    "Search Candidate"
):
    response = requests.post(
        "http://127.0.0.1:8000/search",

        params= {
            "job_description":
            job_description
        }
    )

    data = response.json()

    st.header("Results")

    for candidate in data["results"]:
        st.write(
            f"Rank: "
            f"{candidate['rank']}"
        )

        st.write(
            f"Resume: "
            f"{candidate['candidate']}"
        )

        st.write(
            f"Distance: "
            f"{candidate['distance']}"
        )

        st.write(
            "--------------------"
        )