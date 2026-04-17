import streamlit as st
import PyPDF2
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Screening System",layout="centered")

st.title("Resume Screening System")

# ---------- PDF TEXT EXTRACTION ----------
def extract_text(file):
    text=""
    pdf_reader=PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        if page.extract_text():
            text+=page.extract_text()
    return text

# ---------- KEYWORD EXTRACTION ----------
def get_keywords(text):
    words=text.lower().split()
    return set(words)

# ---------- INPUT ----------
job_desc=st.text_area("📄 Enter Job Description")

uploaded_file=st.file_uploader("📎 Upload Resume (PDF)",type=["pdf"])

# ---------- PROCESS ----------
if st.button("Analyze Resume"):

    if job_desc.strip()=="":
        st.warning("Please enter job description")

    elif uploaded_file is None:
        st.warning("Please upload a resume")

    else:
        resume_text=extract_text(uploaded_file)

        documents=[job_desc,resume_text]

        tfidf=TfidfVectorizer(stop_words="english")
        vectors=tfidf.fit_transform(documents)

        score=cosine_similarity(vectors[0:1],vectors[1:2])[0][0]

        # ---------- DISPLAY SCORE ----------
        st.subheader("📊 Match Score")
        st.write(f"### {score*100:.2f}%")

        # ---------- BAR CHART ----------
        fig,ax=plt.subplots()
        ax.bar(["Match Score"],[score*100])
        ax.set_ylim(0,100)
        st.pyplot(fig)

        # ---------- RESULT ----------
        if score>0.7:
            st.success("Strong Match ✅")
        elif score>0.4:
            st.warning("Average Match ⚠️")
        else:
            st.error("Low Match ❌")

        # ---------- KEYWORD MATCHING ----------
        st.subheader("🔍 Matching Keywords")

        job_keywords=get_keywords(job_desc)
        resume_keywords=get_keywords(resume_text)

        common=job_keywords.intersection(resume_keywords)

        if common:
            st.write(", ".join(list(common)[:20]))
        else:
            st.write("No significant keyword match found")