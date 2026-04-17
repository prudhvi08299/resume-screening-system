import streamlit as st
import PyPDF2
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Analyzer",layout="wide")

st.title("📄 AI Resume Screening System")

# -------- PDF TEXT --------
def extract_text(file):
    text=""
    pdf_reader=PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        if page.extract_text():
            text+=page.extract_text()
    return text

# -------- KEYWORDS --------
def get_keywords(text):
    return set(text.lower().split())

# -------- INPUT SECTION --------
col1,col2=st.columns(2)

with col1:
    job_desc=st.text_area("📌 Job Description",height=200)

with col2:
    uploaded_file=st.file_uploader("📎 Upload Resume (PDF)",type=["pdf"])

st.markdown("---")

# -------- BUTTON --------
if st.button("🔍 Analyze Resume"):

    if job_desc.strip()=="":
        st.warning("Enter job description")

    elif uploaded_file is None:
        st.warning("Upload resume")

    else:
        resume_text=extract_text(uploaded_file)

        documents=[job_desc,resume_text]

        tfidf=TfidfVectorizer(stop_words="english")
        vectors=tfidf.fit_transform(documents)

        score=cosine_similarity(vectors[0:1],vectors[1:2])[0][0]
        percent=score*100

        # -------- SCORE DISPLAY --------
        st.subheader("📊 Match Score")
        st.progress(int(percent))
        st.write(f"### {percent:.2f}%")

        # -------- BAR CHART --------
        fig,ax=plt.subplots()
        ax.barh(["Match"],[percent])
        ax.set_xlim(0,100)
        st.pyplot(fig)

        # -------- RESULT --------
        if score>0.7:
            st.success("🔥 Strong Match")
        elif score>0.4:
            st.warning("⚠️ Moderate Match")
        else:
            st.error("❌ Low Match")

        st.markdown("---")

        # -------- KEYWORD MATCH --------
        st.subheader("🔍 Matching Keywords")

        job_keywords=get_keywords(job_desc)
        resume_keywords=get_keywords(resume_text)

        common=job_keywords.intersection(resume_keywords)

        if common:
            st.write(", ".join(list(common)[:20]))
        else:
            st.write("No strong keyword match")

        st.markdown("---")

        # -------- EXTRA INSIGHT --------
        st.subheader("📌 Analysis")

        st.write(f"- Total Keywords in Job Description: {len(job_keywords)}")
        st.write(f"- Total Keywords in Resume: {len(resume_keywords)}")
        st.write(f"- Matching Keywords: {len(common)}")