import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(file):
    text=""
    pdf_reader=PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        text+=page.extract_text()
    return text

def rank_resumes(job_description,uploaded_files):
    resumes=[]
    names=[]

    for file in uploaded_files:
        text=extract_text_from_pdf(file)
        resumes.append(text)
        names.append(file.name)

    documents=[job_description]+resumes

    tfidf=TfidfVectorizer(stop_words='english')
    vectors=tfidf.fit_transform(documents)

    similarity=cosine_similarity(vectors[0:1],vectors[1:]).flatten()

    ranked=sorted(zip(names,similarity),key=lambda x:x[1],reverse=True)

    return ranked