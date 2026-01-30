# Om Shri Vighna Ganapathi Namosthuthe
# Om Sham Sharavana Bhava
# Om Sham Sharavana Bhava

#!pip install pdfplumber
#!pip install nltk
#!pip install scikit-learn
#!pip install streamlit

import streamlit as st
import pdfplumber
import nltk
from nltk.corpus import stopwords

# Make sure NLTK stopwords are downloaded once
nltk.download('stopwords')

st.title("Resume Filtering App")

# Step-1 - Extract Text from Resume (PDF)

import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Step-2 - Clean & Preprocess Text

import re
import nltk
#from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# STep-3 - Convert Text into Numerical Form (TF-IDF)

from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(resume_text, job_desc):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    return vectors

# Step-4 - Calculate Resume-Job Match Score

from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(vectors):
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return similarity[0][0] * 100

# Step-5 - Full Pipeline Example

resume_text = extract_text_from_pdf(r"C:\Users\saidi\Downloads\SaiDivya_Machine_Learning_Engineer_Resume.pdf")
job_desc = open(r"D:\Sri Krishna\Global_AIEngineer\job_description.txt").read()

resume_clean = clean_text(resume_text)
job_clean = clean_text(job_desc)

vectors = vectorize_text(resume_clean, job_clean)
score = calculate_similarity(vectors)

print(f"Resume Match Score: {score:.2f}%")

# Step-6 - Rank Multiple Resumes (Real-World Feature)

import os

def rank_resumes(resume_folder, job_desc):
    scores = {}
    job_clean = clean_text(job_desc)

    for file in os.listdir(resume_folder):
        text = extract_text_from_pdf(resume_folder + "/" + file)
        clean = clean_text(text)
        vectors = vectorize_text(clean, job_clean)
        score = calculate_similarity(vectors)
        scores[file] = score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Step-7 - Build Simple UI (Streamlit)

import streamlit as st

st.title("AI Resume Screening Tool")

job_desc = st.text_area("Paste Job Description")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

if uploaded_file and job_desc:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    resume_text = extract_text_from_pdf("temp.pdf")
    score = calculate_similarity(
        vectorize_text(
            clean_text(resume_text),
            clean_text(job_desc)
        )
    )

    st.success(f"Resume Match Score: {score:.2f}%")
