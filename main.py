import tkinter as tk 
from tkinter import filedialog
import joblib
import fitz
import re
import spacy 
import pdfplumber
from PyPDF2 import PdfReader

nlp=spacy.load("en_core_web_sm")

rf_classifier_job_recommendation = joblib.load("rf_classifier_job_recommendation.joblib") 
tfidf_vectorizer_job_recommendation=joblib.load("tfidf_vectorizer_job_recommendation.joblib") 
rf_classifier_categorization=joblib.load("rf_classifier_categorization.joblib") 
tfidf_vectorizer_categorization=joblib.load("tfidf_vectorizer_categorization.joblib")
role_encoder = joblib.load("role_encoder.joblib")

def cleanResume(text):
    text=re.sub(r"http\S+"," ",text)
    text=re.sub(r"RT|cc"," ",text)
    text=re.sub(r"#\S+"," ",text)
    text=re.sub(r"@\S+"," ",text)
    text=re.sub(r"[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_{|}~""")," ",text)
    text=re.sub(r"[^\x00-\x7f]"," ",text)
    text=re.sub(r"\s+"," ",text)
    return text.strip()

def extract_text_from_pdf(pdf_path):
    text=""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text=page.extract_text()
            if page_text:
                text+=page_text+"\n"
    return text

def extract_email_from_resume(text):
    email=re.findall(r'[\w\.-]+@[\w\.-]+\.\w+',text)
    if email:
        return email[0]
    return "Not Found"

def extract_contact_number_from_resume(text):
    phone=re.findall(r'(?:\+91[- ]?)?[6-9]\d{9}', text)
    if phone:
        return phone[0]
    return "Not Found"

def extract_name_from_resume(text):
    lines=text.split("\n")
    for line in lines[:8]:
        line=line.strip()
        if(
            len(line.split())>=2
            and "@" not in line
            and "resume" not in line.lower()
            and "objective" not in line.lower()
            and "phone" not in line.lower()
            and "email" not in line.lower()
        ):
            return line
    return "Not Found"

def extract_education_from_resume(text):
    education_keywords=["B.Tech","BTech","Bachelor","M.Tech","MTech","Master","BCA","MCA","B.Sc","M.Sc","MBA","Diploma","PhD"]
    education=[]
    for keyword in education_keywords:
        if keyword.lower() in text.lower():
            education.append(keyword)
    if education:
        return ",".join(set(education))
    return "Not Found"

def extract_skills_from_resume(text):
    skills_list=["Python","Java","C","C++","HTML","CSS","JavaScript","SQL","MySQL","MongoDB","React","Node.js","Machine Learning","Deep Learning","Artificial Intelligence","TensorFlow","Keras","Scikit-learn","Pandas","NumPy","Matplotlib","Power BI","Excel","Data Analysis","Data Science","Figma","Photoshop"]
    found_skills=[]
    for skill in skills_list:
        if skill.lower() in text.lower():
            found_skills.append(skill)
    return found_skills

def predict_category(resume_text):
    resume_text=cleanResume(resume_text)
    print("\n"+"="*60)
    print("TEXT GIVEN TO CATEGORY MODEL")
    print("="*60)
    print(resume_text[:2000])
    resume_tfidf=tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category=rf_classifier_categorization.predict(resume_tfidf)[0]
    print("Predicted Category:",predicted_category)
    return predicted_category

def job_recommendation(skills_text):
    skills_text = cleanResume(skills_text)

    skills_tfidf = tfidf_vectorizer_job_recommendation.transform([skills_text])

    prediction = rf_classifier_job_recommendation.predict(skills_tfidf)

    predicted_job = role_encoder.inverse_transform(prediction)

    return predicted_job[0]

def select_resume():
    file_path=filedialog.askopenfilename(title="Select Resume", filetypes=[("PDF Files","*.pdf")])
    if file_path:
        check_resume(file_path)   


def check_resume(file_path): 
    resume_text=extract_text_from_pdf(file_path) 
    print("="*50) 
    print("Resume Text:") 
    print(resume_text[:3000]) 
    print("="*50)
    category=predict_category(resume_text) 
    skills=extract_skills_from_resume(resume_text) 
    job=job_recommendation(" ".join(skills))
    print("Skills:", skills) 
    print("Skills sent to model:", " ".join(skills)) 
    print("Category:", category) 
    print("Job:", job) 
    name=extract_name_from_resume(resume_text) 
    email=extract_email_from_resume(resume_text) 
    phone=extract_contact_number_from_resume(resume_text) 
    education=extract_education_from_resume(resume_text) 
    result_text = ( 
    f"Name: {name}\n"
    f"\nContact Number: {phone}\n"
    f"\nEmail: {email}\n" 
    f"\nEducation: {education}\n" 
    f"\nSkills: {', '.join(skills)}\n"
    f"\nCategory: {category}\n" 
    f"\nJob Recommendation: {job}\n"
    )
    output_text.delete("1.0",tk.END) 
    output_text.insert(tk.END,result_text)
    
root=tk.Tk() 
root.title("AI Resume Analyzer") 
root.geometry("900x700") 
root.configure(bg="#0b3d91") 
title=tk.Label(root,text="AI Resume Analyzer",font=("Arial", 20, "bold"),bg="#0b3d91",fg="white") 
title.pack(pady=(20,30)) 
upload_btn=tk.Button(root,text="Upload Resume (PDF)",font=("Arial",14),bg="#1E90FF",fg="white",activebackground="#4682B4",activeforeground="white",command=select_resume) 
upload_btn.pack(pady=(0,20)) 
output_text=tk.Text(root,width=85,height=25,font=("Arial",13),bg="#0F4C81",fg="white",insertbackground="white",bd=0,padx=20,pady=20) 
output_text.pack(pady=10)
root.mainloop()