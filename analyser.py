import numpy as np
import matplotlib.pyplot as plt
from math import pi
import json
from PyPDF2 import PdfReader
from docx import Document
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from data.unify_endpoints_data import model_provider, dynamic_provider
from langchain_unify.chat_models import ChatUnify


# Function to extract text from a PDF or DOCX document
def extract_text_from_file(file):
    """Extracts text from a PDF or DOCX document."""
    def extract_text_from_doc(file):
        text = ""
        doc = Document(file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
        return text   

    def extract_text_from_pdf(file):
        text = ""
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()     
        return text
    
    file_extension = file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(file)
    elif file_extension == 'docx':
        return extract_text_from_doc(file)

# Define job titles and associated keywords
job_title_keywords = {
    "Data Scientist": ["machine learning", "data analysis", "python", "statistics"],
    "Software Engineer": ["software development", "programming", "coding", "software architecture"],
    # Add more job titles and associated keywords as needed
}

# Function to suggest job titles based on resume keywords
def suggest_job_titles(resume_text):
    matched_job_titles = []
    for job_title, keywords in job_title_keywords.items():
        if all(keyword in resume_text.lower() for keyword in keywords):
            matched_job_titles.append(job_title)
    return matched_job_titles

# Configure the Streamlit page
st.set_page_config("LLM Resume Analyser", page_icon="🚀")  
st.title("LLM Resume Analyser 🚀")
st.write("Improve your resume with the power of LLMs")

# Initialize session state variables
if 'unify_api_key' not in st.session_state:
    st.session_state.unify_api_key = ''
if 'endpoint' not in st.session_state:
    st.session_state.endpoint = ''
if 'resume_doc' not in st.session_state:
    st.session_state.resume_doc = None
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ''
if 'job_offer_text' not in st.session_state:
    st.session_state.job_offer_text = ''
if 'job_title' not in st.session_state:
    st.session_state.job_title = ''


with st.sidebar:
    st.write("Select your favorite LLM model to assist you in enhancing your career prospects.")
    st.session_state.unify_api_key = st.text_input("Unify API Key*", type="password",
                                                   placeholder="Enter Unify API Key")
    
    with st.expander("Model and Provider Selection"):
        model_name = st.selectbox("Select Model", options=model_provider.keys(), index=20,
                                  placeholder="Model", help="All your favorite models with Unify API")
    
        if st.toggle("Enable Dynamic Routing"):
            provider_name = st.radio("Select a Provider", options=dynamic_provider, 
                                     help="dynamic routing powered by Unify API")
        else:
            provider_name = st.radio("Select a Provider", options=model_provider[model_name], 
                                     help="Providers with access to the selected model")
        model_temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.3, step=0.1)   
    st.session_state.endpoint = f"{model_name}@{provider_name}"
    
    st.session_state.resume_doc = st.file_uploader(label="Upload your Resume*", type=("pdf","docx"), accept_multiple_files=False)

    if st.session_state.resume_doc is not None and st.button("Process resume"): 
        try:
            st.session_state.resume_text = extract_text_from_file(st.session_state.resume_doc)
            st.success("Resume has been successfully processed!")
        except Exception as e:
            st.error("Unable to recognize the document. Please try a compatible format.")

    st.session_state.job_offer_text = st.text_area(label="Job offer description*", key="Job_offer", placeholder="Paste here the job offer description")
    
    st.session_state.job_title = st.text_input("Job title*", key="Job_title", placeholder="enter here your desired job title")
    
    
    
model = ChatUnify(
        model=st.session_state.endpoint,
        unify_api_key=st.session_state.unify_api_key,
        temperature=model_temperature
        )


def feature_match_chain(resume_text, job_offer, job_title):
    feature_match_prompt = PromptTemplate(
        input_variables=["resume_text", "job_offer", "job_title"],
        template="""You are an AI assistant powered by a Language Model, designed to provide guidance for enhancing and optimizing resumes. 
        Your task is to review the provided resume against the given job offer description and job title. 
        Follow the steps below to complete the task:
        1. Make a bullet point list of matching skills and experiences and missing keywords.
        2. Score each section as follows:
            - Soft skills: 3 * (matching soft skill)
            - Hard skills: 2 * (matching hard skill)
            - Experience: 4 * (each relevant experience for the role)
            - Keywords: 20 - 2* (each missing keyword)
        3. Calculate a total score at the end in this way:
            - Total score = (soft_skills_score + hard_skills_score + experience_score + keywords_score)*100/80  
        4. Generate an analysis summary that includes the following information:
                - Whether the candidate's profile aligns with the role description in the job offer with mention to the total score.
                - Highlight the strengths and weaknesses of the applicant in relation to the specified job offer description.
        5. Provide an output in JSON format (without any title as JSON Output:) with the scores previously generated following the template below:
                {{
                    "soft_skills_score": <soft_skills_score>,
                    "hard_skills_score": <hard_skills_score>,
                    "experience_score": <experience_score>,
                    "keywords_score": <keywords_score>,
                }}
        resume_text: {resume_text}
        job_offer: {job_offer}
        job_title: {job_title}"""
    )
    feature_match_chain = LLMChain(llm=model, prompt=feature_match_prompt, verbose=True)
    match_answer = feature_match_chain.run(resume_text=st.session_state.resume_text, 
                                                   job_offer=st.session_state.job_offer_text, 
                                                   job_title=st.session_state.job_title)
    return match_answer


def match_report(match_answer):
    def extract_text_analysis(match_answer):
        # Debugging: Print the raw response
        print("Raw match_answer:", match_answer)
        # Ensure the response contains the expected separator
        if "{" not in match_answer or "}" not in match_answer:
            raise ValueError("Response does not contain expected JSON format.")
        
        # Extract JSON part from the match answer
        json_start = match_answer.index("{")
        json_end = match_answer.rindex("}") + 1
        json_part = match_answer[json_start:json_end]
        text_analysis = match_answer[:json_start].strip()

        # Convert the JSON part into a dictionary
        try:
            scores_dict = json.loads(json_part)
        except json.JSONDecodeError as e:
            print("JSON Decode Error in json_part:", e)
            raise e

        return text_analysis, scores_dict
    
    # Function to create the radar chart
    def create_radar_chart(scores_dict):
        labels = list(scores_dict.keys())
        num_vars = len(labels)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        # Define scores as the list of values in scores_dict
        scores = list(scores_dict.values())
        scores += scores[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], labels)
        ax.set_rlabel_position(0)
        plt.yticks([5, 10, 15, 20], ["5", "10", "15", "20"], color="grey", size=7)
        plt.ylim(0, 20)
        ax.plot(angles, scores, linewidth=2, linestyle='solid')
        ax.fill(angles, scores, 'b', alpha=0.1)
        return fig
    
    # Extract the text analysis and scores dictionary
    text_analysis, scores_dict = extract_text_analysis(match_answer)

    # Create the radar chart
    fig = create_radar_chart(scores_dict)

    match_report = text_analysis, fig, scores_dict
    
    return match_report

feature_suggested_changes_prompt = PromptTemplate(
    input_variables=["resume_text", "job_offer", "job_title"],
    template="""You are an AI assistant powered by a Language Model, designed to provide guidance for enhancing and optimizing resumes. 
    Your task is to review the provided resume against the given job offer description. You must answer in a professional tone.
    1. Make a list of matching skills and experiences and missing keywords.
    2. Make a list of the missing keywords and experiences hidden in the resume but that current resume implies.
    3. With the previous lists as context, build an answer in bullet point format proposing rephrasing and adding or deleting some keywords or experiences to improve the resume match with the job offer.
    {resume_text} {job_offer} {job_title}"""
)
feature_suggested_changes_chain = LLMChain(llm=model, prompt=feature_suggested_changes_prompt, verbose=False)


st.markdown("### Features")

tab1, tab2, tab3 = st.tabs(["Match a job offer", "General Improvements", "Career advice"])

with tab1:
    feature_match_button = st.button("RESUME MATCH WITH THE JOB OFFER")
    feature_suggested_changes_button = st.button("HOW TO IMPROVE MY RESUME?")

with tab2:
    feature_3 = st.button("FEATURE 3")
    feature_4 = st.button("FEATURE 4")
    
with tab3:
    feature_5 = st.button("FEATURE 5")
    feature_6 = st.button("FEATURE 6")
    


with st.container(border=True):
    if feature_match_button:
        if st.session_state.job_title and st.session_state.job_offer_text and st.session_state.resume_text:
            match_answer = feature_match_chain(resume_text=st.session_state.resume_text, 
                                                   job_offer=st.session_state.job_offer_text, 
                                                   job_title=st.session_state.job_title
                                                   )
            # Get the match report
            analysis_text, radar_chart, scores_dict = match_report(match_answer)
    
            # Display the results in a container
            with st.container():
                st.write("### Resume match analysis")
                st.write(analysis_text)
                st.write("### Radar Chart")
                st.pyplot(radar_chart)
                st.write("### Scores")
                st.json(scores_dict)
                
        #    except ValueError as e:
        #        st.error("Something went wrong. Please try again.")  
        #else:
        #    st.warning("Please upload a resume and provide a job offer text and job title to proceed.")
            
    elif feature_suggested_changes_button:
        if st.session_state.job_title and st.session_state.job_offer_text and st.session_state.resume_text:
            try:
                suggested_changes_answer = feature_suggested_changes_chain.run(resume_text=st.session_state.resume_text, 
                                                                              job_offer=st.session_state.job_offer_text, 
                                                                              job_title=st.session_state.job_title)
                st.markdown("### Suggestions to Improve the Resume")
                st.write(suggested_changes_answer)
            except ValueError as e:
                st.error("Something went wrong. Please try again.")
        else:
            st.warning("Please upload a resume and provide a job offer text and job title to proceed.")
    elif feature_3:
        st.write("Feature 3 is not yet implemented")
    elif feature_4:
        st.write("Feature 4 is not yet implemented")
