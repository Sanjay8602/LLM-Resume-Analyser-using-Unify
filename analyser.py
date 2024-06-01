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


# Configure the Streamlit page
st.set_page_config("LLM Resume Analyser", page_icon="🚀")  
st.title("LLM Resume Analyser 🚀")
st.write("Improve your resume with the help of AI!")

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



def feature_match_function(resume_text, job_offer, job_title):
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
            - Keywords: 20 - 1 * (each missing keyword)
        3. Calculate a total score at the end in this way:
            - Total score = (soft_skills_score + hard_skills_score + experience_score + keywords_score)*100/80  
        4. Generate an analysis summary that includes the following information:
                - Whether the candidate's profile aligns with the role description in the job offer with mention to the total score.
                - Highlight the strengths and weaknesses of the applicant in relation to the specified job offer description.
        5. Provide an output in JSON format (without any title as "JSON Output" or "scores") with the scores previously generated following the template below:
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
    feature_match_chain = LLMChain(llm=model, prompt=feature_match_prompt, verbose=False)
    match_answer = feature_match_chain.run(resume_text=st.session_state.resume_text, 
                                                   job_offer=st.session_state.job_offer_text, 
                                                   job_title=st.session_state.job_title)
    print(match_answer) 
    return match_answer


def match_report(match_answer):
    def extract_text_analysis(match_answer):
        # Debugging: Print the raw response
        print("Raw match_answer:", match_answer)
        # Ensure the response contains the expected separator
        if "{" not in match_answer or "}" not in match_answer:
            raise ValueError("Response from this model does not contain expected JSON format.")
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


def suggested_changes_function(resume_text, job_offer, job_title):
    feature_suggested_changes_prompt = PromptTemplate(
        input_variables=["resume_text", "job_offer", "job_title"],
        template="""You are an AI assistant powered by a Language Model, designed to provide guidance for enhancing and optimizing resumes. 
            Your task is to review the provided resume against the given job offer description and job title. 
            Follow the steps below to complete the task:
                1. Make a list of matching soft skills, matching hard skills, matching qualifications, matching experiences.
                2. Make a list of keywords in the job offer and in the resume.
                4. Make a list of the missing keywords in the resume.
                5. make a list of the missing keywords that are missing but that current resume implies and could be added
                6. make a list of the missing experiences that are missing but that current resume implies and could be added.
                7. With the previous lists as context, build a bullet point answer proposing rephrasing and adding or deleting some keywords or experiences to improve the resume match with the job offer.
            resume_text : {resume_text}
        job_offer: {job_offer}
        job_title: {job_title}"""
    )
    feature_suggested_changes_chain = LLMChain(llm=model, prompt=feature_suggested_changes_prompt, verbose=False)
    suggested_changes = feature_suggested_changes_chain.run(resume_text=st.session_state.resume_text,
                                                        job_offer=st.session_state.job_offer_text,
                                                        job_title=st.session_state.job_title)
    return suggested_changes


def skill_list_function (resume_text):
    skill_list_prompt = PromptTemplate(
        input_variables=["resume_text"],
        template="""Extract the following information from the provided resume_text and format it as a JSON object with the following structure:
        {{
        "soft_skills": ["soft_skill1", "soft_skill2", "soft_skill3", "..."],
        "hard_skills": ["hard_skill1", "hard_skill2", "hard_skill3", "..."],
        "keywords": ["keyword1", "keyword2", "keyword3", "..."],
        "experience": ["experience1", "experience2", "experience3", "..."],
        "education_and_certifications": ["education1", "certification1", "certification2", "..."],
        "other_knowledge": ["other_knowledge1", "other_knowledge2", "other_knowledge3", "..."]
        }}
        resume_text:
        {resume_text}
        """
    )
    skill_list_chain = LLMChain(llm=model, prompt=skill_list_prompt, verbose=True)
    skill_list = skill_list_chain.run(resume_text=st.session_state.resume_text)
    
    return skill_list



tab1, tab2, tab3 = st.tabs(["Resume VS Job offer", "Improve my resume", "Career advice"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    feature_match_button = col1.button("REPORT MATCH")
    skills_list_button= col2.button("SKILLS LISTS")
    scores_button = col3.button("SCORES")
    feature_suggested_changes_button = col4.button("SUGGESTED_CHANGES")
with tab2:
    col1, col2 = st.columns(2)
    feature_3 = col1.button("FEATURE 3")
    feature_4 = col2.button("FEATURE 4")
with tab3:
    col1, col2 = st.columns(2)
    feature_5 = col1.button("FEATURE 5")
    feature_6 = col2.button("FEATURE 6")
    
with st.container(border=True):
    if feature_match_button:
        if st.session_state.job_title and st.session_state.job_offer_text and st.session_state.resume_text:
            match_answer = feature_match_function(resume_text=st.session_state.resume_text, 
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
        else:
            st.warning("Please upload a resume and provide a job offer text and job title to proceed.")
    
    elif skills_list_button:
        if st.session_state.resume_text:
            skill_list = skill_list_function(resume_text=st.session_state.resume_text)
            skills_list_text = skill_list.choices[0].text.strip()
            # Print the extracted data
            st.markdown("### JSON Output List?")
            st.write(skill_list)
        else:
            st.warning("Please upload a resume and provide a job offer text and job title to proceed.")
                           
                       
    elif feature_suggested_changes_button:
        if st.session_state.job_title and st.session_state.job_offer_text and st.session_state.resume_text:
            json_answer = suggested_changes_function(resume_text=st.session_state.resume_text, 
                                                   job_offer=st.session_state.job_offer_text, 
                                                   job_title=st.session_state.job_title
                                                   )
            st.markdown("### Suggested Changes")
            st.write(json_answer)
        else:
            st.warning("Please upload a resume and provide a job offer text and job title to proceed.")
            
            
    elif feature_3:
        st.write("Feature 3 is not yet implemented")
    elif feature_4:
        st.write("Feature 4 is not yet implemented")
    elif feature_5:
        st.write("Feature 5 is not yet implemented")
    elif feature_6:
        st.write("Feature 6 is not yet implemented")
        
        
        

