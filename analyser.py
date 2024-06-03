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
st.write("Try many models to see how they behave with your resume and job offer.")

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
if 'scores' not in st.session_state:
    st.session_state.scores = []
if 'user_prompt' not in st.session_state:
    st.session_state.user_prompt = '.'



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
        1. Make list of:
            -matching soft skills
            -matching hard skills 
            -relevant experiences for the position
            -matching education and certifications
            -missing keywords
        2. Score in a range 0 to 100 each category as follows:
            - "Soft skills": 15 * (each matching soft skills)
            - "Hard skills": 10 * (each matching hard skills)
            - "Experience": 20 * (each relevant experience for the position)
            - "Education and certifications": 30 * (each matching education or certification)
            - "Keywords": 100 - 4 * (each missing keyword)
        3. Calculate the match score as follows:
            - Total score over 100 = ("Soft skills" + "Hard skills" + "Experience" + "Education and certifications" + "Keywords")/ 5
        4. Generate an analysis summary that includes the following information:
                - Whether the candidate's profile aligns with the role description in the job offer with mention to the total score.
                - Highlight the strengths and weaknesses of the applicant in relation to the specified job offer description.
        5. Provide an output  titled "scores" in JSON format with the scores previously generated following the template below:
                {{
                    "Soft skills": <soft_skills_score>,
                    "Hard skills": <hard_skills_score>,
                    "Experience": <experience_score>,
                    "Keywords": <keywords_score>,
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
        if "{" not in match_answer or "}" not in match_answer:
            st.warning("Please try again. As small language models sometimes have difficulties following precise parsing instructions. If in 5 attempts the model doesn't rise an answer maybe you should consider highly probable that the model is not able to provide the answer.")
            raise ValueError("Response from this model does not contain expected JSON format.")
        # Extract JSON part from the match answer and convert it to a dictionary
        json_start = match_answer.index("{")
        json_end = match_answer.rindex("}") + 1
        json_part = match_answer[json_start:json_end]
        text_analysis = match_answer[:json_start].strip()
        try:
            scores_dict = json.loads(json_part)
        except json.JSONDecodeError as e:
            st.warning("Please try again. As small language models sometimes have difficulties following precise parsing instructions. If in 5 attempts the model doesn't rise an answer maybe you should consider highly probable that the model is not able to provide the answer.")
            raise ValueError("Response from this model does not contain expected JSON format.")
        return text_analysis, scores_dict
    
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
        plt.yticks([20, 40, 60, 80, 100], ["20", "40","60", "80", "100"], color="grey", size=7)
        plt.ylim(0, 100)
        ax.plot(angles, scores, linewidth=2, linestyle='solid')
        ax.fill(angles, scores, 'r', alpha=0.1)
        return fig
  
    text_analysis, scores_dict = extract_text_analysis(match_answer)
    fig = create_radar_chart(scores_dict)
    match_report = text_analysis, fig, scores_dict
    
    return match_report


def suggested_changes_function(resume_text, job_offer, job_title):
    feature_suggested_changes_prompt = PromptTemplate(
        input_variables=["resume_text", "job_offer", "job_title"],
        template="""You are an AI assistant designed to enhance and optimize resumes to better match specific job offers. 
        Your task is to review the provided resume in light of the given job offer description and job title, and provide detailed suggestions for improvement.
    
        Follow these steps in order:
        1. Identify and list matching soft skills, hard skills, qualifications, and experiences between the resume and the job offer.
        2. Extract and list keywords from both the job offer and the resume.
        3. Identify missing keywords in the resume that are present in the job offer.
        4. Highlight keywords and skills implied by the resume that could be explicitly added.
        5. Identify missing experiences in the resume that are implied and could be explicitly added.
        6. Based on the previous lists, provide specific bullet-point suggestions for rephrasing, adding, or deleting keywords or experiences to enhance the resume's alignment with the job offer.

        Summarize and output only points 4, 5, and 6.

        resume_text: {resume_text}
        job_offer: {job_offer}
        job_title: {job_title}
        """
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
    skill_list_chain = LLMChain(llm=model, prompt=skill_list_prompt, verbose=False)
    skill_list = skill_list_chain.run(resume_text=st.session_state.resume_text)
    
    # Parse JSON string to dictionary
    skill_dict = json.loads(skill_list_json)
    
    return skill_dict

def requirements_list_function (job_offer):
    requirements_list_prompt = PromptTemplate(
        input_variables=["job_offer"],
        template="""Extract the information referring to skills, experience, studies or other relevant keywords from the provided job offer and format it as a JSON object with the following structure:
        {{
        "requirements": ["keyword1", "keyword2", "keyword3", "..."],
        }}
        job_offer:
        {job_offer}
        """
    )
    requirements_list_chain = LLMChain(llm=model, prompt=requirements_list_prompt, verbose=False)
    requirements_list = requirements_list_chain.run(job_offer=st.session_state.job_offer_text)
    
    # Parse JSON string to dictionary
    requirements_dict = json.loads(requirements_list_json)
    
    return requirements_dict

def custom_prompt_function(user_prompt, resume_text, job_offer, job_title):
    custom_prompt = PromptTemplate(
        input_variables=[ "user_prompt","resume_text", "job_offer", "job_title"],
        template="""You are an AI assistant designed to enhance and optimize resumes to better match specific job offers.
        Given the user prompt and with the resume, job offer, and job title as context provide a short answer that addresses the user's query.
        user_prompt: {user_prompt}
        resume_text: {resume_text}
        job_offer: {job_offer}
        job_title: {job_title}
        """
    )
    custom_prompt_chain = LLMChain(llm=model, prompt=custom_prompt, verbose=False)
    custom_QA = custom_prompt_chain.run(user_prompt=st.session_state.user_prompt,
                                        resume_text=st.session_state.resume_text,
                                        job_offer=st.session_state.job_offer_text,
                                        job_title=st.session_state.job_title,
                                            
                                        )
    return custom_QA


# Function to create a radar chart
def create_radar_chart(data):
    # Define categories for radar chart
    categories = ["Soft skills", "Hard skills", "Experience", "Education and certifications", "Keywords"]
    num_vars = len(categories)
    
    # Define angles for radar chart
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Colors for different models
    colors = plt.get_cmap("tab10", len(data))
    
    for idx, entry in enumerate(data):
        scores_dict = entry["score"]
        model = entry["model"]

        # Extract the relevant scores for radar chart
        scores = [
            scores_dict.get("soft_skills_score") or scores_dict.get("Soft skills", 0),
            scores_dict.get("hard_skills_score") or scores_dict.get("Hard skills", 0),
            scores_dict.get("experience_score") or scores_dict.get("Experience", 0),
            scores_dict.get("Education and certifications", 0),  # Some entries may not have this score
            scores_dict.get("keywords_score") or scores_dict.get("Keywords", 0)
        ]
        scores += scores[:1]

        ax.plot(angles, scores, linewidth=2, linestyle='solid', label=model, color=colors(idx))
        ax.fill(angles, scores, color=colors(idx), alpha=0.25)
    
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=7)
    plt.ylim(0, 100)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Model Scores Radar Chart')
    plt.show()



tab1, tab2, tab3, tab4 = st.tabs(["Resume VS Job offer", "Get a job", "Career advice", "custom prompt"])

with tab1:
    col1, col2, col3 = st.columns(3)
    feature_match_button = col1.button("RESUME MATCH")
    Scores_button = col2.button("SESSION SCORES")
    skills_list_button= col3.button("SEMANTIC HEATMAP")
with tab2:
    col1, col2 = st.columns(2)
    feature_suggested_changes_button = col1.button("FINETUNE YOUR RESUME")
    feature_5 = col2.button("TITLE NAMES FOR JOB SEARCH")
with tab3:
    col1, col2, col3 = st.columns(3)
    feature_6 = col1.button("SHORT TERM")
    custom_prompt_button = col2.button("MID TERM")
    feature_7 = col3.button("LONG TERM") 

with tab4:
    user_prompt = st.text_input("User Prompt",placeholder="Enter your prompt here", type="default", key="user_prompt")
    submit_user_prompt_button = st.button("Submit")
    
with st.container(border=True):
    if feature_match_button:
        if st.session_state.job_title and st.session_state.job_offer_text and st.session_state.resume_text:
            match_answer = feature_match_function(resume_text=st.session_state.resume_text, 
                                                   job_offer=st.session_state.job_offer_text, 
                                                   job_title=st.session_state.job_title
                                                   )
            # Get the match report
            analysis_text, radar_chart, scores_dict = match_report(match_answer)
            
            st.session_state.scores.append({'index': len(st.session_state.scores), 'score': scores_dict,
                                            'model': str(model_name),  # Convert the model to a string for storage
                                            'temperature': model_temperature
                                            }) 
              
            # Display the results in a container
            with st.container():
                st.write("### Resume match analysis")
                st.write(analysis_text)
                st.write("### Radar Chart")
                st.pyplot(radar_chart)
                # st.write("### Scores")
                # st.json(scores_dict)
        else:
            st.warning("Please upload a resume and provide a job offer text and job title to proceed.")
    
    elif skills_list_button:
        if st.session_state.resume_text:
            skill_list = skill_list_function(resume_text=st.session_state.resume_text)
            skill_list_text = skill_list.strip()
            
            requirements_list = requirements_list_function(job_offer=st.session_state.job_offer_text)
            requirements_list_text = requirements_list.strip()
            
            # Print the extracted data
            st.markdown("### JSON Output List?")
            st.write(skill_list_text)
            st.write(requirements_list_text)
                     
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
            
    elif submit_user_prompt_button:
        st.session_state.user_prompt = user_prompt
        if st.session_state.job_title and st.session_state.job_offer_text and st.session_state.resume_text:
            answer = custom_prompt_function(user_prompt=st.session_state.user_prompt,
                                        resume_text=st.session_state.resume_text, 
                                        job_offer=st.session_state.job_offer_text, 
                                        job_title=st.session_state.job_title
                                        )
            answer_text = answer.strip()
            with st.container():
                st.markdown("### Custom Prompt Answer")
                st.write(answer_text)       
        else:
            st.warning("Please upload a resume and provide a job offer text and job title to proceed.") 
                   
    elif Scores_button:
        # Call the function to create the radar chart
        st.write("### Model Scores Radar Chart")
        model_names = [entry['model'] for entry in st.session_state.scores]
        num_queries = len(st.session_state.scores)
        st.write(f"In your session, you have conducted {num_queries} queries to these models: {model_names}")
        create_radar_chart(st.session_state.scores)
        st.pyplot(plt)
        
    elif feature_5:
        st.write("Feature 5 is not yet implemented")
    elif feature_6:
        st.write("Feature 6 is not yet implemented")
        
        
        

