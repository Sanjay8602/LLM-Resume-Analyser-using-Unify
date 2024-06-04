import numpy as np
import seaborn as sns
import pandas as pd
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
from sentence_transformers import SentenceTransformer, util


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
if "num_job_offers_input" not in st.session_state:
    st.session_state.num_job_offers_input = 5


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
    
    with st.expander("Resume and job offer inputs"):
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
        template = """You are an AI assistant powered by a Language Model, designed to provide guidance for enhancing and optimizing resumes. 
        Your task is to review the provided resume against the given job offer description and job title. 
        Follow the steps below in order to complete the task:

        step 1. Make a list of:
            - Matching soft skills
            - Matching hard skills
            - Relevant experiences for the position
            - Matching education and certifications
            - Missing keywords

        step 2. Score each category as follows:
            - "Soft skills": 15 points for each matching soft skill (minimum 0 points, maximum 100 points).
            - "Hard skills": 10 points for each matching hard skill(minimum 0 points, maximum 100 points).
            - "Experience": 20 points for each relevant experience for the position(minimum 0 points, maximum 100 points).
            - "Education and certifications": 30 points for each matching education or certification(minimum 0 points, maximum 100 points).
            - "Keywords": 100 minus 4 points for each missing keyword (minimum 0 points, maximum 100 points).


        step 3. Provide the output in two parts:
        1. **Analysis Summary**: An analysis summary of how the candidate's profile aligns with the role description in the job offer, including a reference to the scores, highlighting the strengths and weaknesses of the applicant in relation to the specified job offer description..
        2. **Scores**: A JSON format with the scores for each category using the template below:
            {{
            "Soft skills": <soft_skills_score>,
            "Hard skills": <hard_skills_score>,
            "Experience": <experience_score>,
            "Education and certifications": <education_and_certifications_score>,
            "Keywords": <keywords_score>
            }}
            
        Resume Text: {resume_text}
        Job Offer: {job_offer}
        Job Title: {job_title}
        """
        )
    with st.sidebar.container(border=True):
        st.text(f"Running prompt: {feature_match_prompt.template}")
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
        # Extract JSON part from the match answer and convert it to a dictionary
        json_start = match_answer.index("{")
        json_end = match_answer.rindex("}") + 1
        json_part = match_answer[json_start:json_end]
        text_analysis = match_answer[:json_start].strip()
        try:
            scores_dict = json.loads(json_part)
        except json.JSONDecodeError as e:
            st.warning("Please try again. As small language models sometimes have difficulties following precise parsing instructions. If in 5 attempts the model doesn't rise an answer maybe you should consider highly probable that the model is not able to provide the answer.")
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

        Summarize and output only points 4, 5, and 6(rename them as A, B and C).

        resume_text: {resume_text}
        job_offer: {job_offer}
        job_title: {job_title}
        """
    )
    with st.sidebar.container(border=True):
        st.text(f"Running prompt: {feature_suggested_changes_prompt.template}")
    feature_suggested_changes_chain = LLMChain(llm=model, prompt=feature_suggested_changes_prompt, verbose=False)
    suggested_changes = feature_suggested_changes_chain.run(resume_text=st.session_state.resume_text,
                                                        job_offer=st.session_state.job_offer_text,
                                                        job_title=st.session_state.job_title)
    return suggested_changes


def skills_heatmap_function(resume_text, job_offer):
    # Load pre-trained Sentence-BERT model
    model_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def skill_list_function (resume_text):
        skill_list_prompt = PromptTemplate(
            input_variables=["resume_text"],
            template="""
            First extract the following information from the provided resume_text:
                1. Soft skills list
                2. Hard skills list
                3. General keywords in the resume list
                4. keywords in professional experiences list
                5. keywords in education and certifications list
                6. other relevant knowledge keywords list
            
            With the information extracted from the resume_text provide the output in JSON format using the template below:
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
        with st.sidebar.container(border=True):
            st.text(f"Running prompt: {skill_list_prompt.template}")
        skill_list_chain = LLMChain(llm=model, prompt=skill_list_prompt, verbose=False)
        skill_list = skill_list_chain.run(resume_text=st.session_state.resume_text)
        # Parse JSON string to dictionary
        json_start1 = skill_list.index("{")
        json_end1 = skill_list.rindex("}") + 1
        json_part1 = skill_list[json_start1:json_end1]
        skill_dict = json.loads(json_part1)
        # Check if the output is a dictionary
        if not isinstance(skill_dict, dict):
            st.warning("No skills found. Try again or try another model.")
        return skill_dict

    def requirements_list_function (job_offer):
        requirements_list_prompt = PromptTemplate(
            input_variables=["job_offer"],
            template="""First extract the following information from the provided job_offer and make a list:
                1. "requirements": All the keywords that refers to requirements, skills, experiences or other qualities needed for the job offer.
            
            Provide the output in JSON format using the template below:
            {{
            "requirements": ["keyword1", "keyword2", "keyword3", "..."],
            }}
            job_offer:
            {job_offer}
            """
        )
        with st.sidebar.container(border=True):
            st.text(f"Running prompt: {requirements_list_prompt.template}")
        requirements_list_chain = LLMChain(llm=model, prompt=requirements_list_prompt, verbose=False)
        requirements_list = requirements_list_chain.run(job_offer=st.session_state.job_offer_text)
        # Parse JSON string to dictionary
        json_start2 = requirements_list.index("{")
        json_end2 = requirements_list.rindex("}") + 1
        json_part2 = requirements_list[json_start2:json_end2]
        requirements_dict = json.loads(json_part2)
        # Check if the output is a dictionary
        if not isinstance(requirements_dict, dict):
            st.warning("Output is not a dictionary. Try again or try another model.")
        return requirements_dict
    
    skill_dict = skill_list_function(resume_text=st.session_state.resume_text)
    requirements_dict = requirements_list_function(job_offer=st.session_state.job_offer_text)  
    categories = list(skill_dict.keys())
    requirements = requirements_dict["requirements"]
    
    # Combine all categories into one list
    all_skills = []
    for category in skill_dict:
        all_skills.extend(skill_dict[category])

    st.write(f"Total skills collected: {len(all_skills)}")
    
    # Define similarity function using Sentence-BERT
    def evaluate_similarity(sentence1, sentence2):
        embeddings1 = model_encoder.encode(sentence1, convert_to_tensor=True)
        embeddings2 = model_encoder.encode(sentence2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        return similarity.item()
    # Create similarity matrix
    def create_similarity_matrix(skill_list, requirement_list):
        try:
            matrix = np.zeros((len(skill_list), len(requirement_list)))
            for i, skill in enumerate(skill_list):
                for j, req in enumerate(requirement_list):
                    matrix[i, j] = evaluate_similarity(skill, req)
            return matrix
        except Exception as e:
            st.warning("It didn't work this time, try it again! Take in consideration that small models sometimes struggle when it comes to give a formatted answer.")      
    
    similarity_matrix = create_similarity_matrix(all_skills, requirements)
    
    # Plot heatmap
    def plot_heatmap(matrix, skill_list, requirement_list):
        df = pd.DataFrame(matrix, index=skill_list, columns=requirement_list)
        plt.figure(figsize=(20, 15))
        sns.heatmap(df, annot=False, cmap='viridis', cbar=True, linewidths=.2)
        plt.title('Similarity Heatmap for All Skills Against Requirements')
        plt.xlabel('Requirements')
        plt.ylabel('Skills')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.show()
    plot_heatmap(similarity_matrix, all_skills, requirements)


def job_titles_list_function (resume_text, num_job_offers):
    job_titles_prompt = PromptTemplate(
        input_variables=["resume_text", "num_job_offers"],
        template=""" You are an AI assistant designed to enhance and optimize resumes to better match specific job offers.
        Given a resume ({resume_text}) and an integer ({num_job_offers}):
            1.Identify and return a list of the {num_job_offers} most relevant job titles that best match the skills and experience described in the resume. 
            2.Focus on titles that directly align with the candidate's qualifications and consider factors like keywords, technologies mentioned, and past job roles.
        
        Return just the list of job titles as a bullet-point list.  
        
        resume_text: {resume_text}
        num_job_offers: {num_job_offers}
        """
    )
    with st.sidebar.container(border=True):
        st.text(f"Running prompt: {job_titles_prompt.template}")
    job_titles_chain = LLMChain(llm=model, prompt=job_titles_prompt, verbose=False)
    job_titles = job_titles_chain.run(resume_text=st.session_state.resume_text,
                                        num_job_offers=st.session_state.num_job_offers_input                                           
                                        )
    return job_titles  

def custom_prompt_function(user_prompt, resume_text, job_offer, job_title):
    custom_user_prompt = PromptTemplate(
        input_variables=[ "user_prompt","resume_text", "job_offer", "job_title"],
        template="""You are an AI assistant designed to enhance and optimize resumes to better match specific job offers.
        Given the user prompt as a query to answer and use the resume, job offer, and job title as context to provide a short answer that addresses the user's query.
        user_prompt: {user_prompt}
        resume_text: {resume_text}
        job_offer: {job_offer}
        job_title: {job_title}
        """
    )
    custom_prompt_chain = LLMChain(llm=model, prompt=custom_user_prompt, verbose=False)
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


# UI main structure
tab1, tab2, tab3, tab4 = st.tabs(["Resume VS Job offer", "Get a job", "Career advice", "custom prompt"])

with tab1:
    col1, col2, col3 = st.columns(3)
    feature_match_button = col1.button("RESUME MATCH")
    Scores_button = col2.button("SESSION SCORES")
    semantic_heatmap_button= col3.button("SEMANTIC HEATMAP")
    container1 = st.container(border=True)
    
with tab2:
    col1, col2 = st.columns(2)
    feature_suggested_changes_button = col1.button("TUNE YOUR RESUME")
    feature_suggested_titles_button = col2.button("TITLE NAMES FOR JOB SEARCH")
    st.session_state.num_job_offers_input = col2.slider('Select number of job offers', 1, 20, 5)
    container2 = st.container(border=True)
    
with tab3:
    col1, col2, col3 = st.columns(3)
    feature_6 = col1.button("SHORT TERM")
    feature_7 = col2.button("MID TERM")
    feature_8 = col2.button("LONG TERM") 
    container3 = st.container(border=True)
    
with tab4:
    st.session_state.user_prompt = st.text_area("Try your prompt",placeholder="Enter your prompt here")
    submit_user_prompt_button = st.button("Submit")
    container4 = st.container(border=True)
    
with container1:
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
            
    elif Scores_button:
        # Call the function to create the radar chart
        st.write("### Model Scores Radar Chart")
        model_names = [entry['model'] for entry in st.session_state.scores]
        num_queries = len(st.session_state.scores)
        st.write(f"In your session, you have conducted {num_queries} queries to these models: {model_names}")
        create_radar_chart(st.session_state.scores)
        st.pyplot(plt)
        
    elif semantic_heatmap_button:
        if st.session_state.resume_text and st.session_state.job_offer_text:
            st.write("### Semantic Heatmap")
            st.write("The heatmap represents the semantic similarity matrix between the skills and experiences from the resume and the job offer requirements. (processing time: 1 to 2 minutes aprox.)")
            skills_heatmap_function(resume_text=st.session_state.resume_text, 
                                    job_offer=st.session_state.job_offer_text)
            st.pyplot(plt)
                     
        else:
            st.warning("Please upload a resume and provide a job offer text and job title to proceed.")
with container2:                                                  
    if feature_suggested_changes_button:
        if st.session_state.job_title and st.session_state.job_offer_text and st.session_state.resume_text:
            suggested_changes_answer = suggested_changes_function(resume_text=st.session_state.resume_text, 
                                                   job_offer=st.session_state.job_offer_text, 
                                                   job_title=st.session_state.job_title
                                                   )
            st.markdown("### Suggested Changes")
            st.write("The following suggestions will help you to find other job opportunities that match with your profile")
            suggested_changes_answer_text = suggested_changes_answer.strip() 
            st.write(suggested_changes_answer_text)
            feature_9 = st.button("APPLY CHANGES AND REPEAT ANALYSIS")
            if feature_9:
                st.warning("Feature 8 is not yet implemented")
        else:
            st.warning("Please upload a resume and provide a job offer text and job title to proceed.")
    elif feature_suggested_titles_button:
        if st.session_state.resume_text:
            # st.session_state.num_job_offers_input = st.slider('Select number of job offers', 1, 20, 5)
            # # Check if the slider value has changed
            # if "num_job_offers_input" not in st.session_state or st.session_state.num_job_offers_input != num_job_offers_input:
            #     # Store the current slider value as the previous one for the next run
            #     st.session_state.num_job_offers_input = num_job_offers_input
                
            suggested_job_titles_answer = job_titles_list_function(resume_text=st.session_state.resume_text, 
                                                                    num_job_offers=st.session_state.num_job_offers_input
                                                                    )
            st.markdown("### Suggested Job Titles")
            suggested_job_titles_text= suggested_job_titles_answer.strip()
            st.write(suggested_job_titles_text)
        
        
with container3: 
    if feature_6 or feature_7 or feature_8:
            st.warning("Career advice feature is not yet implemented") 
with container4:           
    if submit_user_prompt_button:
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
                   
    
        
        
        

