
from PyPDF2 import PdfReader
from docx import Document
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from data.unify_endpoints_data import model_provider, dynamic_provider
from langchain_unify.chat_models import ChatUnify
from langchain.chains.combine_documents import create_stuff_documents_chain

# inputs processing
def extract_text_from_file(file):
    """Extracts text from a PDF or DOCX document."""
    
    def extract_text_from_doc(file):
        """ Extracts text from a DOCX document."""
        text = ""
        doc = docx.Document(file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
        return text   

    def extract_text_from_pdf(file):
        """Extracts text from a PDF document."""
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


st.set_page_config("LLM Resume Analyser", page_icon="🚀")  
st.title("LLM Resume Analyser 🚀")
"""Improve your resume with the power of LLMs"""

with st.sidebar:
    """
    Select your favourite LLM model 
    and boost your resume
    """
    # input for Unify API Key
    st.session_state.unify_api_key = st.text_input("Unify API Key*", type="password",
                                                         placeholder="Enter Unify API Key", 
                                                         args=("Unify Key ",)
                                                         )
    # Model and provider selection
    with st.expander("Model and Provider Selection"):
        model_name = st.selectbox("Select Model", options=model_provider.keys(), index=20,
                                    placeholder="Model", 
                                    args=("Model",), help=("All your favourite models with Unify"))
    
        if st.toggle("Enable Dynamic Routing"):
            provider_name = st.radio("Select a Provider", options= dynamic_provider, 
                                     args=("Provider",), help=("dynamic routing powered by Unify API"))
        
        else:
            provider_name = st.radio("Select a Provider", options=model_provider[model_name], 
                                     args=("Provider",), help=("Proviers with access to the selected model"))
        model_temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
        
    # Endpoint as llm@provider      
    st.session_state.endpoint = f"{model_name}@{provider_name}"

    # Document uploader
    st.session_state.resume_doc = st.file_uploader(label="Upload your Resume", type=("pdf","docx"),
                                                 accept_multiple_files=False)
    if st.session_state.resume_doc is not None:
        if st.button("Process resume"): 
            try:
                st.session_state.resume_text = extract_text_from_file(st.session_state.resume_doc)
                st.success("Resume has been successfully processed!")
            except Exception as e:
                st.error("Unable to recognize the document. Please try a compatible format.")

    # Paste Text Job Offer
    st.session_state.job_offer_text = st.text_area(label = "Job_offer_text", key="Job_offer", placeholder="Job offer text")
    
    # Job Title Input
    st.session_state.job_title = st.text_input("Enter the Job Title", key="Job_title", placeholder="Job Title")

 
# Define the LLM model         
model = ChatUnify(
    model=st.session_state.endpoint,
    unify_api_key=st.session_state.unify_api_key,
    temperature= model_temperature
    ) 

# feature 1 prompt
feature_match_prompt=PromptTemplate(
        input_variables=["resume", "job_offer", "job_title"],
        template = """You are an AI assistant powered by a Language Model, designed to provide guidance for enhancing and optimizing resumes. 
        Your task is to review the provided resume against the given job offer description 
        and job title. 
        1. Make a bulletpoint list of matching skills and experiences and missing keywords.
        2. You must answer in a professional tone:
            a. on whether the candidate's profile aligns with the role. 
            b. Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
        {resume_text} {job_description} {job_title}"""
    )

#feature 1 chain
feature_match_chain = LLMChain(llm=model, prompt=feature_match_prompt, verbose=False)

# FEATURES BUTTONS
col1, col2 = st.columns(2)

with col1:
    feature_match_button = st.button("Matching Skills and Experiences")
    feature_3 = st.button("FEATURE 3")

with col2:
    feature_2 = st.button("FEATURE 2")
    feature_4 = st.button("FEATURE 4")


# action for feature 1 button
with st.container(border=True,height=600):
    if feature_match_button:
        if st.session_state.job_title and st.session_state.job_offer_text and st.session_state.resume_text:
            try:
                match_answer = feature_match_chain.run(resume = st.session_state.resume_text, 
                                                               job_offer = st.session_state.job_offer_text, 
                                                               job_title = st.session_state.job_title,
                                                               )
                st.markdown("### Matching Skills and Experiences")
                st.write(match_answer)
            except ValueError as e:
                st.error("something went wrong. Please try again.")  