
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
    st.write("Select your favorite LLM model and boost your resume with the power of AI.")
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
    
    
    
    
    st.session_state.resume_doc = st.file_uploader(label="Upload your Resume", type=("pdf","docx"), accept_multiple_files=False)

    if st.session_state.resume_doc is not None and st.button("Process resume"): 
        try:
            st.session_state.resume_text = extract_text_from_file(st.session_state.resume_doc)
            st.success("Resume has been successfully processed!")
        except Exception as e:
            st.error("Unable to recognize the document. Please try a compatible format.")

    st.session_state.job_offer_text = st.text_area(label="Job offer description", key="Job_offer", placeholder="Paste here the job offer description")
    
    st.session_state.job_title = st.text_input("Job title", key="Job_title", placeholder="enter here your desired job title")
    
    
model = ChatUnify(
        model=st.session_state.endpoint,
        unify_api_key=st.session_state.unify_api_key,
        temperature=model_temperature
        )

feature_match_prompt = PromptTemplate(
    input_variables=["resume", "job_offer", "job_title"],
    template="""You are an AI assistant powered by a Language Model, designed to provide guidance for enhancing and optimizing resumes. 
    Your task is to review the provided resume against the given job offer description and job title. 
    1. Make a bulletpoint list of matching skills and experiences and missing keywords.
    2. You must answer in a professional tone:
        a. on whether the candidate's profile aligns with the role. 
        b. Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
    {resume_text} {job_description} {job_title}"""
)

feature_suggested_changes_prompt = PromptTemplate(
    input_variables=["resume", "job_offer", "job_title"],
    template="""You are an AI assistant powered by a Language Model, designed to provide guidance for enhancing and optimizing resumes. 
    Your task is to review the provided resume against the given job offer description. You must answer in a professional tone.
    1. Make a list of matching skills and experiences and missing keywords.
    2. Make a list of the missing keywords and experiences hidden in the resume but that current resume implies.
    3. With the previous lists as context, build an answer in bullet point format proposing rephrasing and adding or deleting some keywords or experiences to improve the resume match with the job offer.
    {resume_text} {job_description} {job_title}"""
)

feature_match_chain = LLMChain(llm=model, prompt=feature_match_prompt, verbose=False)
feature_suggested_changes_chain = LLMChain(llm=model, prompt=feature_suggested_changes_prompt, verbose=False)

st.markdown("### Features")

tab1, tab2, tab3 = st.tabs(["Resume Analysis", "Advanced Analysis", "Print a Report"])

with tab1:
    feature_match_button = st.button("RESUME MATCH WITH THE JOB OFFER")
    feature_suggested_changes_button = st.button("HOW TO IMPROVE MY RESUME?")

with tab2:
    feature_3 = st.button("FEATURE 3")
    feature_4 = st.button("FEATURE 4")
    


with st.container():
    if feature_match_button:
        if st.session_state.job_title and st.session_state.job_offer_text and st.session_state.resume_text:
            try:
                match_answer = feature_match_chain.run(resume=st.session_state.resume_text, 
                                                       job_offer=st.session_state.job_offer_text, 
                                                       job_title=st.session_state.job_title)
                st.markdown("### Matching Skills and Experiences")
                st.write(match_answer)
            except ValueError as e:
                st.error("Something went wrong. Please try again.")  
        else:
            st.warning("Please upload a resume and provide a job offer text and job title to proceed.")
    elif feature_suggested_changes_button:
        if st.session_state.job_title and st.session_state.job_offer_text and st.session_state.resume_text:
            try:
                suggested_changes_answer = feature_suggested_changes_chain.run(resume=st.session_state.resume_text, 
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
