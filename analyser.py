
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from data.unify_endpoints_data import model_provider, dynamic_provider

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
                                                   placeholder="Enter Unify API Key", args=("Unify Key ",))
    
    # Model and provider selection
    model_name = st.selectbox("Select Model", options=model_provider.keys(), index=20,
                              placeholder="Model", args=("Model",))
    if st.toggle("Enable Dynamic Routing"):
        provider_name = st.selectbox("Select a Provider", options=dynamic_provider,
                                     placeholder="Provider", args=("Provider",))
    else:
        provider_name = st.selectbox("Select a Provider", options=model_provider[model_name],
                                     placeholder="Provider", args=("Provider",))
    st.session_state.endpoint = f"{model_name}@{provider_name}"

    # Document uploader
    st.session_state.resume_doc = st.file_uploader(label="Upload your Resume*", type=("pdf","docx"),
                                                 accept_multiple_files=False)
    # file_upload = st.file_uploader("Upload Resume: ", type=['PDF', 'DOCX']) <-whitout accept_multiple_files=True, without session_state
    
    # Paste Text Job Offer
    st.session_state.job_offer_text = st.text_area(label = "Job_offer_text", key="Job_offer", placeholder="Job offer text")


# if file_upload:
#     try:
#         text = extract_text_from_file(file_upload)
        
#     # TODO
#     # ?  chain = load_qa_chain(llm=llm, chain_type='stuff')
#     # ?  with st.expander("analysis"):
#     # ?      st.markdown("**analysis **")
#     # ?       st.markdown(analyisis anwser)
        
#     except ValueError as e:
#         st.error(str(e))

# def extract_text_from_file(file):
#     file_extension = file.name.split('.')[-1].lower()

#     if file_extension == 'pdf':
#         return extract_text_from_pdf(file)
#     elif file_extension == 'docx':
#         return extract_text_from_doc(file)
    
    
    
# def extract_text_from_doc(file):
#     text = ''
#     doc = docx.Document(file)
#     for paragraph in doc.paragraphs:
#         text += paragraph.text + '\n'
#     return text   


# def extract_pdf(file):
#     """Extracts text from a PDF document.

#     Args:
#         file: pdf file to extract text from.

#     Returns:
#         str: The extracted text from the PDF documents.
#     """
#     text = ""
#     pdf_reader = PdfReader(pdf)
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text


# def split_text(text): 
#     """
#     Splits the input text into chunks of a specified size.

#     Args:
#         text (str): The input text to be split.

#     Returns:
#         list: A list of text chunks.
#     """
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=700,
#         chunk_overlap=200,
#         length_function=len)

#     chunks = text_splitter.split_text(text=text)
#     return chunks

# def faiss_vector_storage(chunks):
#     """Creates a FAISS vector store from the given text chunks.

#     Args:
#         text_chunks: A list of text chunks to be vectorized.

#     Returns:
#         FAISS: A FAISS vector store.
#     """
#     vector_store = None

#     if st.session_state.embedding_model == "HuggingFaceEmbeddings":
#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         vector_store = FAISS.from_texts(chunks, embedding=embeddings)
#     return vector_store