import streamlit as st
from Txt_qa import TxtQA
import PyPDF2
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
import shutil
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import torch
from transformers import AutoTokenizer
import re
from dotenv import load_dotenv
import json
import os
from constants import *
from datetime import datetime

#Initialize timestamp
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Import the PDFConverter class
class PDFConverter:
    def convert_pdf_to_txt(self, raw_file_path, txt_file):
        log(" -- INFO -- convert_pdf_to_txt -- Starting file conversion to txt file")
        try:
            with open(raw_file_path, 'rb') as pdf:
                pdf_reader = PyPDF2.PdfReader(pdf)
                text = ''
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                
                with open(txt_file, 'w', encoding='utf-8') as txt:
                    text = re.sub(r'[^a-zA-Z0-9\séèçàêîôû\'"-.,;:!@#~&\[\]()+=<>?/|\\%]', '', text)
                    txt.write(text)
                
                log(" -- INFO -- convert_pdf_to_txt -- PDF converted to TXT. Saved as {txt_file}")
        except FileNotFoundError:
            print("ERROR -- convert_pdf_to_txt -- File not found.")
        except Exception as e:
            print(f"ERROR -- convert_pdf_to_txt -- An error occurred: {str(e)}")

st.set_page_config(
    page_title='Q&A Bot for PDF',
    page_icon='🔖',
    layout='wide',
    initial_sidebar_state='auto',
)

if "txt_qa_model" not in st.session_state:
    log(" -- INFO -- init_session -- Initializing session state")
    st.session_state["txt_qa_model"]:TxtQA = TxtQA() ## Intialisation   

## To cache resource across multiple session 
@st.cache_resource
def load_llm(llm,load_in_8bit):
    log(" -- INFO -- load_llm -- Loading {llm}")
    if llm == LLM_FLAN_T5_BASE:
        return TxtQA.create_flan_t5_base(load_in_8bit)
    elif llm == LLM_MISTRAL:
        return TxtQA.create_mistral_base(load_in_8bit)
    else:
        raise ValueError("Invalid LLM setting")
    

@st.cache_resource
def load_emb(emb):
    if emb == EMB_INSTRUCTOR_XL:
        return TxtQA.create_instructor_xl()
    elif emb == EMB_SBERT_MPNET_BASE:
        return TxtQA.create_sbert_mpnet()
    elif emb == EMB_SBERT_MINILM:
        pass ##ChromaDB takes care
    else:
        raise ValueError("Invalid embedding setting")


st.title("PDF Q&A (Self hosted LLMs)")

with st.sidebar:
    emb = st.radio("**Select Embedding Model**", [EMB_SBERT_MPNET_BASE], index=0)
    llm = st.radio("**Select LLM Model**", [LLM_FLAN_T5_BASE, LLM_MISTRAL],index=1)
    load_in_8bit = st.radio("**Load 8 bit**", [True, False],index=1)
    uploaded_file = st.file_uploader("**Upload PDF or TXT**", type=["pdf", "txt"])

    if st.button("Submit") and uploaded_file is not None:
        timestamp_start_submit = datetime.now()
        with st.spinner(text="Uploading and Processing File..."):
            raw_file_path = 'C:\\Users\\mathi\\VsCodeRepo\\GptGames\\docGPT-langchain\\gpt_project_v3\\file_cache\\raw_file.pdf'
            converted_file_path = "C:\\Users\\mathi\\VsCodeRepo\\GptGames\\docGPT-langchain\\gpt_project_v3\\file_cache\\file.txt"
            if uploaded_file.type == 'application/pdf':
                # For PDF files, save and convert to text using the PDFConverter class

                # Save the uploaded PDF file to the specified path
                with open(raw_file_path, 'wb') as new_file:
                    new_file.write(uploaded_file.read())

                # Convert the PDF file to text and save it to the specified path
                converter = PDFConverter()
                converter.convert_pdf_to_txt(raw_file_path, "file_cache\\file.txt")
                # CONVERSION DONE
                txt_path = converted_file_path
            else:
                # For TXT files, save and use directly
                file_content = uploaded_file.read()
                # Open a new file in write mode ('w') at the new location
                with open(converted_file_path, 'w', encoding="utf8") as new_file:
                    # Write the content to the new file
                    new_file.write(file_content)
                txt_path = converted_file_path

            st.session_state["txt_qa_model"].config = {
                "txt_path": str(txt_path),  # Pass the path to the text file
                "embedding": emb,
                "llm": llm,
                "load_in_8bit": load_in_8bit
            }
            st.session_state["txt_qa_model"].embedding = load_emb(emb)
            log(" -- INFO -- load_emb -- Loaded embedding method")
            st.session_state["txt_qa_model"].llm = load_llm(llm,load_in_8bit)   
            log(" -- INFO -- load_llm -- Loaded llm")     
            st.session_state["txt_qa_model"].init_embeddings()
            log(" -- INFO -- init_embeddings -- Initiated embeddings")
            st.session_state["txt_qa_model"].init_models()
            log(" -- INFO -- init_models -- Initiated models")
            st.session_state["txt_qa_model"].vector_db_text()
            log(" -- INFO -- vector_db_text -- Vectorized text")
            log(" -- INFO -- Txt file uploaded successfully")
            st.sidebar.success("Txt file uploaded successfully")

            #Process and show time to submit
            timestamp_end_submit = datetime.now()
            process_time_submit = timestamp_end_submit - timestamp_start_submit
            # Calculating total seconds in the timedelta
            total_seconds = process_time_submit.total_seconds()

            # Converting total seconds to minutes and remaining seconds
            minutes, seconds = divmod(total_seconds, 60)

            print(f"Time to submit file: {int(minutes)} min, {int(seconds)} secs")

question = st.text_input('Ask a question', 'What is this document about?')

if st.button("Answer"):
    timestamp_start_answer = datetime.now()

    try:
        log(" -- INFO -- Answer -- Answering question: {question}")
        st.session_state["txt_qa_model"].retreival_qa_chain()      
        answer = st.session_state["txt_qa_model"].answer_query(question)
        st.write(f"{answer}")
    except Exception as e:
        st.error(f"Error answering the question: {str(e)}")
    
    #Process and show time to answer
    timestamp_end_answer = datetime.now()
    process_time_answer = timestamp_end_answer - timestamp_start_answer
    # Calculating total seconds in the timedelta
    total_seconds = process_time_answer.total_seconds()

    # Converting total seconds to minutes and remaining seconds
    minutes, seconds = divmod(total_seconds, 60)

    print(f"Time to answer question: {int(minutes)} min, {int(seconds)} secs")