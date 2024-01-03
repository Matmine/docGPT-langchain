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
from langchain.vectorstores.chroma import Chroma
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
    
# Import the PDFConverter class
class PDFConverter:
    def convert_pdf_to_txt(self, pdf_file, txt_file):
        try:
            with open(pdf_file, 'rb') as pdf:
                pdf_reader = PyPDF2.PdfReader(pdf)
                text = ''
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                
                with open(txt_file, 'w', encoding='utf-8') as txt:
                    txt.write(text)
                print(f"PDF converted to TXT. Saved as {txt_file}")
        except FileNotFoundError:
            print("File not found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

st.set_page_config(
    page_title='Q&A Bot for PDF',
    page_icon='🔖',
    layout='wide',
    initial_sidebar_state='auto',
)

if "_qa_model" not in st.session_state:
    st.session_state["txt_qa_model"]:TxtQA = TxtQA() ## Intialisation   

## To cache resource across multiple session 
@st.cache_resource
def load_llm(llm,load_in_8bit):

    if llm == LLM_OPENAI_GPT35:
        pass
    elif llm == LLM_FLAN_T5_SMALL:
        return TxtQA.create_flan_t5_small(load_in_8bit)
    elif llm == LLM_FLAN_T5_BASE:
        return TxtQA.create_flan_t5_base(load_in_8bit)
    elif llm == LLM_FLAN_T5_LARGE:
        return TxtQA.create_flan_t5_large(load_in_8bit)
    elif llm == LLM_FASTCHAT_T5_XL:
        return TxtQA.create_fastchat_t5_xl(load_in_8bit)
    elif llm == LLM_FALCON_SMALL:
        return TxtQA.create_falcon_instruct_small(load_in_8bit)
    elif llm == LLM_MISTRAL:
        return TxtQA.create_falcon_instruct_small(load_in_8bit)
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
    emb = st.radio("**Select Embedding Model**", [EMB_INSTRUCTOR_XL, EMB_SBERT_MPNET_BASE,EMB_SBERT_MINILM],index=1)
    llm = st.radio("**Select LLM Model**", [LLM_FASTCHAT_T5_XL, LLM_FLAN_T5_SMALL,LLM_FLAN_T5_BASE,LLM_FLAN_T5_LARGE,LLM_FLAN_T5_XL,LLM_FALCON_SMALL,LLM_MISTRAL],index=2)
    load_in_8bit = st.radio("**Load 8 bit**", [True, False],index=1)
    pdf_file = st.file_uploader("**Upload PDF or TXT**", type=["pdf", "txt"])

    if st.button("Submit") and pdf_file is not None:
        with st.spinner(text="Uploading and Processing File..."):
            if pdf_file.type == 'application/pdf':
                # For PDF files, convert to text using the PDFConverter class
                converter = PDFConverter()
                with NamedTemporaryFile(delete=False, suffix='.txt') as tmp_txt:
                    converter.convert_pdf_to_txt(pdf_file, tmp_txt.name)
                    # CONVERSION DONE
                    txt_path = Path(tmp_txt.name)
            else:
                # For TXT files, use directly
                with NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
                    shutil.copyfileobj(pdf_file, tmp)
                    txt_path = Path(tmp.name)

            st.session_state["txt_qa_model"].config = {
                "txt_path": str(txt_path),  # Pass the path to the text file
                "embedding": emb,
                "llm": llm,
                "load_in_8bit": load_in_8bit
            }
            st.session_state["txt_qa_model"].embedding = load_emb(emb)
            st.session_state["txt_qa_model"].llm = load_llm(llm,load_in_8bit)        
            st.session_state["txt_qa_model"].init_embeddings()
            st.session_state["txt_qa_model"].init_models()
            st.session_state["txt_qa_model"].vector_db_text()
            st.sidebar.success("Txt file uploaded successfully")

question = st.text_input('Ask a question', 'What is this document about?')

if st.button("Answer"):
    try:
        st.session_state["txt_qa_model"].retreival_qa_chain()
        answer = st.session_state["txt_qa_model"].answer_query(question)
        st.write(f"{answer}")
    except Exception as e:
        st.error(f"Error answering the question: {str(e)}")