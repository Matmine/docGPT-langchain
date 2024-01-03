from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from constants import *
from transformers import AutoTokenizer
import torch
import os
import re
from datetime import datetime

#Initialize timestamp
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


class TxtQA:
    def __init__(self, config: dict = {}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None

    @classmethod
    def create_sbert_mpnet(cls):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": device})

    @classmethod
    def create_mistral_base(cls, load_in_8bit=False):
        return pipeline(
            task="text2text-generation",
            model = "mistralai/Mistral-7B-v0.1",
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    
    @classmethod
    def create_flan_t5_base(cls, load_in_8bit=False):
        # Wrap it in HF pipeline for use with LangChain
        model="google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )

    # ... (other class methods)

    def init_embeddings(self) -> None:
        log(" -- INFO -- init_embeddings -- Starting init_embeddings")
        if self.config["embedding"] == EMB_OPENAI_ADA:
            self.embedding = OpenAIEmbeddings()
            log(" -- INFO -- init_embeddings -- End of init_embeddings")
        elif self.config["embedding"] == EMB_INSTRUCTOR_XL:
            if self.embedding is None:
                self.embedding = TxtQA.create_instructor_xl()
                log(" -- INFO -- init_embeddings -- End of init_embeddings")
        elif self.config["embedding"] == EMB_SBERT_MPNET_BASE:
            if self.embedding is None:
                self.embedding = TxtQA.create_sbert_mpnet()
                log(" -- INFO -- init_embeddings -- End of init_embeddings")
        else:
            self.embedding = None

    def init_models(self) -> None:
        log(" -- INFO -- init_models -- Starting init_models")
        load_in_8bit = self.config.get("load_in_8bit", False)
        log(" -- INFO -- init_models -- check if llm not initialized")
        if not self.llm:
            if self.config["llm"] == LLM_FLAN_T5_BASE :
                question_t5_template = """
                context: {context}
                question: {question}
                answer: 
                """
                QUESTION_T5_PROMPT = PromptTemplate(
                    template=question_t5_template, input_variables=["context", "question"]
                )
                log(" -- INFO -- init_models -- create_flan_t5_small ")
                self.llm = TxtQA.create_flan_t5_small(load_in_8bit=load_in_8bit)
                self.llm.combine_documents_chain.llm_chain.prompt = QUESTION_T5_PROMPT
            else:
                log(" -- INFO -- init_models -- create_flan_t5_xl")
                self.llm = TxtQA.create_flan_t5_xl(load_in_8bit=load_in_8bit)
            self.llm.combine_documents_chain.verbose = True
            self.llm.return_source_documents = True
        log(" -- INFO -- init_models -- End of init_models")

    def vector_db_text(self) -> None:
        log(" -- INFO -- vector_db_text -- Starting vector_db_text")
        txt_path = self.config.get("txt_path", None)
        log(" -- INFO -- vector_db_text -- Persist Directory")
        persist_directory = self.config.get("persist_directory", None)
        
        if persist_directory and os.path.exists(persist_directory):
            log(" -- INFO -- vector_db_text -- Call chroma for embedding")
            self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
        elif txt_path and os.path.exists(txt_path):
            log(" -- INFO -- vector_db_text -- load txt file")
            loader = TextLoader(txt_path)            
            documents = loader.load()
            log(" -- INFO -- vector_db_text -- split txt file 1/2")
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)            
            texts = text_splitter.split_documents(documents)
            log(" -- INFO -- vector_db_text -- split txt file 2/2")
            text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
            texts = text_splitter.split_documents(texts)
            log(" -- INFO -- vector_db_text -- Call chroma for embedding")
            self.vectordb = Chroma.from_documents(documents=texts, embedding=self.embedding, persist_directory=persist_directory)
            self.vectordb.persist()
            log(" -- INFO -- vector_db_text -- End of vector_db_text")
        else:
            raise ValueError("NO file found")

    def retreival_qa_chain(self):

        if not self.vectordb:
            raise ValueError("Vector database not initialized.")
        log(" -- INFO -- retreival_qa_chain -- Starting retreival QA chain 1/4")
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})
        log(" -- INFO -- retreival_qa_chain --  2/4")
        hf_llm = HuggingFacePipeline(pipeline=self.llm, model_id=self.config["llm"])
        log(" -- INFO -- retreival_qa_chain --  3/4")
        self.qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff", retriever=self.retriever)
        log(" -- INFO -- retreival_qa_chain --  4/4")
        self.qa.combine_documents_chain.verbose = True
        self.qa.return_source_documents = True

    def answer_query(self, question: str) -> str:
        log(" -- INFO -- answer_query --  Starting answer_query 1/3")
        answer_dict = self.qa({"query": question})
        log(" -- INFO -- answer_query -- 2/3")
        answer = answer_dict["result"]
        log(" -- INFO -- answer_query -- 3/3")
        return answer
