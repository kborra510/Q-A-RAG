import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

load_dotenv()

def load_and_split(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def build_vectorstore(chunks, embedding_type="openai"):
    if embedding_type == "openai":
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

def get_llm(model_choice):
    if model_choice == "OpenAI":
        return ChatOpenAI(model_name="gpt-3.5-turbo")
    elif model_choice == "Ollama":
        return Ollama(model="llama3")

def get_qa_chain(vectordb, model_choice):
    retriever = vectordb.as_retriever()
    llm = get_llm(model_choice)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
