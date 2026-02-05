from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os

DOCS_PATH = "E:/chatbot/"
#funcion para leer los archivos
def load_docs():
    docs = []
    for file in os.listdir(DOCS_PATH):
        path = os.path.join(DOCS_PATH, file)
        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif file.endswith(".docx"):
            docs.extend(Docx2txtLoader(path).load())
        elif file.endswith(".txt"):
            docs.extend(TextLoader(path).load())
    return docs

docs = load_docs()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db"
)

print("✅ Documentos indexados correctamente")


import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("🤖 Chat Empresa")

# Embeddings y base
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 4})

# LLM
llm = Ollama(model="mistral")

# Prompt
prompt = ChatPromptTemplate.from_template("""
Sos un asistente de la empresa. Respondé SOLO con la información de contexto.

Contexto:
{context}

Pregunta:
{question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

pregunta = st.text_input("Preguntá algo sobre los documentos...")

if pregunta:
    docs = retriever.invoke(pregunta)
    contexto = format_docs(docs)

    chain = prompt | llm | StrOutputParser()
    respuesta = chain.invoke({
        "context": contexto,
        "question": pregunta
    })

    st.write(respuesta)
