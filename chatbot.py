import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


DOCS_PATH = "./manual-de-instalación-y-configuración-dn100-svb-v1.pdf"



st.title("🤖 Chat Empresa")

# Conexión a la base ya creada por indexar_docs.py
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 4})

llm = Ollama(model="mistral")

prompt = ChatPromptTemplate.from_template("""
Sos un asistente interno de la empresa.
Respondé SOLO usando la información del contexto.

Contexto:
{context}

Pregunta:
{question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


pregunta = st.text_input("Preguntá algo sobre los documentos de la empresa...")

if pregunta:
    docs = retriever.invoke(pregunta)
    contexto = format_docs(docs)

    chain = prompt | llm | StrOutputParser()

    respuesta = chain.invoke({
        "context": contexto,
        "question": pregunta
    })

    st.write(respuesta)


