
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Carpeta donde vas a poner los PDFs, Word, txt de la empresa
DOCS_PATH = "./manual-de-instalación-y-configuración-dn100-svb-v1.pdf"

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


print("📚 Leyendo documentos...")
docs = load_docs()

print("✂️ Dividiendo en chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)

print("🧠 Generando embeddings...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

print("💾 Guardando base vectorial en /db ...")
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db"
)

print("✅ Documentos indexados correctamente")
