import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI  # Fixed import



# ------------------ Load Environment ------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Online mode API key

# ------------------ Config ------------------
DATA_PATH = "./data"          # Folder containing PDFs
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3:latest"

# ------------------ Page Config (MUST BE FIRST) ------------------
st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="centered")

# ------------------ UI ------------------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")  # Load custom CSS after set_page_config

st.title("📄 Chat with Your PDFs")
st.write("Toggle between Online (OpenAI) and Offline (Ollama).")
st.info("⚡ Online mode works faster and often gives better results. Offline mode uses local Ollama model.")
mode = st.radio("Select LLM Mode", ["Online", "Offline"])

# ------------------ Load PDFs ------------------
@st.cache_data
def load_pdfs():
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    return loader.load()

@st.cache_data
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

@st.cache_resource
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

# ------------------ RAG Chain ------------------
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an assistant answering strictly from the context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
)

def build_rag_chain(retriever, mode="Offline"):
    if mode == "Online":
        # Use OpenAI directly with chat model
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=OPENAI_API_KEY)
    else:
        llm = Ollama(model=OLLAMA_MODEL)

    return (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

# ------------------ App Logic ------------------
question = st.text_input("Ask a question about your PDFs")

if st.button("Get Answer") and question:
    with st.spinner("Processing..."):
        docs = load_pdfs()
        chunks = split_documents(docs)
        vector_db = get_vectorstore(chunks)
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        rag_chain = build_rag_chain(retriever, mode)
        answer = rag_chain.invoke(question)

    st.subheader("Answer")
    st.write(answer)