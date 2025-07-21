import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from docx import Document

# Initialize chat history in Streamlit session state if not already present
if "history" not in st.session_state:
    st.session_state.history = []

# Function to extract text from a DOCX file
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to extract text from a TXT file
def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract and split text from uploaded PDF files into chunks
def extract_chunks_from_pdfs(uploaded_files):
    all_chunks = []
    metadata = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for file in uploaded_files:
        pdf_path = f"uploaded/{file.name}"
        os.makedirs("uploaded", exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(file.getbuffer())
        reader = PdfReader(pdf_path)

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                chunks = splitter.split_text(page_text)
                all_chunks.extend(chunks)
                # Store metadata for each chunk (source file and page number)
                metadata.extend([{'source': file.name, 'page': i+1}] * len(chunks))
    return all_chunks, metadata

# Function to create a FAISS vector store from text chunks and metadata
def create_faiss_vector_store(chunks, metadatas, path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Create FAISS vector store and save locally
    vector_store = FAISS.from_texts(chunks, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local(path)

# Function to load a FAISS vector store from disk
def load_faiss_vector_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(path, embeddings,  
                 allow_dangerous_deserialization=True)
    return vector_store

# Function to build a conversational QA chain using the vector store and LLM
def build_conversational_qa_chain(vector_store_path="faiss_index"):
    vector_store = load_faiss_vector_store(vector_store_path)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model="llama3.2")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        # Add chain_type_kwargs={"verbose": True} for logging/debugging
    )
    return qa_chain

# Streamlit UI logic starts here
if __name__ == "__main__":
    st.title("Multi-Document RAG Chatbot")
    st.write("Upload one or more PDF files and ask questions based on their combined content. Answers will include document and page references.")

    # File uploader for PDFs, DOCX, and TXT files
    uploaded_files = st.file_uploader("Upload your files (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    # Track if the QA chain is ready in session state
    if "qa_chain_ready" not in st.session_state:
        st.session_state.qa_chain_ready = False

    # If files are uploaded and QA chain is not ready, process files and build vector store and QA chain
    if uploaded_files and not st.session_state.qa_chain_ready:
        with st.spinner("Extracting text, splitting, and building vector store..."):
            chunks, metadatas = extract_chunks_from_pdfs(uploaded_files)
            create_faiss_vector_store(chunks, metadatas)
            st.session_state.qa_chain = build_conversational_qa_chain()
            st.session_state.qa_chain_ready = True
        st.success("Chatbot is ready! Ask your questions below.")

    # If QA chain is ready, display chat interface
    if st.session_state.get("qa_chain_ready"):

        # Display chat history if available
        if st.session_state["history"]:
            st.write("### Chat History")
            for entry in st.session_state["history"]:
                st.markdown(f"**You:** {entry['q']}")
                st.markdown(f"**Bot:** {entry['a']}")
                if "sources" in entry and entry["sources"]:
                    st.markdown(f"<sup><i>Sources: {entry['sources']}</i></sup>", unsafe_allow_html=True)
            st.markdown("---")

        # Input box for user question
        question = st.text_input("Ask a question about the uploaded PDFs:")
        if question:
            # Prepare chat history for context
            chat_history = [(entry["q"], entry["a"]) for entry in st.session_state["history"]]
            with st.spinner("Searching and generating answer..."):
                response = st.session_state.qa_chain({
                    "question": question,
                    "chat_history": chat_history
                    })
            answer = response["answer"]
            sources = []
            st.success(answer)
            # Display sources used for the answer
            if "source_documents" in response and response["source_documents"]:
                st.info("Sources used for this answer:")
                for doc in response["source_documents"]:
                    meta = doc.metadata
                    sources = f"{meta.get('source', 'Unknown file')}, page {meta.get('page', '?')}"
                    sources_str = "; ".join(sources)
            else:
                sources_str = ""

            # Append current Q&A to chat history
            st.session_state.history.append({"q": question, "a": answer, "sources": sources_str})

    else:
        st.info("Please upload one or more PDFs to get started.")