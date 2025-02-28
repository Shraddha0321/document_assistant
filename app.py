import os
import streamlit as st
from datetime import datetime
import pyperclip  # Enables copying of text

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


# ---------- CONFIGURATION ----------

PDF_FOLDER = "document_store/pdfs"
os.makedirs(PDF_FOLDER, exist_ok=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query}
Context: {document_context}
Answer:
"""

EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)


# ---------- FUNCTIONS ----------

def save_uploaded_file(uploaded_file) -> str:
    """Save the uploaded file to a local folder and return its path."""
    file_path = os.path.join(PDF_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    """Load the PDF using PDFPlumberLoader, returning a list of Document objects."""
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    """Split raw documents into smaller chunks for vector indexing."""
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    """Add the document chunks to the in-memory vector store."""
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query, k=3):
    """Find `k` related documents from the vector store for the given query."""
    return DOCUMENT_VECTOR_DB.similarity_search(query, k=k)

def generate_answer(user_query, context_documents):
    """Generate a concise answer using the context documents."""
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    
    return response_chain.invoke({
        "user_query": user_query, 
        "document_context": context_text
    })


# ---------- STREAMLIT APP CONFIGURATION ----------

st.set_page_config(
    page_title="📘 Document Assistant Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- THEME TOGGLE ----------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

theme_toggle = st.sidebar.toggle("🌗 Toggle Theme", value=(st.session_state.theme == "dark"))

if theme_toggle:
    st.session_state.theme = "dark"
else:
    st.session_state.theme = "light"

# Define theme-based colors
THEME_COLORS = {
    "dark": {
        "bg_color": "#0E1117",
        "text_color": "#FFFFFF",
        "button_color": "#00FFAA",
        "input_bg": "#1E1E1E",
        "border_color": "#3A3A3A"
    },
    "light": {
        "bg_color": "#FFFFFF",
        "text_color": "#000000",
        "button_color": "#007BFF",
        "input_bg": "#F0F0F0",
        "border_color": "#CCCCCC"
    }
}

theme = st.session_state.theme
colors = THEME_COLORS[theme]

# Apply custom CSS based on the theme
st.markdown(f"""
    <style>
    body {{
        background-color: {colors['bg_color']};
    }}
    .stApp {{
        background-color: {colors['bg_color']};
        color: {colors['text_color']};
    }}
    .stFileUploader {{
        background-color: {colors['input_bg']};
        border: 1px solid {colors['border_color']};
        border-radius: 10px;
        padding: 15px;
    }}
    h1, h2, h3 {{
        color: {colors['button_color']} !important;
    }}
    .copy-button {{
        background-color: {colors['button_color']};
        border: none;
        color: black;
        padding: 6px 12px;
        border-radius: 8px;
        cursor: pointer;
        font-weight: bold;
    }}
    .edit-textbox {{
        width: 100%;
        background-color: {colors['input_bg']};
        color: {colors['text_color']};
        padding: 10px;
        border-radius: 8px;
        border: 1px solid {colors['border_color']};
    }}
    .sidebar-header {{
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        padding: 10px;
        color: {colors['button_color']};
    }}
    .sidebar-message {{
        background-color: {colors['input_bg']};
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        font-size: 16px;
        color: {colors['text_color']};
    }}
    </style>
    """, unsafe_allow_html=True)


# ---------- SIDEBAR: Chat History ----------

st.sidebar.markdown("<div class='sidebar-header'>📝 Chat History</div>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    with st.sidebar.expander(f"🔹 {chat['query'][:40]}..."):
        st.markdown(f"<div class='sidebar-message'><b>User:</b> {chat['query']}<br><b>Assistant:</b> {chat['response']}</div>", unsafe_allow_html=True)


# ---------- MAIN TITLE & FILE UPLOAD ----------

st.title("📘 Document Assistant Bot")
st.markdown("### Upload a PDF and start chatting!")

if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = ""

uploaded_pdf = st.file_uploader(
    label="📄 Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf and not st.session_state.doc_loaded:
    with st.spinner("Processing your PDF..."):
        saved_path = save_uploaded_file(uploaded_pdf)
        raw_docs = load_pdf_documents(saved_path)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)
        st.session_state.doc_loaded = True  
        st.session_state.uploaded_filename = uploaded_pdf.name

    st.success(f"✅ Document '{st.session_state.uploaded_filename}' processed successfully! Ask questions below.")


# ---------- CHAT FUNCTIONALITY ----------

if st.session_state.doc_loaded:
    st.markdown(f"**📄 Current Document:** `{st.session_state.uploaded_filename}`")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        timestamp = datetime.now().strftime("%H:%M:%S")

        with st.chat_message("user"):
            st.write(f"🕒 `{timestamp}` - **You:** {user_input}")

        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input, k=3)
            ai_response = generate_answer(user_input, relevant_docs)

        st.session_state.chat_history.append({
            "query": user_input,
            "response": ai_response,
            "timestamp": timestamp
        })

        with st.chat_message("assistant", avatar="🤖"):
            st.write(f"🕒 `{timestamp}` - **Assistant:** {ai_response}")

            edited_text = st.text_area("Edit Response", ai_response, height=100)
            if st.button("📋 Copy"):
                pyperclip.copy(edited_text)
                st.toast("Copied to clipboard!")
