import streamlit as st
import os
from rag_engine import RAGEngine
from ingest_data import ingest_pdf
import tempfile

# Page configuration
st.set_page_config(page_title="AI Resume Chatbot", page_icon="📄", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput > div > div > input {
        color: #ffffff;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize RAG Engine
@st.cache_resource
def get_rag_engine():
    return RAGEngine()

try:
    engine = get_rag_engine()
except Exception as e:
    st.error(f"Failed to initialize RAG Engine: {e}")
    st.stop()

# Sidebar for file upload
with st.sidebar:
    st.title("📂 Document Manager")
    uploaded_file = st.file_uploader("Upload a PDF for RAG", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner("Processing PDF and Storing Embeddings..."):
                # Save uploaded file to temp directory
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # Ingest
                success = ingest_pdf(tmp_path)
                os.remove(tmp_path)
                
                if success:
                    st.success("✅ Document Ingested Successfully!")
                else:
                    st.error("❌ Ingestion Failed.")

st.title("📄 AI Resume Chatbot")
st.caption("Ask me anything about the uploaded documents!")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = engine.generate_answer(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating response: {e}")
