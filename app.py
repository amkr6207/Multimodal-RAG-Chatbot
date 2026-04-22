import streamlit as st
import os
import hashlib
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

# Session state defaults
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = {}
if "selected_doc_ids" not in st.session_state:
    st.session_state.selected_doc_ids = []

# Sidebar for file upload
with st.sidebar:
    st.title("📂 Document Manager")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs for RAG",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Ingest Selected Documents"):
            success_count = 0
            failed_files = []

            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    tmp_path = None
                    try:
                        file_bytes = uploaded_file.getvalue()
                        doc_id = hashlib.sha256(file_bytes).hexdigest()[:16]

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(file_bytes)
                            tmp_path = tmp.name

                        result = ingest_pdf(tmp_path, doc_id=doc_id, source_name=uploaded_file.name)
                        if result:
                            st.session_state.ingested_docs[doc_id] = uploaded_file.name
                            success_count += 1
                        else:
                            failed_files.append(uploaded_file.name)
                    except Exception:
                        failed_files.append(uploaded_file.name)
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.remove(tmp_path)

            if success_count:
                st.success(f"✅ Ingested {success_count} document(s).")
            if failed_files:
                st.error(f"❌ Failed: {', '.join(failed_files)}")

    if st.session_state.ingested_docs:
        st.subheader("Indexed Documents")
        doc_options = {
            f"{name} ({doc_id[:8]})": doc_id
            for doc_id, name in st.session_state.ingested_docs.items()
        }

        selected_labels = st.multiselect(
            "Filter retrieval to these files",
            options=list(doc_options.keys()),
            default=list(doc_options.keys()),
        )
        st.session_state.selected_doc_ids = [doc_options[label] for label in selected_labels]

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

st.title("📄 AI Resume Chatbot")
st.caption("Ask me anything about the uploaded documents!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What would you like to know?"):
    if not st.session_state.ingested_docs:
        st.error("Please upload and ingest at least one document first.")
        st.stop()

    doc_filter = st.session_state.selected_doc_ids
    if not doc_filter:
        st.error("Please select at least one indexed file in the sidebar.")
        st.stop()

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = engine.generate_answer(prompt, doc_ids=doc_filter)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating response: {e}")
