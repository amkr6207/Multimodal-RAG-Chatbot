# 📄 Multimodal RAG Chatbot (Groq + MongoDB Atlas)

A high-performance Retrieval-Augmented Generation (RAG) system that allows users to chat with their PDF documents. This project stands out by supporting **Multimodal Ingestion**, meaning it can "see" and describe charts, graphs, and images within your PDFs.

## 🚀 Key Features
- **Semantic Text Search**: Uses local `all-MiniLM-L6-v2` embeddings for fast, zero-cost text retrieval.
- **Multimodal Vision**: Automatically extracts images from PDFs and captions them using **Groq's Llama 3.2 Vision** model.
- **Vector Database**: Scalable vector storage using **MongoDB Atlas Vector Search**.
- **Ultra-Fast LLM**: Powers conversations with **Llama 3.3 (70B)** via Groq, providing near-instant responses.
- **Streamlit UI**: A clean, modern chat interface for easy document management and interaction.

## 🛠️ Tech Stack
- **AI Models**: Groq (Llama 3.3 & Llama 3.2 Vision), HuggingFace (Local Embeddings).
- **Database**: MongoDB Atlas (Vector Search).
- **Orchestration**: LangChain.
- **Interface**: Streamlit.
- **PDF Processing**: PyMuPDF (fitz), PyPDF.

## 📋 Architecture
1. **Ingestion**: PDFs are split into text chunks. Images are extracted and sent to Groq Vision for detailed captioning.
2. **Indexing**: Both text and image captions are converted into 384-dimensional vectors and stored in MongoDB Atlas.
3. **Retrieval**: When a user asks a question, the system performs a similarity search in MongoDB to find the most relevant context (text or visuals).
4. **Generation**: Groq's Llama 3.3 synthesizes the retrieved context into a clear, accurate answer.

## ⚙️ Setup Instructions
1. **Clone the repo**:
   ```bash
   git clone <your-repo-url>
   cd Multimodal-RAG-Chatbot
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment Variables**:
   Create a `.env` file based on `.env.example` and add your:
   - `GROQ_API_KEY`
   - `MONGODB_ATLAS_CLUSTER_URI`
4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

---
*Created by Aman - Data Science Portfolio Project*
