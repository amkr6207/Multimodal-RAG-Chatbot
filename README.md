# 📄 Multimodal RAG Chatbot (Groq + MongoDB Atlas)

A Retrieval-Augmented Generation (RAG) application for asking questions about PDF documents. During ingestion, it extracts PDF text and uses a Groq-hosted vision model to turn charts and images into searchable text descriptions.

## 🚀 Key Features
- **Semantic Text Search**: Uses local `all-MiniLM-L6-v2` embeddings for semantic retrieval.
- **Multimodal Ingestion**: Extracts PDF images and captions them with a configurable Groq vision model.
- **Vector Database**: Stores text chunks, image descriptions, embeddings, and metadata in MongoDB Atlas Vector Search.
- **Configurable Generation**: Uses a configurable Groq-hosted model to answer questions from retrieved context.
- **Source Citations**: Appends deduplicated PDF filenames and page numbers from retrieved metadata.
- **Streamlit UI**: A clean, modern chat interface for easy document management and interaction.

## 🛠️ Tech Stack
- **AI Models**: Groq (`qwen/qwen3.6-27b` by default), Hugging Face (`all-MiniLM-L6-v2` local embeddings).
- **Database**: MongoDB Atlas (Vector Search).
- **Orchestration**: LangChain.
- **Interface**: Streamlit.
- **PDF Processing**: PyMuPDF (fitz), PyPDF.

## 📋 Architecture
1. **Ingestion**: PDFs are split into text chunks. Images are extracted and sent to Groq Vision for detailed captioning.
2. **Indexing**: Both text and image captions are converted into 384-dimensional vectors and stored in MongoDB Atlas.
3. **Retrieval**: When a user asks a question, the system performs a similarity search in MongoDB to find the most relevant context (text or visuals).
4. **Generation**: The configured Groq model generates an answer from the retrieved context, and Python appends the retrieved filename/page citations.

## ⚙️ Setup Instructions
1. **Clone the repo**:
   ```bash
   git clone https://github.com/amkr6207/Multimodal-RAG-Chatbot.git
   cd Multimodal-RAG-Chatbot
   ```
2. **Create and Activate Environment**:
   ```bash
   conda create -n chatbot-env python=3.12 -y
   conda activate chatbot-env
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment Variables**:
   Create a `.env` file based on `.env.example` and add your:
   - `GROQ_API_KEY`
   - `GROQ_VISION_MODEL` (defaults to `qwen/qwen3.6-27b`)
   - `GROQ_GENERATION_MODEL` (defaults to `qwen/qwen3.6-27b`)
   - `MONGODB_ATLAS_CLUSTER_URI`
5. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Vision Ingestion Behavior

Image captioning uses `GROQ_VISION_MODEL`. If one or more images cannot be captioned, the application stores the available text and successful image captions, then reports a partial-ingestion warning instead of silently claiming complete multimodal success.

Groq-hosted model availability can change. Check the [Groq vision documentation](https://console.groq.com/docs/vision) before changing the configured model.

## Tests

The automated tests use mocks and do not call Groq, MongoDB, or Hugging Face services.

```bash
pip install -r requirements-dev.txt
pytest
```

At startup, the application validates that `GROQ_API_KEY` and `MONGODB_ATLAS_CLUSTER_URI` are present and that the MongoDB URI has a valid scheme. This validation does not make external network requests.

## Current Limitations

- Image understanding is caption-based; the original images are not passed to the final answer model.
- Citations identify retrieved chunks; they are not claim-level attribution from the generation model.
- Provider availability, rate limits, and network failures can cause partial image ingestion.
- The project does not yet include a RAG quality or latency benchmark.
