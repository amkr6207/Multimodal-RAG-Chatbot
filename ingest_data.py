import os
import pymongo
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
import fitz  # PyMuPDF
from groq import Groq
import base64
import io
from PIL import Image

# Load environment variables
load_dotenv()

# Configuration
DB_NAME = os.getenv("DB_NAME", "rag_chatbot")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_embeddings")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME", "vector_index")
MONGODB_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_image_caption(image_bytes):
    """Use Groq Llama-3 Vision to caption an image"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image, chart, or diagram found in a PDF document in detail for a search index."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error captioning image: {e}")
        return ""

def extract_images_and_caption(file_path):
    """Extract images from PDF and return list of Document chunks with captions"""
    from langchain_core.documents import Document
    doc = fitz.open(file_path)
    image_chunks = []
    
    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            print(f"Captioning image {img_index+1} on page {page_index+1}...")
            caption = get_image_caption(image_bytes)
            
            if caption:
                metadata = {"source": file_path, "page": page_index + 1, "type": "image"}
                image_chunks.append(Document(page_content=f"[Image/Chart Description]: {caption}", metadata=metadata))
                
    return image_chunks

def ingest_pdf(file_path):
    print(f"--- Processing: {file_path} ---")
    
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()
    
    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    # 2b. Extract and Caption Images
    print("Searching for images/charts...")
    image_chunks = extract_images_and_caption(file_path)
    if image_chunks:
        print(f"Added {len(image_chunks)} image captions to index.")
        chunks.extend(image_chunks)
    
    print(f"Total processing units: {len(chunks)}")

    # 3. Setup Embeddings (Local Model)
    try:
        print("Initializing local embedding model (HuggingFace)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 4. Connect to MongoDB and Store
        client = pymongo.MongoClient(MONGODB_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        print("Storing embeddings in MongoDB Atlas...")
        vector_search = MongoDBAtlasVectorSearch.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection=collection,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
        )
        print("✅ Ingestion successful!")
        return vector_search
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        if "quota" in str(e).lower() or "429" in str(e):
             print("TIP: Your Gemini API is hitting quota limits. You might need to enable 'Pay-as-you-go' (still has a free tier) or use a different embedding model.")
        return None

if __name__ == "__main__":
    # The file is in the current project folder
    resume_path = "Aman_Resume.pdf"
    if os.path.exists(resume_path):
        ingest_pdf(resume_path)
    else:
        print(f"❌ File not found: {resume_path}")
