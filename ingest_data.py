import base64
import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import fitz  # PyMuPDF
import pymongo
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configuration
DB_NAME = os.getenv("DB_NAME", "rag_chatbot")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_embeddings")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME", "vector_index")
MONGODB_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")


class VisionCaptionError(RuntimeError):
    """Raised when the configured vision provider cannot caption an image."""


@dataclass
class ImageExtractionResult:
    """Image-captioning output, including failures that should be shown to users."""

    chunks: list[Document] = field(default_factory=list)
    images_found: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def images_captioned(self) -> int:
        return len(self.chunks)

    @property
    def images_failed(self) -> int:
        return len(self.warnings)


@dataclass
class IngestionResult:
    """Summary of a completed ingestion operation."""

    vector_search: Any
    doc_id: str
    text_chunks: int
    images_found: int
    images_captioned: int
    images_failed: int
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "partial_success" if self.warnings else "success"


def _image_mime_type(extension: str | None) -> str:
    normalized = (extension or "jpeg").lower()
    if normalized == "jpg":
        normalized = "jpeg"
    if normalized not in {"jpeg", "png", "webp", "gif"}:
        normalized = "jpeg"
    return f"image/{normalized}"


def get_image_caption(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    *,
    client: Groq | None = None,
    model: str | None = None,
) -> str:
    """Caption an image with the configured Groq vision model."""
    selected_model = model or GROQ_VISION_MODEL
    if not selected_model:
        raise VisionCaptionError("GROQ_VISION_MODEL is not configured.")

    try:
        groq_client = client or Groq(api_key=GROQ_API_KEY)
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        response = groq_client.chat.completions.create(
            model=selected_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this image, chart, or diagram found in a PDF "
                                "document in detail for a search index."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
        )
        caption = response.choices[0].message.content
        if not caption or not caption.strip():
            raise VisionCaptionError(
                f"Vision model '{selected_model}' returned an empty caption."
            )
        return caption.strip()
    except VisionCaptionError:
        raise
    # Wrap SDK/network failures in the module's provider-specific exception.
    except Exception as e:  # noqa: BLE001
        raise VisionCaptionError(
            f"Vision model '{selected_model}' failed: {e}"
        ) from e


def extract_images_and_caption(
    file_path: str,
    metadata_fields: dict[str, Any] | None = None,
    *,
    captioner: Callable[..., str] | None = None,
) -> ImageExtractionResult:
    """Extract and caption PDF images while preserving partial failures."""
    metadata_fields = metadata_fields or {}
    caption_image = captioner or get_image_caption
    result = ImageExtractionResult()

    with fitz.open(file_path) as pdf_document:
        for page_index in range(len(pdf_document)):
            page = pdf_document[page_index]

            for image_index, image in enumerate(page.get_images(full=True)):
                result.images_found += 1
                page_number = page_index + 1
                image_number = image_index + 1

                try:
                    extracted_image = pdf_document.extract_image(image[0])
                    image_bytes = extracted_image["image"]
                    mime_type = _image_mime_type(extracted_image.get("ext"))
                    print(
                        f"Captioning image {image_number} on page {page_number}..."
                    )
                    caption = caption_image(image_bytes, mime_type=mime_type)
                # One malformed image must not prevent usable PDF text from indexing.
                except Exception as error:  # noqa: BLE001
                    warning = (
                        f"Image {image_number} on page {page_number} could not be "
                        f"captioned: {error}"
                    )
                    print(f"Warning: {warning}")
                    result.warnings.append(warning)
                    continue

                metadata = {
                    "source": file_path,
                    "page": page_number,
                    "type": "image",
                }
                metadata.update(metadata_fields)
                result.chunks.append(
                    Document(
                        page_content=f"[Image/Chart Description]: {caption}",
                        metadata=metadata,
                    )
                )

    return result


def ingest_pdf(file_path, doc_id=None, source_name=None):
    print(f"--- Processing: {file_path} ---")
    if not doc_id:
        with open(file_path, "rb") as f:
            doc_id = hashlib.sha256(f.read()).hexdigest()[:16]
    source_name = source_name or os.path.basename(file_path)
    
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()
    
    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["source_name"] = source_name
    text_chunk_count = len(chunks)

    # 2b. Extract and Caption Images
    print("Searching for images/charts...")
    image_result = extract_images_and_caption(
        file_path,
        metadata_fields={"doc_id": doc_id, "source_name": source_name}
    )
    if image_result.chunks:
        print(f"Added {image_result.images_captioned} image captions to index.")
        chunks.extend(image_result.chunks)

    if image_result.warnings:
        print(
            "Vision ingestion completed with warnings: "
            f"{image_result.images_failed} of {image_result.images_found} image(s) failed."
        )
    
    print(f"Total processing units: {len(chunks)}")

    # 3. Setup Embeddings (Local Model)
    try:
        print("Initializing local embedding model (HuggingFace)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # 4. Connect to MongoDB and Store
        client = pymongo.MongoClient(MONGODB_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        collection.delete_many({"doc_id": doc_id})
        
        print("Storing embeddings in MongoDB Atlas...")
        vector_search = MongoDBAtlasVectorSearch.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection=collection,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
        )
        print("✅ Ingestion successful!")
        return IngestionResult(
            vector_search=vector_search,
            doc_id=doc_id,
            text_chunks=text_chunk_count,
            images_found=image_result.images_found,
            images_captioned=image_result.images_captioned,
            images_failed=image_result.images_failed,
            warnings=image_result.warnings,
        )
    # Convert embedding/database failures into the function's existing failure result.
    except Exception as e:  # noqa: BLE001
        print(f"❌ Ingestion failed: {e}")
        if "quota" in str(e).lower() or "429" in str(e):
            print("TIP: A provider rate limit was reached. Wait and try again.")
        return None


if __name__ == "__main__":
    # The file is in the current project folder
    resume_path = "Aman_Resume.pdf"
    if os.path.exists(resume_path):
        ingest_pdf(resume_path)
    else:
        print(f"❌ File not found: {resume_path}")
