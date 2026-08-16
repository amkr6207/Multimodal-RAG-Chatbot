from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import ingest_data


class FakePdfDocument:
    def __init__(self):
        self.pages = [FakePage([(11,), (22,)])]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index):
        return self.pages[index]

    def extract_image(self, xref):
        images = {
            11: {"image": b"png-image", "ext": "png"},
            22: {"image": b"jpeg-image", "ext": "jpg"},
        }
        return images[xref]


class FakePage:
    def __init__(self, images):
        self.images = images

    def get_images(self, full=True):
        assert full is True
        return self.images


def test_get_image_caption_uses_configured_model_and_mime_type():
    create = MagicMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  A chart.  "))]
        )
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    caption = ingest_data.get_image_caption(
        b"image-data",
        mime_type="image/png",
        client=client,
        model="vision-model",
    )

    assert caption == "A chart."
    request = create.call_args.kwargs
    assert request["model"] == "vision-model"
    image_url = request["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")


def test_get_image_caption_exposes_provider_failure():
    create = MagicMock(side_effect=RuntimeError("model unavailable"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(ingest_data.VisionCaptionError, match="model unavailable"):
        ingest_data.get_image_caption(
            b"image-data",
            client=client,
            model="retired-model",
        )


def test_get_image_caption_rejects_empty_provider_response():
    create = MagicMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  "))]
        )
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(ingest_data.VisionCaptionError, match="empty caption"):
        ingest_data.get_image_caption(
            b"image-data",
            client=client,
            model="vision-model",
        )


def test_extract_images_reports_partial_success(monkeypatch):
    monkeypatch.setattr(ingest_data.fitz, "open", lambda _: FakePdfDocument())
    captioner = MagicMock(
        side_effect=[
            "Revenue increased to 25 lakh in 2024.",
            ingest_data.VisionCaptionError("rate limited"),
        ]
    )

    result = ingest_data.extract_images_and_caption(
        "report.pdf",
        metadata_fields={"doc_id": "doc-1", "source_name": "report.pdf"},
        captioner=captioner,
    )

    assert result.images_found == 2
    assert result.images_captioned == 1
    assert result.images_failed == 1
    assert "rate limited" in result.warnings[0]
    assert result.chunks[0].page_content.startswith("[Image/Chart Description]")
    assert result.chunks[0].metadata == {
        "source": "report.pdf",
        "page": 1,
        "type": "image",
        "doc_id": "doc-1",
        "source_name": "report.pdf",
    }
    assert captioner.call_args_list[0].kwargs["mime_type"] == "image/png"
    assert captioner.call_args_list[1].kwargs["mime_type"] == "image/jpeg"


def test_ingest_pdf_returns_warning_summary_without_external_calls(monkeypatch):
    text_chunk = Document(page_content="Revenue increased.", metadata={"page": 0})
    image_chunk = Document(
        page_content="[Image/Chart Description]: Revenue chart.",
        metadata={"page": 2, "type": "image"},
    )
    image_result = ingest_data.ImageExtractionResult(
        chunks=[image_chunk],
        images_found=2,
        warnings=["Image 2 on page 2 could not be captioned: rate limited"],
    )

    loader = MagicMock()
    loader.load.return_value = [Document(page_content="Revenue increased.")]
    monkeypatch.setattr(ingest_data, "PyPDFLoader", MagicMock(return_value=loader))

    splitter = MagicMock()
    splitter.split_documents.return_value = [text_chunk]
    monkeypatch.setattr(
        ingest_data,
        "RecursiveCharacterTextSplitter",
        MagicMock(return_value=splitter),
    )
    monkeypatch.setattr(
        ingest_data,
        "extract_images_and_caption",
        MagicMock(return_value=image_result),
    )
    monkeypatch.setattr(ingest_data, "HuggingFaceEmbeddings", MagicMock())

    collection = MagicMock()
    database = MagicMock()
    database.__getitem__.return_value = collection
    mongo_client = MagicMock()
    mongo_client.__getitem__.return_value = database
    monkeypatch.setattr(
        ingest_data.pymongo,
        "MongoClient",
        MagicMock(return_value=mongo_client),
    )

    vector_search = object()
    store_documents = MagicMock(return_value=vector_search)
    monkeypatch.setattr(
        ingest_data.MongoDBAtlasVectorSearch,
        "from_documents",
        store_documents,
    )

    result = ingest_data.ingest_pdf(
        "report.pdf",
        doc_id="doc-1",
        source_name="report.pdf",
    )

    assert result.status == "partial_success"
    assert result.text_chunks == 1
    assert result.images_found == 2
    assert result.images_captioned == 1
    assert result.images_failed == 1
    assert "rate limited" in result.warnings[0]
    assert text_chunk.metadata["doc_id"] == "doc-1"
    assert text_chunk.metadata["source_name"] == "report.pdf"
    collection.delete_many.assert_called_once_with({"doc_id": "doc-1"})
    stored_documents = store_documents.call_args.kwargs["documents"]
    assert stored_documents == [text_chunk, image_chunk]
