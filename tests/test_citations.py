from unittest.mock import MagicMock

from langchain_core.documents import Document

import rag_engine


def test_retrieve_documents_preserves_page_metadata():
    document = Document(
        page_content="Revenue increased.",
        metadata={"source_name": "report.pdf", "page_number": 3},
    )
    engine = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
    engine.vector_search = MagicMock()
    engine.vector_search.similarity_search.return_value = [document]

    result = engine.retrieve_documents("What changed?")

    assert result == [document]
    assert result[0].metadata["source_name"] == "report.pdf"
    assert result[0].metadata["page_number"] == 3


def test_build_context_labels_retrieved_chunks():
    documents = [
        Document(
            page_content="Revenue increased.",
            metadata={"source_name": "report.pdf", "page_number": 3},
        )
    ]

    context = rag_engine.build_context(documents)

    assert "[Retrieved source 1: report.pdf, page 3]" in context
    assert "Revenue increased." in context


def test_build_sources_deduplicates_filename_and_page():
    documents = [
        Document(
            page_content="First chunk",
            metadata={"source_name": "report.pdf", "page_number": 2},
        ),
        Document(
            page_content="Second chunk",
            metadata={"source_name": "report.pdf", "page_number": 2},
        ),
        Document(
            page_content="Third chunk",
            metadata={"source_name": "appendix.pdf", "page_number": 1},
        ),
    ]

    sources = rag_engine.build_sources(documents)

    assert sources == ("\n\n**Sources**\n- report.pdf, page 2\n- appendix.pdf, page 1")


def test_build_sources_normalizes_legacy_page_metadata():
    documents = [
        Document(
            page_content="Text",
            metadata={"source_name": "report.pdf", "page": 0, "type": "text"},
        ),
        Document(
            page_content="Image",
            metadata={"source_name": "report.pdf", "page": 2, "type": "image"},
        ),
    ]

    sources = rag_engine.build_sources(documents)

    assert "- report.pdf, page 1" in sources
    assert "- report.pdf, page 2" in sources


def test_build_sources_does_not_expose_temporary_path():
    document = Document(
        page_content="Text",
        metadata={"source": "/tmp/private-upload.pdf", "page_number": 1},
    )

    sources = rag_engine.build_sources([document])

    assert sources == "\n\n**Sources**\n- Unknown document, page 1"
    assert "/tmp/" not in sources
