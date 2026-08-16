from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import rag_engine


def make_engine(response="Answer"):
    create = MagicMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
        )
    )
    engine = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
    engine.groq_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    engine.retrieve_documents = MagicMock(
        return_value=[
            Document(
                page_content="Retrieved PDF context",
                metadata={"source_name": "report.pdf", "page_number": 2},
            )
        ]
    )
    return engine, create


def test_build_prompt_requests_only_a_concise_final_answer():
    prompt = rag_engine.build_prompt("Document evidence", "What happened?")

    assert "Return only the final answer" in prompt
    assert "Do not include reasoning" in prompt
    assert "Do not create citations" in prompt
    assert "Do not infer facts" in prompt
    assert "When information is ambiguous" in prompt
    assert "Document evidence" in prompt
    assert "What happened?" in prompt


def test_generate_answer_uses_configured_generation_model(monkeypatch):
    monkeypatch.setattr(rag_engine, "GROQ_GENERATION_MODEL", "qwen/qwen3.6-27b")
    engine, create = make_engine()

    answer = engine.generate_answer("What does the chart show?", doc_ids=["doc-1"])

    assert answer == "Answer\n\n**Sources**\n- report.pdf, page 2"
    assert (
        "[Retrieved source 1: report.pdf, page 2]"
        in (create.call_args.kwargs["messages"][0]["content"])
    )
    assert create.call_args.kwargs["model"] == "qwen/qwen3.6-27b"
    assert create.call_args.kwargs["reasoning_effort"] == "none"
    assert (
        create.call_args.kwargs["max_completion_tokens"]
        == rag_engine.MAX_COMPLETION_TOKENS
    )


def test_generate_answer_does_not_send_qwen_reasoning_option_to_other_models(
    monkeypatch,
):
    monkeypatch.setattr(rag_engine, "GROQ_GENERATION_MODEL", "another-model")
    engine, create = make_engine()

    engine.generate_answer("What happened?")

    assert "reasoning_effort" not in create.call_args.kwargs


def test_generate_answer_rejects_empty_provider_response():
    engine, _ = make_engine(response="  ")

    with pytest.raises(rag_engine.GenerationError, match="empty response"):
        engine.generate_answer("What happened?")


def test_generate_answer_rejects_visible_reasoning():
    engine, _ = make_engine(response="<think>internal reasoning</think>Final answer")

    with pytest.raises(rag_engine.GenerationError, match="visible reasoning"):
        engine.generate_answer("What happened?")


def test_generate_answer_wraps_provider_failure():
    engine, create = make_engine()
    create.side_effect = RuntimeError("provider unavailable")

    with pytest.raises(rag_engine.GenerationError, match="invalid response"):
        engine.generate_answer("What happened?")


def test_generate_answer_skips_provider_when_retrieval_is_empty():
    engine = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
    engine.groq_client = MagicMock()
    engine.retrieve_documents = MagicMock(return_value=[])

    answer = engine.generate_answer("Unknown question", doc_ids=["doc-1"])

    assert "couldn't find relevant context" in answer
    engine.groq_client.chat.completions.create.assert_not_called()
