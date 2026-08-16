from types import SimpleNamespace
from unittest.mock import MagicMock

import rag_engine


def test_generate_answer_uses_configured_generation_model(monkeypatch):
    monkeypatch.setattr(rag_engine, "GROQ_GENERATION_MODEL", "configured-model")
    create = MagicMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))]
        )
    )
    engine = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
    engine.groq_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    engine.get_context = MagicMock(return_value="Retrieved PDF context")

    answer = engine.generate_answer("What does the chart show?", doc_ids=["doc-1"])

    assert answer == "Answer"
    assert create.call_args.kwargs["model"] == "configured-model"


def test_generate_answer_skips_provider_when_retrieval_is_empty():
    engine = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
    engine.groq_client = MagicMock()
    engine.get_context = MagicMock(return_value="")

    answer = engine.generate_answer("Unknown question", doc_ids=["doc-1"])

    assert "couldn't find relevant context" in answer
    engine.groq_client.chat.completions.create.assert_not_called()
