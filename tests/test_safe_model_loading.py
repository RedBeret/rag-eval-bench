"""The model wrapper must never opt into executing remote Python code."""

import sys
from types import SimpleNamespace

from rag.index.embedder import SentenceTransformerEmbedder


def test_model_loading_disables_remote_code_and_is_lazy(monkeypatch):
    calls = []

    class Model:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

        def encode(self, texts, **kwargs):
            return SimpleNamespace(tolist=lambda: [[1.0, 0.0] for _ in texts])

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    embedder = SentenceTransformerEmbedder("local-test-model")
    assert calls == []
    assert embedder.embed(["one"]) == [[1.0, 0.0]]
    assert embedder.embed(["two"]) == [[1.0, 0.0]]
    assert calls == [("local-test-model", {"trust_remote_code": False})]
