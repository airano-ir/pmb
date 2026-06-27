"""doctor diagnostics."""
from __future__ import annotations

import pmb.cli.doctor as doctor
from pmb.config import Config


def test_check_embedding_model_uses_configured_model(tmp_path, monkeypatch):
    """Regression: doctor must check the CONFIGURED embedding model, not a
    hardcoded all-MiniLM-L6-v2 - otherwise the default multilingual model
    always reports 'not cached' even when it is."""

    def _no_ws(*a, **k):
        raise RuntimeError("isolate: fall back to default Config()")

    # detect_workspace is imported inside the check; patch it on its module.
    monkeypatch.setattr("pmb.core.workspace.detect_workspace", _no_ws, raising=False)

    hf = tmp_path / "hf"
    monkeypatch.setenv("HF_HOME", str(hf))

    model = Config().get("embedding.model")  # default: multilingual MiniLM-L12
    assert "MiniLM" in model

    # The OLD hardcoded model (all-MiniLM-L6-v2) being cached must NOT satisfy
    # the check when a DIFFERENT model is configured.
    (hf / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2").mkdir(parents=True)
    res = doctor.check_embedding_model()
    assert res["status"] == "warn"
    assert model in res["msg"]

    # Caching the CONFIGURED model satisfies it.
    (hf / "hub" / ("models--" + model.replace("/", "--"))).mkdir(parents=True)
    res2 = doctor.check_embedding_model()
    assert res2["status"] == "ok"
    assert model in res2["msg"]


def test_check_embedding_model_skips_hf_check_for_non_st_backend(tmp_path, monkeypatch):
    """With an ollama/fastembed embedding backend there is no HuggingFace cache
    to look for, so the check should report ok instead of a false warning."""

    def _no_ws(*a, **k):
        raise RuntimeError("isolate")

    monkeypatch.setattr("pmb.core.workspace.detect_workspace", _no_ws, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty_hf"))

    # Force a non-sentence-transformers backend via a stub config.
    real_get = Config.get

    def _patched_get(self, key):
        if key == "embedding.backend":
            return "ollama"
        return real_get(self, key)

    monkeypatch.setattr(Config, "get", _patched_get)
    res = doctor.check_embedding_model()
    assert res["status"] == "ok"
    assert "ollama" in res["msg"]
