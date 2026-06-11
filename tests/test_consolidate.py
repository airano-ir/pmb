"""Tests for LLM-based consolidation. Uses MockLLM — no API calls."""
from __future__ import annotations

from pmb.core.engine import Engine
from pmb.health.consolidate import (
    _parse_llm_json,
    cluster_events,
)


class MockLLM:
    """Deterministic stand-in for the Anthropic client."""

    def __init__(self, *, consolidate: bool = True, summary: str = "Generalized rule",
                 confidence: float = 0.9, reasoning: str = "they all say the same thing",
                 raise_error: bool = False):
        self._consolidate = consolidate
        self._summary = summary
        self._confidence = confidence
        self._reasoning = reasoning
        self._raise = raise_error
        self.calls: list[list[str]] = []

    def consolidate(self, events_text: list[str]) -> dict:
        self.calls.append(list(events_text))
        if self._raise:
            raise RuntimeError("simulated LLM failure")
        return {
            "consolidate": self._consolidate,
            "summary": self._summary,
            "confidence": self._confidence,
            "reasoning": self._reasoning,
        }


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------


def test_parse_clean_json():
    out = _parse_llm_json('{"consolidate": true, "summary": "x", "confidence": 0.9, "reasoning": "y"}')
    assert out["consolidate"] is True
    assert out["summary"] == "x"
    assert out["confidence"] == 0.9


def test_parse_fenced_json():
    out = _parse_llm_json('```json\n{"consolidate": false, "summary": "", "confidence": 0.1, "reasoning": "no"}\n```')
    assert out["consolidate"] is False
    assert out["confidence"] == 0.1


def test_parse_with_prose():
    out = _parse_llm_json('Sure! Here is my answer:\n{"consolidate": true, "summary": "rule", "confidence": 0.7, "reasoning": "obvious"}')
    assert out["summary"] == "rule"


def test_parse_garbage_falls_back():
    out = _parse_llm_json("this is not json at all")
    assert out["consolidate"] is False
    assert out["confidence"] == 0.0


# ----------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------


def test_cluster_empty_workspace_returns_nothing(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    assert cluster_events(eng) == []


def test_cluster_groups_similar_events(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    # Four very similar fact statements + one unrelated
    eng.record_fact("user prefers no comments in code")
    eng.record_fact("don't add docstrings, user thinks they're noise")
    eng.record_fact("strip comments before commit per user preference")
    eng.record_fact("comments are not welcome in this repo")
    eng.record_fact("unrelated: deploy with kubectl apply -f")

    clusters = cluster_events(eng, similarity_threshold=0.3, min_cluster_size=3)
    assert len(clusters) >= 1
    # The first cluster should be the "no comments" cluster, size >= 3
    top = max(clusters, key=lambda c: len(c))
    assert len(top) >= 3


def test_cluster_respects_min_size(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    # Only 2 similar — below min_cluster_size=3
    eng.record_fact("user uses postgres")
    eng.record_fact("postgres on port 5432")
    assert cluster_events(eng, min_cluster_size=3) == []


def test_cluster_respects_threshold(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("user prefers no comments")
    eng.record_fact("deploy via terraform")
    eng.record_fact("postgres on port 5432")
    # Very high threshold — none of these are highly similar
    assert cluster_events(eng, similarity_threshold=0.99, min_cluster_size=2) == []


# ----------------------------------------------------------------------
# Run consolidation
# ----------------------------------------------------------------------


def test_consolidation_stores_fact_and_archives_sources(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulids = [
        eng.record_fact("user prefers no comments in code"),
        eng.record_fact("don't add docstrings, user dislikes them"),
        eng.record_fact("strip comments before commit"),
    ]
    llm = MockLLM(consolidate=True, summary="User prefers code without comments.", confidence=0.9)

    result = eng.consolidate(llm=llm, similarity_threshold=0.3, min_cluster_size=3)

    assert result["n_clusters_found"] >= 1
    assert result["n_consolidated"] >= 1
    # The new fact exists
    rs = result["results"]
    stored = [r for r in rs if r["consolidated"]]
    assert stored
    new_ulid = stored[0]["new_ulid"]
    new_ev = eng.events.get_by_ulid(new_ulid)
    assert new_ev is not None
    assert "comments" in new_ev.content.lower()
    assert new_ev.importance == 0.85
    assert new_ev.metadata["consolidated_from"]
    # Sources archived
    archived = stored[0]["archived_source_ulids"]
    assert len(archived) >= 3
    for u in archived:
        ev = eng.events.get_by_ulid(u)
        assert ev.archived_at is not None


def test_consolidation_dry_run_does_not_write(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("user prefers no comments in code")
    eng.record_fact("don't add docstrings")
    eng.record_fact("strip comments before commit")
    llm = MockLLM(consolidate=True, summary="No comments.", confidence=0.9)

    before = eng.events.count(eng.workspace.id)
    result = eng.consolidate(llm=llm, similarity_threshold=0.3,
                              min_cluster_size=3, dry_run=True)
    after = eng.events.count(eng.workspace.id)
    assert before == after  # nothing stored
    assert result["n_archived"] == 0
    # But the LLM was asked
    assert len(llm.calls) >= 1


def test_consolidation_skips_low_confidence(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("user prefers no comments in code")
    eng.record_fact("don't add docstrings")
    eng.record_fact("strip comments before commit")
    llm = MockLLM(consolidate=True, summary="Maybe?", confidence=0.3)

    result = eng.consolidate(llm=llm, similarity_threshold=0.3, min_cluster_size=3)
    assert result["n_consolidated"] == 0
    # Sources NOT archived when not consolidated
    assert result["n_archived"] == 0


def test_consolidation_skips_when_llm_says_no(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("user prefers no comments")
    eng.record_fact("don't add docstrings")
    eng.record_fact("strip comments")
    llm = MockLLM(consolidate=False, summary="", confidence=0.95)

    result = eng.consolidate(llm=llm, similarity_threshold=0.3, min_cluster_size=3)
    assert result["n_consolidated"] == 0
    assert result["n_archived"] == 0


def test_consolidation_handles_llm_error(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("a")
    eng.record_fact("a")
    eng.record_fact("a")
    llm = MockLLM(raise_error=True)
    result = eng.consolidate(llm=llm, similarity_threshold=0.4, min_cluster_size=3)
    # Result records the failure but doesn't crash
    assert result["n_consolidated"] == 0
    if result["results"]:
        assert "llm error" in result["results"][0]["reasoning"]


def test_ollama_ping_failure_returns_false(monkeypatch):
    """Ping should not raise on connection failure — used in auto-detection."""
    from pmb.health.consolidate import OllamaClient
    # Point at a port nothing's on
    assert OllamaClient.ping(base_url="http://127.0.0.1:1", timeout=0.5) is False


def test_ollama_consolidate_via_mocked_urlopen(monkeypatch):
    """Drive OllamaClient.consolidate with a mocked HTTP layer — no network."""
    import json as _json
    import urllib.request

    from pmb.health.consolidate import OllamaClient

    expected = {
        "response": _json.dumps({
            "consolidate": True,
            "summary": "User prefers no comments.",
            "confidence": 0.9,
            "reasoning": "they all say the same",
        })
    }

    class _Resp:
        status = 200
        def read(self_inner):
            return _json.dumps(expected).encode("utf-8")
        def __enter__(self_inner): return self_inner
        def __exit__(self_inner, *a): return False

    def _fake_urlopen(req, timeout=None):
        # Sanity: we hit /api/generate with a POST
        assert req.full_url.endswith("/api/generate")
        assert req.method == "POST"
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    client = OllamaClient(model="llama3.1:8b")
    out = client.consolidate(["one", "two", "three"])
    assert out["consolidate"] is True
    assert "comments" in out["summary"].lower()
    assert out["confidence"] == 0.9


def test_ollama_unreachable_raises_runtimeerror(monkeypatch):
    import urllib.error
    import urllib.request

    from pmb.health.consolidate import OllamaClient

    def _fail(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    client = OllamaClient()
    try:
        client.consolidate(["a"])
    except RuntimeError as e:
        assert "Ollama unreachable" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_resolve_auto_falls_back_to_ollama(monkeypatch):
    """No claude on PATH, no API key, ollama up → ollama."""
    from pmb.health import consolidate as C
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(C.ClaudeCLIClient, "available", staticmethod(lambda *a, **kw: False))
    monkeypatch.setattr(C.OllamaClient, "ping", staticmethod(lambda *a, **kw: True))
    client = C.resolve_llm_client(backend="auto")
    assert isinstance(client, C.OllamaClient)


def test_resolve_auto_raises_when_nothing_available(monkeypatch):
    from pmb.health import consolidate as C
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(C.ClaudeCLIClient, "available", staticmethod(lambda *a, **kw: False))
    monkeypatch.setattr(C.OllamaClient, "ping", staticmethod(lambda *a, **kw: False))
    try:
        C.resolve_llm_client(backend="auto")
    except RuntimeError as e:
        msg = str(e)
        assert "ANTHROPIC_API_KEY" in msg
        assert "ollama" in msg.lower()
        assert "claude" in msg.lower()
    else:
        raise AssertionError("expected RuntimeError")


def test_resolve_explicit_ollama_when_down(monkeypatch):
    from pmb.health import consolidate as C
    monkeypatch.setattr(C.OllamaClient, "ping", staticmethod(lambda *a, **kw: False))
    try:
        C.resolve_llm_client(backend="ollama")
    except RuntimeError as e:
        assert "not reachable" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_claude_cli_available_check(monkeypatch):
    import shutil

    from pmb.health.consolidate import ClaudeCLIClient
    monkeypatch.setattr(shutil, "which", lambda cmd: "/fake/path/claude" if cmd == "claude" else None)
    assert ClaudeCLIClient.available() is True
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    assert ClaudeCLIClient.available() is False


def test_claude_cli_consolidate_via_mocked_subprocess(monkeypatch):
    """Mock subprocess so we never spawn a real claude process."""
    import subprocess as sp

    from pmb.health.consolidate import ClaudeCLIClient

    expected = '{"consolidate": true, "summary": "User prefers no comments.", "confidence": 0.9, "reasoning": "all three say so"}'

    class _R:
        returncode = 0
        stdout = expected
        stderr = ""

    captured = {}
    def _fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["timeout"] = kwargs.get("timeout")
        return _R()

    monkeypatch.setattr(sp, "run", _fake_run)
    client = ClaudeCLIClient(command="claude", model="haiku", timeout=60)
    out = client.consolidate(["msg1", "msg2", "msg3"])
    assert out["consolidate"] is True
    assert "comments" in out["summary"].lower()
    # Sanity: argv shape
    assert captured["argv"][0] == "claude"
    assert "-p" in captured["argv"]
    assert "--model" in captured["argv"]
    assert "haiku" in captured["argv"]
    assert "--no-session-persistence" in captured["argv"]


def test_claude_cli_propagates_exit_code(monkeypatch):
    import subprocess as sp

    from pmb.health.consolidate import ClaudeCLIClient
    class _R:
        returncode = 1
        stdout = ""
        stderr = "Some error happened on stderr"
    monkeypatch.setattr(sp, "run", lambda *a, **kw: _R())
    client = ClaudeCLIClient()
    try:
        client.consolidate(["x"])
    except RuntimeError as e:
        assert "exited 1" in str(e)
        assert "Some error" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_claude_cli_handles_missing_binary(monkeypatch):
    import subprocess as sp

    from pmb.health.consolidate import ClaudeCLIClient
    def _fail(*a, **kw):
        raise FileNotFoundError("claude not found")
    monkeypatch.setattr(sp, "run", _fail)
    try:
        ClaudeCLIClient().consolidate(["x"])
    except RuntimeError as e:
        assert "PATH" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_claude_cli_handles_timeout(monkeypatch):
    import subprocess as sp

    from pmb.health.consolidate import ClaudeCLIClient
    def _timeout(*a, **kw):
        raise sp.TimeoutExpired(cmd="claude", timeout=1)
    monkeypatch.setattr(sp, "run", _timeout)
    try:
        ClaudeCLIClient(timeout=1).consolidate(["x"])
    except RuntimeError as e:
        assert "timed out" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_resolve_auto_prefers_claude_cli_over_anthropic(monkeypatch):
    """When `claude` is on PATH, auto picks it even if API key is set."""
    from pmb.health import consolidate as C
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(C.ClaudeCLIClient, "available", staticmethod(lambda *a, **kw: True))
    client = C.resolve_llm_client(backend="auto")
    assert isinstance(client, C.ClaudeCLIClient)


def test_resolve_auto_falls_through_claude_to_anthropic(monkeypatch):
    """No claude in PATH but API key set → anthropic."""
    from pmb.health import consolidate as C
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(C.ClaudeCLIClient, "available", staticmethod(lambda *a, **kw: False))
    monkeypatch.setattr(C.AnthropicHaikuClient, "__init__", lambda self, *a, **kw: None)
    monkeypatch.setattr(C.OllamaClient, "ping", staticmethod(lambda *a, **kw: False))
    client = C.resolve_llm_client(backend="auto")
    assert isinstance(client, C.AnthropicHaikuClient)


def test_resolve_explicit_claude_when_missing(monkeypatch):
    from pmb.health import consolidate as C
    monkeypatch.setattr(C.ClaudeCLIClient, "available", staticmethod(lambda *a, **kw: False))
    try:
        C.resolve_llm_client(backend="claude")
    except RuntimeError as e:
        assert "not in PATH" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_resolve_unknown_backend_raises():
    from pmb.health import consolidate as C
    try:
        C.resolve_llm_client(backend="gpt5")
    except ValueError as e:
        assert "unknown backend" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_consolidation_preserves_pinned_sources(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    pinned = eng.record_fact("user prefers no comments")
    eng.pin(pinned)
    eng.record_fact("don't add docstrings")
    eng.record_fact("strip comments")
    llm = MockLLM(consolidate=True, summary="No comments preferred.", confidence=0.9)

    eng.consolidate(llm=llm, similarity_threshold=0.3, min_cluster_size=3)
    # Pinned event should remain active
    p = eng.events.get_by_ulid(pinned)
    assert p.archived_at is None
    assert p.importance >= 0.99
