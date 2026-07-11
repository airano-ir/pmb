"""Tests for the agent-wrapper scaffold. No network calls."""
from __future__ import annotations

from pmb.agent_wrapper.budget import TokenBudget
from pmb.agent_wrapper.policy import DropOldestNarrative


def test_budget_counts_messages_monotonically():
    b = TokenBudget(window=1000, target_max=0.75)
    n1 = b.count_messages([{"role": "user", "content": "hello"}])
    n2 = b.count_messages([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ])
    assert n2 > n1


def test_budget_threshold_calculated():
    b = TokenBudget(window=1000, target_max=0.5)
    assert b.threshold == 500


def test_budget_should_compact_under_threshold():
    b = TokenBudget(window=100_000, target_max=0.75)
    msgs = [{"role": "user", "content": "small"}]
    assert b.should_compact(msgs) is False


def test_budget_should_compact_over_threshold():
    b = TokenBudget(window=200, target_max=0.5)  # threshold = 100 tokens ≈ 350 chars
    big = "x" * 5000
    msgs = [{"role": "user", "content": big}]
    assert b.should_compact(msgs) is True


def test_drop_oldest_keeps_last_n():
    b = TokenBudget(window=200, target_max=0.5)
    p = DropOldestNarrative(b, keep_last_n=3)
    msgs = [
        {"role": "user", "content": "x" * 1000} for _ in range(10)
    ]
    out = p.compact(msgs)
    # At least keep_last_n preserved
    assert len(out) >= 3
    # Last messages preserved (same dicts)
    assert out[-1] is msgs[-1]
    assert out[-3] is msgs[-3]


def test_drop_oldest_noop_under_budget():
    b = TokenBudget(window=100_000, target_max=0.75)
    p = DropOldestNarrative(b)
    msgs = [{"role": "user", "content": "small"}]
    assert p.compact(msgs) == msgs


def test_loop_module_imports_without_anthropic_call():
    """Importing should not fail and not contact any network."""
    from pmb.agent_wrapper import loop  # noqa
    from pmb.agent_wrapper.loop import AgentConfig
    cfg = AgentConfig(model="haiku", window=1000)
    assert cfg.model == "haiku"
    assert cfg.window == 1000


# ----------------------------------------------------------------------
# Selective compression policy
# ----------------------------------------------------------------------


def test_classify_decision_message():
    from pmb.agent_wrapper.policy import classify_message
    msg = {"role": "user", "content": "We decided to go with Postgres 17"}
    assert classify_message(msg) == "decision"


def test_classify_fact_message():
    from pmb.agent_wrapper.policy import classify_message
    msg = {"role": "assistant", "content": "The answer is to use a connection pool"}
    assert classify_message(msg) == "fact"


def test_classify_fact_explicit_marker():
    from pmb.agent_wrapper.policy import classify_message
    msg = {"role": "user", "content": "[FACT] db = Postgres 17"}
    assert classify_message(msg) == "fact"


def test_classify_error_message():
    from pmb.agent_wrapper.policy import classify_message
    msg = {"role": "assistant",
           "content": "Traceback (most recent call last): File ..."}
    assert classify_message(msg) == "error"


def test_classify_narrative_default():
    from pmb.agent_wrapper.policy import classify_message
    msg = {"role": "user", "content": "Could you also check the formatting"}
    assert classify_message(msg) == "narrative"


def test_classify_system_role():
    from pmb.agent_wrapper.policy import classify_message
    assert classify_message({"role": "system", "content": "..."}) == "system"


def test_classify_tool_result_blocks():
    from pmb.agent_wrapper.policy import classify_message
    msg = {"role": "user", "content": [
        {"type": "tool_result", "content": "ok"}
    ]}
    assert classify_message(msg) == "tool_result"


def test_selective_keeps_decisions_drops_narrative():
    from pmb.agent_wrapper.budget import TokenBudget
    from pmb.agent_wrapper.policy import SelectivePolicy
    budget = TokenBudget(window=2000, target_max=0.5)  # threshold ≈ 1000 tokens
    # Small max_summary_chars so summaries are well under threshold and survive backoff
    policy = SelectivePolicy(budget, keep_last_n=2, max_summary_chars=120)

    msgs = [
        {"role": "user", "content": "x" * 3000},  # large narrative
        {"role": "user", "content": "we decided to use Postgres 17"},  # decision
        {"role": "user", "content": "y" * 3000},  # large narrative
        {"role": "user", "content": "the answer is X"},  # fact
        {"role": "user", "content": "last 1"},
        {"role": "user", "content": "last 2"},
    ]
    out = policy.compact(msgs)
    contents = [
        m.get("content") if isinstance(m.get("content"), str) else ""
        for m in out
    ]
    # decision and fact are preserved
    assert any("Postgres" in c for c in contents)
    assert any("the answer is X" in c for c in contents)
    # last 2 are preserved
    assert any("last 1" in c for c in contents)
    assert any("last 2" in c for c in contents)
    # at least one summary was produced
    assert any(c.startswith("[COMPRESSED NARRATIVE]") for c in contents)
    # The original 1500-char narratives no longer present verbatim
    assert not any(c.startswith("x" * 100) for c in contents)


def test_selective_no_compaction_under_budget():
    from pmb.agent_wrapper.budget import TokenBudget
    from pmb.agent_wrapper.policy import SelectivePolicy
    budget = TokenBudget(window=100_000, target_max=0.75)
    policy = SelectivePolicy(budget)
    msgs = [{"role": "user", "content": "small"}]
    assert policy.compact(msgs) == msgs


def test_selective_uses_llm_for_summary_when_provided():
    from pmb.agent_wrapper.budget import TokenBudget
    from pmb.agent_wrapper.policy import SelectivePolicy

    class _StubLLM:
        called = False
        def consolidate(self, texts):
            _StubLLM.called = True
            return {"consolidate": True, "summary": "AI-summarized content",
                    "confidence": 0.9, "reasoning": ""}

    budget = TokenBudget(window=400, target_max=0.5)
    policy = SelectivePolicy(budget, llm=_StubLLM(), keep_last_n=2,
                              max_summary_chars=80)
    msgs = [{"role": "user", "content": "x" * 800}] * 4 + [
        {"role": "user", "content": "we decided X"},
        {"role": "user", "content": "tail"},
    ]
    out = policy.compact(msgs)
    assert _StubLLM.called
    assert any(
        isinstance(m.get("content"), str)
        and "AI-summarized content" in m["content"]
        for m in out
    )


def test_ollama_transport_calls_api_chat(monkeypatch):
    """AgentLoop with transport=ollama hits /api/chat on the configured URL."""
    import json
    import urllib.request as urlreq

    from pmb.agent_wrapper.loop import AgentConfig, AgentLoop

    captured = {}

    class _Resp:
        def __enter__(self_inner): return self_inner
        def __exit__(self_inner, *a): return False
        def read(self_inner):
            return json.dumps({"message": {"role": "assistant", "content": "hi from llama"}}).encode("utf-8")

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr(urlreq, "urlopen", _fake_urlopen)

    # Use a stub engine so no real PMB workspace is touched
    class _StubEngine:
        class workspace:
            id = "test"
            name = "test"
        def recall(self, q, top_k=5):
            class _P:
                results = []
                def to_text(self_inner, max_results=5): return ""
            return _P()
        def remember(self, q, r): pass

    cfg = AgentConfig(
        model="llama3.1:8b",
        transport="ollama",
        ollama_url="http://remote-server:11434",
        selective_compression=False,
        persist_turns=False,
    )
    loop = AgentLoop(engine=_StubEngine(), config=cfg)
    out = loop.turn("hello")

    assert out == "hi from llama"
    assert captured["url"] == "http://remote-server:11434/api/chat"
    assert captured["body"]["model"] == "llama3.1:8b"
    assert captured["body"]["stream"] is False
    # system prompt + user message
    roles = [m["role"] for m in captured["body"]["messages"]]
    assert roles[0] == "system"
    assert "user" in roles


def test_openai_transport_calls_chat_completions(monkeypatch):
    """AgentLoop with transport=openai hits chat completions with no SDK."""
    import json
    import urllib.request as urlreq

    from pmb.agent_wrapper.loop import AgentConfig, AgentLoop

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("PMB_OPENAI_MODEL", raising=False)
    captured = {}

    class _Resp:
        def __enter__(self_inner): return self_inner
        def __exit__(self_inner, *a): return False
        def read(self_inner):
            return json.dumps({"choices": [{"message": {"content": "hi from openai"}}]}).encode("utf-8")

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["auth"] = req.get_header("Authorization")
        return _Resp()

    monkeypatch.setattr(urlreq, "urlopen", _fake_urlopen)

    class _StubEngine:
        class workspace:
            id = "test"
            name = "test"
        def recall(self, q, top_k=5):
            class _P:
                results = []
                def to_text(self_inner, max_results=5): return ""
            return _P()
        def remember(self, q, r): pass

    cfg = AgentConfig(
        model="haiku",
        transport="openai",
        selective_compression=False,
        persist_turns=False,
    )
    out = AgentLoop(engine=_StubEngine(), config=cfg).turn("hello")

    assert out == "hi from openai"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["model"] == "gpt-4o-mini"
    roles = [m["role"] for m in captured["body"]["messages"]]
    assert roles[0] == "system"
    assert "user" in roles


def test_ollama_transport_unreachable_raises(monkeypatch):
    import urllib.error
    import urllib.request as urlreq

    from pmb.agent_wrapper.loop import AgentConfig, AgentLoop

    def _fail(req, timeout=None):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(urlreq, "urlopen", _fail)

    class _StubEngine:
        class workspace:
            id = "test"; name = "test"
        def recall(self, q, top_k=5):
            class _P:
                results = []
                def to_text(self_inner, max_results=5): return ""
            return _P()
        def remember(self, q, r): pass

    cfg = AgentConfig(
        transport="ollama",
        ollama_url="http://does-not-exist:11434",
        selective_compression=False,
        persist_turns=False,
    )
    loop = AgentLoop(engine=_StubEngine(), config=cfg)
    try:
        loop.turn("hi")
    except RuntimeError as e:
        assert "unreachable" in str(e).lower()
    else:
        raise AssertionError("expected RuntimeError")


def test_selective_falls_back_when_llm_raises():
    from pmb.agent_wrapper.budget import TokenBudget
    from pmb.agent_wrapper.policy import SelectivePolicy

    class _BadLLM:
        def consolidate(self, texts):
            raise RuntimeError("llm broke")

    budget = TokenBudget(window=400, target_max=0.5)
    policy = SelectivePolicy(budget, llm=_BadLLM(), keep_last_n=2,
                              max_summary_chars=80)
    msgs = [{"role": "user", "content": "x" * 800}] * 3 + [
        {"role": "user", "content": "tail 1"},
        {"role": "user", "content": "tail 2"},
    ]
    out = policy.compact(msgs)
    # Should still produce a summary via heuristic fallback
    contents = [
        m.get("content") if isinstance(m.get("content"), str) else ""
        for m in out
    ]
    assert any(c.startswith("[COMPRESSED NARRATIVE]") for c in contents)
