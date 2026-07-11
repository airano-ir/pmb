"""
Minimal chat loop: LLM backend + PMB for memory.

What it does right now:

  1. Bind to current PMB workspace (auto-detect from cwd).
  2. On startup, fetch a recall pack for the first user message and inject
     it into the system prompt.
  3. Talk to the configured LLM in turns. Every assistant reply is persisted via
     `engine.remember(user, assistant)`.
  4. Before each turn, check budget; if over, run the compression policy.
  5. On exit (Ctrl-D / Ctrl-C / `/exit`), optionally run `pmb consolidate`.

This is enough to demonstrate end-to-end "agent + PMB + custom compaction"
on a tiny scale. It is NOT a Claude Code replacement - no tool use, no
file editing, no multi-turn tool chaining. See `PLAN.md` for what's
missing.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from pmb.agent_wrapper.budget import TokenBudget
from pmb.agent_wrapper.policy import (
    CompressionPolicy,
    DropOldestNarrative,
    SelectivePolicy,
)
from pmb.core.engine import Engine

SYSTEM_BASE = """\
You are a coding assistant with persistent project memory.
You have access to a local PMB store via injected recall context below.

When the user asks a project-specific question, the latest recall pack is
already in the system prompt - use it, cite specific points, do not
invent details that aren't in either the recall or the live conversation.
"""


@dataclass
class AgentConfig:
    model: str = "haiku"
    window: int = 200_000
    target_max: float = 0.75
    recall_top_k: int = 5
    persist_turns: bool = True
    consolidate_on_exit: bool = False
    transport: str = "auto"  # auto | claude | anthropic | openai | ollama
    selective_compression: bool = True
    ollama_url: str | None = None  # honours PMB_OLLAMA_URL / OLLAMA_HOST


@dataclass
class ChatTurn:
    role: str
    content: str
    importance: float = 0.5


class AgentLoop:
    def __init__(
        self,
        engine: Engine | None = None,
        config: AgentConfig | None = None,
        policy: CompressionPolicy | None = None,
    ):
        self.engine = engine or Engine()
        self.config = config or AgentConfig()
        self.budget = TokenBudget(
            window=self.config.window, target_max=self.config.target_max,
        )
        # Selective policy needs an LLM for summarization - wire the same
        # backend selector consolidation uses, so both share a transport.
        if policy is not None:
            self.policy = policy
        elif self.config.selective_compression:
            llm = self._try_make_summary_llm()
            self.policy = SelectivePolicy(self.budget, engine=self.engine, llm=llm)
        else:
            self.policy = DropOldestNarrative(self.budget)
        self.messages: list[dict] = []
        self._system: str = SYSTEM_BASE
        self._transport = self._resolve_transport()

    # -------- Transport resolution --------
    def _resolve_transport(self) -> str:
        """Pick how we actually call the model. Mirrors consolidate's logic
        but cached once per session."""
        import os
        import shutil
        t = (self.config.transport or "auto").lower()
        if t == "auto":
            if shutil.which("claude"):
                return "claude"
            if os.environ.get("ANTHROPIC_API_KEY"):
                return "anthropic"
            if os.environ.get("OPENAI_API_KEY"):
                return "openai"
            # Last fallback: Ollama, if a server responds
            from pmb.health.consolidate import OllamaClient
            if OllamaClient.ping(self.config.ollama_url or os.environ.get("PMB_OLLAMA_URL") or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"):
                return "ollama"
            return "anthropic"  # will error on call if no key; caller sees clear msg
        return t

    def _try_make_summary_llm(self):
        """Best-effort LLM for narrative summarization in SelectivePolicy.

        Falls back to None - SelectivePolicy will use a head+tail heuristic
        when no LLM is available.
        """
        try:
            from pmb.health.consolidate import resolve_llm_client
            return resolve_llm_client(backend=self.config.transport)
        except Exception:
            return None

    # -------- LLM call --------
    def _call_llm(self) -> str:
        if self._transport == "claude":
            return self._call_claude_cli()
        if self._transport == "ollama":
            return self._call_ollama()
        if self._transport == "openai":
            return self._call_openai_api()
        return self._call_anthropic_api()

    def _call_ollama(self) -> str:
        """Talk to Ollama at config.ollama_url (or env vars) via /api/chat.

        Honours `PMB_OLLAMA_URL`, `OLLAMA_HOST`, and PMB_OLLAMA_MODEL the
        same way the consolidation backend does. Multi-turn natively
        supported here because /api/chat takes a `messages` list.
        """
        import json
        import os
        import urllib.error
        import urllib.request

        base = (
            self.config.ollama_url
            or os.environ.get("PMB_OLLAMA_URL")
            or os.environ.get("OLLAMA_HOST")
            or "http://localhost:11434"
        ).rstrip("/")
        model = (
            self.config.model
            if self.config.model not in ("haiku", "sonnet", "opus")
            else os.environ.get("PMB_OLLAMA_MODEL") or "llama3.1:8b"
        )
        # Flatten our messages list (which may contain content blocks) to plain text
        flat: list[dict] = []
        for m in self.messages:
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            flat.append({"role": m.get("role", "user"), "content": str(content)})

        payload = {
            "model": model,
            "stream": False,
            "messages": [{"role": "system", "content": self._system}, *flat],
            "options": {"temperature": 0.4},
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/chat",
            data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama unreachable at {base} ({e}). "
                f"Start it with `ollama serve` (and set PMB_OLLAMA_URL "
                f"to its address if it's on another host)."
            )
        data = json.loads(raw)
        msg = data.get("message") or {}
        return msg.get("content", "")

    def _call_anthropic_api(self) -> str:
        from anthropic import Anthropic
        client = Anthropic()
        # Map common short names to full versioned ids for the API
        model_id = {
            "haiku": "claude-haiku-4-5",
            "sonnet": "claude-sonnet-4-6",
            "opus": "claude-opus-4-5",
        }.get(self.config.model, self.config.model)
        resp = client.messages.create(
            model=model_id,
            max_tokens=2048,
            system=self._system,
            messages=self.messages,
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text"))

    def _call_openai_api(self) -> str:
        from pmb.health.consolidate import openai_chat_completion

        flat: list[dict[str, str]] = []
        for m in self.messages:
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            flat.append({"role": m.get("role", "user"), "content": str(content)})
        return openai_chat_completion(
            [{"role": "system", "content": self._system}, *flat],
            model=self.config.model,
            max_tokens=2048,
            temperature=0.4,
            timeout=180,
        )

    def _call_claude_cli(self) -> str:
        """Send the rendered conversation through `claude -p`."""
        import subprocess
        prompt = self._render_conversation_for_cli()
        # Security: the rendered prompt carries recalled memory (untrusted).
        # No tools + no permission bypass so injected content can't make the
        # spawned agent act on the system. The wrapper only wants text back.
        # See SECURITY.md.
        argv = [
            "claude", "-p",
            "--model", self.config.model,
            "--no-session-persistence",
            "--allowed-tools", "",
            "--disable-slash-commands",
            prompt,
        ]
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=180,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            tail = "\n".join(result.stderr.strip().splitlines()[-3:])
            raise RuntimeError(f"claude CLI failed (exit {result.returncode}): {tail[:400]}")
        return result.stdout.strip()

    def _render_conversation_for_cli(self) -> str:
        """Flatten system + messages into one prompt string for `claude -p`.

        We own the message history (the wrapper's job is selective
        compression on the way in) so each call sends the full current
        state. claude -p has no persistent session of its own.
        """
        parts = [self._system, ""]
        for m in self.messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            parts.append(f"--- {role} ---")
            parts.append(str(content))
            parts.append("")
        parts.append("--- ASSISTANT ---")
        return "\n".join(parts)

    # -------- One turn --------
    def turn(self, user_msg: str) -> str:
        # Inject recall context lazily for first user message (or anytime we
        # don't yet have a pack in the system prompt).
        if "[Memory recall" not in self._system:
            pack = self.engine.recall(user_msg, top_k=self.config.recall_top_k)
            if pack.results:
                self._system = SYSTEM_BASE + "\n\n" + pack.to_text(
                    max_results=self.config.recall_top_k
                )

        self.messages.append({"role": "user", "content": user_msg})

        # Compress if needed BEFORE the call so we stay inside the window
        self.messages = self.policy.compact(self.messages)

        assistant = self._call_llm()
        self.messages.append({"role": "assistant", "content": assistant})

        if self.config.persist_turns:
            try:
                self.engine.remember(user_msg, assistant)
            except Exception:
                # Memory persistence should never block the chat
                pass
        return assistant

    # -------- Interactive REPL --------
    def repl(self) -> None:
        print("PMB chat. /exit to quit, /stats for memory, /recall <q> to probe memory.")
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line == "/exit":
                break
            if line == "/stats":
                print(self.engine.stats())
                continue
            if line.startswith("/recall "):
                q = line[len("/recall "):].strip()
                pack = self.engine.recall(q, top_k=5)
                print(pack.to_text(max_results=5))
                continue
            try:
                reply = self.turn(line)
            except Exception as e:
                print(f"[error] {e}", file=sys.stderr)
                continue
            print("agent>", reply)

        if self.config.consolidate_on_exit:
            try:
                result = self.engine.consolidate(dry_run=False)
                print(f"[consolidation] {result['n_consolidated']} new facts, "
                      f"{result['n_archived']} sources archived")
            except Exception as e:
                print(f"[consolidation skipped] {e}", file=sys.stderr)
