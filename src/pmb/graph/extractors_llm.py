"""LLM-backed entity extractor - Claude CLI / Ollama / Codex CLI at write time.

This is the "as graphify does" option: send each event's text to a local LLM,
get back a small JSON list of clean concept names, and use those as graph
nodes. Trades the "no LLM in the write path" promise for a much cleaner
knowledge graph (real concepts like "Claude Code", "JWT auth", "deep
research" instead of regex tokens like "built", "ideas", "stol").

Optional - activated by config:

    pmb config set graph.extractor llm:claude        # Claude Code CLI
    pmb config set graph.extractor llm:ollama        # local Ollama model
    pmb config set graph.extractor llm:codex         # OpenAI Codex CLI

All providers shell out to a CLI binary that must be on PATH. We never block
on an LLM longer than `graph.llm_timeout_s` (default 30 s); on timeout / non-
zero exit / malformed JSON we silently fall back to the regex backend so a
broken LLM never stalls a record_batch.

We extend the regex backend rather than replace it - files + techs still go
through the fast regex paths, the LLM only fills `concepts` / `persons` /
`orgs` / `places` / `products`. That keeps known-set lookups (50 techs, file
extensions) exact and only burns LLM cycles on the open-vocab part.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Iterable

from pmb.graph.entities import (
    _POSIXPATH_RE,
    _STOPWORDS,
    _WINPATH_RE,
    EntityExtractor,
    ExtractedEntities,
    _normalize_path,
    extract_file_paths,
    extract_techs,
)

log = logging.getLogger(__name__)


# Tight single-event schema. Each rule earns its place in tokens.
_PROMPT_TMPL = """Extract entities. JSON only, no prose.
{{"persons":[],"orgs":[],"places":[],"products":[],"concepts":[]}}
≤{max_n} per list. Lowercase unless proper noun. Skip verbs/adjectives/file-paths/generics.

TEXT:
{text}"""

# Batched prompt: N events → ONE call. Saves the ~20 s claude CLI startup +
# the shared system tokens, so a batch of 10 events ≈ 1 single-event cost.
_BATCH_PROMPT_TMPL = """Extract entities from each numbered TEXT below.
Return JSON: {{"results":[{{...}},{{...}}, ...]}} - one object PER TEXT in input order.
Each object: {{"persons":[],"orgs":[],"places":[],"products":[],"concepts":[]}}
≤{max_n} per list. Lowercase unless proper noun. Skip verbs/adjectives/paths/generics.
Output JSON only, no prose, no markdown.

{texts}"""


def _make_prompt(text: str, max_n: int) -> str:
    # Hard cap so a 5000-char fact doesn't blow up the prompt.
    snippet = text[:3000]
    return _PROMPT_TMPL.format(text=snippet, max_n=max_n)


def _make_batch_prompt(texts: list[str], max_n: int) -> str:
    """Format N texts into one prompt. Each gets capped at 1500 chars so a
    batch of 20 events stays well under the model's input window."""
    body = "\n\n".join(
        f"TEXT {i+1}:\n{(t or '')[:1500]}"
        for i, t in enumerate(texts)
    )
    return _BATCH_PROMPT_TMPL.format(texts=body, max_n=max_n)


# Tolerant JSON extractor - the model sometimes wraps output in ```json fences.
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.IGNORECASE)
_BARE_OBJ_RE = re.compile(r"\{[\s\S]+?\}")


def _parse_json(raw: str) -> dict:
    """Pull the first JSON object out of `raw`, regardless of fences or noise."""
    if not raw:
        raise ValueError("empty LLM output")
    s = raw.strip()
    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()
    # If the model added prose around the JSON, salvage the first {...}.
    if not s.startswith("{"):
        m2 = _BARE_OBJ_RE.search(s)
        if m2:
            s = m2.group(0)
    return json.loads(s)


def _clean_list(seq, max_n: int) -> list[str]:
    """Coerce / dedupe / cap a list-of-strings from the LLM."""
    if not isinstance(seq, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in seq:
        if not isinstance(item, str):
            continue
        name = " ".join(item.split()).strip()
        if not name or len(name) < 2:
            continue
        low = name.lower()
        if low in _STOPWORDS or low in seen:
            continue
        seen.add(low)
        out.append(name)
        if len(out) >= max_n:
            break
    return out


# ----------------------------------------------------------------------
# Provider invocations - each returns the raw stdout text or raises.
# ----------------------------------------------------------------------

def _run_claude_cli(prompt: str, timeout: float, model: str = "") -> str:
    cmd = shutil.which("claude")
    if not cmd:
        raise RuntimeError("`claude` CLI not on PATH")
    # Security: entity extraction feeds untrusted event text into the prompt.
    # Give the spawned agent NO tools and do NOT bypass permissions so an
    # injected payload can't drive it into running Bash/Edit/Write. This is a
    # text-in/JSON-out call - it never needs tools. See SECURITY.md.
    argv = [
        cmd, "-p", "--no-session-persistence",
        "--allowed-tools", "",
        "--disable-slash-commands",
    ]
    # haiku / sonnet / opus / explicit anthropic id - pass through so the user
    # can pick a cheap model (entity extraction doesn't need opus).
    if model:
        argv += ["--model", model]
    argv.append(prompt)
    # Force ASCII cwd: on Windows, Claude CLI walks the cwd looking for
    # config/session files. If the path has non-ASCII chars (e.g. a Cyrillic
    # "Desktop" folder) it fails with a "path not found" OS error. TEMP is
    # guaranteed ASCII and writable.
    safe_cwd = os.environ.get("TEMP") or os.environ.get("TMP") or os.getcwd()
    r = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace", cwd=safe_cwd,
    )
    if r.returncode != 0:
        raise RuntimeError(f"claude exit {r.returncode}: {(r.stderr or '')[:200]}")
    return r.stdout or ""


def _run_ollama_cli(prompt: str, model: str, timeout: float) -> str:
    cmd = shutil.which("ollama")
    if not cmd:
        raise RuntimeError("`ollama` CLI not on PATH")
    argv = [cmd, "run", model, prompt]
    r = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        raise RuntimeError(f"ollama exit {r.returncode}: {(r.stderr or '')[:200]}")
    return r.stdout or ""


def _run_codex_cli(prompt: str, timeout: float) -> str:
    # OpenAI Codex CLI: `codex exec` is the non-interactive one-shot runner.
    cmd = shutil.which("codex")
    if not cmd:
        raise RuntimeError("`codex` CLI not on PATH")
    argv = [cmd, "exec", "--quiet", prompt]
    r = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        raise RuntimeError(f"codex exit {r.returncode}: {(r.stderr or '')[:200]}")
    return r.stdout or ""


class LLMExtractor(EntityExtractor):
    """LLM-extraction backend.

    Layer order on each event:
      1. files  ← fast regex (file-path patterns)
      2. techs  ← fast regex (closed KNOWN_TECHS set)
      3. open vocab (persons / orgs / places / products / concepts) ← LLM CLI
    """

    backend_name = "llm"
    DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"

    def __init__(
        self,
        provider: str = "claude",
        max_concepts: int = 5,
        timeout_s: float = 30.0,
        model: str = "",
        ollama_model: str | None = None,
    ):
        super().__init__(max_concepts=max_concepts)
        self.provider = provider.lower().strip()
        self.timeout = float(timeout_s)
        # `model` is the model id passed to the CLI. For Claude: "haiku" /
        # "sonnet" / full id. For Ollama it overrides DEFAULT_OLLAMA_MODEL.
        self.model = (model or "").strip()
        self.ollama_model = (ollama_model or self.model
                             or self.DEFAULT_OLLAMA_MODEL)
        self.backend_name = f"llm:{self.provider}"
        # Verify the CLI is on PATH at construction time so the factory can
        # fall back to regex cleanly if it isn't.
        self._verify_cli()

    def _verify_cli(self) -> None:
        binaries = {"claude": "claude", "ollama": "ollama", "codex": "codex"}
        bin_name = binaries.get(self.provider)
        if not bin_name:
            raise RuntimeError(f"unknown LLM provider {self.provider!r}")
        if not shutil.which(bin_name):
            raise RuntimeError(f"`{bin_name}` CLI not on PATH for graph.extractor=llm:{self.provider}")

    def _call(self, prompt: str) -> str:
        if self.provider == "claude":
            return _run_claude_cli(prompt, self.timeout, self.model)
        if self.provider == "ollama":
            return _run_ollama_cli(prompt, self.ollama_model, self.timeout)
        if self.provider == "codex":
            return _run_codex_cli(prompt, self.timeout)
        raise RuntimeError(f"unknown provider {self.provider!r}")

    def extract(self, text: str, files_hint: Iterable[str] = ()) -> ExtractedEntities:
        # Always run the cheap regex layers first - they're exact and have
        # zero cost. The LLM only does the open-vocab part.
        files = list(dict.fromkeys(
            [*extract_file_paths(text), *(_normalize_path(f) for f in files_hint)]
        ))
        techs = extract_techs(text)

        if not text or len(text.strip()) < 8:
            return ExtractedEntities(files=files, techs=techs, concepts=[])

        # Strip paths before the LLM sees them (saves prompt tokens and
        # stops the model "extracting" Users/Roaming as concepts).
        scrubbed = _WINPATH_RE.sub(" ", text)
        scrubbed = _POSIXPATH_RE.sub(" ", scrubbed)

        prompt = _make_prompt(scrubbed, self.max_concepts)
        try:
            raw = self._call(prompt)
            payload = _parse_json(raw)
        except subprocess.TimeoutExpired:
            log.warning("graph.extractor=%s timed out after %.0fs - falling back to regex on this event",
                        self.backend_name, self.timeout)
            return EntityExtractor(self.max_concepts).extract(text, files_hint)
        except Exception as e:
            log.warning("graph.extractor=%s failed (%s) - falling back to regex on this event",
                        self.backend_name, e)
            return EntityExtractor(self.max_concepts).extract(text, files_hint)

        return ExtractedEntities(
            files=files,
            techs=techs,
            concepts=_clean_list(payload.get("concepts"), self.max_concepts),
            persons=_clean_list(payload.get("persons"),  self.max_concepts),
            orgs=_clean_list(payload.get("orgs"),     self.max_concepts),
            places=_clean_list(payload.get("places"),   self.max_concepts),
            products=_clean_list(payload.get("products"), self.max_concepts),
        )

    def extract_batch(
        self,
        items: list[tuple[str, tuple]],
    ) -> list[ExtractedEntities]:
        """Extract entities for N events in ONE LLM call.

        `items` is a list of (text, files_hint) pairs. Returns a list of
        ExtractedEntities aligned with the input order.

        Why this matters: a single `claude -p` call pays ~20 s of CLI startup
        and ~10-15 s of inference. A batch of 20 events pays the same startup
        once + slightly more inference - so per-event cost drops from ~30 s
        to ~2 s. Same trick on tokens: the schema/system prompt is shared
        across all N events instead of repeated N times (~30 % token win on
        top of the latency win).

        Any per-event failure (LLM returned fewer results than expected,
        malformed entry, etc.) falls back to the regex extractor for that one
        slot - the rest of the batch is unaffected.
        """
        n = len(items)
        if n == 0:
            return []
        if n == 1:
            text, files_hint = items[0]
            return [self.extract(text, files_hint)]

        # Cheap layers per-event first - regex is exact, no point burning LLM
        # tokens on file paths and known techs.
        files_per: list[list[str]] = []
        techs_per: list[list[str]] = []
        scrubbed_texts: list[str] = []
        for text, files_hint in items:
            files = list(dict.fromkeys(
                [*extract_file_paths(text), *(_normalize_path(f) for f in files_hint)]
            ))
            techs = extract_techs(text)
            files_per.append(files)
            techs_per.append(techs)
            s = _WINPATH_RE.sub(" ", text or "")
            s = _POSIXPATH_RE.sub(" ", s)
            scrubbed_texts.append(s)

        prompt = _make_batch_prompt(scrubbed_texts, self.max_concepts)
        try:
            raw = self._call(prompt)
            payload = _parse_json(raw)
        except subprocess.TimeoutExpired:
            log.warning("graph.extractor=%s BATCH timed out after %.0fs - "
                        "falling back to regex for %d events",
                        self.backend_name, self.timeout, n)
            return [EntityExtractor(self.max_concepts).extract(t, fh)
                    for (t, fh) in items]
        except Exception as e:
            log.warning("graph.extractor=%s BATCH failed (%s) - falling back "
                        "to regex for %d events", self.backend_name, e, n)
            return [EntityExtractor(self.max_concepts).extract(t, fh)
                    for (t, fh) in items]

        results_raw = payload.get("results")
        if not isinstance(results_raw, list):
            log.warning("graph.extractor=%s BATCH returned no `results` array - "
                        "falling back to regex for %d events", self.backend_name, n)
            return [EntityExtractor(self.max_concepts).extract(t, fh)
                    for (t, fh) in items]

        # Pad / truncate to match the batch size. Missing slots fall back to
        # regex for that one event so we never silently lose data.
        out: list[ExtractedEntities] = []
        for i in range(n):
            slot = results_raw[i] if i < len(results_raw) else None
            if isinstance(slot, dict):
                out.append(ExtractedEntities(
                    files=files_per[i],
                    techs=techs_per[i],
                    concepts=_clean_list(slot.get("concepts"), self.max_concepts),
                    persons=_clean_list(slot.get("persons"),  self.max_concepts),
                    orgs=_clean_list(slot.get("orgs"),     self.max_concepts),
                    places=_clean_list(slot.get("places"),   self.max_concepts),
                    products=_clean_list(slot.get("products"), self.max_concepts),
                ))
            else:
                # Per-slot fallback - the other N-1 events still benefit.
                text, files_hint = items[i]
                out.append(EntityExtractor(self.max_concepts).extract(text, files_hint))
        return out
