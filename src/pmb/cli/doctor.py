"""
`pmb doctor` - diagnose install and runtime state.

Prints a checklist for the user: Python, deps, embedding model, git,
PMB_HOME writability, MCP config hint. No fixes, just reporting.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional  # noqa: F401

REQUIRED_DEPS = [
    "sentence_transformers",
    "lancedb",
    "fastmcp",
    "rank_bm25",
    "typer",
    "yaml",
    "numpy",
]

MIN_PY = (3, 11)


def _ok(msg: str) -> dict:
    return {"status": "ok", "msg": msg}


def _warn(msg: str) -> dict:
    return {"status": "warn", "msg": msg}


def _fail(msg: str) -> dict:
    return {"status": "fail", "msg": msg}


def check_python() -> dict:
    v = sys.version_info
    if (v.major, v.minor) < MIN_PY:
        return _fail(f"Python {v.major}.{v.minor} - requires >= {MIN_PY[0]}.{MIN_PY[1]}")
    return _ok(f"Python {v.major}.{v.minor}.{v.micro}")


def check_deps() -> dict:
    missing: list[str] = []
    versions: dict[str, str] = {}
    for dep in REQUIRED_DEPS:
        try:
            importlib.import_module(dep)
            # Try to read a version (best-effort, package name differs from module sometimes)
            for candidate in (dep, dep.replace("_", "-")):
                try:
                    versions[dep] = importlib.metadata.version(candidate)
                    break
                except importlib.metadata.PackageNotFoundError:
                    continue
        except ImportError:
            missing.append(dep)
    if missing:
        return _fail(f"missing deps: {', '.join(missing)} - run `pip install -e .[dev]`")
    return _ok("all deps importable: " + ", ".join(f"{k}={v}" for k, v in versions.items()))


def check_embedding_model() -> dict:
    """Look for a cached sentence-transformers model so first recall is fast."""
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    candidates = [
        hf_cache / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2",
        hf_cache / "transformers",
    ]
    for p in candidates:
        if p.exists():
            return _ok(f"embedding model cache found: {p}")
    return _warn(
        "embedding model not cached yet - first recall will download ~80MB. "
        f"Cache root: {hf_cache}"
    )


def check_pmb_home() -> dict:
    from pmb.core.workspace import DEFAULT_PMB_HOME
    home = Path(os.environ.get("PMB_HOME", DEFAULT_PMB_HOME))
    try:
        home.mkdir(parents=True, exist_ok=True)
        # Write probe
        probe = home / ".doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as e:
        return _fail(f"PMB_HOME not writable: {home} ({e})")
    return _ok(f"PMB_HOME writable: {home}")


def check_consolidation_backend() -> dict:
    """Report whether consolidation has a working LLM backend."""
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    try:
        from pmb.health.consolidate import (
            DEFAULT_OLLAMA_MODEL,
            ClaudeCLIClient,
            OllamaClient,
        )
        claude_cli = ClaudeCLIClient.available()
        ollama_up = OllamaClient.ping()
    except Exception:
        claude_cli = False
        ollama_up = False

    available = []
    if claude_cli:
        available.append("claude-cli")
    if has_anthropic:
        available.append("anthropic")
    if ollama_up:
        available.append("ollama")

    if available:
        primary = available[0]
        return _ok(
            f"`pmb consolidate` ready - available: {', '.join(available)}. "
            f"`auto` will use [{primary}]."
        )
    return _warn(
        "No LLM backend for consolidation. Either:\n"
        "  - install Claude Code so `claude` is in PATH (no key needed), or\n"
        "  - set ANTHROPIC_API_KEY for direct API access, or\n"
        f"  - run `ollama serve` + `ollama pull {DEFAULT_OLLAMA_MODEL}`.\n"
        "Everything else in PMB works without it."
    )


def check_git() -> dict:
    git = shutil.which("git")
    if not git:
        return _warn("git not found in PATH - git sync will be a no-op")
    try:
        out = subprocess.run(
            [git, "--version"], capture_output=True, text=True, timeout=2,
        )
        return _ok(out.stdout.strip())
    except Exception as e:
        return _warn(f"git found but not callable: {e}")


def check_current_workspace() -> dict:
    """Best-effort: if cwd is a known PMB workspace, summarize."""
    try:
        from pmb.core.workspace import detect_workspace
        ws = detect_workspace()
    except Exception as e:
        return _warn(f"workspace detection failed: {e}")

    meta_exists = ws.meta_path.exists()
    if not meta_exists:
        return _warn(
            f"no workspace metadata at {ws.storage_dir} yet - run `pmb init`"
        )

    try:
        from pmb.core.events import EventStore
        store = EventStore(ws.db_path)
        n_active = store.count(ws.id, include_archived=False)
        n_total = store.count(ws.id, include_archived=True)
    except Exception as e:
        return _warn(f"workspace meta ok but store unreadable: {e}")

    return _ok(
        f"workspace '{ws.name}' (id {ws.id[:12]}, source {ws.source}): "
        f"{n_active} active / {n_total} total events"
    )


def check_recent_errors() -> dict:
    """Surface swallowed-exception breadcrumbs from the error_log (B5) so a
    silently-degrading path is visible instead of invisible."""
    try:
        from pmb.core.errlog import error_counts
        from pmb.core.workspace import detect_workspace
        ws = detect_workspace()
        if not ws.db_path.exists():
            return _ok("no error log yet")
        counts = error_counts(ws.db_path, since_s=24 * 3600)
    except Exception as e:
        return _ok(f"error log unavailable: {e}")
    if not counts:
        return _ok("no errors in the last 24h")
    total = sum(counts.values())
    top = ", ".join(f"{k}={v}" for k, v in list(counts.items())[:5])
    return _warn(f"{total} swallowed error(s) in last 24h - {top}")


def check_quality_flags() -> dict:
    """Surface how many incoming facts the write-time quality gate (#E2,
    default ON) flagged as suspect_junk recently, so the user can audit for
    false positives and review/clean them with `pmb declutter`."""
    try:
        import sqlite3
        import time as _t

        from pmb.core.workspace import detect_workspace
        ws = detect_workspace()
        if not ws.db_path.exists():
            return _ok("no events yet")
        cutoff = _t.time() - 30 * 86400.0
        with sqlite3.connect(str(ws.db_path)) as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM events WHERE timestamp >= ? "
                "AND metadata_json LIKE '%\"quality_flag\": \"suspect_junk\"%'",
                (cutoff,)).fetchone()[0]
    except Exception as e:
        return _ok(f"quality-flag count unavailable: {e}")
    if not n:
        return _ok("no facts flagged as junk in the last 30d")
    return _ok(f"{n} fact(s) flagged suspect_junk (30d) - review with "
               f"`pmb declutter`")


def mcp_config_hint() -> dict:
    """Print Claude Code MCP config snippet for the user to copy."""
    pmb_mcp = shutil.which("pmb-mcp")
    if pmb_mcp:
        server: dict = {"command": pmb_mcp}
    else:
        server = {"command": sys.executable, "args": ["-m", "pmb.mcp.server"]}
    server["env"] = {"PMB_CWD": str(Path.cwd())}
    snippet = {"mcpServers": {"pmb": server}}
    return _ok(
        "MCP server entry - LOCAL (paste into Claude Code mcp.json):\n"
        + json.dumps(snippet, indent=2, ensure_ascii=False)
    )


def mcp_remote_config_hint(remote: str) -> dict:
    """Print SSH-tunneled MCP config when PMB lives on another machine.

    `remote` is `user@host:/abs/path/to/repo`. We synthesise an entry
    that spawns `ssh` locally; ssh runs `pmb-mcp` on the server with
    the right PMB_CWD; stdin/stdout are tunneled so MCP just works.
    """
    if ":" not in remote:
        return _warn(
            f"--remote must be user@host:/abs/path/to/repo (got {remote!r})"
        )
    target, remote_cwd = remote.split(":", 1)
    # Default to system PATH `pmb-mcp` on the remote; user can override.
    # Quote the remote_cwd so spaces / unicode don't trip the shell.
    remote_cmd = f'PMB_CWD="{remote_cwd}" pmb-mcp'
    snippet = {
        "mcpServers": {
            "pmb-remote": {
                "command": "ssh",
                "args": [target, remote_cmd],
            }
        }
    }
    return _ok(
        "MCP server entry - REMOTE (PMB and Ollama live on the server):\n"
        + json.dumps(snippet, indent=2, ensure_ascii=False)
        + "\n\nPrereqs:\n"
        + f"  - passwordless SSH to {target} (key in ~/.ssh/authorized_keys)\n"
        + "  - `pmb-mcp` on the server's PATH (or replace with absolute path,\n"
        + "    e.g. `/home/user/pmb/.venv/bin/pmb-mcp`)\n"
        + f"  - the repo at {remote_cwd} exists on the server and `pmb init`\n"
        + "    has been run there.\n"
    )


def check_multilingual_model() -> dict:
    """P1-2: warn when an English-only embedding model is in use on a
    workspace whose contents are mostly non-Latin. The most common
    failure mode for integrators who override the default model for
    speed without realising their data is multilingual.
    """
    try:
        from pmb.config import Config
        from pmb.health.multilingual_check import evaluate
        from pmb.workspace import detect_workspace
        ws = detect_workspace()
        ws.ensure_dirs()
        cfg = Config(workspace_dir=ws.storage_dir, pmb_home=ws.pmb_home)
        model = (
            cfg.get("embedding.fastembed_model")
            if cfg.get("embedding.backend") == "fastembed"
            else cfg.get("embedding.model")
        )
        result = evaluate(ws.db_path, model)
        if result["severity"] == "ok":
            return {
                "status": "ok",
                "msg": f"model `{model}` looks appropriate for content "
                       f"({int(result['non_latin_ratio'] * 100)}% non-Latin, "
                       f"sampled {result['n_sampled']} events)",
            }
        msg = result["warning"] or "multilingual concern"
        if result.get("recommendation"):
            msg += " " + result["recommendation"].replace("\n", " ")
        return {"status": "warn", "msg": msg}
    except Exception as e:
        return {"status": "ok", "msg": f"(skipped: {type(e).__name__})"}


def check_graph_extractor() -> dict:
    """Report the active entity-graph extractor backend and whether its
    requirements (spaCy model / Claude / Ollama / Codex CLI) are satisfied.
    """
    try:
        from pmb.config import Config
        cfg = Config()
        backend = (cfg.get("graph.extractor") or "regex").strip().lower()
    except Exception as e:
        return {"status": "warn", "msg": f"could not read config ({e})"}
    if backend in ("regex", "", "default"):
        return {"status": "ok",
                "msg": "backend `regex` (default: fast, offline, no deps)"}
    if backend == "spacy":
        try:
            import spacy  # noqa: F401
        except ImportError:
            return {"status": "warn",
                    "msg": "backend `spacy` selected but spaCy not installed → "
                           "falls back to regex. Run: pip install spacy && "
                           "python -m spacy download en_core_web_sm"}
        # Probe at least one model
        try:
            import spacy
            for name in ("en_core_web_sm", "en_core_web_md", "xx_ent_wiki_sm"):
                try:
                    spacy.load(name)
                    return {"status": "ok",
                            "msg": f"backend `spacy` ready (model: {name})"}
                except OSError:
                    continue
        except Exception:
            pass
        return {"status": "warn",
                "msg": "backend `spacy` selected but no model installed → "
                       "falls back to regex. Run: python -m spacy download en_core_web_sm"}
    if backend.startswith("llm:"):
        import shutil
        provider = backend.split(":", 1)[1]
        bins = {"claude": "claude", "ollama": "ollama", "codex": "codex"}
        bin_name = bins.get(provider)
        if not bin_name:
            return {"status": "warn", "msg": f"unknown LLM provider {provider!r} → falls back to regex"}
        if not shutil.which(bin_name):
            return {"status": "warn",
                    "msg": f"backend `{backend}` selected but `{bin_name}` "
                           f"CLI not on PATH → falls back to regex"}
        return {"status": "ok",
                "msg": f"backend `{backend}` ready (`{bin_name}` CLI on PATH)"}
    return {"status": "warn", "msg": f"unknown graph.extractor={backend!r} → falls back to regex"}


def check_channel_weights() -> dict:
    """F3: surface the X2 channel-weights SUGGESTION the tick proposed (never
    auto-applied - the user inspects and applies it explicitly)."""
    try:
        import json as _json

        from pmb.core.workspace import detect_workspace
        ws = detect_workspace()
        p = ws.storage_dir / "channel_weights_suggestion.json"
        if not p.exists():
            return {"status": "ok",
                    "msg": "no suggestion yet (recall.weight_learning off or "
                           "too little feedback)"}
        s = _json.loads(p.read_text(encoding="utf-8"))
        w = s.get("weights", {})
        wtxt = ", ".join(f"{k}:{float(v):.2f}" for k, v in w.items())
        return {"status": "warn",
                "msg": f"learned {{{wtxt}}} from {s.get('n_useful')}/"
                       f"{s.get('n_samples')} useful samples - apply with: "
                       f"pmb config set recall.channel_weights '{_json.dumps(w)}'"}
    except Exception:
        return {"status": "ok", "msg": "channel-weights check skipped"}


def check_schema() -> dict:
    """Report the on-disk schema version vs the code's SCHEMA_VERSION, so an
    out-of-date or newer-than-code db is visible (upgrade/downgrade safety)."""
    try:
        import sqlite3

        from pmb.core.events import SCHEMA_VERSION
        from pmb.core.workspace import detect_workspace
        ws = detect_workspace()
        if not ws.db_path.exists():
            return _ok("no workspace db yet")
        with sqlite3.connect(str(ws.db_path)) as conn:
            row = conn.execute("SELECT MAX(version) FROM migrations").fetchone()
        cur = int((row[0] if row else 0) or 0)
    except Exception as e:
        return _ok(f"schema check skipped: {e}")
    if cur == SCHEMA_VERSION:
        return _ok(f"schema v{cur} (current)")
    if cur < SCHEMA_VERSION:
        return _warn(f"schema v{cur} < code v{SCHEMA_VERSION} - auto-migrates on next open")
    return _warn(
        f"schema v{cur} > code v{SCHEMA_VERSION} - db written by a NEWER PMB; "
        "downgrade is not supported, upgrade PMB"
    )


def check_daemon() -> dict:
    """Is a warm PMB daemon live? Uses the same discovery the hooks use. Cold =
    each per-turn hook pays a model load and skips semantic/cross-lingual recall."""
    try:
        from pmb.mcp.registry import find_live_daemon
        d = find_live_daemon()
    except Exception as e:
        return _ok(f"daemon probe skipped: {e}")
    if d:
        return _ok(
            f"warm daemon live (pid {d.get('pid','?')}, port {d.get('port','?')}) "
            "- hooks served warm; semantic + cross-lingual recall active"
        )
    return _warn(
        "no warm daemon - per-turn hooks run COLD (model not loaded, so semantic "
        "and cross-lingual recall are skipped that turn). Start one with "
        "`pmb warmup` or `pmb daemon start`."
    )


def check_write_backlog() -> dict:
    """Async write/embed backlog: pending embeds, dead-letters, and undrained
    write-outbox rows. A growing backlog means recent writes are not searchable
    yet (the index is behind)."""
    try:
        from pmb.core.workspace import detect_workspace
        ws = detect_workspace()
        if not ws.db_path.exists():
            return _ok("no workspace db yet")
    except Exception as e:
        return _ok(f"backlog check skipped: {e}")
    parts: list[str] = []
    sev = "ok"
    try:
        from pmb.core.embed_queue import PersistentEmbedQueue
        q = PersistentEmbedQueue(ws.db_path)
        pend = q.pending_count()
        dead = q.dead_letter_count()
        if pend:
            parts.append(f"{pend} embed(s) pending")
            sev = "warn" if pend > 50 else sev
        if dead:
            parts.append(f"{dead} embed dead-letter(s)")
            sev = "warn"
    except Exception:
        pass
    try:
        import sqlite3
        with sqlite3.connect(str(ws.db_path)) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM write_outbox "
                "WHERE status IN ('pending','processing')"
            ).fetchone()
        ob = int((row[0] if row else 0) or 0)
        if ob:
            parts.append(f"{ob} outbox row(s) undrained")
            sev = "warn" if ob > 50 else sev
    except Exception:
        pass  # write_outbox is v7+; older dbs simply have no backlog here
    if not parts:
        return _ok("write/embed queues drained (no backlog)")
    msg = "; ".join(parts) + " - normally drains in the background"
    return _warn(msg) if sev == "warn" else _ok(msg)


def check_tool_profile() -> dict:
    """Report which MCP tool surface an agent sees (PMB_TOOL_PROFILE)."""
    prof = (os.environ.get("PMB_TOOL_PROFILE") or "default").lower().strip()
    counts = {"minimal": "13", "lean": "26", "default": "30", "full": "all"}
    if prof not in counts:
        return _warn(
            f"unknown PMB_TOOL_PROFILE={prof!r} - server falls back to default (30)"
        )
    note = {
        "minimal": "essentials only",
        "lean": "set by `pmb connect claude-code` (hooks cover the read-status tools)",
        "default": "day-to-day surface",
        "full": "every tool incl. admin (debug/dev)",
    }[prof]
    return _ok(f"MCP tool profile `{prof}` ({counts[prof]} tools) - {note}")


def check_ald() -> dict:
    """ALD (Anchor->Lexicon Distillation) field coverage: the self-compiling
    cold-path lexicon built from this machine's own traffic. Lightweight - reads
    config + the anchor/shadow tables, never loads the embedding model."""
    try:
        from pmb.config import Config
        from pmb.core.workspace import detect_workspace
        from pmb.maintenance.distill import ald_field_report
        ws = detect_workspace()
        ws.ensure_dirs()
        cfg = Config(workspace_dir=ws.storage_dir, pmb_home=ws.pmb_home)

        class _Shim:
            pass

        sh = _Shim()
        sh.workspace = ws
        sh.config = cfg
        r = ald_field_report(sh)
    except Exception as e:
        return _ok(f"ALD report skipped: {e}")
    if not r.get("enabled"):
        return _warn(
            "ALD fire-logging OFF (lang.anchor_log) - the cold-path lexicon will "
            "not self-compile from your traffic. Turn it on to let PMB learn your "
            "languages: `pmb config set lang.anchor_log true`."
        )
    lex, fires = r["lexicon"], r["fires"]
    fp = r.get("false_positive_rate") or {}
    worst = max(fp.values()) if fp else 0.0
    msg = (f"lexicon {lex['entries']} entries / {lex['categories']} categories; "
           f"{fires['total']} fires over {fires['span_days']}d; "
           f"worst intent FP {worst:.0%}")
    drop = r["pruning"]["shadow_would_drop"]
    if drop:
        return _warn(msg + f"; shadow gate would prune: {', '.join(drop)}")
    return _ok(msg)


def run_doctor(remote: str | None = None) -> list[tuple[str, dict]]:
    """Run all checks. Returns list of (label, result_dict)."""
    checks: list[tuple[str, dict]] = [
        ("Python", check_python()),
        ("Dependencies", check_deps()),
        ("Embedding model cache", check_embedding_model()),
        ("Multilingual fit", check_multilingual_model()),
        ("Graph extractor", check_graph_extractor()),
        ("PMB_HOME", check_pmb_home()),
        ("Git", check_git()),
        ("Consolidation backend", check_consolidation_backend()),
        ("Current workspace", check_current_workspace()),
        ("Schema version", check_schema()),
        ("Warm daemon", check_daemon()),
        ("Write/embed backlog", check_write_backlog()),
        ("MCP tool profile", check_tool_profile()),
        ("ALD coverage", check_ald()),
        ("Recent errors (24h)", check_recent_errors()),
        ("Quality gate (30d)", check_quality_flags()),
        ("Channel weights (F3)", check_channel_weights()),
        ("MCP config hint", mcp_config_hint()),
    ]
    if remote:
        checks.append(("MCP remote config", mcp_remote_config_hint(remote)))
    return checks


def print_doctor(console, remote: str | None = None) -> int:
    """Pretty-print doctor results. Returns non-zero if any check failed."""
    from rich.panel import Panel
    from rich.table import Table

    results = run_doctor(remote=remote)
    n_fail = sum(1 for _, r in results if r["status"] == "fail")
    n_warn = sum(1 for _, r in results if r["status"] == "warn")

    table = Table(show_header=True, header_style="bold magenta",
                  title=f"PMB Doctor  ({platform.system()} {platform.release()})")
    table.add_column("Check", style="cyan", width=24)
    table.add_column("Status", width=8)
    table.add_column("Detail", overflow="fold")
    for label, r in results:
        color = {"ok": "green", "warn": "yellow", "fail": "red"}[r["status"]]
        table.add_row(label, f"[{color}]{r['status']}[/]", r["msg"])
    console.print(table)

    summary_color = "red" if n_fail else ("yellow" if n_warn else "green")
    summary = f"[{summary_color}]{len(results) - n_fail - n_warn} ok, {n_warn} warn, {n_fail} fail[/]"
    console.print(Panel.fit(summary, title="Summary"))
    return 1 if n_fail else 0
