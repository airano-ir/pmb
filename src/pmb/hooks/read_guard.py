"""Read-Guard - block redundant in-context re-reads of unchanged files.

A PreToolUse guard for `Read`: when the agent is about to read a file it ALREADY
read this session, whose content is unchanged (sha match) AND was read recently
(within a recency window, so it is almost certainly still in the live context),
the read is denied with a reason. The agent then uses what it already has
instead of dumping the same file into the context window again.

Conservative by design - a CHANGED file, a NEW file, or one last read long ago
(possibly compacted out of context) is always allowed. This targets a cost the
snapshot/memo features missed: the SAME file re-read several times in one
session (common right after a compaction) re-enters context and is then re-sent
on every subsequent turn.

The decision logic is a pure function so it is cheap to test; the warm daemon
holds the per-session ledger and the config gate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha1(path: str) -> str | None:
    """SHA1 of a file's bytes, or None if it can't be read."""
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def evaluate(
    ledger: dict, session: str, file_path: str, cur_sha: str | None, window: int,
) -> dict:
    """Decide allow/deny for a Read and RECORD it in `ledger` (mutated in place).

    Deny only when the same file was read before this session, is byte-identical
    now, and was read within the last `window` reads (so the content is very
    likely still in the live context). Returns {"decision", "reason"}.
    """
    sess = ledger.setdefault(session or "", {"seq": 0, "files": {}})
    sess["seq"] += 1
    seq = sess["seq"]
    prev = sess["files"].get(file_path)

    decision, reason = "allow", ""
    if (cur_sha and prev and prev.get("sha") == cur_sha
            and (seq - int(prev.get("seq", 0))) <= window):
        decision = "deny"
        reason = (
            f"You already read {file_path} earlier this session (step "
            f"{prev.get('seq')}) and it is unchanged. Re-reading duplicates it "
            f"in the context window, which is re-sent every turn. Use what you "
            f"already have; if you need a part you no longer have, Read it with "
            f"a line range (offset/limit)."
        )
    # Record this attempt so the next identical read is caught and the recency
    # window slides forward.
    sess["files"][file_path] = {"sha": cur_sha, "seq": seq}
    return {"decision": decision, "reason": reason}


def deny_payload(reason: str) -> dict:
    """The PreToolUse JSON that tells Claude Code to block the tool call and
    feed `reason` back to the model."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }
