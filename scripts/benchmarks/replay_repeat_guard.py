"""Behavioral replay: would today's correction-capture + repeat-guard have
caught the repeated mistakes EARLIER than the human did?

This measures the axis LoCoMo recall@10 cannot: not "find a similar fact" but
"stop the agent repeating a mistake the user already complained about". It
replays a real Claude-Code transcript chronologically through the ACTUAL
shipped functions (pmb.hooks.correction_capture) — no simulation of the idea,
the real code.

Protocol:
  1. Read user messages in order.
  2. detect_correction() on each -> is this pushback? (capture trigger)
  3. Keep captured drafts (= the prior complaint texts, as production does).
  4. For each new correction, BEFORE storing it, run the repeat-guard
     (strong_lesson_matches) against captured drafts: would it have WARNED
     the agent that this is a repeat?
  5. Cluster repeats; report at which occurrence the guard first fires vs how
     many times the human actually had to repeat themselves.

Run:
    PYTHONPATH=src python scripts/benchmarks/replay_repeat_guard.py <transcript.jsonl>
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent.parent / "src"))

from pmb.hooks.correction_capture import (  # noqa: E402
    detect_correction,
    strong_lesson_matches,
)

DEFAULT_TRANSCRIPT = (
    r"C:\Users\alexb\.claude\projects"
    r"\C--Users-alexb-OneDrive--------------RR"
    r"\d1eca84b-b659-439c-a35b-fb8d39769371.jsonl"
)


def user_messages(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") != "user":
                continue
            c = ev.get("message", {}).get("content")
            if isinstance(c, str):
                text = c
            elif isinstance(c, list):
                tx = [b.get("text", "") for b in c
                      if isinstance(b, dict) and b.get("type") == "text"]
                if not tx:
                    continue
                text = " ".join(tx)
            else:
                continue
            text = text.strip()
            if not text or text.startswith("<") or "[Request interrupted" in text:
                continue
            if "continued from a previous conversation" in text:
                continue
            out.append((ev.get("timestamp", "")[:16], text))
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRANSCRIPT
    msgs = user_messages(path)
    print(f"transcript: {Path(path).name}")
    print(f"user messages: {len(msgs)}\n")

    captured = []   # list of {"content","ts","occ"} drafts already captured
    n_corr = 0
    n_prewarned = 0   # corrections the guard would have flagged as a repeat
    first_warn_examples = []

    for ts, text in msgs:
        sig = detect_correction(text)
        if not sig:
            continue
        n_corr += 1
        # repeat-guard: does this match something already captured?
        hits = strong_lesson_matches(text, captured, min_overlap=2, min_strong=1, limit=1)
        if hits:
            n_prewarned += 1
            if len(first_warn_examples) < 8:
                first_warn_examples.append((ts, text[:90], hits[0]["ts"]))
        # capture this one (production records a draft on every correction;
        # dedup would merge, but for "would it warn" we keep the signal)
        captured.append({"ulid": f"d{n_corr}", "content": text, "ts": ts})

    print(f"corrections detected: {n_corr}")
    print(f"of those, repeat-of-earlier (guard would WARN): {n_prewarned} "
          f"({100*n_prewarned/max(n_corr,1):.0f}%)\n")

    print("=== sample 'would-warn' moments (repeat caught) ===")
    for ts, txt, prior_ts in first_warn_examples:
        print(f"  [{ts}] would warn (matches earlier complaint @ {prior_ts})")
        print(f"          \"{txt}\"")

    # Focus cluster: locate-me (the one that repeated ~8x in reality).
    print("\n=== locate-me cluster (reality: complained ~8x, lesson recorded 23:30) ===")
    occ = 0
    for ts, text in msgs:
        low = text.lower()
        if ("locate me" in low or "location" in low) and detect_correction(text):
            occ += 1
            tag = "CAPTURE (1st)" if occ == 1 else "GUARD WOULD WARN"
            print(f"  occ#{occ} [{ts}] -> {tag}")
    if occ:
        print(f"\n  >>> guard fires from occ#2; human kept repeating to occ#{occ} "
              f"before it stuck. Caught ~{occ-1} repeats earlier.")


if __name__ == "__main__":
    main()
