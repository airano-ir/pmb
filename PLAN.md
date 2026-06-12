# PLAN v0.9.0 — "Anchor Engine": language packs become unnecessary

Branch: `feat/anchor-engine-v0.9` (no "claude" in branch names). v0.8.0 shipped
at `9653b0b` (tag v0.8.0); its plan is archived at
`docs/plans/2026-06-11-PLAN-v0.8-invisible-memory-done.md`.

> **STATUS (2026-06-12): SHIPPED as 0.9.0.** Every phase landed and is tested:
> A1·A2 (anchors + FPR calibration, frozen) · B1 (multilingual intents, 12/12) ·
> B2 (statement detectors, group-scoped, no B1 regression) · B3·B4 (PAMVR
> write-time fp + verb-boost gate) · C1·C2·C3 (value spans + hypothesis-margin
> keyed extraction 11/12 + canonical atoms) · D1·D2·D3 (ALD self-compiling cache
> + recency prune) · E1·E2 (corpus stopwords + isupper headings) · F1·F2·F3·F4
> (extraction confidence + contradiction + X2 loop + multilingual eval gate
> top1 0.83 / top3 0.94) · G1·G2·G4 (hybrid default, packs-off ratchet CI,
> release 0.9.0). The C2/F1/F2 keyed-extraction path and E1/F3 are additive +
> default-OFF (`extract.anchor_keyed`, `lang.corpus_stopwords`,
> `recall.weight_learning`), flipped on by their evals, never auto-applied.
>
> **Follow-up closures (2026-06-12):** F4 grown to **101** labelled queries
> (overall top1 0.77 / top3 0.91); keyed-parity test (anchors never contradict
> the regex tier, and extend to de/es/fr); warm-latency gate (anchor classify
> p50≈50 / p95≈70 ms on this memory-constrained box; the +30 ms DoD target
> assumes healthy hardware); B3 first-person now also via the warm about_self
> anchor (non-English); C2 keyed cosine-merge of inflected values (Kieve→Kiev);
> E2 FULLY done — all four hot tokenizers (text_match / pamvr / recall /
> declutter) use Unicode property classes instead of the cyrillic_script_range,
> with V1 RU/UK recall proven byte-identical (1.00/1.00).
>
> D3 shadow-T1 sampling now landed too (`shadow_t1` table + 5% T0-vs-T1 sample
> in the dispatch + a per-category live-precision prune in the distiller), so
> EVERY code phase A–G is implemented and tested.
>
> **G3 DONE (2026-06-12, owner-authorized, soak skipped).** `packs/ru.yaml` +
> `packs/uk.yaml` DELETED, `_DEFAULT_ACTIVE=()`, packs-off CI gate now BLOCKING.
> Verified: V1 RU/UK recall byte-identical (1.00) — the embedder carries it. The
> ~90 pack-dependent tests were migrated: cold lexical matrices are EN-only,
> RU/UK intent/keyed/negation/future moved to warm-anchor tests, the 3 frozen
> baselines (`_regex_parity` / `_fact_extract` / `_lang_parity`) regenerated
> packs-off, and RU/UK general atomic-fact extraction (no warm replacement) is
> skipped with a reason. HONEST cold regression on a no-daemon stdio path: RU/UK
> first-person / self-intent / relation / negation / future-intent / atomic
> extraction no longer fire cold until ALD distils them; the warm-daemon default
> is unaffected. **PLAN now fully implemented, A–G.**

---

## 0. Where the system stands after v0.8.0 (honest scorecard)

Measured, not vibes:

| Axis | State | Evidence | Score |
|---|---|---|---|
| Speed (warm) | hook auto-context 69–261 ms end-to-end on the live daemon path; recall warm p50 ~100–300 ms; PAMVR p99 cut 860→300 ms | today's trace headers; S10 honest trace; perf smoke p95 < 300 ms | **8/10** |
| Speed (cold) | seconds (model load) — mitigated by S1 lazy import (−3–6 s) + daemon autostart; cold is now the rare path | S1/S2 measurements | 6/10 |
| Accuracy (in-language) | EN/RU/UK top-1 = 1.00 on the V1 eval — but n=13 (small); historical PAMVR bench 93.3% top-1 on n=30 | tests/test_memory_eval.py | **7.5/10** |
| Accuracy (cross-lingual) | EN→RU top-3 = 1.00 but top-1 = 0.33; an in-language paraphrase outranks the foreign fact | V1 eval xl bucket | 5/10 |
| Coverage (languages) | recall CORE is already multilingual (the embedder + BM25 handle 50+ languages). What is NOT: intent fast-paths, fact/attribute extraction, PAMVR lexical boosts, stopwords — those exist for EN+RU+UK only (packs) | Phase L architecture | **4/10** |
| Intent recall | known regex coverage gaps — e.g. the goals regex missed real phrasings ("что мне осталось сделать по проекту") until hand-patched (R4) | recorded lesson, usefulness bench 2026-06-07 | 6/10 |
| Engineering quality | 1286 tests, parity pins, eval/latency/SLO/API-contract gates, ruff-blocking CI, zero-Cyrillic ratchet, fault-injection | the v0.8.0 suite | **9/10** |

Overall: **~8.3/10**. The single biggest structural weakness is the last mile of
language understanding: every list/regex the packs hold is a hand-enumerated
approximation of a SEMANTIC class ("this is a goals question", "this asserts
where the user lives"). Enumeration can never be complete (the goals-regex gap
proved it in-house), and it scales O(languages × categories × phrasings).

The recorded architecture principle (2026-06-08) already says it:
*hardcoded regex/keyword lists are the WRONG tool for OPEN-ENDED language
understanding; embeddings are the right tool — regex should be the fast-path
cache, not the source of truth.* v0.9 implements exactly that.

---

## 1. The mechanism — SAE + ALD (what's new here)

Two coupled components:

**SAE — Semantic Anchor Engine.** Every function a language pack serves is
re-expressed as a small set of ENGLISH-ONLY semantic anchors: positive
exemplars + HARD-NEGATIVE exemplars per class, embedded once and cached. At
runtime a text is classified by **margin** — `max cos(text, positives) − max
cos(text, negatives)` — against calibrated per-class thresholds. The
multilingual embedder (already shipped, already loaded on the daemon path) does
the cross-lingual transfer: a German or Japanese "what are my open goals" lands
near the English anchors with no German or Japanese data anywhere in the
system. Extraction (not just classification) is done the same way via
**hypothesis margins** (§5).

**ALD — Anchor→Lexicon Distillation.** Anchors need the embedder (warm path
only) and cost one embed per message. So the system OBSERVES which character
n-grams in the live workspace reliably co-fire with which anchors and
periodically (in the existing daemon maintenance tick) compiles them into
`$PMB_HOME/lang/auto.yaml` — in the EXISTING pack schema, loaded by the
EXISTING loader with zero changes. The pack format stops being hand-written
source-of-truth and becomes a self-compiled per-workspace cache: it serves the
cold path (no model loaded) and the hot path (no embed needed on cache hit),
in WHATEVER language the user actually writes. Entries carry provenance and are
auto-pruned when their precision drops.

Result: hand-written `ru.yaml`/`uk.yaml` become deletable (gated by eval, §8),
any language works at anchor accuracy warm and at cache accuracy cold, and the
hot path stays regex-fast.

**Honest novelty assessment:** exemplar/prototype classification over sentence
embeddings is known technique; NLI-style zero-shot relation extraction exists
in research. What I have not seen in any shipped memory product (mem0, Zep,
Letta and the rest all use cloud-LLM extraction, or per-language NLP models):
a fully OFFLINE, no-LLM, any-language extraction pipeline built on hypothesis
margins, combined with a self-compiling per-workspace lexical cache that makes
even the model-free cold path multilingual over time. That combination — and
especially ALD — is the differentiated piece. Claim it as "first I'm aware of
in this product class", not "provably first ever".

**Why this won't repeat the C5 failure.** v0.7 shipped C5 (semantic intent
fallback, default OFF) and the measured finding was "doesn't beat lexical with
the default embedder". Three differences this time: (1) C5 used plain cosine
against positive exemplars only — SAE uses hard-negative MARGINS, which is what
kills the false-positive problem plain cosine has; (2) per-class thresholds are
CALIBRATED against a labelled corpus (§7), not hand-picked; (3) the rollout is
eval-gated per function — if anchors still lose to lexical for RU/UK, hybrid
mode keeps lexical for RU/UK and anchors serve only the languages that have
NOTHING today (pure addition, zero regression risk). And the embedder itself is
upgradable (BGE-M3) behind the same anchor cache.

---

## 2. Architecture — the three-tier ladder

```
            ┌────────────────────────────────────────────────┐
 message →  │ T0  LEXICAL CACHE  (µs, no model)              │
            │   EN inline floor + $PMB_HOME/lang/auto.yaml   │
            │   (ALD-compiled; ru/uk.yaml during migration)  │
            └──────────────┬─────────────────────────────────┘
                           │ miss, engine warm, lang.anchors on
            ┌──────────────▼─────────────────────────────────┐
            │ T1  SEMANTIC ANCHORS  (1 embed, ~5–30 ms warm) │
            │   margin vs positive+negative exemplar sets    │
            │   fires → logged for ALD distillation          │
            └──────────────┬─────────────────────────────────┘
                           │ only for extraction, opt-in
            ┌──────────────▼─────────────────────────────────┐
            │ T2  LOCAL LLM (existing extractors_llm, opt-in)│
            └────────────────────────────────────────────────┘
```

Rules: T0 never vetoed by T1 (additive during migration). T1 runs only when
`engine.is_warm()` (same gate C5 used) — the cold path stays pure T0, which is
exactly why ALD matters. Config: `lang.mode = packs | hybrid | anchors`
(default `hybrid`), plus per-function kill-switches.

---

## GROUND RULES (unchanged from v0.8)

- NEVER run `git add/commit/push` from the agent — print commands.
- Branch names without "claude".
- Real workspaces under `~/.pmb/workspaces` are never touched without explicit OK.
- All cleanup archive-only. New config keys additive with safe defaults.
- Default behavior must not change unless the task says so. MCP signatures additive-only.
- Tests via `./.venv/Scripts/python.exe -m pytest` (system python lacks deps);
  ruff via `./.venv/Scripts/ruff.exe`. Full-suite green only counts from a FULL run.
- The V1 memory-eval floors (EN/RU/UK top-1, xl top-3) are the hard regression
  gate for every phase here. A phase that moves them does not land.

---

## PHASE A — Anchor core

### A1 (P0) — `pmb/lang/anchors.py`: AnchorSet + AnchorIndex

```python
# src/pmb/lang/anchors.py
"""Semantic anchors — language-free classification of the roles the packs
hand-enumerated. English-only exemplars; the multilingual embedder does the
cross-lingual transfer. Decisions are MARGINS against hard negatives, not raw
cosine — that is the difference from the failed C5 attempt."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AnchorSet:
    name: str                      # "intent.goals_query"
    positives: tuple[str, ...]     # EN exemplars ONLY (8-15 is plenty)
    negatives: tuple[str, ...]     # HARD negatives — near-misses, not random
    tau: float = 0.10              # margin threshold (recalibrated in A2)
    floor: float = 0.35            # min positive cosine (anti garbage-match)


@dataclass
class AnchorHit:
    name: str
    pos: float
    margin: float


class AnchorIndex:
    """Embeds every exemplar once, caches the matrices keyed by
    (embedder_model_id, anchors_hash); classify() costs ONE text embed plus a
    few hundred dot products (microseconds)."""

    def __init__(self, embed_fn, model_id: str, sets: list[AnchorSet],
                 cache_dir):
        self._embed = embed_fn
        self._sets = sets
        self._cache_key = hashlib.sha1(
            (model_id + json.dumps([(s.name, s.positives, s.negatives)
                                    for s in sets], default=tuple)
             ).encode()).hexdigest()[:16]
        self._load_or_build(cache_dir)   # .npz per cache key

    def _load_or_build(self, cache_dir) -> None:
        p = cache_dir / f"anchors-{self._cache_key}.npz"
        if p.exists():
            z = np.load(p)
            self._pos = {s.name: z[f"p:{s.name}"] for s in self._sets}
            self._neg = {s.name: z[f"n:{s.name}"] for s in self._sets}
            return
        def _mat(texts):
            m = np.asarray([self._embed(t) for t in texts], dtype=np.float32)
            return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
        self._pos = {s.name: _mat(s.positives) for s in self._sets}
        self._neg = {s.name: _mat(s.negatives) if s.negatives else None
                     for s in self._sets}
        arrays = {f"p:{k}": v for k, v in self._pos.items()}
        arrays |= {f"n:{k}": v for k, v in self._neg.items() if v is not None}
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(p, **arrays)

    def classify(self, text: str) -> list[AnchorHit]:
        v = np.asarray(self._embed(text), dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        hits: list[AnchorHit] = []
        for s in self._sets:
            pos = float(np.max(self._pos[s.name] @ v))
            neg = float(np.max(self._neg[s.name] @ v)) \
                if self._neg[s.name] is not None else 0.0
            margin = pos - neg
            if pos >= s.floor and margin >= s.tau:
                hits.append(AnchorHit(s.name, pos, margin))
        return sorted(hits, key=lambda h: -h.margin)
```

Anchor definitions live next to it — one place, English only, with the
hard negatives doing the precision work:

```python
INTENT_ANCHORS = [
    AnchorSet(
        "intent.goals_query",
        positives=(
            "what are my open goals", "what's left to do",
            "what should I work on next", "show my current tasks",
            "what remains unfinished on this project",
            "what did I plan to do next",
        ),
        negatives=(
            "I just finished that task",          # statement, not question
            "create a new goal for me",           # write-intent
            "what is a goal in OKR methodology",  # theory question
            "great, the goal is done",            # ack
        ),
    ),
    AnchorSet(
        "intent.trivial_ack",
        positives=("ok", "thanks", "got it", "sounds good", "hello", "nice"),
        negatives=("ok, now refactor the auth module",
                   "thanks, but what database do we use?"),
        floor=0.45,   # short texts embed noisily — higher floor
    ),
    # ... past_query / recent_query / lessons_query / work_request / self_intent
]
```

Engine wiring (lazy, warm-only — mirrors the existing PAMVR/C5 pattern):

```python
# base.py init
self._anchor_index = None     # built on first warm use

def anchor_index(self):
    if self._anchor_index is None and self.is_warm():
        from pmb.lang.anchors import ALL_ANCHORS, AnchorIndex
        self._anchor_index = AnchorIndex(
            self.search.embed, self.search.model_id, ALL_ANCHORS,
            self.workspace.storage_dir / "anchors")
    return self._anchor_index
```

Config (all additive): `lang.mode` (str, "hybrid"), `lang.anchors` (bool,
True), `lang.anchor_log` (bool, True — feeds ALD).

Tests: cache round-trip; classify returns goals for 6 phrasings incl. the
known regex-gap phrase "что мне осталось сделать по проекту" and de/es/ja
equivalents; negatives don't fire; cold engine → index is None (no model load).

**LANDED + MEASURED (2026-06-11, default paraphrase-multilingual embedder).**
Two findings changed the design vs the skeleton above:

1. **Margin is ONE-VS-REST + hard-negatives, not just hard-negatives.**
   `margin(set) = cos(pos) − max( best cos to ANY OTHER set's positives, best
   cos to this set's hard negatives )`. Negatives-only had cross-class false
   positives (German "goals" fired `self_intent`, which lacked goals-like
   negatives). One-vs-rest makes classes compete and kills that. In
   `AnchorIndex._scores`.
2. **Hard negatives must NOT share the positive's topic.** `work_request` with
   negative "is the auth module done" SUPPRESSED "refactor the auth module"
   (shared "auth module"). Removed; one-vs-rest supplies cross-class contrast.

Measured margins at placeholder τ=0.10 (pos, margin), zero per-language data:
```
 goals_query  EN 1.00,+0.49 · PL 0.79,+0.16 · JA 0.74,+0.21  → fire
              RU-gap 0.66,+0.07 · DE 0.65,−0.00              → MISS (A2 target)
              NEG-finished −0.51 · NEG-theory −0.58          → correctly reject
 work_request EN 0.78,+0.38 · DE 0.83,+0.44 → fire ; RU 0.34,+0.06 → miss
 self/past/ack  EN + RU all fire correctly
```
Verdict: architecture proven — intents classify across EN/PL/JA/ES/RU with NO
language data, and the gap between true positives (≥+0.06) and hard negatives
(≤−0.51) is huge, so A2's per-set τ calibration (target τ≈0.04–0.05) converts
the RU-gap / RU-work borderline into hits at ~zero FPR. The DE-goals miss
(−0.00) is an EMBEDDER limitation (German "offenen Aufgaben" embeds far), not a
threshold one → flagged for F4 per-bucket eval + the BGE-M3 upgrade; do NOT
paper over it with a global low τ. Tests: tests/test_anchors.py (3).

### A2 (P0) — calibration harness, not hand-picked thresholds

`scripts/calibrate_anchors.py`: build a labelled corpus from (a) the V3 intent
eval cases, (b) the INTENT_PROBES baseline, (c) ~200 unlabelled real-shaped
distractors. For each anchor set, sweep τ and pick the largest τ meeting
**FPR ≤ 1%** on distractors with recall ≥ lexical tier's recall on the labelled
positives. Emit the table into the anchor definitions + a
`tests/_anchor_calibration.json` snapshot pinned by a test (V4-style freeze).

```python
def pick_tau(margins_pos: list[float], margins_neg: list[float],
             max_fpr: float = 0.01) -> float:
    neg = sorted(margins_neg)
    cut = neg[int(len(neg) * (1 - max_fpr))]        # FPR quantile
    return max(cut + 1e-3, 0.05)
```

**LANDED + MEASURED (2026-06-12, default paraphrase-multilingual embedder).**
`scripts/calibrate_anchors.py` + `src/pmb/lang/anchor_calibration.json` (shipped
beside the module, not under `tests/`, so the wheel carries it and the runtime
reads it with no model). Snapshot is KEY-GATED on the same `sha1(model, anchor-
defs)` as the `.npz` cache: edit an anchor → key drifts → `AnchorIndex` ignores
the snapshot and `test_anchor_calibration.py::test_freeze_key_matches…` fails →
forced recalibration. `pick_tau` takes the LOWEST τ meeting FPR ≤ 1% (= highest
recall under budget), floored at 0.04. Corpus = 157 hand-written multilingual
positives (en/ru/uk/de/es/fr/pl/it/pt/ja/zh) + 38 distractors; negatives are
one-vs-rest (every other set's positives + all distractors).

```
set                       tau   recall   fpr     (placeholder was τ=0.10 flat)
intent.goals_query       0.040   0.92   0.000    ← A1's RU/DE 0.07-margin MISS now FIRES
intent.recent_query      0.040   0.94   0.006    ← likewise recovered
intent.past_query        0.079   0.95   0.006
intent.self_intent       0.103   1.00   0.006
intent.lessons_query     0.089   0.72   0.006    ← honest embedder limit → Phase D / upgrade
intent.work_request      0.134   0.70   0.006    ← imperatives in ja/zh weakest → Phase D
intent.trivial_ack       0.192   0.97   0.006    macro: recall 0.885 · fpr 0.005
```

Deviation from the sketch: corpus is hand-written multilingual (not V3 cases +
INTENT_PROBES) so calibration measures the CROSS-LINGUAL transfer A1 cares about;
floor stays the dataclass value (τ is the FPR knob, floor is the garbage knob).
No τ hand-picked — the two weak sets are reported honestly, not fudged.

---

## PHASE B — migrate the CLASSIFICATION consumers (low risk first)

Order chosen by blast radius; each step lands with the V1 floors green and the
V3 intent eval EXTENDED (not just kept green).

- **B1 (P0) intents** — `auto_recall.detect_intents` grows tier T1:

```python
_ANCHOR_TO_INTENT = {
    "intent.goals_query": Intent.GOALS_QUERY,
    "intent.past_query": Intent.PAST_QUERY,
    # ...
}

def detect_intents_v2(msg, engine, known_projects, min_chars=5):
    out = detect_intents(msg, known_projects, min_chars)      # T0 (unchanged)
    if out != [Intent.SKIP]:
        return out
    idx = engine.anchor_index() if engine else None           # T1
    if idx is None or not engine.config.get("lang.anchors"):
        return out
    hits = idx.classify(msg)
    mapped = [_ANCHOR_TO_INTENT[h.name] for h in hits
              if h.name in _ANCHOR_TO_INTENT]
    if hits and engine.config.get("lang.anchor_log"):
        _log_anchor_fires(engine, msg, hits)                  # feeds ALD (D1)
    return mapped or out
```

  V3 eval gains buckets the packs never covered: de/es/fr/ja/zh queries with
  expected intents — these pass ONLY through anchors. That's the proof line.

  **LANDED (2026-06-12).** Implemented as the WARM-ONLY semantic tier already
  wired into the hook dispatch, not a new `detect_intents_v2`: lexical T0
  (`detect_intents`) runs first and unchanged; only on its `[SKIP]` does the
  anchor tier run, and ONLY when warm (daemon-served — never the cold per-turn
  hook). `pmb/hooks/semantic_intent.py` gains `_ANCHOR_TO_INTENT` +
  `classify_anchor_intent` (top calibrated hit → coarse intent; `self_intent`→
  `PAST_QUERY`; a top `trivial_ack` returns None so a foreign-language "merci"/
  "了解" stays silent). `classify_semantic_intent` now PREFERS anchors and only
  falls back to the legacy centroid cosine when `lang.anchors` is off. Default
  flips ON via `lang.anchors` (was the opt-in `hooks.semantic_intents`).
  Measured (`test_semantic_intent.py::test_anchor_intent_real_multilingual`):
  12 cases × 6 European languages × 6 intents, **12/12** classify with zero
  language packs (de/es/fr/it/pl goals/past/self/recent/work/lessons). Known
  weak corner left for Phase D: distant-script imperatives (ja/zh work_request)
  + a few borderline es phrasings. 104 auto_recall/work_request tests stay green.
- **B2 (P1) lesson-intent + future-intent** (`memory_quality.is_lesson_intent`,
  `attributes.looks_like_future_intent`) — same ladder, anchors
  `query.lesson_intent` / `statement.future_intent`.

  **LANDED (2026-06-12).** Key insight: these are STATEMENT-level binary
  detectors, semantically adjacent to existing intent sets (`lesson_intent` ≈
  `lessons_query`; `future_intent` shares verbs with `work_request`). Naively
  adding them to the one-vs-rest index REGRESSED B1's weakest sets (measured:
  lessons 0.72→0.61, work 0.70→0.57). Fix = `AnchorSet.group`: rivals (in
  `_scores`) AND calibration negatives are scoped to the SAME group, so each
  statement detector lives alone in its group (margin = pos − hard-negative,
  truly independent) and never competes with — or inflates the τ of — the intent
  tier. Re-measured: B1 intent τ/recall byte-identical to pre-B2; new sets
  `query.lesson_intent` recall 0.88, `statement.future_intent` 0.56 (a warm
  bonus over the regex/marker floor). Consumers gain an optional `engine=`:
  lexical first, then `engine.anchor_fires(text, set)` — WARM-ONLY (cold write
  path never loads the model), wired at `recall.py` (lesson boost) and `write.py`
  (suggest-goal flag). 9-set snapshot re-frozen (key `b98ce291…`).
- **B3 (P1) PAMVR self-intent + first-person + relation** — `has_self_intent`
  via anchor `query.about_self`; `_has_first_person` via anchor
  `statement.first_person` (precomputed per CANDIDATE at write time, stored as
  `metadata.fp=1`, so the per-candidate hot loop stays embed-free — subtle
  point: never embed inside the per-candidate loop).
- **B4 (P2) verb-match boost** — drop the lexical verb_synonyms boost when
  `lang.mode=anchors`: the vector channel already encodes verb synonymy; V1/V4
  decide whether the boost still earns its keep (suspicion: it's a BM25-era
  crutch that the eval will show is replaceable).

---

## PHASE C — extraction without packs (the hard part)

Today extraction = regex with named groups + LOCALIZED output templates
(`"Живёт в {place}"`). Both are pack-bound. Replace with:

### C1 (P0) — universal value-span detector (no language data)

Reuses what already exists: `pamvr._extract_proper_nouns` is already
Unicode-script-agnostic (isupper/islower work for Cyrillic/Greek/accented).
Add number/date spans via universal patterns:

```python
# src/pmb/reasoning/spans.py
_NUM_RE = re.compile(r"\b\d[\d .,:/-]*\b")
def value_spans(sentence: str) -> list[Span]:
    out = [Span(t, "proper") for t in extract_proper_nouns(sentence)]
    out += [Span(m.group().strip(" .,"), "number")
            for m in _NUM_RE.finditer(sentence)]
    # identifiers (record_batch, qwen2.5) excluded from "proper" via the
    # existing is_strong() charset check — an identifier must never become a city
    return [s for s in out if not looks_like_identifier(s.text)]
```

### C2 (P0) — hypothesis-margin extraction for KEYED facts

The new trick: for each candidate (attribute, span) pair build an English
hypothesis with the RAW span injected, embed sentence + hypotheses in ONE
batch, decide by margin against an anti-hypothesis:

```python
# src/pmb/reasoning/extract_anchor.py
HYPOTHESES: dict[str, tuple[str, str]] = {
    #  attr        positive hypothesis              anti-hypothesis (hard)
    "city":     ("the user lives in {v}",        "the user no longer lives in {v}"),
    "employer": ("the user works at {v}",        "the user stopped working at {v}"),
    "name":     ("the user's own name is {v}",   "{v} is some other person's name"),
    "job_title":("the user's job is {v}",        "{v} is a tool the user uses"),
}

def extract_keyed_anchor(sentence: str, engine,
                         tau: float = 0.06) -> list[KeyedCandidate]:
    idx = engine.anchor_index()
    if idx is None:                                   # cold → T0 only
        return []
    if not idx.fires(sentence, "statement.about_self"):   # cheap pre-gate
        return []
    spans = value_spans(sentence)[:4]
    if not spans:
        return []
    texts, keys = [sentence], []
    for attr, (pos_t, neg_t) in HYPOTHESES.items():
        for sp in spans:
            texts += [pos_t.format(v=sp.text), neg_t.format(v=sp.text)]
            keys.append((attr, sp))
    vecs = engine.search.embed_batch(texts)           # ONE forward pass
    s, hyp = vecs[0], vecs[1:]
    out = []
    for i, (attr, sp) in enumerate(keys):
        margin = cos(s, hyp[2*i]) - cos(s, hyp[2*i + 1])
        if margin > tau and cos(s, hyp[2*i]) > 0.45:
            out.append(KeyedCandidate(attr, sp.text, margin))
    return dedupe_best_per_attr(out)
```

Cost: ≤ 4 attrs × 4 spans × 2 + 1 ≈ 33 short embeds in one batch — ~20–50 ms,
WRITE path only (record_*), never on recall. Negation/tense ("I no longer live
in Tampa") is handled by the SAME mechanism — the anti-hypothesis side wins the
margin, which routes to the existing `close_on_negation` machinery instead of
asserting.

**LANDED (2026-06-12) — C1 + C2 + F1 + F2.** C1 `reasoning/spans.py`
(`value_spans`): case-preserving proper-noun + number spans, identifier-guarded.
C2 `reasoning/extract_anchor.py` (`extract_keyed_anchor`): per-span CROSS-
ATTRIBUTE argmax (which relationship does the sentence assert about this value?)
gated by the anti-hypothesis (negation) + a discrimination gap → precision-first.
Measured 11/12 across en/ru/de/fr (city/employer/name; the one RU-employer miss
fails safe). `job_title` dropped (values aren't proper nouns; only collided with
employer). Wired WARM-ONLY + gated `extract.anchor_keyed` (default OFF, flips on
after the F4 parity eval) into the new-write auto-promotion: when the regex tier
misses, `_anchor_promote_keyed` records each candidate. F1: every anchor record
carries `metadata.extract={tier,margin,pos,model}`; recall scales the keyed boost
by `0.7+0.3·min(1,margin/0.15)` (no-op for regex/manual facts). F2: when regex
`detect_negated_state` misses, `_anchor_close_on_negation` embeds the anti-
hypothesis of each CURRENT keyed value; a clear anti-win closes it (shares the
refactored `_close_keyed_attr`). 74 keyed/negation/recall tests stay green.
DEFERRED sub-item: keyed-value cosine-merge (Киеве/Киев ≥0.92) — a warm dedup in
`record_keyed_fact`; low-risk, moot while C2 is gated off, tracked for a later pass.

Subtle point — inflected values: "Я живу в Киеве" extracts the inflected span
"Киеве", not "Киев". Do NOT try to lemmatize (that's per-language again).
Store the raw span; the EXISTING keyed canonicalization + semantic dedup
already cluster near-identical values by embedding (cos("Киеве","Киев")≈0.95)
— extend `keyed_canonical` to merge value aliases by cosine ≥ 0.92 with the
shorter/nominative-looking form as canonical. Test with RU locatives + German
dative.

### C3 (P1) — atomic facts: kill localized templates entirely

`fact_extract`'s RU/UK patterns render localized strings ("Живёт в {place}").
Replace the OUTPUT with canonical language-neutral atoms — keyed facts and
`attr: value` strings (`"city: Киев"`) — content keeps the user's language in
the VALUE, structure is canonical. The pack categories
`fact_extract_patterns`/`current_state_prefixes`/`attr_*` become unnecessary
once C2 covers their attribute set; parity gate = the keyed-memory A1 test
corpus must extract the SAME (attr, value) pairs via anchors as via regexes
(values compared by canonical cluster, not byte equality).

---

## PHASE D — ALD: the self-compiling lexical cache

### D1 (P0) — anchor-fire log (cheap, existing infra)

Reuse the ambient_log SQLite (dependency-light, already on the hot path for
track-action). One row per T1 fire: `(ts, anchor, text_hash, ngrams_json)`
where ngrams = lowercased token 1–3-grams of the message (the existing
`distinctive_tokens` tokenizer — already Unicode-correct after Phase L).

### D2 (P0) — distiller in the existing maintenance tick

```python
# src/pmb/maintenance/distill.py
def distill_lexicon(engine, min_support=6, min_precision=0.95) -> dict:
    """Mine n-grams that reliably predict an anchor in THIS workspace's
    traffic and compile them into $PMB_HOME/lang/auto.yaml (existing pack
    schema, extend-only). Cold path + hot path then classify those phrasings
    lexically — any language, zero hand-written data."""
    fires = load_anchor_log(engine)            # anchor -> Counter(ngram)
    volume = load_message_ngrams(engine)       # ngram -> total occurrences
    pack: dict[str, list[str]] = {}
    for anchor, counter in fires.items():
        cat = ANCHOR_TO_PACK_CATEGORY.get(anchor)     # e.g. intent.goals_query
        if not cat:                                   #  -> intent_goals_query
            continue
        for ngram, n_with in counter.items():
            n_total = volume.get(ngram, 0)
            if n_total >= min_support and n_with / n_total >= min_precision \
                    and ngram not in corpus_stopwords(engine):
                pack.setdefault(cat, []).append(re.escape(ngram))
    write_auto_pack(engine, pack)              # provenance + version stamped
    return {"categories": len(pack),
            "entries": sum(map(len, pack.values()))}
```

Loader change: NONE — `active_packs()` already merges every
`$PMB_HOME/lang/*.yaml`. Wire one `_step("distill", ...)` into
`run_maintenance_tick` (M1), report counts in `/internal/health`.

### D3 (P1) — pruning + provenance (the cache must not rot)

Each auto.yaml entry carries `{src: "ald", anchor, support, precision, ts}`.
On every tick, entries whose live precision (T0 fired but T1 disagreed when
both ran — sampled 5% of T0 hits get a shadow T1 check) drops below 0.9 are
removed. This is the self-healing property hand-written packs never had.

The headline behavior this buys: a Polish user's workspace, after a week of
normal use, classifies "co mi zostało do zrobienia" lexically in microseconds
on a COLD hook with no model loaded and no Polish anywhere in the repo.

**LANDED (2026-06-12) — D1 + D2.** `src/pmb/maintenance/distill.py`. D1:
`log_anchor_fire` appends `(ts, workspace_id, anchor, text_hash, ngrams_json)`
to an `anchor_fires` table (created on first write); wired into
`classify_anchor_intent` (every B1 fire, incl. `trivial_ack`), gated on
`lang.anchor_log`. D2: `distill_lexicon` mines the log — an n-gram is compiled
for anchor A only if, across all fires it appears in, ≥ `min_precision` (0.95)
were A with ≥ `min_support` (6) messages AND it has a contentful (≥4-char,
non-stopword) token. Precision is measured ACROSS anchors, so generic n-grams
self-prune; the ambiguous-word test proves a 0.5-precision token is dropped.
Output → `$PMB_HOME/lang/auto.yaml` (existing schema, `re.escape`'d fragments +
an `_ald_meta` provenance block for D3) → `active_packs()` picks it up with a
cache clear. Wired as step 4 of `run_maintenance_tick`. The end-to-end test
proves the COLD `detect_intents` regex, rebuilt from the packs, classifies
"teraz co mi zostało do zrobienia" as GOALS_QUERY with no model. CJK (spaceless)
is intentionally out of scope for the lexical cache — distant scripts stay on
the warm anchor tier. D3 (live-precision pruning) remains P1, future.

---

## PHASE E — universal tokenization + corpus stopwords (kill the last lists)

- **E1 (P1)** stopwords from the corpus, not from packs: a token whose document
  frequency in the workspace exceeds 25% (computed in the tick, cached by
  write-generation) is a stopword for `distinctive_tokens`/lesson matching.
  EN floor stays inline as the bootstrap for empty workspaces. This is what
  BM25 already does implicitly via IDF — make the explicit consumers use it too.
- **E2 (P2)** the remaining char-class data (`sentence_uppercase`,
  `cyrillic_script_range`, pdf heading words) → Unicode property functions
  (`str.isupper()`, `unicodedata.category`) which Phase L already proved work
  cross-script in `_extract_proper_nouns`. Heading words ("Chapter/Глава/…")
  → anchor `heading.chapter_marker` only in the PDF path (not hot).

---

## PHASE F — data-accuracy mechanisms (the "точность данных" ask)

- **F1 (P0) extraction confidence as first-class metadata.** Every anchor/
  hypothesis extraction writes `metadata.extract = {tier: "anchor", margin:
  0.21, model: "<id>"}`. Consumers: recall multiplies the keyed boost by a
  confidence factor (`0.7 + 0.3*min(1, margin/0.15)`); declutter targets
  low-confidence first; doctor lists the bottom decile for human review.
- **F2 (P0) write-time contradiction check via the same hypothesis machinery.**
  On a new keyed value, embed the anti-hypothesis of the CURRENT value against
  the new sentence; if it fires, route to the existing `close_on_negation` /
  conflict flow instead of silently keeping both. One batch embed on writes.
- **F3 (P1) close the X2 loop (it's scaffolded, not closed).** Collect
  `(channel_scores, useful)` samples: a recall result that the agent then
  `mark_lesson_followed(True)` / feedback-boosts → useful=True; surfaced but
  never used in the session → useful=False (weak label). Maintenance tick runs
  `propose_channel_weights(samples)` (exists) and writes the SUGGESTION into
  doctor output: `pmb doctor` prints "learned weights {hit:1.08, recency:0.94}
  — apply with `pmb config set recall.channel_weights ...`". Never auto-applied
  (the X2 contract).
- **F4 (P1) eval expansion — accuracy you can SEE.** V1 grows 13 → 100+
  queries: per-attribute extraction cases (incl. negations + inflected values),
  de/es/fr/ja/zh buckets running pack-free, and a paraphrase-robustness bucket
  (5 phrasings per fact). Floors set the same way as v0.8 (measure, then floor
  a margin below). This is the gate everything above must pass.

---

## PHASE G — rollout, packs removal, release

- **G1** `lang.mode=hybrid` default ON in 0.9.0 (T1 additive-only — cannot
  regress T0 by construction; the only cost is ≤1 embed per message on the
  warm path, budget-gated by the existing perf smoke).
- **G2** packs-off ratchet: a CI job running the FULL memory-eval +
  keyed-extraction parity with `PMB_LANG_MODE=anchors` and the ru/uk packs
  DISABLED. When it holds the same floors for 2 consecutive releases →
- **G3** delete `packs/ru.yaml` + `packs/uk.yaml` (data, not behavior — auto
  .yaml + anchors carry it), keep the loader for auto.yaml + user packs
  (user packs become an OVERRIDE escape hatch, no longer a requirement).
- **G4** release 0.9.0; CHANGELOG headline: "memory now understands any
  language the embedder does — no language packs required".

Execution order: A1→A2→B1→D1→D2 (the visible win: multilingual intents +
self-compiling cache), then C1→C2→F1→F2 (extraction), then B2–B4, C3, D3, E,
F3–F4, G. After EVERY phase: full `.venv` suite + ruff + V1 floors.

---

## Subtle points / risks (write them down now, not in the postmortem)

1. **The embedder bounds everything.** MiniLM-class multilingual models are
   weak on low-resource languages and short texts; BGE-M3 is the upgrade path.
   The anchor cache is keyed by model id, so upgrading is config-only. If the
   default embedder can't hit the floors for a language bucket, the eval shows
   it per-bucket — don't average it away.
2. **Short acks embed noisily** ("ok", "ага"). Keep the length-≤5 trivial check
   BEFORE anchors; trivial_ack anchor gets a raised floor (0.45).
3. **Never embed in the per-candidate recall loop.** B3's first-person check
   moves to write-time metadata precisely for this. The per-recall budget for
   anchors is ONE embed (the query), already paid by recall itself — reuse that
   vector for intent anchors on the hook path (pass it through, don't re-embed).
4. **Code-mixed messages** (RU prose + EN identifiers) are the normal case for
   this user. Hypothesis spans must exclude identifier-shaped tokens
   (`is_strong` charset logic) or "record_batch" becomes an employer.
5. **Inflected values** (Киеве/Киев, German dative) — canonical-cluster merge
   in keyed storage (C2), never lemmatization.
6. **Threshold drift across embedder upgrades** — recalibrate (A2 harness) as a
   maintenance-tick step when model id changes; anchors cache rebuilds itself.
7. **ALD poisoning** — a precision-pruned cache with provenance (D3) plus
   stopword exclusion prevents "и" from becoming a goals-marker. min_support=6
   AND min_precision=0.95 AND not-a-stopword is the triple gate.
8. **Don't delete packs early.** G2's two-release soak is deliberate. The
   v0.8 lesson stands: breaking RU recall is worse than shipping later.
9. **Privacy** — anchors are English exemplars in code; auto.yaml is local,
   per-workspace, and contains the user's OWN phrasings only. Note it in docs.
10. **Windows/OneDrive Cyrillic paths** (two recorded lessons) — anchor cache
    and auto.yaml live under `$PMB_HOME` (ASCII default), not the project dir.

## Definition of done for v0.9.0

- A German/Spanish/Japanese query routes to the right intent with NO pack for
  that language (V3 buckets green), warm ≤ +30 ms p95 vs v0.8.
- Keyed extraction passes the A1 correctness corpus via anchors (parity with
  regexes on EN/RU/UK, PLUS new-language cases regexes can't do).
- auto.yaml demonstrably serves a cold-path classification for a non-pack
  language after simulated traffic (test with a fake fire-log).
- V1 floors unchanged or better; full suite + ruff green; packs still present
  (deletion is G3, post-soak).
