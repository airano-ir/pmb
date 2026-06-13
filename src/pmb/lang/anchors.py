"""Semantic Anchor Engine (SAE) - language-free classification of the roles the
hand-written lang packs enumerate.

Every function a pack serves (is this a goals question? a trivial ack? a
self-referential statement?) is re-expressed as a small set of ENGLISH-ONLY
exemplars: POSITIVES plus HARD NEGATIVES. A text is classified by MARGIN -
`max cos(text, positives) − max cos(text, negatives)` - against a calibrated
per-set threshold, with a floor on the positive similarity to reject garbage
matches. The multilingual embedder already shipped (and already loaded on the
daemon path) does the cross-lingual transfer: a German or Japanese "what are my
open goals" lands near the English anchors with NO German/Japanese data
anywhere in the system.

Why margins, not raw cosine (the v0.7 C5 attempt used raw cosine and lost to
lexical): the hard negatives subtract the "looks vaguely on-topic" baseline, so
a statement like "I just finished that goal" does not fire `goals_query` even
though it shares the word "goal". Thresholds are CALIBRATED (scripts/
calibrate_anchors.py, Phase A2) at FPR ≤ 1%, not hand-picked.

Cost: exemplars are embedded ONCE and cached to an `.npz` keyed by
(embedder model id, anchor-definition hash). `classify()` is then one text
embed plus a few hundred dot products. Used WARM-ONLY (engine.is_warm()); the
cold path stays on the lexical packs, which is exactly what Phase D's
Anchor→Lexicon Distillation keeps multilingual over time.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

# Calibration snapshot lives beside this module so it ships in the wheel and the
# runtime can read it with no model load. scripts/calibrate_anchors.py writes it;
# tests/test_anchor_calibration.py freezes it.
CALIBRATION_FILE = "anchor_calibration.json"


@dataclass(frozen=True)
class AnchorSet:
    """One classifiable role. `name` is dotted (`intent.goals_query`).
    `positives`/`negatives` are ENGLISH exemplars (8-15 positives is plenty;
    negatives should be NEAR-MISSES, not random sentences). `tau` is the margin
    threshold (recalibrated in A2); `floor` is the minimum positive cosine.

    `group` scopes the one-vs-rest competition: a set's rivals are only OTHER
    sets in the SAME group. So the query-intent sets compete among themselves
    (B1) while the statement detectors (B2 - is-this-a-lesson / future-plan)
    compete among themselves, and the two tasks never suppress each other - a
    future-plan statement must not have to out-score `goals_query`."""
    name: str
    positives: tuple[str, ...]
    negatives: tuple[str, ...] = ()
    tau: float = 0.10
    floor: float = 0.35
    group: str = "intent"


@dataclass(frozen=True)
class AnchorHit:
    name: str
    pos: float       # best positive cosine
    margin: float    # pos − best negative cosine


def _norm(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


class AnchorIndex:
    """Embeds every exemplar once (cached), then classifies by margin.

    `embed_batch` maps a list[str] → np.ndarray[n, dim] (the engine's
    `search.embed_batch`). `model_id` keys the cache so an embedder upgrade
    rebuilds it automatically.
    """

    def __init__(self, embed_batch, model_id: str, sets: list[AnchorSet],
                 cache_dir):
        from pathlib import Path
        self._embed_batch = embed_batch
        self._model_id = str(model_id)
        self._sets = list(sets)
        sig = json.dumps([(s.name, s.positives, s.negatives) for s in self._sets],
                         ensure_ascii=False, sort_keys=True)
        self._key = hashlib.sha1((str(model_id) + "\n" + sig).encode("utf-8")
                                 ).hexdigest()[:16]
        self._pos: list[np.ndarray] = []
        self._neg: list[np.ndarray | None] = []
        # Thresholds default to the dataclass placeholders; A2's calibration
        # snapshot (keyed by the SAME (model, anchor-defs) hash as the .npz
        # cache) overrides them per set. Editing an anchor's exemplars changes
        # the key, so a stale snapshot is ignored until recalibrated - the
        # freeze test then fails loudly rather than silently scoring new anchors
        # against thresholds tuned for the old ones.
        self._tau: dict[str, float] = {s.name: s.tau for s in self._sets}
        self._floor: dict[str, float] = {s.name: s.floor for s in self._sets}
        self.calibrated = False
        self._load_or_build(Path(cache_dir))
        self._load_calibration()

    def _load_calibration(self) -> None:
        """Override per-set tau/floor from the committed calibration snapshot
        (scripts/calibrate_anchors.py, Phase A2) when it matches THIS index's
        (model, anchor-defs) key. Best-effort: a missing or mismatched snapshot
        leaves the dataclass placeholders untouched."""
        from pathlib import Path
        p = Path(__file__).with_name(CALIBRATION_FILE)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        if data.get("key") != self._key:
            return  # snapshot is for different anchors/embedder → ignore
        for name, row in (data.get("sets") or {}).items():
            if name in self._tau:
                if "tau" in row:
                    self._tau[name] = float(row["tau"])
                if "floor" in row:
                    self._floor[name] = float(row["floor"])
        self.calibrated = True

    def _load_or_build(self, cache_dir) -> None:
        p = cache_dir / f"anchors-{self._key}.npz"
        try:
            if p.exists():
                z = np.load(p)
                self._pos = [z[f"p{i}"] for i in range(len(self._sets))]
                self._neg = [z[f"n{i}"] if f"n{i}" in z else None
                             for i in range(len(self._sets))]
                return
        except Exception:
            pass  # corrupt cache → rebuild
        arrays: dict[str, np.ndarray] = {}
        for i, s in enumerate(self._sets):
            pos = _norm(np.asarray(self._embed_batch(list(s.positives)),
                                   dtype=np.float32))
            self._pos.append(pos)
            arrays[f"p{i}"] = pos
            if s.negatives:
                neg = _norm(np.asarray(self._embed_batch(list(s.negatives)),
                                       dtype=np.float32))
                self._neg.append(neg)
                arrays[f"n{i}"] = neg
            else:
                self._neg.append(None)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(p, **arrays)
        except Exception:
            pass  # cache is an optimisation; never fail classification on it

    def _scores(self, text: str) -> list[tuple[AnchorSet, float, float]]:
        """Per set: pos = best cosine to its positives; margin = pos minus the
        STRONGEST competing signal - the best positive of any other set IN THE
        SAME GROUP (one-vs-rest within the task, so a class must out-score its
        rivals) OR this set's own hard negatives (which catch same-topic-but-
        wrong-act cases other classes can't, e.g. 'I finished that goal').
        Whichever is higher wins, because both are reasons NOT to fire."""
        v = np.asarray(self._embed_batch([text]), dtype=np.float32)[0]
        v = v / (np.linalg.norm(v) + 1e-9)
        pos = [float(np.max(self._pos[i] @ v)) for i in range(len(self._sets))]
        out = []
        for i, s in enumerate(self._sets):
            rivals = max((pos[j] for j, r in enumerate(self._sets)
                          if j != i and r.group == s.group), default=0.0)
            hardneg = float(np.max(self._neg[i] @ v)) if self._neg[i] is not None else 0.0
            margin = pos[i] - max(rivals, hardneg)
            out.append((s, pos[i], margin))
        return out

    def score_map(self, text: str) -> dict[str, AnchorHit]:
        """Every set's (pos, margin) for one text, fired or not. The calibration
        harness (A2) sweeps tau over these; the fire-log (D1) records them."""
        return {s.name: AnchorHit(s.name, pos, margin)
                for (s, pos, margin) in self._scores(text)}

    def classify(self, text: str) -> list[AnchorHit]:
        """Every set whose positive floor AND calibrated margin threshold are
        met, best margin first."""
        hits = [AnchorHit(s.name, pos, margin)
                for (s, pos, margin) in self._scores(text)
                if pos >= self._floor[s.name] and margin >= self._tau[s.name]]
        return sorted(hits, key=lambda h: -h.margin)

    def fires(self, text: str, name: str) -> bool:
        """Cheap single-set check (extraction pre-gate)."""
        for (s, pos, margin) in self._scores(text):
            if s.name == name:
                return pos >= self._floor[name] and margin >= self._tau[name]
        return False


# ── Anchor definitions (English-only). The hard negatives do the precision work.
# Thresholds here are PLACEHOLDERS; scripts/calibrate_anchors.py (A2) overwrites
# `tau`/`floor` from a labelled corpus at FPR ≤ 1%.

INTENT_ANCHORS: list[AnchorSet] = [
    AnchorSet(
        "intent.goals_query",
        positives=(
            "what are my open goals", "what's left to do",
            "what should I work on next", "show my current tasks",
            "what remains unfinished on this project",
            "what did I plan to do next", "what's still pending",
            "remaining work on this",
        ),
        negatives=(
            "I just finished that task", "create a new goal for me",
            "what is a goal in OKR methodology", "great, the goal is done",
            "mark this task complete",
        ),
    ),
    AnchorSet(
        "intent.past_query",
        positives=(
            "what did I do yesterday", "why did we choose this",
            "what did we decide about the database", "who is Alice",
            "where did I put the config", "what did I work on last week",
        ),
        negatives=(
            "what should I do next", "create a new note",
            "what is the capital of France",
        ),
    ),
    AnchorSet(
        "intent.recent_query",
        positives=(
            "what did we just discuss", "what are we doing right now",
            "what just happened", "where did I leave off",
            "what was I working on a moment ago",
        ),
        negatives=("what did I do last month", "what's the plan for tomorrow"),
    ),
    AnchorSet(
        "intent.lessons_query",
        positives=(
            "what are the project conventions", "do we have a rule about commits",
            "what's the convention here", "what lessons did we learn",
            "how should I do this in this project",
        ),
        negatives=("what is a coding convention in general",
                   "fix the bug in auth"),
    ),
    AnchorSet(
        "intent.work_request",
        positives=(
            "fix the auth module", "refactor this function",
            "add a test for the parser", "implement the new endpoint",
            "let's deploy the service", "clean up this code",
            "optimize the query", "rewrite the parser", "build the new feature",
        ),
        # NOTE: hard negatives for work_request must NOT share the work's topic
        # (e.g. "is the auth module done" suppressed "refactor the auth module").
        # One-vs-rest already supplies the cross-class competition; the hard
        # negative here is the QUESTION-about-past-work case only.
        negatives=("what did we fix last week",),
    ),
    AnchorSet(
        "intent.self_intent",
        positives=(
            "who am I", "where do I live", "what's my name",
            "what do I prefer", "what is my job",
        ),
        negatives=("where does Alice live", "who is the CEO of Stripe"),
    ),
    AnchorSet(
        "intent.trivial_ack",
        positives=("ok", "thanks", "got it", "sounds good", "hello", "nice",
                   "cool", "great"),
        negatives=("ok, now refactor the auth module",
                   "thanks, but what database do we use?"),
        floor=0.45,   # short texts embed noisily → higher floor
    ),
]

# B2 - STATEMENT-level binary detectors. Each lives in its OWN group, so it has
# NO rivals (margin = pos − hard-negative only): they are independent yes/no
# detectors, not a one-vs-rest classification, and must not compete with the
# intent tier or each other. Consumed warm-only by memory_quality.is_lesson_intent
# (a recall boost) and reasoning.attributes.looks_like_future_intent (a write-time
# suggest-goal flag); the cold path keeps its lexical floor.
STATEMENT_ANCHORS: list[AnchorSet] = [
    AnchorSet(
        "query.lesson_intent",
        positives=(
            "how should I do this in this project",
            "what's the convention for commits here",
            "what's the best way to structure this",
            "is there a rule about this",
            "how do we usually handle errors",
            "what should I avoid here",
            "are there any gotchas with this setup",
            "what's the right approach for tests",
            "how is this project configured",
        ),
        negatives=(
            "fix the auth module", "what did we do yesterday",
            "the build passed", "what time is the meeting",
        ),
        group="lesson_intent",
    ),
    AnchorSet(
        "statement.future_intent",
        positives=(
            "next we'll add the export feature",
            "the plan is to migrate to Postgres",
            "we're going to refactor the parser",
            "I will write the tests tomorrow",
            "going forward we should cache results",
            "the next step is to wire up the API",
            "we intend to support multi-region",
            "let's plan to ship by Friday",
        ),
        negatives=(
            "we added the export feature yesterday",
            "we decided to use Postgres",
            "the parser was refactored last week",
            "the tests are already written",
        ),
        group="future_intent",
    ),
    # C2 pre-gate / B3 first-person: does this STATEMENT assert a personal
    # attribute of the USER? Fires the keyed hypothesis-extraction pass and is
    # reused as the warm first-person signal. Own group → independent (pos −
    # hardneg), never competes with intents.
    AnchorSet(
        "statement.about_self",
        positives=(
            "I live in Tampa", "my name is Alex", "I work at Stripe",
            "I'm allergic to peanuts", "my favorite colour is blue",
            "I was born in Kyiv", "I prefer dark mode",
            "my job is software engineer", "I moved to Berlin last year",
        ),
        negatives=(
            "Alice lives in Paris", "the CEO is John Smith",
            "we decided to use Postgres", "the build passed yesterday",
            "the cat is sleeping on the sofa",
        ),
        group="about_self",
    ),
]

ALL_ANCHORS: list[AnchorSet] = [*INTENT_ANCHORS, *STATEMENT_ANCHORS]
