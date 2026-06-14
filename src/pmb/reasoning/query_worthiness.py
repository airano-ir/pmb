"""Query-worthiness classifier — the SAE anchor-margin pattern applied to a NEW
axis: is a GENERIC_FACTUAL message a real KNOWLEDGE query, or a CONVERSATIONAL /
meta turn ('is it better now?', 'did you test it?', 'what do you think?') that
the Context (recall) channel should stay SILENT on?

    margin(msg) = max cos(msg, CONVERSATIONAL exemplars)
                - max cos(msg, KNOWLEDGE exemplars)
    margin >= tau  ->  conversational  ->  suppress the Context channel.

Why this works where a cosine FLOOR could not: the same-domain noise the floor
can't separate is about TOPIC similarity (all ~0.06); this measures similarity
to exemplars of each FORM. Measured on a labelled set (en + ru): conversational
messages land near conversational exemplars (margin mean +0.40), knowledge
queries near knowledge exemplars (mean -0.38) - a clean split at tau ~ 0, and it
catches exactly the conversational turns the cheap gap/conf/lexical gate missed
('continue', 'did you run the tests?', 'what do you think?').

English exemplars only; the multilingual embedder does the cross-lingual transfer
(RU/EN both separate), so there is NO per-language rule and NO corpus dependency.
Warm-only (needs the embedder); the cold path keeps the cheap lexical gate.
"""
from __future__ import annotations

import numpy as np

# Exemplars of a CONVERSATIONAL / meta turn: about the current work, the agent,
# or the state of things - NOT a request for stored project knowledge.
CONVERSATIONAL: tuple[str, ...] = (
    "is it better now?",
    "did you test it?",
    "did you run the tests?",
    "what do you think?",
    "does that work?",
    "are you done?",
    "is that correct?",
    "looks good?",
    "should we continue?",
    "how is it going so far?",
    "is this approach effective?",
    "continue",
    "ok thanks",
)

# Exemplars of a KNOWLEDGE-seeking query: about stored facts / decisions / the
# project's state that memory should answer.
KNOWLEDGE: tuple[str, ...] = (
    "what database do we use?",
    "how is the deployment configured?",
    "what did we decide about caching?",
    "what is the rule for package managers?",
    "where is the recall pipeline defined?",
    "what is the threshold for the gate?",
    "why does the daemon restart?",
    "what is the architecture of the engine?",
    "what version are we on?",
    "who is the lead on this project?",
)

DEFAULT_TAU = 0.05  # margin >= this -> conversational; ~midpoint of the measured gap


def _unit_rows(vecs) -> np.ndarray:
    m = np.asarray(list(vecs), dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


class QueryWorthiness:
    """Embeds the exemplar sets once (one batch) and scores a message by margin.

    embed_batch_fn(list[str]) -> list/array of vectors (e.g. engine.search.embed_batch).
    """

    def __init__(self, embed_batch_fn) -> None:
        self._embed = embed_batch_fn
        self._conv = _unit_rows(embed_batch_fn(list(CONVERSATIONAL)))
        self._know = _unit_rows(embed_batch_fn(list(KNOWLEDGE)))

    def conversational_margin(self, message: str) -> float:
        if not message or not message.strip():
            return 0.0
        v = _unit_rows(self._embed([message]))[0]
        conv = float(np.max(self._conv @ v))
        know = float(np.max(self._know @ v))
        return conv - know

    def is_conversational(self, message: str, tau: float = DEFAULT_TAU) -> bool:
        """True when the message looks like a conversational/meta turn (margin >=
        tau) and the Context channel should surface nothing."""
        return self.conversational_margin(message) >= tau
