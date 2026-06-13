"""
Adaptive Layer Routing (Improvement E).

PMB has 6 semantic layers stored in events:
  raw events, entities, reflections, causation edges, arcs, atomic facts,
  bi-temporal index.

Different question types are best answered by different layers:
  "What port does Postgres use?"        → atomic facts (direct lookup)
  "When did Alice meet Bob?"            → temporal index + facts
  "What happened after the migration?"  → causation edges + raw
  "Tell me about the auth refactor"     → arcs + raw
  "Why did Bob leave?"                  → reflections + inferential search

This module:
  1. Classifies the query into one or more intent types via cheap regex.
  2. Returns per-layer multipliers that the recall pipeline applies to
     base scores, biasing which layer surfaces in the top-K.

It does NOT replace existing recall layers - it tunes their relative
contribution PER QUERY. Layers stay always-on (for safety) but their
weight is dynamic.

Cost: <0.1ms (regex over query string). No LLM.

Why this matters:
  Without routing, atomic facts (Improvement D) dominate top-K because they
  outnumber other event types ~15:1. Great for direct lookups, bad for
  narrative questions (where raw session context is richer). Routing
  re-balances based on query intent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

# ----------------------------------------------------------------------
# Intent detection patterns
# ----------------------------------------------------------------------

_TEMPORAL_RE = re.compile(
    r"\b(when|date|day|month|year|after|before|during|since|until|"
    r"yesterday|today|tomorrow|last|next|ago|earlier|previously|"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)

_MULTIHOP_RE = re.compile(
    r"\b(after|before|because|due to|caused|led to|then|next|earlier|"
    r"previously|why did|what happened (?:after|when)|following|preceding|"
    r"subsequently|as a result|consequence|prior to|in response|reaction|"
    r"triggered|prompted|since|until)\b",
    re.IGNORECASE,
)

_NARRATIVE_RE = re.compile(
    r"\b(history of|tell me about|what's been|how did .* end up|"
    r"summary of|overview|the story of|arc|journey|evolution|"
    r"what (?:has )?happened (?:with|to)|all about|what do (?:we|you) know about|"
    r"talk through|walk me through|recap|context)\b",
    re.IGNORECASE,
)

_INFERENTIAL_RE = re.compile(
    r"\b(why|would|could|should|might|imply|suggest|seem|appear|"
    r"is .* (?:a|an) |consider(?:ed)? |type of|kind of)\b",
    re.IGNORECASE,
)

# Identity intent: query is about the USER or a named person, not a topic.
# Examples: "Who is the user?", "Where do I live?", "What languages does
# the user use?", "What's my favourite editor?", "Who am I working with?".
# When this fires, facts starting with possessive markers ("User's...",
# the user's own name, "My...") get a small boost, because the user is
# asking about the entity, not about something topical.
#
# Without this gate, the personal-marker boost was firing on ANY query
# touching a fact whose content starts that way - e.g. "What's the JWT
# lifetime?" returning "<name>'s terminal is Wezterm..." at top-1 just
# because the matching event happened to start with a possessive.
#
# NOTE: the query-side patterns reference only GENERIC self-words
# ("user", "i", "me", "my"). The USER'S OWN NAME ("who is Bob",
# "where does Bob live") is matched dynamically via
# `_identity_re_for_names`, fed by the mined user-name cache - we never
# hardcode a specific personal name here.

# Shared fragments - reused by the static identity regex AND the
# per-user-name dynamic regex so the two never drift.
_ID_ATTRS = (
    r"name|email|stack|language|languages|editor|terminal|preference|"
    r"preferences|home|location|allergy|allergies"
)
_ID_TOOLS = r"languages?|tools?|stacks?"
_ID_LIKES = r"like|love|hate|prefer|avoid"

_IDENTITY_RE = re.compile(
    r"\b("
    r"who is (?:the |my )?(?:user|i|me)\b|"
    r"who am i\b|"
    r"who (?:is|are) we\b|"
    r"who (?:do|did) i work|"
    r"who (?:'?s|s) on the team\b|"
    r"who is on the team\b|"
    rf"what(?:'?s)? (?:my|the user'?s?) (?:{_ID_ATTRS})|"
    r"where (?:does|do) (?:the user|i) (?:live|work)|"
    rf"what (?:{_ID_TOOLS}) (?:does|do) (?:the user|i) (?:use|prefer)|"
    rf"what (?:does|do) (?:the user|i) (?:{_ID_LIKES})|"
    r"who am i working with\b"
    r")",
    re.IGNORECASE,
)


@lru_cache(maxsize=128)
def _identity_re_for_names(names_key: tuple[str, ...]) -> re.Pattern | None:
    """Build (cached) an identity regex for the user's OWN names, mirroring
    the static patterns that used to hardcode a single name. `names_key` is
    a sorted tuple of lowercased names (hashable, for the lru_cache)."""
    names = [re.escape(n) for n in names_key if n]
    if not names:
        return None
    alt = "|".join(names)
    return re.compile(
        r"\b("
        rf"who is (?:the |my )?(?:{alt})\b|"
        rf"what(?:'?s)? (?:{alt})'?s? (?:{_ID_ATTRS})|"
        rf"where (?:does|do) (?:{alt}) (?:live|work)|"
        rf"what (?:{_ID_TOOLS}) (?:does|do) (?:{alt}) (?:use|prefer)|"
        rf"what (?:does|do) (?:{alt}) (?:{_ID_LIKES})"
        r")",
        re.IGNORECASE,
    )

# Historical intent: "what did we use before", "previous", "old setup",
# "before the migration", "originally", "used to". Fires when the user
# explicitly asks about the PAST state of something whose current state
# has been superseded. We slightly boost older events and avoid letting
# the latest pinned/recency-heavy fact dominate.
_HISTORICAL_RE = re.compile(
    r"\b("
    # "what/where/how did X (before|previously|originally|formerly)" anywhere in the tail
    r"(?:what|where|how) (?:did|do|does|were|was) .*\b(?:before|previously|originally|formerly|prior)\b|"
    # Explicit "previous/former/old/original/prior" qualifier on subject
    r"(?:previous|former|old|original|prior|legacy) (?:setup|stack|deployment|target|approach|system|database|provider|host)|"
    # "what was our prior X"
    r"what (?:was|were) (?:our|the) (?:previous|former|old|original|prior)\b|"
    # "before the migration", "before we moved"
    r"before (?:the |we |they |our )?(?:migration|move|switch|change|move to|switch to)|"
    # "we used to / formerly used / previously used"
    r"\b(?:used to|formerly used|previously used|once used)\b"
    r")\b",
    re.IGNORECASE,
)


# Decision intent: "what did we decide", "did we agree", "team decision",
# "final call", "what's the verdict", "what was concluded", "resolution".
# When this fires, events of type=activity with kind ∈ {decision, decided,
# agreed, resolved, concluded} should outrank arguments / positions for the
# same topic. Without this, a query like "what did the team decide about X"
# can return the LOUDEST argument instead of the actual decision activity.
_DECISION_RE = re.compile(
    r"\b("
    r"what (?:did|do|does) .* decide|"
    r"what (?:was|is|were) (?:the )?decision|"
    r"what'?s? (?:the )?(?:decision|verdict|conclusion|outcome|resolution)|"
    r"did (?:we|they|the team) (?:agree|decide|settle|resolve)|"
    r"(?:team|final|our) (?:decision|verdict|conclusion|call)|"
    r"what (?:was|is) (?:the )?(?:agreed|chosen|picked|settled)|"
    r"what (?:was|did) .* (?:agreed|decided|concluded|resolved)|"
    r"what (?:are|were) we going with|"
    r"how (?:did|do) (?:we|they|the team) decide"
    r")\b",
    re.IGNORECASE,
)

# "Direct lookup" pattern - short factual question
_DIRECT_RE = re.compile(
    r"^\s*(what|who|where|which) (is|are|was|were|did) ",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

@dataclass
class LayerWeights:
    """Multiplier for each layer's contribution to final recall score.
    1.0 = unchanged. 0.0 = disable layer's boost. 2.0 = double weight."""
    facts_boost: float = 1.0         # event_type='fact_atom' importance
    reflections_boost: float = 1.0   # event_type='reflection' importance
    raw_boost: float = 1.0           # event_type='qa'/'fact'/'event' etc.
    graph_boost_mul: float = 1.0     # multiplier on graph_boost config
    causation_boost_mul: float = 1.0 # multiplier on causation_boost
    arc_boost_mul: float = 1.0       # multiplier on arc_boost
    temporal_boost_mul: float = 1.0  # multiplier on temporal_boost
    ppr_weight_mul: float = 1.0      # multiplier on ppr_weight
    # Multiplier for `activity` events whose `metadata.kind` marks them as
    # the conclusive outcome (decision / agreed / resolved / concluded).
    # Fired by the "decision" intent; lets the actual team decision outrank
    # the loudest argument when the user asks "what did we decide".
    decision_boost: float = 1.0
    # When "historical" intent fires, we want OLDER events (which usually
    # carry the superseded state) to outrank the latest pinned facts.
    # Implemented as: a negative offset on the recency-half-life multiplier
    # (so newer is no longer rewarded) plus a flat boost for older events.
    historical_recency_mul: float = 1.0   # 1.0 = normal recency curve;
                                          # <1.0 weakens recency boost;
                                          # >1.0 strengthens it.
    older_event_bonus: float = 0.0        # additive bonus per "older than
                                          # median timestamp" candidate.
    # When "identity" intent fires, facts whose content starts with a
    # personal-possessive marker ("User's", the user's own name, "I am ",
    # "My ", etc.) get a small multiplicative boost. Default 1.0 = no boost;
    # the router bumps to ~1.15 on identity-shaped queries.
    identity_marker_boost: float = 1.0


@dataclass
class QueryIntent:
    query: str
    types: list[str] = field(default_factory=list)
    weights: LayerWeights = field(default_factory=LayerWeights)
    rationale: str = ""


# ----------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------

class QueryRouter:
    """Classifies queries and returns layer multipliers.

    Pure stateless - same query → same weights.

    Multiple intent types can fire simultaneously; in that case
    multipliers compose (max of competing values, since boosts are
    additive in scoring).
    """

    def classify(
        self, query: str, user_names: set[str] | None = None
    ) -> QueryIntent:
        if not query or not query.strip():
            return QueryIntent(query=query, types=["direct"], rationale="empty")

        q = query.strip()
        types: list[str] = []
        notes: list[str] = []

        if _TEMPORAL_RE.search(q):
            types.append("temporal")
            notes.append("temporal pattern matched")
        if _MULTIHOP_RE.search(q):
            types.append("multi_hop")
            notes.append("multi-hop pattern matched")
        if _NARRATIVE_RE.search(q):
            types.append("narrative")
            notes.append("narrative pattern matched")
        if _INFERENTIAL_RE.search(q):
            types.append("inferential")
            notes.append("inferential pattern matched")
        if _DECISION_RE.search(q):
            types.append("decision")
            notes.append("decision-intent pattern matched")
        if _HISTORICAL_RE.search(q):
            types.append("historical")
            notes.append("historical-intent pattern matched")
        id_hit = bool(_IDENTITY_RE.search(q))
        if not id_hit and user_names:
            dyn = _identity_re_for_names(tuple(sorted(user_names)))
            id_hit = bool(dyn is not None and dyn.search(q))
        if id_hit:
            types.append("identity")
            notes.append("identity-intent pattern matched")

        # Direct lookup if nothing else strongly fired, or query is short
        # factual ("what is X?" style)
        if not types or (_DIRECT_RE.match(q) and len(q.split()) <= 8):
            types.append("direct")
            notes.append("direct lookup pattern")

        # Compose weights
        weights = LayerWeights()

        if "direct" in types:
            # Direct lookups benefit from atomic facts (mem0-style).
            # Raw context is less important; arcs/causation hurt.
            weights.facts_boost = 1.5
            weights.arc_boost_mul = 0.5
            weights.causation_boost_mul = 0.5
            weights.ppr_weight_mul = 0.5

        if "temporal" in types:
            # Temporal questions need date-anchored events.
            # Boost temporal proximity heavily; facts also help for date lookup.
            weights.temporal_boost_mul = max(weights.temporal_boost_mul, 2.0)
            weights.facts_boost = max(weights.facts_boost, 1.3)
            weights.arc_boost_mul = min(weights.arc_boost_mul, 0.7)

        if "multi_hop" in types:
            # Multi-hop needs causation walk + PPR; less single-fact reliance.
            weights.causation_boost_mul = max(weights.causation_boost_mul, 2.0)
            weights.ppr_weight_mul = max(weights.ppr_weight_mul, 1.5)
            weights.graph_boost_mul = max(weights.graph_boost_mul, 1.5)
            weights.facts_boost = max(weights.facts_boost, 1.2)

        if "narrative" in types:
            # Narrative queries want arc summaries + raw context.
            # Penalize fragmented atomic facts.
            weights.arc_boost_mul = max(weights.arc_boost_mul, 2.0)
            weights.raw_boost = max(weights.raw_boost, 1.5)
            weights.facts_boost = min(weights.facts_boost, 0.6)

        if "inferential" in types:
            # "Why / would / should" - reflections capture interpretations.
            weights.reflections_boost = max(weights.reflections_boost, 1.5)
            weights.raw_boost = max(weights.raw_boost, 1.2)

        if "decision" in types:
            # "What did we decide / agree / conclude" - the answer is a
            # decision activity, not the loudest argument. Boost activity
            # events whose metadata.kind is in the decision family.
            # 1.8 picked empirically: enough to outrank typical surface-form
            # competitors (~1.5 raw_boost on similar phrasing) but not so
            # high that an off-topic decision wins.
            weights.decision_boost = max(weights.decision_boost, 1.8)
            # Reduce raw boost slightly so arguments don't beat the decision
            # purely on lexical match.
            weights.raw_boost = min(weights.raw_boost, 0.9)

        if "historical" in types:
            # "What did we use before" - the LATEST pinned fact is the wrong
            # answer; we want the superseded state. Flatten the recency
            # boost and add a small bonus for older events.
            weights.historical_recency_mul = min(weights.historical_recency_mul, 0.25)
            weights.older_event_bonus = max(weights.older_event_bonus, 0.10)

        if "identity" in types:
            # User-about-self / user-about-team query. Facts starting with
            # personal possessives are the most likely match.
            weights.identity_marker_boost = max(weights.identity_marker_boost, 1.15)

        return QueryIntent(
            query=query,
            types=types,
            weights=weights,
            rationale="; ".join(notes),
        )
