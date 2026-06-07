"""spaCy-backed entity extractor — POS filter + NER, no LLM, fully offline.

Replaces the regex 'concepts' layer (which keeps every 4-letter token) with:
  - POS filter: only NOUN / PROPN tokens
  - NER: PERSON, ORG, GPE/LOC, PRODUCT, WORK_OF_ART, EVENT

Files + tech keep their fast regex paths — spaCy can't beat known-set lookup
on a 50-entry tech list, and posix-style paths are proper-noun-shaped anyway.

Optional dep. Activate with:

    pip install spacy
    python -m spacy download en_core_web_sm        # English
    python -m spacy download xx_ent_wiki_sm        # multilingual fallback

then in config:

    pmb config set graph.extractor spacy

If neither model is installed, the engine logs a warning and falls back to
the regex backend (in `entities.make_extractor`).
"""
from __future__ import annotations

import logging
import threading
from typing import Iterable, Optional

from pmb.graph.entities import (
    EntityExtractor, ExtractedEntities,
    extract_file_paths, extract_techs, _normalize_path, _STOPWORDS,
    _WINPATH_RE, _POSIXPATH_RE,
)

log = logging.getLogger(__name__)

_NLP = None
_NLP_LOCK = threading.Lock()
_NLP_MODEL_NAME: Optional[str] = None


def _load_nlp():
    """Load the first available spaCy model. Cached process-wide."""
    global _NLP, _NLP_MODEL_NAME
    if _NLP is not None:
        return _NLP
    with _NLP_LOCK:
        if _NLP is not None:
            return _NLP
        try:
            import spacy
        except ImportError as e:
            raise RuntimeError(
                "spaCy not installed. Run: pip install spacy\n"
                "  and: python -m spacy download en_core_web_sm"
            ) from e
        # Try the small English model first (it's what most users will
        # `pmb doctor`-install), then the multilingual NER-only fallback.
        candidates = ("en_core_web_sm", "en_core_web_md", "xx_ent_wiki_sm")
        last_err = None
        for name in candidates:
            try:
                # Disable tagger/lemmatizer/parser pieces we don't use to keep
                # ~40 % of init time and memory.
                _NLP = spacy.load(name, disable=["lemmatizer", "attribute_ruler"])
                _NLP_MODEL_NAME = name
                log.info("graph.extractor=spacy loaded %s", name)
                return _NLP
            except OSError as e:  # model not present on disk
                last_err = e
        raise RuntimeError(
            "No spaCy model found. Install one:\n"
            "  python -m spacy download en_core_web_sm\n"
            f"(last error: {last_err})"
        )


# spaCy NER label → PMB graph kind. Anything not in this map is dropped.
_NER_TO_KIND = {
    "PERSON": "person",
    "PER":    "person",
    "ORG":    "org",
    "GPE":    "place",
    "LOC":    "place",
    "PRODUCT": "tool",
    "WORK_OF_ART": "topic",
    "EVENT":  "event",
}


class SpacyExtractor(EntityExtractor):
    """POS + NER backend. Concepts get POS-filtered, named entities get
    promoted to person / org / place / tool kinds.
    """

    backend_name = "spacy"

    def __init__(self, max_concepts: int = 8):
        super().__init__(max_concepts=max_concepts)
        # Force a load at construction time so a missing model surfaces
        # immediately (the factory then falls back to regex), not on the
        # first write where it would silently corrupt the graph.
        _load_nlp()

    def extract(self, text: str, files_hint: Iterable[str] = ()) -> ExtractedEntities:
        nlp = _load_nlp()
        # Files + techs: keep their fast regex paths.
        files = list(dict.fromkeys(
            [*extract_file_paths(text), *(_normalize_path(f) for f in files_hint)]
        ))
        techs = extract_techs(text)

        # Strip absolute paths before spaCy sees them — same rationale as the
        # regex backend: AppData / Рабочий стол / home/user shouldn't end up
        # as PROPN entities.
        scrubbed = _WINPATH_RE.sub(" ", text)
        scrubbed = _POSIXPATH_RE.sub(" ", scrubbed)
        # Cap input to spaCy's default 1M-char limit even though we'll never
        # see anywhere near that — defensive.
        if len(scrubbed) > 200_000:
            scrubbed = scrubbed[:200_000]

        doc = nlp(scrubbed)

        # ---- NER ----
        persons: list[str] = []
        orgs: list[str] = []
        places: list[str] = []
        products: list[str] = []
        for ent in doc.ents:
            kind = _NER_TO_KIND.get(ent.label_)
            if not kind:
                continue
            name = " ".join(ent.text.split()).strip()
            if len(name) < 2:
                continue
            lower = name.lower()
            if lower in _STOPWORDS:
                continue
            if kind == "person" and lower not in [p.lower() for p in persons]:
                persons.append(name)
            elif kind == "org" and lower not in [o.lower() for o in orgs]:
                orgs.append(name)
            elif kind == "place" and lower not in [p.lower() for p in places]:
                places.append(name)
            elif kind == "tool" and lower not in [p.lower() for p in products]:
                products.append(name)

        # ---- Concepts via POS filter ----
        tech_set = {t.lower() for t in techs}
        file_blob = " ".join(files).lower()
        already = (
            {p.lower() for p in persons}
            | {o.lower() for o in orgs}
            | {p.lower() for p in places}
            | {p.lower() for p in products}
            | tech_set
        )
        concepts: list[str] = []
        seen: set[str] = set()
        for tok in doc:
            if tok.pos_ not in ("NOUN", "PROPN"):
                continue
            if tok.is_stop or tok.is_punct or tok.is_space or tok.like_num:
                continue
            t = tok.text.lower().strip()
            if len(t) < 4 or t in _STOPWORDS or t in seen or t in already:
                continue
            if t in file_blob:
                continue
            seen.add(t)
            concepts.append(t)
            if len(concepts) >= self.max_concepts:
                break

        return ExtractedEntities(
            files=files,
            techs=techs,
            concepts=concepts,
            persons=persons,
            orgs=orgs,
            places=places,
            products=products,
        )
