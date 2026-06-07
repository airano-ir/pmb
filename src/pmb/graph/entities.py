"""
Entity extraction from raw event text.

Rule-based and fast — runs at event-write time. Designed so we never block
on an LLM call during normal use. Deep semantic extraction can be added
later in the consolidation pass.

Three layers:

1. **File paths.** Regex for posix-style paths (`src/auth.py`, `tests/test_x.py`)
   and bare filenames with a programming-language extension. Plus the
   `files_changed` metadata field already populated by git sync.

2. **Tech names.** Closed-set match against `KNOWN_TECHS`. Trades recall for
   precision — better to miss `Bun` than to call every English word a tech.
   Case-insensitive but only on word boundaries.

3. **Concepts.** Conservative noun-like tokens (length ≥ 4, not in stopword
   list, not pure-digit). Capped per event to avoid bloat.

All entity names are normalized: lowercased, whitespace-collapsed. Files
keep their case to stay matchable against real file paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# Known technologies — closed set, lowercase canonical name → display alias list.
# Match against any alias on word boundaries.
KNOWN_TECHS: dict[str, list[str]] = {
    "postgres": ["postgres", "postgresql", "pg"],
    "mysql": ["mysql"],
    "sqlite": ["sqlite"],
    "redis": ["redis"],
    "mongodb": ["mongodb", "mongo"],
    "kafka": ["kafka"],
    "rabbitmq": ["rabbitmq"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    "fastapi": ["fastapi"],
    "django": ["django"],
    "flask": ["flask"],
    "starlette": ["starlette"],
    "react": ["react"],
    "nextjs": ["nextjs", "next.js"],
    "vue": ["vue", "vuejs"],
    "vite": ["vite"],
    "webpack": ["webpack"],
    "rollup": ["rollup"],
    "esbuild": ["esbuild"],
    "tailwind": ["tailwind", "tailwindcss"],
    "typescript": ["typescript", "ts"],
    "python": ["python"],
    "rust": ["rust"],
    "golang": ["golang", "go"],
    "node": ["node", "nodejs"],
    "bun": ["bun"],
    "deno": ["deno"],
    "jwt": ["jwt"],
    "oauth": ["oauth", "oauth2"],
    "graphql": ["graphql"],
    "rest": ["rest"],
    "grpc": ["grpc"],
    "pytest": ["pytest"],
    "jest": ["jest"],
    "vitest": ["vitest"],
    "lancedb": ["lancedb"],
    "qdrant": ["qdrant"],
    "pinecone": ["pinecone"],
    "chroma": ["chroma", "chromadb"],
    "anthropic": ["anthropic", "claude"],
    "openai": ["openai", "gpt", "gpt-4", "gpt-4o", "codex"],
    "ollama": ["ollama"],
    "huggingface": ["huggingface", "hf"],
    "llama": ["llama", "llama3"],
    "qwen": ["qwen"],
    "mistral": ["mistral"],
}


# Build a single combined regex for all aliases for one pass over text
def _build_tech_regex() -> re.Pattern:
    aliases = []
    for variants in KNOWN_TECHS.values():
        for v in variants:
            aliases.append(re.escape(v))
    # Sort longest-first so 'nextjs' matches before 'next'
    aliases.sort(key=len, reverse=True)
    pattern = r"(?<![\w/.-])(?:" + "|".join(aliases) + r")(?!\w)"
    return re.compile(pattern, re.IGNORECASE)


_TECH_RE = _build_tech_regex()


# File path: posix-style with a known extension, or bare filename.ext
_FILE_RE = re.compile(
    r"(?<![\w/])"
    r"(?:[A-Za-z0-9_\-./]+/)*"               # optional dirs
    r"[A-Za-z0-9_\-.]+"
    r"\.(?:py|js|ts|tsx|jsx|go|rs|java|kt|swift|rb|php|c|h|cpp|hpp|cs|"
    r"sql|sh|ps1|yml|yaml|toml|json|md|html|css|scss|vue|svelte|conf|"
    r"ini|env|lock|cfg|exe|dll|so|dylib|bat|cmd)\b"
)

# Windows-style absolute paths: C:\foo\bar\..., \\server\share\..., or any
# token containing 2+ backslashes / forward slashes. The concept extractor
# strips these wholesale before tokenizing so AppData/Roaming/Users etc.
# don't pollute the concept set. The Unicode `\w` char-class catches Cyrillic
# / Greek / accented folder names too — without it `C:\Users\…\Рабочий стол\`
# would leak "Рабочий" and "стол" into the concept set.
_WINPATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\)[\w\-.\\/ :]+",
    re.IGNORECASE | re.UNICODE,
)
# Posix-style absolute paths too, including any path segment containing 2+
# slashes — works on macOS / Linux logs and on git diff headers.
_POSIXPATH_RE = re.compile(
    r"/(?:[\w\-.]+/){1,}[\w\-.]*",
    re.UNICODE,
)


_STOPWORDS = {
    # Articles, demonstratives, pronouns
    "the", "a", "an", "this", "that", "these", "those", "it", "we", "you",
    "they", "them", "their", "his", "her", "hers", "him", "she", "him",
    "your", "yours", "mine", "ours", "us", "our",
    # Auxiliary / modal verbs
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "should", "could", "may",
    "might", "must", "can", "cannot", "shall", "want", "wants", "wanted",
    "need", "needs", "needed", "got", "get", "gets", "getting", "make",
    "makes", "made", "making", "take", "takes", "took", "taken",
    # Conjunctions / discourse
    "and", "or", "but", "if", "then", "so", "because", "since", "though",
    "although", "however", "while", "until", "unless", "even", "either",
    "neither", "nor", "yet", "still",
    # Prepositions
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "as",
    "into", "onto", "upon", "over", "under", "above", "below", "between",
    "among", "against", "around", "about", "after", "before", "behind",
    "during", "through", "throughout", "across", "along", "via", "near",
    # Question / Wh-words
    "what", "where", "who", "whom", "whose", "when", "how", "why", "which",
    "whether",
    # Dialogue roles (filter so "user" / "agent" / "assistant" don't bloat)
    "user", "agent", "assistant", "system", "bot", "human", "ai",
    "chatbot", "model", "tool",
    # Generic technical terms — too common to be useful as concept nodes
    "code", "function", "class", "method", "value", "name", "data",
    "type", "object", "instance", "default", "config", "setting",
    "settings", "param", "params", "arg", "args",
    # Action / connector words
    "using", "used", "use", "uses", "based", "set", "sets", "setting",
    "got", "gets", "done", "doing",
    # Time-related fragments
    "today", "yesterday", "tomorrow", "now", "later", "soon", "ago",
    "week", "weeks", "month", "months", "year", "years", "day", "days",
    "hour", "hours", "minute", "minutes",
    # Conversation filler
    "okay", "yeah", "yes", "sure", "right", "good", "great", "fine",
    "thanks", "thank", "please", "hello", "hi", "hey", "bye",
    "sounds", "looks", "seems", "say", "says", "said", "tell", "told",
    "ask", "asked", "talk", "talked", "speak", "spoke",
    # Common verbs (past tense often pollutes)
    "decided", "decides", "deciding", "switched", "switches", "switching",
    "broke", "breaks", "broken", "made", "makes", "making",
    "ran", "runs", "running", "went", "goes", "going",
    "came", "comes", "coming", "lived", "lives", "living",
    "added", "adds", "adding", "removed", "removes", "removing",
    "flew", "flies", "flying", "moved", "moves", "moving",
    # Misc descriptors
    "primary", "secondary", "old", "new", "latest", "current", "previous",
    "first", "second", "third", "last", "next", "final", "initial",
    "main", "core", "basic", "simple", "complex", "important", "key",
    # Self-referential project nouns
    "project", "session", "layer", "feature", "module", "service",
    "endpoint", "endpoints", "request", "response",
    "frontend", "backend", "fullstack", "client", "server", "database",
    "cache", "queue", "worker", "pipeline", "monorepo", "platform",
    "framework", "library", "package", "bundle", "build", "deploy",
    "deployment", "release", "version", "support",
    # Common path / Windows folder tokens that survive as bare concepts
    "appdata", "roaming", "programfiles", "programdata", "users",
    "desktop", "documents", "downloads", "system32", "windows", "apps",
    "onedrive", "dropbox", "icloud", "googledrive",
    # Russian / Ukrainian folder names that leak from `C:\Users\...\Рабочий стол\`
    # paths even after the Windows-path stripper has done its pass.
    "рабочий", "стол", "документы", "загрузки", "рабочий стол",
    # Acronym fragments that aren't useful as nodes
    "lgbtq", "lgbt", "covid",
    # Generic English nouns / qualifiers that clog the concept set.
    # Almost every event mentions "ideas", "work", "thing" — they're noise.
    "idea", "ideas", "thing", "things", "way", "ways", "fact", "facts",
    "stuff", "lots", "couple", "anything", "everything", "something",
    "anyone", "someone", "everyone", "nobody", "somebody", "anybody",
    "place", "places", "area", "areas", "point", "points", "side", "sides",
    "case", "cases", "kind", "kinds", "sort", "sorts",
    # Adjectives commonly extracted as concepts
    "full", "only", "deep", "high", "low", "long", "short", "real", "fake",
    "easy", "hard", "smart", "dumb", "open", "closed", "available",
    "different", "similar", "various", "many", "much", "more", "less",
    "free", "paid", "private", "public", "local", "remote", "global",
    "specific", "general", "common", "rare", "weird", "strange",
    # Past-tense verbs commonly picked up by the noun-shaped regex
    "built", "created", "installed", "uninstalled", "opened", "closed",
    "deleted", "finished", "started", "completed", "fixed", "broke",
    "broken", "removed", "added", "called", "sent", "received",
    "received", "loaded", "saved", "exported", "imported", "renamed",
    "moved", "copied", "pulled", "pushed", "merged", "rebased", "reverted",
    "logged", "configured", "updated", "upgraded", "downgraded",
    "deployed", "shipped", "released", "tested", "verified",
    "checked", "skipped", "tracked", "pinned", "archived",
    "implemented", "applied", "rejected", "accepted", "approved",
    "found", "lost", "missed", "caught", "hit", "tried", "failed",
    "worked", "stopped", "paused", "resumed", "restored",
    # Reporting / dialogue past-tense
    "explained", "described", "mentioned", "noted", "shared", "showed",
    "claimed", "argued", "proposed", "suggested", "agreed", "disagreed",
}


# Hardening (H5): Unicode-aware concept matcher. The previous `[a-z]…`
# regex silently dropped every Cyrillic, Greek, accented Latin etc. word —
# which meant the entity-graph layer was effectively single-lingual even
# though the embedder / BM25 already handle 50+ languages. `\w` with
# re.UNICODE matches any letter or digit in any script; `\d` is excluded
# from the head so pure-numeric tokens don't enter the graph.
_CONCEPT_RE = re.compile(r"(?u)\b[^\W\d_]\w{3,}\b", re.UNICODE)

# L2: multi-word noun phrases. Captures 2-3 consecutive capitalized words
# ("Claude Code", "Deep Research", "JWT Auth Bug") so they end up as one
# entity instead of fragmenting into "claude" + "code" + "deep" + "research".
# Allows lower-case connectors ("Bank of America"). Anchored at a capital
# letter to keep precision high; not perfect but cheap and high-signal.
_PHRASE_RE = re.compile(
    r"\b[A-Z][\w\-]+(?:\s+(?:of|the|de|la|and|&)\s+)?"
    r"(?:\s+[A-Z][\w\-]+){1,2}\b",
    re.UNICODE,
)


def extract_phrases(text: str, max_n: int = 6) -> list[str]:
    """Find capitalized 2-3 word phrases — proper-noun-shaped concepts.

    Examples it catches: "Claude Code", "Deep Research", "JWT Auth Bug",
    "Bank of America". Filtered against the stop-list so "The Code" /
    "Code Review" don't slip through if every component is a stopword.
    """
    out: list[str] = []
    seen: set[str] = set()
    # Strip absolute paths first — they're full of capitalized folder names
    # that would otherwise turn into "Users Alexb", "Program Files" etc.
    scrubbed = _WINPATH_RE.sub(" ", text)
    scrubbed = _POSIXPATH_RE.sub(" ", scrubbed)
    for m in _PHRASE_RE.finditer(scrubbed):
        phrase = " ".join(m.group(0).split())
        key = phrase.lower()
        if key in seen:
            continue
        # Reject if EVERY word is a stopword/connector — that's a false hit
        # on something like "The Code" or "The Future". A multi-word phrase
        # like "Claude Code" / "Deep Research" should survive: even though
        # "code"/"deep" are single-word stopwords, the proper-noun first
        # element gives the phrase meaning.
        words = [w.lower() for w in re.findall(r"[\w\-]+", phrase)]
        meaningful = [w for w in words if w not in _STOPWORDS
                      and w not in ("of", "the", "de", "la", "and")]
        if not meaningful:
            continue
        seen.add(key)
        out.append(phrase)
        if len(out) >= max_n:
            break
    return out


def extract_file_paths(text: str) -> list[str]:
    """Find file paths inside text. Case-preserving."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _FILE_RE.finditer(text):
        norm = m.group(0).replace("\\", "/").strip()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def extract_techs(text: str) -> list[str]:
    """Return canonical tech names (lowercased) mentioned in text."""
    seen: set[str] = set()
    # Map alias → canonical for fast lookup
    alias_to_canon: dict[str, str] = {}
    for canon, aliases in KNOWN_TECHS.items():
        for a in aliases:
            alias_to_canon[a.lower()] = canon

    for m in _TECH_RE.finditer(text):
        canon = alias_to_canon.get(m.group(0).lower())
        if canon and canon not in seen:
            seen.add(canon)
    return sorted(seen)


def extract_concepts(text: str, max_n: int = 8) -> list[str]:
    """
    Conservative concept extraction: lowercase tokens of length ≥ 4 that
    aren't stopwords, aren't already tech names (those go to extract_techs),
    aren't pure digits, and aren't substrings of file paths.

    Capped at `max_n` to keep edge density manageable.
    """
    tech_set = set(extract_techs(text))
    file_blob = " ".join(extract_file_paths(text)).lower()

    # Strip Windows AND posix-style paths BEFORE tokenizing so AppData/Roaming/
    # Users / Рабочий-стол / home / opt don't leak into the concept set.
    scrubbed = _WINPATH_RE.sub(" ", text)
    scrubbed = _POSIXPATH_RE.sub(" ", scrubbed)

    seen: set[str] = set()
    out: list[str] = []
    for m in _CONCEPT_RE.finditer(scrubbed.lower()):
        tok = m.group(0)
        if tok in _STOPWORDS or tok in seen:
            continue
        if tok in tech_set:
            continue
        if tok.isdigit():
            continue
        if tok in file_blob:
            continue
        # Reject short hex strings — they're nearly always session-tag
        # noise (e.g. "be8974" from seed scripts) rather than real concepts.
        # A 4-8 char token made entirely of [0-9a-f] with at least one digit
        # is suspicious; if it ALSO has no English-letter pattern, drop it.
        if 4 <= len(tok) <= 8 and all(c in "0123456789abcdef" for c in tok):
            # Has at least one digit → it's hex tag, not a word like "face".
            if any(c.isdigit() for c in tok):
                continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_n:
            break
    return out


@dataclass
class ExtractedEntities:
    files: list[str]
    techs: list[str]
    concepts: list[str]
    # spaCy / LLM backends can populate these too. Regex backend leaves them
    # empty. all_named() emits them as their proper graph-kind, so the
    # entity graph distinguishes a PERSON node from a CONCEPT node.
    persons: list[str] = None  # type: ignore[assignment]
    orgs: list[str] = None     # type: ignore[assignment]
    places: list[str] = None   # type: ignore[assignment]
    products: list[str] = None # type: ignore[assignment]

    def __post_init__(self):
        # dataclass defaults must be immutable; promote None → fresh list.
        if self.persons  is None: self.persons  = []
        if self.orgs     is None: self.orgs     = []
        if self.places   is None: self.places   = []
        if self.products is None: self.products = []

    def all_named(self) -> list[tuple[str, str]]:
        """Returns [(kind, name), ...] in stable order."""
        out = [("file", f) for f in self.files]
        out += [("tech", t) for t in self.techs]
        out += [("person", p) for p in self.persons]
        out += [("org", o) for o in self.orgs]
        out += [("place", p) for p in self.places]
        out += [("tool", p) for p in self.products]
        out += [("concept", c) for c in self.concepts]
        return out


class EntityExtractor:
    """Stateless wrapper around the three regex layers — the default backend.

    Sub-classes can override `extract` to add spaCy POS-filter / NER / LLM
    extraction. The engine always talks to this interface so swapping the
    backend is config-only — no recall code changes.
    """

    backend_name = "regex"

    def __init__(self, max_concepts: int = 8):
        self.max_concepts = max_concepts

    def extract(self, text: str, files_hint: Iterable[str] = ()) -> ExtractedEntities:
        """
        Extract entities. `files_hint` lets the caller pass git-sync's
        `files_changed` metadata so file entities don't depend on regex
        finding them inside the formatted commit text.
        """
        files = list(dict.fromkeys(
            [*extract_file_paths(text), *(_normalize_path(f) for f in files_hint)]
        ))
        techs = extract_techs(text)
        # L2: pull multi-word noun phrases first ("Claude Code" → 1 node).
        # Their tokens are then suppressed from the single-word concept layer
        # so we don't get both "claude code" AND "claude" + "code".
        phrases = extract_phrases(text, max_n=max(2, self.max_concepts // 2))
        phrase_tokens = {
            w for ph in phrases for w in ph.lower().split() if len(w) >= 4
        }
        single_concepts = [
            c for c in extract_concepts(text, max_n=self.max_concepts + 4)
            if c not in phrase_tokens
        ][: self.max_concepts]
        # Phrases live alongside concepts; viz colours them the same.
        concepts = [*phrases, *single_concepts]
        return ExtractedEntities(files=files, techs=techs, concepts=concepts)

    def extract_batch(
        self,
        items: "list[tuple[str, tuple]]",
    ) -> list[ExtractedEntities]:
        """Default batch is just per-event loop — overridden by LLMExtractor
        to send N events in one CLI call. Lets the engine always call
        `extract_batch` without checking the backend type.
        """
        return [self.extract(t, fh) for (t, fh) in items]


def _normalize_path(p: str) -> str:
    return p.replace("\\", "/").strip()


# ----------------------------------------------------------------------
# Backend factory — picks the right extractor based on `graph.extractor`
# config. The default is "regex" (the implementation above). Other choices
# load their backend module lazily so an extra dep (spaCy / LLM CLI) is
# only required when the user actually opts in.
# ----------------------------------------------------------------------

def make_extractor(config=None, max_concepts: int = 8) -> EntityExtractor:
    """Return the extractor selected by `config.get('graph.extractor')`.

    Recognised values:
      - "regex" (default): fast, offline, no deps
      - "spacy":  POS-filter + NER via spaCy; needs `pip install pmb-ai[spacy]`
      - "llm:claude":  Claude Code CLI extracts concepts at write time
      - "llm:ollama":  local Ollama model extracts concepts at write time
      - "llm:codex":   OpenAI Codex CLI extracts concepts at write time

    Unknown values silently fall back to regex (the safest choice — the rest
    of the pipeline keeps working).
    """
    backend = "regex"
    if config is not None:
        try:
            backend = (config.get("graph.extractor") or "regex").strip().lower()
        except Exception:
            backend = "regex"

    if backend in ("regex", "", "default"):
        return EntityExtractor(max_concepts=max_concepts)

    if backend == "spacy":
        try:
            from pmb.graph.extractors_spacy import SpacyExtractor
            return SpacyExtractor(max_concepts=max_concepts)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "graph.extractor=spacy unavailable (%s) — falling back to regex", e)
            return EntityExtractor(max_concepts=max_concepts)

    if backend.startswith("llm:"):
        provider = backend.split(":", 1)[1]
        try:
            from pmb.graph.extractors_llm import LLMExtractor
            _get = getattr(config, "get", lambda *_: None)
            return LLMExtractor(
                provider=provider,
                max_concepts=int(_get("graph.llm_max_concepts") or 5),
                timeout_s=float(_get("graph.llm_timeout_s") or 30),
                model=str(_get("graph.llm_model") or ""),
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "graph.extractor=%s unavailable (%s) — falling back to regex", backend, e)
            return EntityExtractor(max_concepts=max_concepts)

    # Unrecognised — keep working with the safe default.
    import logging
    logging.getLogger(__name__).warning(
        "Unknown graph.extractor=%r — using regex", backend)
    return EntityExtractor(max_concepts=max_concepts)
