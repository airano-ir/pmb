from __future__ import annotations

import math
import re
import time
from typing import Any

from pmb import lang as _lang
from pmb.core import circuit_breaker as _breaker
from pmb.core.engine.types import (
    RecallPack,
    RecallResult,
    _collapse_reflections,
    _looks_multihop,
)
from pmb.core.events import (
    Event,
)
from pmb.core.recall_cache import make_recall_cache_key
from pmb.core.search import SearchHit
from pmb.reasoning.pamvr import (
    apply_pamvr as _pamvr_apply,
)
from pmb.reasoning.pamvr import (
    prepare_query_features as _pamvr_prepare,
)
from pmb.reasoning.scoring import combine_base_score as _combine_base_score
from pmb.reasoning.scoring import importance_factor as _imp_factor
from pmb.reasoning.user_names import (
    mine_user_names_from_db as _mine_user_names,
)
from pmb.signals.decay import boost_on_recall

# Personal-attribute intent gate (R6). EN question words / first-person markers
# inline; the RU/UK equivalents (which/when/where/my in RU & UK) live in packs
# (recall_qwords / recall_first_person) so this module stays Cyrillic-free (L1).
# Built ONCE at import, not per recall.
_QWORD_RE = re.compile(
    r"\b(" + "|".join(["where", "when", "why", "what", "who", "which", "how"]
                      + [str(x) for x in _lang.merged_list("recall_qwords")]) + r")\b",
    re.IGNORECASE,
)
_ATTR_RE = re.compile(
    r"\b(" + "|".join(["i", "my", "me", "mine", "myself", "user", "user_", "u", "user'?s"]
                      + [str(x) for x in _lang.merged_list("recall_first_person")]) + r")\b",
    re.IGNORECASE,
)
# E2: lexical-overlap tokenizer uses a Unicode letters+digits class (`[^\W_]` =
# word char minus underscore - any script), so no enumerated Cyrillic range.
_WORD_NUM = r"[^\W_]+"


def _result_in_project(r, project_lc: str) -> bool:
    """True if a RecallResult is tied to `project_lc` (case-insensitive
    substring against project_name / project / project_path metadata)."""
    meta = r.metadata if isinstance(r.metadata, dict) else {}
    for k in ("project_name", "project"):
        v = meta.get(k)
        if isinstance(v, str) and project_lc in v.lower():
            return True
    pp = meta.get("project_path")
    return isinstance(pp, str) and project_lc in pp.lower()


class RecallMixin:
    def _ensure_user_names_state(self) -> None:
        if not hasattr(self, "_user_names_cache"):
            self._user_names_cache: set[str] = set()
            self._user_names_dirty = True       # force a first-call refresh
            self._writes_since_names = 0
            self._user_names_last_refresh = 0.0

    def _get_user_names(self) -> set[str]:
        """Lowercased set of names the user calls themselves, mined from
        "My name is X" / "my name is …" facts (any language). Cached.

        Refreshed when: (a) first call, (b) a name-statement write marked the
        cache dirty - so a newly recorded name takes effect on the very NEXT
        recall, not after 25 more events, (c) 25 writes have accrued, or (d) a
        300 s TTL elapsed (covers a name written by ANOTHER process). No
        per-recall ``SELECT COUNT(*)`` - the common path is a flag check, which
        matters in a long-lived daemon serving many recalls.

        Single source of truth for BOTH the PAMVR self-reference rescue and the
        identity-marker boost in recall(), so neither path hardcodes a specific
        personal name. Safe to call when PAMVR is disabled.
        """
        self._ensure_user_names_state()
        now = time.time()
        if (self._user_names_dirty
                or self._writes_since_names >= 25
                or (now - self._user_names_last_refresh) > 300.0):
            try:
                self._user_names_cache = _mine_user_names(self.workspace.db_path)
            except Exception:
                pass
            self._user_names_dirty = False
            self._writes_since_names = 0
            self._user_names_last_refresh = now
        return self._user_names_cache

    def mark_user_names_dirty(self) -> None:
        """Write-path hook: a name-statement fact was just recorded, so the
        next _get_user_names() must re-mine immediately."""
        self._ensure_user_names_state()
        self._user_names_dirty = True

    def note_user_names_write(self) -> None:
        """Write-path hook: advance the cheap write-since-refresh counter (no
        SQL) so the cache still refreshes periodically without a name signal."""
        self._ensure_user_names_state()
        self._writes_since_names += 1

    def _active_count_cached(self) -> int:
        """S9: active-event COUNT(*) memoized by the recall-cache write
        generation. recall() reads `n_total_in_workspace` several times per
        call; the count only moves on a write, which already bumps the
        generation, so this turns N COUNT(*) per recall into a dict lookup."""
        gen = getattr(self.recall_cache, "_generation", 0)
        cached = getattr(self, "_active_count_cache", None)
        if cached is not None and cached[0] == gen:
            return cached[1]
        n = self.events.count(self.workspace.id, include_archived=False)
        self._active_count_cache = (gen, n)
        return n

    def recall(
        self,
        query: str,
        top_k: int | None = None,
        recency_half_life_days: float | None = None,
        graph_boost: float | None = None,
        rerank: bool | None = None,
        rerank_top_n: int | None = None,
        _skip_decompose: bool = False,
    ) -> RecallPack:
        """D4 singleflight wrapper: collapse concurrent IDENTICAL top-level
        recalls (daemon multi-client / agent fan-out) so the work runs once -
        followers wait for the leader and reuse its result. On timeout or a
        leader error they compute independently, so this can never deadlock.
        Recursive sub-query calls (``_skip_decompose=True``) and the disabled
        config bypass it entirely. All real work is in ``_recall_impl``."""
        if _skip_decompose or not self.config.get("recall.singleflight"):
            return self._recall_impl(
                query, top_k, recency_half_life_days, graph_boost, rerank,
                rerank_top_n, _skip_decompose)
        import threading as _th
        if top_k is None:
            top_k = self.config.get("recall.top_k")
        key = (self.workspace.id, (query or "").strip().lower(), int(top_k))
        leader = False
        with self._sf_lock:
            ev = self._sf_inflight.get(key)
            if ev is None:
                ev = _th.Event()
                self._sf_inflight[key] = ev
                leader = True
        if leader:
            try:
                res = self._recall_impl(
                    query, top_k, recency_half_life_days, graph_boost, rerank,
                    rerank_top_n, _skip_decompose)
                ev._sf_result = res          # carried on the event, no dict to GC
                ev._sf_ok = True
                return res
            except BaseException:
                ev._sf_ok = False
                raise
            finally:
                with self._sf_lock:
                    self._sf_inflight.pop(key, None)
                ev.set()
        # follower: wait for the leader, fall back to own compute on miss
        deadline_s = float(self.config.get("recall.smart_deadline_ms") or 15000) / 1000.0
        if ev.wait(timeout=deadline_s) and getattr(ev, "_sf_ok", False):
            return ev._sf_result
        return self._recall_impl(
            query, top_k, recency_half_life_days, graph_boost, rerank,
            rerank_top_n, _skip_decompose)

    def _recall_impl(
        self,
        query: str,
        top_k: int | None = None,
        recency_half_life_days: float | None = None,
        graph_boost: float | None = None,
        rerank: bool | None = None,
        rerank_top_n: int | None = None,
        _skip_decompose: bool = False,
    ) -> RecallPack:
        # Resolve defaults from config (per-workspace > global > schema default)
        if top_k is None:
            top_k = self.config.get("recall.top_k")
        if recency_half_life_days is None:
            recency_half_life_days = self.config.get("recall.recency_half_life_days")
        if graph_boost is None:
            graph_boost = self.config.get("recall.graph_boost")
        if rerank is None:
            rerank = self.config.get("recall.rerank")
        if rerank_top_n is None:
            rerank_top_n = self.config.get("recall.rerank_top_n")

        # Adaptive Query Decomposition (RAG-Fusion / IRCoT style).
        # Triggered only when query looks multi-hop and feature is on.
        # The user's recall() call is the entry point; recursive sub-query
        # calls pass _skip_decompose=True to prevent infinite recursion.
        if (
            not _skip_decompose
            and self.config.get("recall.adaptive_decompose")
            and _looks_multihop(query)
        ):
            fused = self._recall_with_decomposition(
                query,
                top_k=top_k,
                recency_half_life_days=recency_half_life_days,
                graph_boost=graph_boost,
                rerank=rerank,
                rerank_top_n=rerank_top_n,
            )
            if fused is not None:
                return fused
            # Fallthrough: decomposition failed → run original single-shot

        # Pattern-based query splitting (Improvement UU). Cheap, no LLM -
        # catches compound queries like "why X and why Y" / "X because Y"
        # (the RU/UK split markers live in the lang packs).
        # When a split fires, each sub-query runs through the normal recall
        # pipeline and results are fused via RRF. Saves ~30pp on compound
        # queries that single-shot recall would otherwise diffuse.
        #
        # Hardening note (H1): we intentionally do NOT wrap this whole block
        # in a bare `except: pass` - silent fallback hides real bugs (a
        # missing RecallPack field would have looked indistinguishable from
        # "no split fired"). We only catch the specific failure modes we
        # accept (sub-recall raising, fusion edge cases). Constructor /
        # programmer errors propagate so tests catch them.
        if not _skip_decompose and self.config.get("recall.pattern_split"):
            from pmb.reasoning.query_split import rrf_fuse, split_query

            sub_queries = split_query(query)
            self._pattern_split_last_fired = len(sub_queries) > 1
            if len(sub_queries) > 1:
                sub_packs = []
                sub_recall_ok = True
                for sq in sub_queries:
                    try:
                        sp = self.recall(
                            sq,
                            top_k=max(top_k, 10),
                            recency_half_life_days=recency_half_life_days,
                            graph_boost=graph_boost,
                            rerank=rerank,
                            rerank_top_n=rerank_top_n,
                            _skip_decompose=True,
                        )
                    except Exception:
                        sub_recall_ok = False
                        break
                    sub_packs.append(sp)
                if sub_recall_ok and sub_packs:
                    # Round-robin interleave per-sub top results. Each
                    # sub-pack's top-1 is guaranteed a slot in the final
                    # top-N, which is what the user expects for compound
                    # queries - "X and Y" should surface BOTH X-answer
                    # and Y-answer near the top, not have RRF blend them
                    # into a single diluted ranking.
                    #
                    # We still keep RRF as a tiebreaker on duplicates that
                    # both sub-packs found, so the fusion isn't naive.
                    rank_lists = [[r.ulid for r in sp.results] for sp in sub_packs]
                    fused_scored = rrf_fuse(rank_lists, top_n=top_k * 4)

                    # Resolve ulid -> result (use first occurrence across packs)
                    ulid_to_res: dict[str, Any] = {}
                    for sp in sub_packs:
                        for r in sp.results:
                            if r.ulid not in ulid_to_res:
                                ulid_to_res[r.ulid] = r

                    # Round-robin: take top-1 of pack 0, then top-1 of
                    # pack 1, then top-2 of pack 0, etc. Dedup as we go.
                    seen_ulids: set[str] = set()
                    out_results = []
                    max_per_pack = max(top_k, 5)
                    for rank in range(max_per_pack):
                        for sp in sub_packs:
                            if rank < len(sp.results):
                                r = sp.results[rank]
                                if r.ulid not in seen_ulids:
                                    out_results.append(r)
                                    seen_ulids.add(r.ulid)
                                    if len(out_results) >= top_k:
                                        break
                        if len(out_results) >= top_k:
                            break
                    # If still under top_k, fill from RRF order
                    if len(out_results) < top_k:
                        for ulid, _s in fused_scored:
                            if ulid in seen_ulids:
                                continue
                            r = ulid_to_res.get(ulid)
                            if r is not None:
                                out_results.append(r)
                                seen_ulids.add(ulid)
                                if len(out_results) >= top_k:
                                    break
                    if out_results:
                        # Take workspace metadata from the first sub-pack so
                        # the returned RecallPack has all required fields.
                        meta_src = sub_packs[0]
                        pack = RecallPack(
                            query=query,
                            workspace_name=meta_src.workspace_name,
                            workspace_id=meta_src.workspace_id,
                            results=out_results,
                            n_total_in_workspace=meta_src.n_total_in_workspace,
                            elapsed_ms=sum(getattr(sp, "elapsed_ms", 0.0) for sp in sub_packs),
                        )
                        self.recall_cache.put(
                            make_recall_cache_key(
                                query,
                                top_k,
                                recency_half_life_days,
                                graph_boost,
                                rerank,
                                rerank_top_n,
                            ),
                            pack,
                        )
                        self._pattern_split_last_returned = True
                        return pack
            self._pattern_split_last_returned = False

        # LRU cache hit - short-circuit the whole pipeline. The cache is
        # invalidated automatically on any event write via bump_generation().
        cache_key = make_recall_cache_key(
            query,
            top_k,
            recency_half_life_days,
            graph_boost,
            rerank,
            rerank_top_n,
        )
        cached = self.recall_cache.get(cache_key)
        if cached is not None:
            return cached
        """
        Search relevant memory by query.

        Pipeline:
        1. HybridSearch returns top_k*5 candidates ranked by BM25+vec only.
        2. Graph expansion: extract entities from query, pull additional
           candidate ulids from `graph_event_entities` (1-hop traversal).
        3. Batched SQL fetch for all candidates with the archived filter.
        4. Importance multiplier + recency boost + graph_boost applied in Python.
        5. Touch + reinforcement boost for the final hits.

        graph_boost: additive bonus for events surfaced by the graph
        (0..1, default 0.15). Set to 0 to disable graph augmentation.
        """
        t0 = time.perf_counter()

        # Stage -0.5: typo correction (Improvement K). For each query token,
        # if there's a known entity within edit-distance 2, substitute.
        # Catches "Aliceee" → "alice", "Postgers" → "postgres", etc.
        # Cheap: ~90 entities × short Levenshtein → microseconds.
        typo_corrections = []
        if self.config.get("recall.typo_correction"):
            try:
                import sqlite3

                from pmb.reasoning.typo_fix import correct_query

                with sqlite3.connect(self.workspace.db_path) as conn:
                    rows = conn.execute(
                        "SELECT kind, name FROM graph_entities "
                        "WHERE workspace_id = ? AND n_mentions >= 1",
                        (self.workspace.id,),
                    ).fetchall()
                known = [(r[0], r[1]) for r in rows]
                corrected, typo_corrections = correct_query(query, known)
                if typo_corrections:
                    query = corrected  # use corrected query for the rest of pipeline
            except Exception:
                pass

        # User-name set (mined from "My name is X" facts in any language,
        # cached, refreshed every 25 writes). Computed ONCE per recall and
        # reused by the router (identity-by-name), PAMVR self-reference
        # rescue, and the identity-marker boost - so no path hardcodes a
        # personal name.
        user_names = self._get_user_names()

        # Stage 0: Adaptive Layer Routing (Improvement E). Classify the query
        # and get per-layer multipliers; applied later in scoring loop.
        # Cheap - pure regex.
        layer_weights = None
        if self.config.get("recall.adaptive_routing"):
            try:
                from pmb.reasoning.router import QueryRouter

                intent = QueryRouter().classify(query, user_names=user_names)
                layer_weights = intent.weights
            except Exception:
                layer_weights = None

        # Stage 0.5: Predictive cache check (Improvement F).
        # ~3-5ms cosine over pre-baked questions. If we have a near-identical
        # cached query, return its top-K ulids directly - bypassing all
        # downstream stages. This is the "intuitive answer" path.
        if self.config.get("recall.predictive_enabled"):
            try:
                from pmb.reasoning.predictive import (
                    best_match,
                    load_entries,
                    mark_hit,
                )

                entries = load_entries(
                    self.workspace.db_path,
                    self.workspace.id,
                    max_age_days=self.config.get("recall.predictive_ttl_days") or 7.0,
                )
                if entries and self.search.is_ready():
                    q_emb = self.search.embed(query)
                    threshold = self.config.get("recall.predictive_threshold") or 0.85
                    hit = best_match(entries, q_emb, threshold=threshold)
                    if hit is not None:
                        entry, sim = hit
                        # Hydrate the cached ulids back into events
                        rows = self.events.get_many(
                            entry.top_ulids,
                            workspace_id=self.workspace.id,
                            only_active=True,
                        )
                        if rows:
                            # Build a results pack
                            results: list[RecallResult] = []
                            for u in entry.top_ulids[:top_k]:
                                ev = rows.get(u)
                                if not ev:
                                    continue
                                results.append(
                                    RecallResult(
                                        ulid=ev.ulid,
                                        content=ev.content,
                                        score=float(sim),
                                        bm25_score=0.0,
                                        vec_score=float(sim),
                                        importance=ev.importance,
                                        recency_score=0.0,
                                        timestamp=ev.timestamp,
                                        event_type=ev.event_type,
                                        metadata=ev.metadata,
                                    )
                                )
                            # Record cache hit stats
                            try:
                                mark_hit(self.workspace.db_path, entry.id)
                            except Exception:
                                pass
                            n_total = self._active_count_cached()
                            pack = RecallPack(
                                query=query,
                                workspace_name=self.workspace.name,
                                workspace_id=self.workspace.id,
                                results=results,
                                n_total_in_workspace=n_total,
                                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                            )
                            return pack
            except Exception:
                pass  # cache failure → normal recall

        # Read-your-writes: embeds parked while the process was cold (the
        # passive worker never loads the model - the Codex-timeout fix) are
        # flushed HERE, because recall is a legitimate model consumer: it
        # embeds the query on the next line anyway. Bounded inline drain so a
        # record→recall sequence sees its own vectors; big backlogs finish via
        # the worker / warmup kick.
        try:
            if self._embed_queue:
                _ = self.search.model   # recall pays this load regardless
                self._drain_embed_inline(max_items=64)
        except Exception as e:
            from pmb.core.errlog import log_error
            log_error(self.workspace.db_path, "recall_embed_drain", e)
        # Stage 1: search by BM25+vec only (no importance/recency in search core)
        raw_hits: list[SearchHit] = self.search.search(
            query=query,
            top_k=top_k * 5,  # headroom for the archived filter
            crosslingual_damp=float(
                self.config.get("recall.crosslingual_bm25_damp") or 1.0),
        )

        # Stage 2: graph expansion - entities in the query may surface events
        # that BM25/vec missed entirely. We weight each matched entity by
        # rarity (IDF-ish): an event that matches multiple rare entities
        # gets a stronger boost than one that matches a single common word.
        # Improvement C: temporal anchor - parse a date reference from the
        # query when it looks temporal. Used later to boost candidates with
        # nearby event_time.
        temporal_anchor: float | None = None
        if self.config.get("recall.temporal_enabled"):
            try:
                from pmb.reasoning.temporal import (
                    extract_event_time,
                    is_temporal_query,
                )

                if is_temporal_query(query):
                    temporal_anchor = extract_event_time(query)
            except Exception:
                temporal_anchor = None

        graph_ulids: set[str] = set()
        graph_weights: dict[str, float] = {}  # ulid -> sum of IDF weights
        if graph_boost > 0:
            q_ext = self.entity_extractor.extract(query)
            q_names = [n for _, n in q_ext.all_named()]
            # Optional LLM query expansion for abstract queries that the
            # rule-based extractor missed. Opt-in via config.
            if self.config.get("recall.graph_expansion_llm"):
                try:
                    from pmb.graph.expansion import expand_query
                    from pmb.health.consolidate import resolve_llm_client

                    extra = expand_query(
                        query,
                        self.workspace.storage_dir,
                        llm=resolve_llm_client(
                            backend=self.config.get("consolidate.backend"),
                        ),
                    )
                    for e in extra:
                        if e and e not in q_names:
                            q_names.append(e)
                except Exception:
                    # Expansion is best-effort; never block recall on LLM errors
                    pass
            if q_names:
                matched = self.graph.find_entities_by_name(
                    self.workspace.id,
                    q_names,
                )
                if matched:
                    # IDF-ish: 1 / log(2 + n_mentions). Rare entity (1 mention)
                    # → ~0.91; common entity (e.g. 50 mentions) → ~0.25.
                    eid_to_idf: dict[int, float] = {}
                    for e in matched:
                        if e.id is None:
                            continue
                        eid_to_idf[e.id] = 1.0 / math.log(2.0 + max(0, e.n_mentions))
                    # Bump pair limit (was top_k * 20 = 200 - too tight for
                    # multi-hop, where the right event may be ranked far down
                    # in raw graph traversal but emerges via multi-entity bonus
                    # below). 100x top_k gives ample room without paying for it.
                    pairs = self.graph.event_entity_pairs(
                        list(eid_to_idf),
                        limit=top_k * 100,
                    )
                    event_matched_eids: dict[str, set[int]] = {}
                    for ulid_x, eid in pairs:
                        idf = eid_to_idf.get(eid, 0.0)
                        graph_weights[ulid_x] = graph_weights.get(ulid_x, 0.0) + idf
                        event_matched_eids.setdefault(ulid_x, set()).add(eid)
                    # Multi-hop bonus: an event mentioning N distinct query
                    # entities gets weight × (1 + 0.5*(N-1)). Two entities =
                    # 1.5x, three = 2x. This is the bridge for multi-hop
                    # questions: "what did Alice do in December?" - answer
                    # event has both 'alice' AND 'december', so it dominates
                    # single-entity matches.
                    multi_bonus = self.config.get("recall.multi_entity_bonus") or 0.5
                    for ulid_x, ent_set in event_matched_eids.items():
                        n = len(ent_set)
                        if n > 1 and multi_bonus > 0:
                            graph_weights[ulid_x] *= 1.0 + multi_bonus * (n - 1)
                    graph_ulids = set(graph_weights.keys())

        # Stage 2.4: Personalized PageRank (HippoRAG). Cheap multi-hop boost
        # that uses the entity graph we already built. Diffuses probability
        # from query entities through the graph for many hops in one shot -
        # this is what lets the answer event (which only mentions ONE query
        # entity directly but is multi-hop close to the rest) surface.
        #
        # GATING: experiments show PPR helps multi-hop queries but ADDS NOISE
        # to single-entity lookups (where exact match should dominate). We
        # apply PPR only when query has multi-hop intent OR mentions 2+
        # known graph entities.
        ppr_event_scores: dict = {}
        if self.config.get("recall.ppr_enabled"):
            try:
                from pmb.graph.ppr import (
                    personalized_pagerank,
                    score_events_by_ppr,
                )

                ppr_graph = self._get_ppr_graph()
                if ppr_graph is not None:
                    # Extract query entities independently (Stage 2 may have skipped)
                    q_ext_ppr = self.entity_extractor.extract(query)
                    q_names_ppr = [n for _, n in q_ext_ppr.all_named()]
                    seed_eids: list[int] = []
                    seed_w: list[float] = []
                    if q_names_ppr:
                        matched_ppr = self.graph.find_entities_by_name(
                            self.workspace.id,
                            q_names_ppr,
                        )
                        for e in matched_ppr:
                            if e.id is None:
                                continue
                            seed_eids.append(e.id)
                            # IDF-ish - rare entity weighted more
                            seed_w.append(1.0 / math.log(2.0 + max(0, e.n_mentions)))
                    # Intent gate: PPR only when query is multi-hop or has 2+
                    # matched entities. Single-entity exact-match queries are
                    # better served by raw hybrid search.
                    ppr_should_run = (
                        len(seed_eids) >= 2
                        or _looks_multihop(query)
                        or self.config.get("recall.ppr_always")
                    )
                    if seed_eids and ppr_should_run:
                        ppr_scores = personalized_pagerank(
                            ppr_graph,
                            seed_eids,
                            seed_weights=seed_w,
                            alpha=self.config.get("recall.ppr_alpha") or 0.5,
                            iterations=self.config.get("recall.ppr_iters") or 20,
                        )
                        ppr_event_scores = score_events_by_ppr(
                            ppr_graph,
                            ppr_scores,
                        )
            except Exception as _e:
                ppr_event_scores = {}

        # Stage 2.5: causation walk for multi-hop queries.
        # Cheap: one SQL hit, walks up to 1 hop forward/backward from top
        # search hits. Detects multi-hop intent via regex; harmless on
        # simple queries (no edges → no expansion).
        causation_ulids: set[str] = set()
        if self.config.get("recall.causation_walk") and raw_hits and _looks_multihop(query):
            try:
                from pmb.reasoning.causation import walk_edges

                seed = [h.ulid for h in raw_hits[: max(3, top_k // 2)]]
                causation_ulids = walk_edges(
                    self.workspace.db_path,
                    seed,
                    direction="both",
                    hops=1,
                    min_confidence=0.4,
                    max_results=top_k * 3,
                )
                # Don't double-count search hits
                causation_ulids -= {h.ulid for h in raw_hits}
            except Exception:
                causation_ulids = set()

        # Stage 2.6: narrative arc expansion. If query looks narrative
        # ("tell me about X", "history of Y"), search arc summaries via
        # BM25-ish substring match and inject member events of best-matching
        # arc into the candidate pool. Very cheap - no LLM.
        arc_ulids: set[str] = set()
        arc_summaries_hit: list[dict] = []
        if self.config.get("recall.arc_expansion"):
            try:
                from pmb.reasoning.arcs import (
                    events_in_arc as _events_in_arc,
                )
                from pmb.reasoning.arcs import (
                    list_arcs as _list_arcs,
                )
                from pmb.reasoning.arcs import looks_narrative

                # Always try; the term-match below is cheap. But weight it
                # more heavily if query 'looks narrative'.
                arcs_now = _list_arcs(
                    self.workspace.db_path, self.workspace.id, status="active", limit=200
                )
                if arcs_now:
                    q_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
                    q_terms = {t for t in q_terms if len(t) > 2}
                    scored_arcs = []
                    for arc in arcs_now:
                        blob = (arc.title + " " + (arc.summary or "")).lower()
                        # Count distinct query terms appearing in the blob
                        n_match = sum(1 for t in q_terms if t in blob)
                        if n_match == 0:
                            continue
                        # Bonus if narrative-looking query
                        score = n_match * (2.0 if looks_narrative(query) else 1.0)
                        scored_arcs.append((arc, score))
                    scored_arcs.sort(key=lambda t: -t[1])
                    for arc, _ in scored_arcs[:2]:
                        arc_ulids.update(_events_in_arc(self.workspace.db_path, arc.id))
                        arc_summaries_hit.append(
                            {
                                "arc_id": arc.id,
                                "title": arc.title,
                                "summary": (arc.summary or "")[:300],
                            }
                        )
                    arc_ulids -= {h.ulid for h in raw_hits}
            except Exception:
                arc_ulids = set()

        # PPR candidate expansion: top events by PPR score that aren't already
        # candidates. Caps at top_k * 2 to avoid blowing up the fetch.
        ppr_top_ulids: set[str] = set()
        if ppr_event_scores:
            ppr_top_n = min(top_k * 3, 100)
            top_by_ppr = sorted(ppr_event_scores.items(), key=lambda kv: -kv[1])[:ppr_top_n]
            ppr_top_ulids = {u for u, _ in top_by_ppr}

        if (
            not raw_hits
            and not graph_ulids
            and not causation_ulids
            and not arc_ulids
            and not ppr_top_ulids
        ):
            n_total = self._active_count_cached()
            return RecallPack(
                query=query,
                workspace_name=self.workspace.name,
                workspace_id=self.workspace.id,
                results=[],
                n_total_in_workspace=n_total,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Stage 3: batched fetch - union of search + graph + causation + arc hits
        seen_for_dedupe = {h.ulid for h in raw_hits}
        all_ulids: list[str] = [h.ulid for h in raw_hits]
        for u in graph_ulids:
            if u not in seen_for_dedupe:
                all_ulids.append(u)
                seen_for_dedupe.add(u)
        for u in causation_ulids:
            if u not in seen_for_dedupe:
                all_ulids.append(u)
                seen_for_dedupe.add(u)
        for u in arc_ulids:
            if u not in seen_for_dedupe:
                all_ulids.append(u)
                seen_for_dedupe.add(u)
        for u in ppr_top_ulids:
            if u not in seen_for_dedupe:
                all_ulids.append(u)
                seen_for_dedupe.add(u)

        # Inject active keyed-facts into the candidate pool when the query
        # looks like a personal-attribute question. Lexical / vector overlap
        # is often poor (e.g. query "where do I live" vs content "user city:
        # Warsaw") so without this the keyed fact never reaches the ranking
        # stage, and the keyed-fact-boost has nothing to amplify. This is
        # a cheap SQL scan (~5ms on a 1000-event workspace) gated on a tight
        # regex so non-personal queries pay nothing.
        # Two-token intent gate: needs BOTH a question word AND a
        # personal-pronoun/user-cue. Either alone fires too often -
        # "how does X work" should not inject keyed-facts (no personal
        # cue), and "user works on PMB" should not inject either (no
        # question cue).
        _qword_re = _QWORD_RE
        _attr_re = _ATTR_RE
        # R6: the SAME personal-intent signal that gates keyed-fact INJECTION
        # below also gates the keyed-fact BOOST in the scoring loop, so a
        # topical query like "Warsaw deployment timezone" no longer rockets
        # "user city: Warsaw" to the top just because it shares a token.
        _personal_query = bool(_qword_re.search(query) and _attr_re.search(query))
        if _personal_query:
            try:
                import sqlite3 as _sql
                with _sql.connect(str(self.workspace.db_path)) as _c:
                    _c.row_factory = _sql.Row
                    _keyed_rows = _c.execute(
                        "SELECT ulid FROM events "
                        "WHERE workspace_id = ? "
                        "AND archived_at IS NULL "
                        "AND metadata_json LIKE '%\"keyed_fact_key\"%' "
                        "ORDER BY timestamp DESC LIMIT 20",
                        (self.workspace.id,),
                    ).fetchall()
                for _r in _keyed_rows:
                    if _r["ulid"] not in seen_for_dedupe:
                        all_ulids.append(_r["ulid"])
                        seen_for_dedupe.add(_r["ulid"])
            except Exception:
                pass

        rows = self.events.get_many(
            all_ulids,
            workspace_id=self.workspace.id,
            only_active=True,
        )

        # Stage 4: rerank with importance + recency + graph boost
        now = time.time()
        half_life_sec = recency_half_life_days * 86400.0

        # Improvement B: pre-compute the set of *content* tokens from the
        # query. Used below as a small multiplicative boost when a candidate
        # event's content contains ALL the unique meaningful tokens of the
        # query - direct lexical match in the right context.
        # Stopwords removed because they would over-fire on generic queries.
        _STOP = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "of",
            "in",
            "on",
            "to",
            "for",
            "and",
            "or",
            "with",
            "we",
            "i",
            "you",
            "do",
            "does",
            "did",
            "what",
            "who",
            "where",
            "when",
            "why",
            "how",
            "by",
            "at",
            "from",
            "as",
            "be",
            "have",
            "has",
            "had",
            "this",
            "that",
            "these",
            "those",
            "it",
            "our",
            "my",
            "your",
            "their",
            "his",
            "her",
            "about",
            "any",
            "some",
            "all",
            "more",
            "than",
            "but",
            "not",
            "now",
            "use",
            "used",
            "using",
        }
        q_tokens = {
            t
            for t in re.findall(_WORD_NUM, (query or "").lower())
            if len(t) > 2 and t not in _STOP
        }

        # Precompute PAMVR query features ONCE per recall, not per
        # candidate. Cuts p99 from ~860ms to ~300ms on multilingual stress.
        pamvr_features = None
        if self._pamvr_enabled:
            try:
                pamvr_features = _pamvr_prepare(
                    query,
                    vocab_bridges=self._vocab_bridges,
                    user_names=user_names,
                    # B4: drop the lexical verb-synonym boost in anchors mode.
                    verb_match=(self.config.get("lang.mode") != "anchors"),
                )
            except Exception:
                pamvr_features = None

        search_hits_by_ulid = {h.ulid: h for h in raw_hits}
        scored: list[tuple[SearchHit, Event, float, float]] = []
        for ulid, ev in rows.items():
            h = search_hits_by_ulid.get(ulid)
            if h is None:
                # Graph-only hit - synthesize a SearchHit at low base score so
                # graph evidence alone can still surface a strong importance event.
                h = SearchHit(
                    ulid=ulid,
                    score=0.0,
                    bm25_score=0.0,
                    vec_score=0.0,
                    importance=ev.importance,
                    recency_score=0.0,
                )
            # X2: the importance factor carries the importance channel weight,
            # computed ONCE here so the base score AND every boost term below use
            # the SAME (consistently-weighted) factor. Identity weights →
            # 0.5 + 0.5*importance (byte-identical to before).
            importance_factor = _imp_factor(ev.importance, self._channel_weights)
            age_sec = max(0.0, now - ev.timestamp)
            recency = math.exp(-age_sec * math.log(2) / half_life_sec)
            # Historical-intent: when the user asks "what did we use before",
            # the latest pinned fact is the wrong answer. We weaken the
            # recency reward (multiplier <1.0) and add a small bonus to OLDER
            # events. Both come from QueryRouter.classify().
            rec_mul = layer_weights.historical_recency_mul if layer_weights else 1.0
            # X1/X2: the base channel combination lives in one explainable
            # function, fed the already-weighted importance_factor so importance
            # scaling is consistent with the boosts. Identity weights →
            # byte-identical to `h.score * importance_factor * (1+0.2*recency*rec_mul)`.
            base = _combine_base_score(
                h.score, importance_factor, recency, rec_mul=rec_mul,
                weights=self._channel_weights)
            if layer_weights and layer_weights.older_event_bonus > 0:
                # invert recency: 1.0 for very old, 0.0 for fresh.
                base += layer_weights.older_event_bonus * (1.0 - recency) * importance_factor
            # Graph augmentation: bonus proportional to summed IDF of matched
            # entities. Rare-entity matches dominate, common-entity matches
            # contribute little.
            w = graph_weights.get(ulid, 0.0) if graph_boost > 0 else 0.0
            if w > 0:
                gb_mul = layer_weights.graph_boost_mul if layer_weights else 1.0
                # Saturate the summed-IDF weight so a multi-entity match nudges
                # PROPORTIONALLY rather than leapfrogging the base relevance.
                # Raw w (sum of per-entity IDF x the multi-entity bonus) is the
                # ONE uncalibrated additive term and could exceed the base score
                # for events that merely share several query entities - sinking
                # the genuinely-relevant top result (measured on LoCoMo). w/(1+w)
                # bounds the contribution to (0, graph_boost*imp).
                w_eff = w / (1.0 + w)
                base += graph_boost * gb_mul * w_eff * importance_factor
            # Causation augmentation: small additive bonus for events surfaced
            # by walking event_edges from raw hits. Multi-hop unlock.
            if ulid in causation_ulids:
                causation_boost = self.config.get("recall.causation_boost") or 0.10
                cb_mul = layer_weights.causation_boost_mul if layer_weights else 1.0
                base += causation_boost * cb_mul * importance_factor * (1.0 + 0.2 * recency)
            # Arc augmentation: events that belong to a matching arc get a
            # narrative coherence bonus. Smaller than causation since arc
            # membership is a softer signal.
            if ulid in arc_ulids:
                arc_boost = self.config.get("recall.arc_boost") or 0.08
                ab_mul = layer_weights.arc_boost_mul if layer_weights else 1.0
                base += arc_boost * ab_mul * importance_factor * (1.0 + 0.2 * recency)
            # PPR augmentation: events with high PPR mass from query entities
            # get a boost. This is the multi-hop unlock - events that don't
            # appear in raw search but are graph-close to query entities.
            if ppr_event_scores:
                pscore = ppr_event_scores.get(ulid, 0.0)
                if pscore > 0:
                    ppr_weight = self.config.get("recall.ppr_weight") or 0.5
                    pw_mul = layer_weights.ppr_weight_mul if layer_weights else 1.0
                    base += ppr_weight * pw_mul * min(1.0, pscore * 100.0) * importance_factor
            # Improvement C: temporal-proximity boost. If query is temporal
            # and event has a parsed event_time, boost by exp-decay distance
            # to the query's anchor.
            if temporal_anchor is not None and ev.metadata:
                ev_t = ev.metadata.get("event_time") if isinstance(ev.metadata, dict) else None
                if ev_t is not None:
                    try:
                        from pmb.reasoning.temporal import temporal_proximity_boost

                        tprox = temporal_proximity_boost(
                            event_time=float(ev_t),
                            query_anchor_time=temporal_anchor,
                            half_life_days=self.config.get("recall.temporal_half_life_days")
                            or 14.0,
                        )
                        if tprox > 0:
                            t_weight = self.config.get("recall.temporal_boost") or 0.20
                            tb_mul = layer_weights.temporal_boost_mul if layer_weights else 1.0
                            base += t_weight * tb_mul * tprox * importance_factor
                    except Exception:
                        pass
            # Improvement E: per-layer event-type boost. Routes the query
            # toward the right semantic layer based on intent.
            if layer_weights:
                et = ev.event_type
                if et == "fact_atom":
                    base *= layer_weights.facts_boost
                elif et == "reflection":
                    base *= layer_weights.reflections_boost
                else:
                    # raw events (qa, fact, event, git, etc.)
                    base *= layer_weights.raw_boost
            # Keyed-fact boost. Facts recorded via record_keyed_fact
            # (subject/attribute/value pattern, e.g. "user.city=Warsaw")
            # represent the CURRENT canonical value of a personal
            # attribute. On personal-attribute queries ("where do I
            # live", "what is my X") these should rank above generic
            # facts that share vocabulary. Old values are archived by
            # supersession so we never boost a stale value here.
            #
            # Two-step boost:
            #   (a) MIN FLOOR - keyed-facts have weak text overlap with
            #       natural-language questions ("where do I live" vs
            #       "user.city: Warsaw"). Bring the base up to a floor
            #       so it competes with topical matches.
            #   (b) ADDITIVE - recency-weighted, importance-scaled.
            #   (c) MULTIPLICATIVE - keyed-facts answering personal
            #       questions are USUALLY the right answer. Bump above
            #       merely-topical hits.
            _boost_keyed = (
                ev.metadata and isinstance(ev.metadata, dict)
                and ev.metadata.get("keyed_fact_key")
                # R6: only boost on a personal-attribute query, unless the gate
                # is disabled (then it's the old always-boost behaviour).
                and (_personal_query
                     or not self.config.get("recall.keyed_boost_personal_only"))
            )
            if _boost_keyed:
                keyed_boost = self.config.get("recall.keyed_fact_boost") or 0.35
                # F1: scale the boost by EXTRACTION CONFIDENCE when present. A
                # hypothesis-extracted keyed fact (C2) carries metadata.extract.
                # margin; a low-margin extraction gets a smaller boost. Regex /
                # manual keyed facts have no margin → conf 1.0 (unchanged).
                conf = 1.0
                if self.config.get("extract.confidence_recall") and isinstance(ev.metadata, dict):
                    _ex = ev.metadata.get("extract")
                    if isinstance(_ex, dict) and isinstance(_ex.get("margin"), (int, float)):
                        conf = 0.7 + 0.3 * min(1.0, max(0.0, float(_ex["margin"])) / 0.15)
                # (a) Floor - keyed-fact text is short ("user city: X")
                # so vector + BM25 base often underestimates.
                base = max(base, 0.50)
                # (b) additive - recency and importance scaled, confidence-weighted.
                base += keyed_boost * importance_factor * (1.0 + 0.3 * recency) * conf
                # (c) multiplicative - keyed-fact-on-personal-query is
                # almost always the right answer (full 1.4 at conf=1.0).
                base *= 1.0 + 0.4 * conf
            # Improvement B: query-keyword overlap boost. If most of the
            # meaningful tokens of the query are present in this event's
            # content, this event is a likely DIRECT match - boost it.
            # The boost is proportional to the overlap fraction so a tiny
            # overlap doesn't move the score much, but a near-full overlap
            # gives a meaningful nudge. Capped at 1.25x to avoid letting
            # one keyword count as a definitive answer.
            if q_tokens:
                content_lower = (ev.content or "").lower()
                # Cheap substring check: count which query tokens appear
                # anywhere in the content. Skips word-boundary work; good
                # enough at this score-tweak granularity.
                n_hit = sum(1 for t in q_tokens if t in content_lower)
                if n_hit >= 2:
                    # Overlap fraction over QUERY tokens, capped via min().
                    overlap = n_hit / max(1, len(q_tokens))
                    # Multiplier: 1.0 (no overlap) up to 1.25 (full match).
                    base *= 1.0 + min(0.25, 0.25 * overlap)

            # Personal-marker boost: facts that literally start with a
            # personal possessive ("User's", the user's own name, "I am",
            # "My", "I work on") get boosted ONLY when the query itself has
            # identity intent (router classified it as "identity"). Without
            # this gate the boost mis-fires on topical queries that just
            # happen to surface an identity-shaped fact (e.g. "What's the
            # JWT lifetime?" surfacing "<name>'s terminal is Wezterm" top-1).
            if (
                ev.event_type == "fact"
                and layer_weights
                and layer_weights.identity_marker_boost > 1.0
            ):
                content_head = (ev.content or "")[:60].lower()
                # Generic first-person / role markers - language-functional,
                # NOT personal data, so they stay inline.
                is_identity = (
                    content_head.startswith("user's ")
                    or content_head.startswith("user ")
                    or content_head.startswith("i am ")
                    or content_head.startswith("my ")
                    or content_head.startswith("i work ")
                    or content_head.startswith("i prefer ")
                )
                # A fact that opens with the user's OWN name ("Bob's
                # terminal is ...") is an identity fact too. Driven by the
                # mined user-name cache (user_names) - never a hardcoded
                # personal-name literal.
                if not is_identity and user_names:
                    first_tok = content_head.split(" ", 1)[0]
                    if first_tok.endswith("'s"):
                        first_tok = first_tok[:-2]
                    if first_tok in user_names:
                        is_identity = True
                if is_identity:
                    base *= layer_weights.identity_marker_boost
            # Decision-intent boost: activity events marked as a
            # decision/agreement should outrank arguments on the same
            # topic when the user asks "what did we decide". The router
            # only sets decision_boost > 1.0 when the query phrasing
            # actually asks for a decision.
            # record_activity() stores the kind under `activity_kind`
            # in metadata (not `kind`); record_batch uses the same path,
            # so we check both for safety.
            if layer_weights and ev.event_type == "activity" and layer_weights.decision_boost > 1.0:
                meta = ev.metadata or {}
                kind = meta.get("activity_kind") or meta.get("kind") or ""
                if kind in ("decision", "decided", "agreed", "resolved", "concluded"):
                    base *= layer_weights.decision_boost
            # PAMVR (Predicate-Aware Multi-View Reranking) - 14 small
            # content-based boost rules empirically tuned to lift top-1
            # accuracy from ~60% to 93%+. Verb match, entity strict,
            # vocab bridges, topic constraint, time-duration, etc. See
            # pmb.reasoning.pamvr for the full rule set + research data.
            if self._pamvr_enabled:
                base = _pamvr_apply(
                    query,
                    ev,
                    base,
                    vocab_bridges=self._vocab_bridges,
                    query_features=pamvr_features,  # reuse precomputed
                )
            scored.append((h, ev, base, recency))

        # Stage 3.25: collapse reflections onto their source events.
        # Reflections are meta - they exist to make sources findable via
        # their LLM-generated 'might_answer' text. The final result list
        # should always be source events, with the reflection's score
        # added to the source's. If the source isn't in candidates yet,
        # the reflection is REPLACED by the source (preserving the score).
        # If source is missing entirely (deleted), the reflection still
        # acts as a useful fallback result.
        if self.config.get("recall.collapse_reflections"):
            scored = _collapse_reflections(scored, self.events, self.workspace.id)

        # Lesson-intent boost (#1): on how-to/convention queries, gently lift
        # lesson & failure memories so the agent applies them / avoids repeats.
        # SCOPED to events with kind=lesson/failure ONLY - datasets without
        # such events (LoCoMo) are provably unaffected. Config-gated.
        if self.config.get("recall.lesson_boost"):
            try:
                from pmb.memory_quality import is_lesson_intent

                if is_lesson_intent(query, engine=self):
                    lf = float(self.config.get("recall.lesson_boost_factor") or 1.3)
                    scored = [
                        (
                            h,
                            ev,
                            (
                                base * lf
                                if (ev.metadata or {}).get("kind") in ("lesson", "failure")
                                else base
                            ),
                            recency,
                        )
                        for (h, ev, base, recency) in scored
                    ]
            except Exception as e:
                # never let a boost crash recall; log so a systemic failure shows
                from pmb.core.errlog import log_error
                log_error(self.workspace.db_path, "recall_boost", e)

        scored.sort(key=lambda t: -t[2])

        # Stage 3.5: optional cross-encoder reranker over top-N candidates.
        # The hybrid scoring is good at *finding* the right neighbourhood;
        # the cross-encoder is better at picking the single most-relevant doc.
        #
        # Improvement #1 + VV - SMART gated rerank:
        #
        #   Original gate:  fire when (top1 - top3) score gap is < epsilon.
        #   The intuition: when BM25+vector clearly disagree on the order,
        #   the cross-encoder is helpful; when they agree, the CE causes
        #   regressions (LoCoMo loses 17pp with always-on rerank).
        #
        #   VV upgrade: ADD a "confidence-required" stage after the CE
        #   produces scores. Only commit the reranked order if the new
        #   top-1's CE score beats the previous top-1's by >= swap_margin.
        #   When confidence is low, KEEP the hybrid order - the cross-
        #   encoder isn't sure enough to override it.
        #
        #   Net effect: gating fires more often (helps where hybrid was
        #   ambiguous) but commits more conservatively (no LoCoMo
        #   regression because clear winners stay on top).
        gated_rerank = False
        if (
            not rerank
            and self.config.get("recall.rerank_when_close")
            and self.search.reranker is not None
            and len(scored) >= 3
        ):
            top1_score = scored[0][2]
            top3_score = scored[2][2]
            eps = float(self.config.get("recall.rerank_close_epsilon") or 0.05)
            if (top1_score - top3_score) < eps:
                gated_rerank = True

        if (rerank or gated_rerank) and self.search.reranker is not None and len(scored) > 1:
            top_n = scored[: max(top_k, min(rerank_top_n, len(scored)))]
            remainder = scored[len(top_n) :]
            prev_top1_ulid = top_n[0][1].ulid
            # Build a temporary fake SearchHit list to feed the reranker
            tmp_hits = [
                SearchHit(
                    ulid=ev.ulid,
                    score=score,
                    bm25_score=h.bm25_score,
                    vec_score=h.vec_score,
                    importance=ev.importance,
                    recency_score=recency,
                    raw_vec=getattr(h, "raw_vec", 0.0),
                )
                for h, ev, score, recency in top_n
            ]
            ev_by_ulid = {ev.ulid: ev for _, ev, _, _ in top_n}
            recency_by_ulid = {ev.ulid: rc for _, ev, _, rc in top_n}
            reranked = self.search.rerank(
                query,
                tmp_hits,
                text_for_ulid=lambda u: ev_by_ulid[u].to_text(),
            )
            # VV confidence gate: when gating is on (not full rerank=True),
            # only ACCEPT a swap of position 1 if the cross-encoder is
            # genuinely confident. Otherwise revert to the hybrid order
            # for that slot, leaving the rest of the rerank intact.
            if gated_rerank and not rerank and len(reranked) >= 2:
                new_top1_score = float(reranked[0].score)
                # find prev top-1's score in the reranked list (could be at any pos)
                prev_top1_in_rerank = next(
                    (i for i, h in enumerate(reranked) if h.ulid == prev_top1_ulid),
                    None,
                )
                if prev_top1_in_rerank is not None and prev_top1_in_rerank > 0:
                    prev_top1_score = float(reranked[prev_top1_in_rerank].score)
                    swap_margin = float(self.config.get("recall.rerank_swap_margin") or 0.20)
                    if (new_top1_score - prev_top1_score) < swap_margin:
                        # CE not confident enough - restore prev top-1 at pos 0
                        old_idx = prev_top1_in_rerank
                        reranked = [reranked[old_idx]] + [
                            h for i, h in enumerate(reranked) if i != old_idx
                        ]
            top_n = [(h, ev_by_ulid[h.ulid], h.score, recency_by_ulid[h.ulid]) for h in reranked]
            scored = top_n + remainder

        # Stage 3.6: optional LLM-as-judge rerank (Improvement XX).
        # Asks a small local LLM (qwen2.5:1.5b by default, via Ollama)
        # to pick the single best candidate from the current top-N.
        # OFF by default - adds ~100-300ms but lifts top-1 by 5-15pp on
        # hard queries where lexical+semantic+CE all give close scores.
        if self.config.get("recall.llm_rerank") and len(scored) >= 2:
            try:
                from pmb.health.consolidate import OllamaClient
                from pmb.reasoning.llm_rerank import (
                    DEFAULT_LLM_RERANK_MODEL,
                    llm_rerank_top_k,
                )

                llm_top_n = int(self.config.get("recall.llm_rerank_top_n") or 10)
                window = scored[:llm_top_n]
                # Lazy-create the client; reuse on subsequent calls
                if not hasattr(self, "_llm_rerank_client"):
                    self._llm_rerank_client = OllamaClient(
                        model=(
                            self.config.get("recall.llm_rerank_model") or DEFAULT_LLM_RERANK_MODEL
                        ),
                        timeout=float(self.config.get("recall.llm_rerank_timeout") or 5.0),
                    )
                cands = [ev for _h, ev, _s, _r in window]
                pick = llm_rerank_top_k(
                    query,
                    cands,
                    self._llm_rerank_client,
                    max_candidates=llm_top_n,
                )
                if pick is not None and pick > 0:
                    # Move picked candidate to position 0, preserve rest
                    chosen = window[pick]
                    new_window = [chosen] + [x for i, x in enumerate(window) if i != pick]
                    scored = new_window + scored[llm_top_n:]
            except Exception:
                # Best-effort: any failure (model down, parse error) falls
                # back to whatever the previous stage produced.
                pass

        scored = scored[:top_k]

        # Stage 4: hydrate + reinforcement. Collect side effects so we
        # can flush them in a single SQLite transaction at the end -
        # this is the biggest latency win on warm recall.
        from pmb.core.events import (
            PROMOTE_EPISODIC_TO_SEMANTIC_ACCESS,
            PROMOTE_WORKING_TO_EPISODIC_ACCESS,
            TIER_EPISODIC,
            TIER_SEMANTIC,
            TIER_WORKING,
        )

        results: list[RecallResult] = []
        touches: list[str] = []
        importance_updates: list[tuple[str, float]] = []
        tier_promotions: list[tuple[str, str]] = []
        for h, ev, final_score, recency in scored:
            touches.append(ev.ulid)
            new_imp = boost_on_recall(ev.importance, final_score)
            if new_imp > ev.importance + 0.001:
                importance_updates.append((ev.ulid, new_imp))
            # Tier promotion: repeated successful retrieval moves a memory
            # up the hierarchy (working → episodic → semantic). The +1
            # here accounts for the touch we're about to apply.
            new_access = ev.access_count + 1
            if ev.tier == TIER_WORKING and new_access >= PROMOTE_WORKING_TO_EPISODIC_ACCESS:
                tier_promotions.append((ev.ulid, TIER_EPISODIC))
            elif ev.tier == TIER_EPISODIC and new_access >= PROMOTE_EPISODIC_TO_SEMANTIC_ACCESS:
                tier_promotions.append((ev.ulid, TIER_SEMANTIC))
            results.append(
                RecallResult(
                    ulid=ev.ulid,
                    event_type=ev.event_type,
                    content=ev.content,
                    metadata=ev.metadata,
                    timestamp=ev.timestamp,
                    score=final_score,
                    bm25_score=h.bm25_score,
                    vec_score=h.vec_score,
                    importance=ev.importance,
                    recency_score=recency,
                    raw_vec=getattr(h, "raw_vec", 0.0),  # R3 absolute cosine
                )
            )
        # Improvement #6: enqueue touches for the deferred flusher instead
        # of writing synchronously. Under concurrent recalls this turns
        # 16 lock-acquisitions per second into ~4 (one per flush tick).
        # S9: tier promotions ride the touch-flusher batch (was one
        # connection+txn per promoted ulid). They land in the SAME single
        # transaction as the access_count / importance updates.
        self._enqueue_touches(touches, importance_updates, tier_promotions)
        # ...unless touch_async is off: drain inline so access_count /
        # importance / last_accessed / tier are visible the instant recall()
        # returns. Keeps side-effects deterministic for tests and single-shot
        # CLI callers that read them right after recall.
        if not self.config.get("recall.touch_async"):
            self._drain_touch_buffer()

        n_total = self._active_count_cached()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        pack = RecallPack(
            query=query,
            workspace_name=self.workspace.name,
            workspace_id=self.workspace.id,
            results=results,
            n_total_in_workspace=n_total,
            elapsed_ms=elapsed_ms,
        )
        # Spreading activation - prime graph neighbours of every hit. Like
        # priming in human associative memory: thinking about X makes
        # related concepts easier to retrieve next time.
        if self.config.get("recall.spreading_activation") and results:
            hit_evs = [ev for _, ev, _, _ in scored[: len(results)]]
            _spread_boost = self.config.get("recall.spreading_boost")
            _spread_hl = self.config.get("recall.spreading_half_life_hours")

            def _spread() -> None:
                try:
                    from pmb.graph.spreading import apply_spreading_activation
                    apply_spreading_activation(
                        self, hit_events=hit_evs,
                        boost=_spread_boost, half_life_hours=_spread_hl)
                except Exception:
                    # Priming is best-effort; never block recall on it
                    pass

            # S9: the pack is already built, so priming is a pure side effect on
            # FUTURE recalls - get it OFF the synchronous return in the long-lived
            # daemon (touch_async on) by running it on a background thread. When
            # touch_async is off (tests / one-shot CLI) run inline so priming is
            # deterministic and visible immediately.
            if self.config.get("recall.touch_async"):
                import threading as _th
                _th.Thread(target=_spread, daemon=True,
                           name="pmb-spreading").start()
            else:
                _spread()
        # The model is necessarily loaded by this point (we just searched).
        # If embeds queued while the process was cold are still parked (the
        # passive worker may have deadline-exited before this lazy load),
        # kick the drain now so their vectors land without waiting for
        # warmup() or the next write.
        try:
            if self._embed_queue:
                self._kick_embed_drain()
        except Exception as e:
            from pmb.core.errlog import log_error
            log_error(self.workspace.db_path, "recall_kick_drain", e)
        # Stash for next time. Writes bump the generation so future stale
        # cache hits are dropped on read.
        self.recall_cache.put(cache_key, pack)
        # F3 (gated, off by default): log the TOP result's channels as a sample;
        # a later 'useful' feedback flips it positive and the tick proposes
        # channel weights. One cheap insert per recall when on; zero cost off.
        try:
            if self.config.get("recall.weight_learning") and results and scored:
                from pmb.reasoning.weight_learning import record_channel_sample
                top_ev = scored[0][1]
                record_channel_sample(
                    self,
                    {"hit": float(getattr(results[0], "score", 0.0) or 0.0),
                     "importance": float(getattr(top_ev, "importance", 0.0) or 0.0)},
                    ulid=getattr(top_ev, "ulid", None), useful=False)
        except Exception:
            pass
        return pack

    def pin(self, ulid: str, importance: float = 1.0):
        self.events.pin(ulid, importance)

    def forget(self, ulid: str):
        self.events.archive(ulid)

    def unforget(self, ulid: str):
        self.events.unarchive(ulid)

    def purge(self, ulid: str) -> bool:
        """HARD delete (irreversible): drop the vector, the SQLite row, and the
        event's graph links, then invalidate the recall cache. Returns True if a
        row was actually removed. For reversible deletion use `forget`."""
        try:
            self.search.remove(ulid)
        except Exception:
            pass
        deleted = self.events.purge(ulid)
        try:
            self.recall_cache.bump_generation()
        except Exception:
            pass
        return deleted

    def delete_event(self, ulid: str, hard: bool = False) -> dict:
        """Unified delete used by the CLI + dashboard.

        hard=False -> archive (reversible; restore with `unforget`).
        hard=True  -> purge permanently (vector + row + graph links)."""
        if hard:
            ok = self.purge(ulid)
            return {"ulid": ulid, "mode": "hard", "ok": bool(ok)}
        self.events.archive(ulid)
        try:
            self.recall_cache.bump_generation()
        except Exception:
            pass
        return {"ulid": ulid, "mode": "soft", "ok": True}

    def recall_smart(
        self,
        query: str,
        top_k: int | None = None,
        confidence_threshold: float = 0.5,
        max_escalations: int = 2,
        deadline_ms: float | None = None,
        **kwargs,
    ) -> RecallPack:
        """Deadline-bounded auto-escalating recall for the INTERACTIVE path.

        Contract (the fix for the 120s foreground hang):
          * ONE wall-clock budget (`recall.smart_deadline_ms`, default 15s).
            No stage may START once it's exhausted; the best result we
            already have is returned immediately.
          * The fast path is LOCAL ONLY - BM25+vec+graph, plus a local
            cross-encoder rerank only if that model is already warm. It
            never resolves an LLM client / spawns the Claude CLI, so a hung
            backend cannot stall an interactive answer.
          * LLM query-decomposition is OPT-IN (`recall.smart_allow_llm`)
            and, when enabled, runs strictly inside the remaining budget
            (the LLM call's own timeout is clamped to the time left). For an
            unconditional deep pass, call `recall_deep()`.

        The returned pack carries `.escalation` diagnostics (which stages
        ran + why we stopped) so a caller knows NOT to fan out more recalls.
        """
        # We set _skip_decompose explicitly on every internal recall() call;
        # drop any caller-supplied value so it can't collide as a dup kwarg.
        kwargs.pop("_skip_decompose", None)
        if top_k is None:
            top_k = self.config.get("recall.top_k")
        if deadline_ms is None:
            deadline_ms = self.config.get("recall.smart_deadline_ms") or 15000
        t0 = time.perf_counter()
        deadline = t0 + (float(deadline_ms) / 1000.0)

        def _budget_left() -> float:
            return deadline - time.perf_counter()

        stages: list[str] = []

        def _finish(pack: RecallPack, reason: str) -> RecallPack:
            try:
                pack.escalation = {
                    "stages": stages,
                    "stopped": reason,
                    "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                    "deadline_ms": float(deadline_ms),
                    "confidence": pack.confidence,
                }
            except Exception:
                pass
            return pack

        # Stage 1 - fast LOCAL recall. _skip_decompose guarantees we never
        # drop into the LLM decomposition path even if recall.adaptive_
        # decompose is enabled globally; the interactive answer stays local.
        stages.append("local")
        pack = self.recall(query, top_k=top_k, _skip_decompose=True, **kwargs)
        best = pack
        if pack.confidence >= confidence_threshold:
            return _finish(best, "confidence_met")

        # Stage 2 - cheap local escalation: wider candidate pool + a local
        # cross-encoder rerank, but ONLY if the reranker model is already
        # loaded (never block the interactive path on a cold model load).
        if max_escalations >= 1 and _budget_left() > 0:
            if getattr(self.search, "reranker", None) is not None:
                stages.append("local_rerank")
                try:
                    pack2 = self.recall(
                        query,
                        top_k=max(top_k * 2, 25),
                        rerank=True,
                        _skip_decompose=True,
                    )
                    if pack2.confidence > best.confidence:
                        best = pack2
                    if best.confidence >= confidence_threshold:
                        return _finish(best, "confidence_met")
                except Exception:
                    pass
            else:
                stages.append("rerank_skipped_cold")

        # Stage 3 - OPT-IN bounded LLM decomposition. OFF by default. Even
        # when on it runs inside the remaining budget, and is skipped while
        # the LLM backend is in cooldown after a recent timeout.
        if (
            max_escalations >= 2
            and self.config.get("recall.smart_allow_llm")
            and _budget_left() > 2.0
        ):
            if _breaker.is_open("llm"):
                # backend tripped (#12) → don't pay the latency, stay local
                stages.append("llm_skipped_breaker_open")
            else:
                stages.append("llm_decompose")
                try:
                    rh = kwargs.get("recency_half_life_days") or self.config.get(
                        "recall.recency_half_life_days"
                    )
                    gb = kwargs.get("graph_boost")
                    gb = gb if gb is not None else self.config.get("recall.graph_boost")
                    rr = kwargs.get("rerank")
                    rr = rr if rr is not None else self.config.get("recall.rerank")
                    rtn = kwargs.get("rerank_top_n") or self.config.get("recall.rerank_top_n")
                    packd = self._recall_with_decomposition(
                        query,
                        top_k=top_k,
                        recency_half_life_days=rh,
                        graph_boost=gb,
                        rerank=rr,
                        rerank_top_n=rtn,
                        budget_s=_budget_left(),
                    )
                    if packd and packd.confidence > best.confidence:
                        best = packd
                except Exception:
                    pass  # breaker bookkeeping happens inside the call

        reason = "deadline_hit" if _budget_left() <= 0 else "exhausted_escalations"
        return _finish(best, reason)

    def breaker_status(self) -> dict:
        """Circuit-breaker state per backend (#12). Surfaced via the `stats`
        tool / dashboard so a temporarily-disabled deep backend is visible."""
        return _breaker.status()

    def recall_deep(
        self,
        query: str,
        top_k: int | None = None,
        deadline_ms: float | None = None,
        **kwargs,
    ) -> RecallPack:
        """Explicit DEEP recall: always ATTEMPTS LLM query-decomposition
        (RAG-Fusion), bounded by a deadline (default 2x the interactive
        budget). Use only when the caller knowingly wants the slow, thorough
        pass - never on the latency-sensitive interactive path. Falls back
        to a single-shot local recall if decomposition is unavailable."""
        kwargs.pop("_skip_decompose", None)
        if top_k is None:
            top_k = self.config.get("recall.top_k")
        if deadline_ms is None:
            deadline_ms = (self.config.get("recall.smart_deadline_ms") or 15000) * 2.0
        t0 = time.perf_counter()
        budget_s = float(deadline_ms) / 1000.0
        rh = kwargs.get("recency_half_life_days") or self.config.get(
            "recall.recency_half_life_days"
        )
        gb = kwargs.get("graph_boost")
        gb = gb if gb is not None else self.config.get("recall.graph_boost")
        rr = kwargs.get("rerank")
        rr = rr if rr is not None else self.config.get("recall.rerank")
        rtn = kwargs.get("rerank_top_n") or self.config.get("recall.rerank_top_n")
        try:
            packd = self._recall_with_decomposition(
                query,
                top_k=top_k,
                recency_half_life_days=rh,
                graph_boost=gb,
                rerank=rr,
                rerank_top_n=rtn,
                budget_s=budget_s,
            )
        except Exception:
            packd = None
        if packd is not None and packd.results:
            try:
                packd.escalation = {
                    "stages": ["llm_decompose"],
                    "stopped": "deep_done",
                    "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                }
            except Exception:
                pass
            return packd
        # Fallback: single-shot local recall (no LLM)
        return self.recall(query, top_k=top_k, _skip_decompose=True, **kwargs)

    def recall_scoped(
        self,
        query: str,
        project: str | None = None,
        top_k: int | None = None,
        **kwargs,
    ) -> RecallPack:
        """recall(), optionally scoped to a single project (issue #7).

        `project` is a FILTER over the unified user memory - NOT a separate
        store or workspace. When given, we over-fetch and keep only events
        tied to that project (project_name / project / project_path metadata,
        case-insensitive substring). When None/empty this is plain recall(),
        so default behaviour and the recall hot-path/cache are untouched.

        This is the additive first step of separating "user memory" from
        "project scope"; merging per-project workspaces into one memory is a
        separate, opt-in migration."""
        if top_k is None:
            top_k = self.config.get("recall.top_k")
        if not project:
            return self.recall(query, top_k=top_k, **kwargs)
        pj = str(project).strip().lower()
        wide = self.recall(query, top_k=max(top_k * 6, 30), **kwargs)
        kept = [r for r in wide.results if _result_in_project(r, pj)][:top_k]
        pack = RecallPack(
            query=wide.query,
            workspace_name=wide.workspace_name,
            workspace_id=wide.workspace_id,
            results=kept,
            n_total_in_workspace=wide.n_total_in_workspace,
            elapsed_ms=wide.elapsed_ms,
        )
        pack.escalation = {
            "stages": ["local", "project_filter"],
            "stopped": "project_filtered",
            "project": pj,
            "n_before_filter": len(wide.results),
            "n_after_filter": len(kept),
        }
        return pack

    def _recall_with_decomposition(
        self,
        query: str,
        top_k: int,
        recency_half_life_days: float,
        graph_boost: float,
        rerank: bool,
        rerank_top_n: int,
        budget_s: float | None = None,
    ) -> RecallPack | None:
        """Run query decomposition + sub-query retrieval + RRF merge.
        Returns None if decomposition couldn't run (LLM unavailable etc.)
        so caller can fall back to single-shot recall.

        `budget_s`, when given, clamps the resolved LLM client's own timeout
        to the remaining wall-clock budget - so a slow Claude CLI / Ollama
        call can never outlive the recall_smart deadline."""
        # Circuit breaker (#12): skip the LLM entirely while it's tripped.
        if _breaker.is_open("llm"):
            return None
        _thr = self.config.get("recall.breaker_threshold") or 2
        _cd = self.config.get("recall.breaker_cooldown_s") or 60.0
        try:
            from pmb.health.consolidate import resolve_llm_client
            from pmb.reasoning.decompose import QueryDecomposer, reciprocal_rank_fuse

            try:
                llm = resolve_llm_client(
                    backend=self.config.get("consolidate.backend"),
                )
            except Exception:
                _breaker.record_failure("llm", _thr, _cd, "resolve_llm_client failed")
                return None
            if budget_s is not None and hasattr(llm, "timeout"):
                try:
                    llm.timeout = max(1.0, min(float(llm.timeout), float(budget_s)))
                except Exception:
                    pass
            decomposer = QueryDecomposer(
                llm,
                cache_dir=self.workspace.storage_dir,
            )
            decomp = decomposer.decompose(query)
            _breaker.record_success("llm")  # the LLM responded - backend healthy
            if len(decomp.sub_queries) <= 1:
                # Decomposer said it's already single-hop; cancel decomposition
                return None

            # Run each sub-query through the standard pipeline. Skip
            # decomposition on these calls to prevent recursion.
            t0 = time.perf_counter()
            rankings: list[list[str]] = []
            all_pack_results: dict = {}  # ulid -> RecallResult (any sub-q)
            for sq in decomp.sub_queries:
                sub_pack = self.recall(
                    sq,
                    top_k=max(top_k, 15),
                    recency_half_life_days=recency_half_life_days,
                    graph_boost=graph_boost,
                    rerank=rerank,
                    rerank_top_n=rerank_top_n,
                    _skip_decompose=True,
                )
                rankings.append([r.ulid for r in sub_pack.results])
                for r in sub_pack.results:
                    all_pack_results.setdefault(r.ulid, r)

            # Fuse rankings into one
            fused = reciprocal_rank_fuse(rankings, k=60)[:top_k]
            results = [all_pack_results[u] for u, _ in fused if u in all_pack_results]

            n_total = self._active_count_cached()
            return RecallPack(
                query=query,
                workspace_name=self.workspace.name,
                workspace_id=self.workspace.id,
                results=results,
                n_total_in_workspace=n_total,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as e:
            import logging

            _breaker.record_failure("llm", _thr, _cd, str(e))
            logging.getLogger(__name__).info("Decomposition failed: %s", e)
            return None

    def _attach_event_time(self, ev) -> None:
        """Parse content for explicit date references and store as
        metadata.event_time (UTC timestamp). Free, no LLM."""
        if not self.config.get("recall.temporal_enabled"):
            return
        try:
            from pmb.reasoning.temporal import extract_event_time

            t = extract_event_time(ev.content or "", reference_now=ev.timestamp)
            if t is None:
                return
            # Update metadata in-place via direct SQL
            import json as _j
            import sqlite3

            with sqlite3.connect(self.workspace.db_path) as conn:
                row = conn.execute(
                    "SELECT metadata_json FROM events WHERE ulid = ?",
                    (ev.ulid,),
                ).fetchone()
                if row is None:
                    return
                try:
                    meta = _j.loads(row[0]) if row[0] else {}
                except Exception:
                    meta = {}
                meta["event_time"] = float(t)
                conn.execute(
                    "UPDATE events SET metadata_json = ? WHERE ulid = ?",
                    (_j.dumps(meta), ev.ulid),
                )
            # Also bump in-memory metadata if dataclass instance is reused
            if hasattr(ev, "metadata") and isinstance(ev.metadata, dict):
                ev.metadata["event_time"] = float(t)
        except Exception:
            pass
