# PMB — Implementation Tasks (instructions for Claude Opus)

> Prepared 2026-06-09 on top of `main` @ 5702d09 (v0.5.0, PR #15 merged).
> Все задачи согласованы с владельцем проекта. Выполнять по порядку, по одной,
> с полным прогоном тестов после каждой группы.

---

## 0. GROUND RULES — read before touching anything

### Environment
- Windows 11, repo: `C:\Users\alexb\OneDrive\Рабочий стол\pmb` (package `pmb-ai`).
- Python venv: `.venv` → run everything as `./.venv/Scripts/python.exe -m ...`.
- Install/deps: `uv`. Lint: `Ruff`. Tests: `pytest` (+ `pytest-asyncio`).
- Console output with Cyrillic: prefix commands with `$env:PYTHONUTF8 = "1"`.
- The embedding model has a one-time cold load (~15-20s). In timing-sensitive
  tests, warm it OUTSIDE the timed region (see `tests/test_recall_smart_deadline.py:77-79`).

### Git — HARD RULES
- **NEVER run `git add` / `git commit` / `git push`.** When work is ready,
  print the exact commands for the user to run themselves.
- `git checkout` / `git pull` / branch creation are allowed.
- Start from fresh `main`, create branch **`chore/hardcode-ux-memory`**
  (branch names must NOT contain the word "claude").

### Real user data — HARD RULES
- The real workspace lives under `~/.pmb/workspaces/...`. **NEVER write to,
  repair, or archive anything in the real workspace without an explicit OK
  from the user in chat.** For verification, copy the workspace directory to
  a temp location and point `PMB_HOME` at the copy.
- All cleanup operations must be **archive-only** (reversible), never delete.

### Compatibility — HARD RULES
- Default behavior must not change unless a task explicitly says so.
  New config keys are additive with safe defaults. New CLI flags default to
  current behavior. Existing MCP tool signatures: additive args only.

### Test discipline
- Full suite: `./.venv/Scripts/python.exe -m pytest -q` (~6-8 min).
- Pre-existing flaky tests (NOT your regressions, do not chase them):
  `test_auto_consolidate*`, `test_rehearse_max_cap*`, one `temporal` test,
  and `test_mcp_e2e` flickers under load. Baseline as of v0.5.0: ~914 passed,
  3-4 flaky.
- **Never declare the suite green from a subset run.** Always finish with the
  full suite and compare against the baseline above.
- Every task below ships with its own new/updated tests.

---

## TASK 1 (P0, small) — Remove "alex" literals from the identity boost

**File:** `src/pmb/core/engine/recall.py:903-919`.

**Problem:** the identity-marker boost hardcodes a specific person's name:
```python
content_head.startswith("alex ")
content_head.startswith("alex's ")
```
This is a leaked personal name in library code. It only works for a user
named Alex and silently fails for everyone else.

**The replacement mechanism already exists in the same file:**
- `recall.py:19-20` imports `mine_user_names_from_db as _mine_user_names`
  (from `src/pmb/reasoning/user_names.py`).
- `recall.py:734-757` maintains `self._user_names_cache: set[str]`, refreshed
  every 25 new events, and already passes it to PAMVR as `user_names=`.

**What to do:**
1. Read `recall.py:720-930` fully to understand cache lifecycle and where the
   boost block sits relative to it (the cache is populated in a different
   code path — make sure it is available/initialized where the boost runs;
   if not, factor the cache-refresh into a small helper `_get_user_names()`
   used by both call sites).
2. Replace the two `"alex"` literals with a check: the first token of
   `content_head` (stripped of a trailing `'s`) is in `_get_user_names()`
   (lowercased comparison).
3. KEEP the generic English first-person/role markers as-is
   (`"user's "`, `"user "`, `"i am "`, `"my "`, `"i work "`, `"i prefer "`) —
   they are language-functional, not personal data.
4. Tests (`tests/test_identity_boost_names.py`):
   - Engine where a fact "My name is Bob" was recorded → fact
     `"Bob's terminal is wezterm"` receives the identity boost on an
     identity-intent query, while `"Zorg's terminal is wezterm"` does not.
   - No fact-name recorded → no name-based boost, generic markers still work.
   - Grep-test or plain assertion: the string `"alex"` no longer appears in
     `src/pmb/core/engine/recall.py`.

**Acceptance:** no personal-name literals in `recall.py`; existing recall
tests pass; new tests green.

---

## TASK 2 (P0, small) — Empty `DEFAULT_NAMED_ENTITIES` in PAMVR

**File:** `src/pmb/reasoning/pamvr.py:118` and its single consumer at `:270`:
```python
DEFAULT_NAMED_ENTITIES = {"alex", "bob", "carol", "dana", "alice", "stripe", "adyen"}
...
f.entities = named_entities or DEFAULT_NAMED_ENTITIES   # line 270
```

**Problem:** test/benchmark names leaked into the production default of the
ranker. The dynamic `_PROPER_NOUN_RE` (pamvr.py:124) + the learned
`user_names` already cover real entity detection.

**What to do:**
1. Set `DEFAULT_NAMED_ENTITIES = frozenset()` (keep the symbol for back-compat
   imports; update the comment to say entities come from the caller /
   proper-noun extraction).
2. `:270` — `named_entities or DEFAULT_NAMED_ENTITIES` then resolves to the
   empty set; verify by reading the surrounding function that an empty
   `f.entities` degrades gracefully (entity check becomes a no-op rather than
   penalising everything). If any code path treats "no entities" differently
   from "empty set", normalize it.
3. Fix the tests that implicitly relied on the default list (run
   `rg -l "DEFAULT_NAMED_ENTITIES|carol|adyen" tests/`): inject entities
   explicitly via the `named_entities` parameter in those tests.
4. Sanity on data: with a COPY of the real workspace, run 3 recall queries
   that previously worked (e.g. "where do I live", one project query, one
   person query) and confirm top-3 results unchanged.

**Acceptance:** default is empty; no behavior change on the real-corpus copy;
full suite at baseline.

---

## TASK 3 (P1) — English-only CLI + status panel + loading feedback

### 3a. Translate user-facing Russian to English
Typer renders command docstrings as `--help` text, so Russian docstrings leak
into the user-facing CLI. Confirmed locations (re-verify with
`rg -n "\p{Cyrillic}" src/pmb/cli`):
- `src/pmb/cli/main.py:5-12` — module docstring (command list in Russian).
- `src/pmb/cli/commands/capture.py:81,88,133,172,204,530,675,683` — docstrings/comments.
- `src/pmb/cli/commands/health.py:55,124,147` — docstrings.
- `src/pmb/cli/commands/maintenance.py:150,585,599,617,769,805` — docstrings.
- `src/pmb/cli/commands/manage.py:965` — docstring.
- `src/pmb/cli/commands/ambient.py:329` — Russian usage example → English example.

**Do NOT touch** (functional multilingual DATA, not UI):
- `src/pmb/reasoning/pamvr.py` synonym sets and `_NOT_PROPER`;
- `src/pmb/reasoning/attributes.py` detection patterns;
- `src/pmb/reasoning/user_names.py` name patterns;
- `src/pmb/cli/connect.py:111,135` — trigger words («запомни» etc.) inside the
  agent-instructions template; they are functional triggers for RU-speaking users.
- Anything under `tests/` (Cyrillic fixtures are intentional).

**Acceptance:** `rg -n "\p{Cyrillic}" src/pmb/cli` shows ONLY `connect.py`
template lines; `pmb --help` and every subcommand help read in English.

### 3b. `pmb` status panel (no args)
Make bare `pmb` (typer callback with `invoke_without_command=True`, keeping
`--help` intact) print a rich Panel/Table dashboard:
- version (`pmb.__version__`), active workspace: name, short id, and HOW it
  was resolved (env `PMB_WORKSPACE` / project `.pmb/workspace.yaml` /
  persisted default / personal fallback — see Task 4);
- storage path + sizes (events.sqlite, vectors.lance);
- event counts by type (facts / lessons / goals / activities) + pinned count;
- running MCP servers via `src/pmb/mcp/registry.py:list_servers()` (pid,
  transport, RSS MB) — reuse, don't duplicate, the `pmb mcp status` logic;
- embedding model warm/cold (`engine.is_warm()` style probe — do NOT trigger
  the 15-20s load from the panel; report "cold" instead);
- hint footer: `pmb --help` for commands.

### 3c. Loading feedback
Wrap the known slow paths in `rich.console.Console.status(...)` spinners with
a clear label ("loading embedding model (first run only)…", "rebuilding BM25
index…", "migrating workspaces…"). Slow paths to cover: first `pmb recall`
/ `pmb remember` in a cold process, `pmb index`, `pmb migrate-workspaces`,
`pmb compact`. Rich auto-disables spinners when not a TTY — no CI breakage.

**Tests:** CliRunner smoke tests: bare `pmb` exits 0 and contains workspace
name; help strings contain no Cyrillic (regex assertion over `--help` output
of the main app and each registered subcommand).

---

## TASK 4 (P1) — Workspace switching UX

**Problem:** switching to `personal` is not discoverable. Today resolution is
(see `src/pmb/core/workspace.py:5,129,143`): `PMB_WORKSPACE` env override →
project detection → fallback. There is NO `use` command; `pmb workspaces`
(manage.py:965) only lists. The `pmb workspace ...` subapp
(`src/pmb/cli/commands/workspace.py`) is currently git-sync only (init/push/pull).

**What to do (additive, do not break `detect_workspace()` callers):**
1. Read `src/pmb/core/workspace.py` fully first.
2. Add a persisted default: file `PMB_HOME/current_workspace` (single line,
   workspace name) or config key — pick the file (works without config load).
   New resolution order: `PMB_WORKSPACE` env → explicit project
   `.pmb/workspace.yaml` → **persisted default** → personal/auto fallback.
   The new step slots in BEFORE the final fallback only, so every existing
   setup resolves exactly as before until the user opts in.
3. New commands on the existing `workspace_app`:
   - `pmb workspace use <name>` — validate the name exists (offer close
     matches if not), write the persisted default, print confirmation panel.
   - `pmb workspace use --clear` — remove the persisted default.
   - `pmb workspace current` — active workspace + which resolution rule won.
4. `pmb workspaces` list: mark the active one (`*`), add footer hint
   "switch: pmb workspace use <name>".
5. Show resolution source in the Task-3b panel.

**Tests:** resolution-order unit tests (env beats persisted; persisted beats
fallback; absent file = old behavior byte-for-byte); CLI tests for
use/current/--clear; unknown name → exit 1 with suggestion.

---

## TASK 5 (P1) — Tombstone cleanup: negation/"unknown" facts must be superseded

**The real-corpus case (workspace `0019ea88`):** the fact
`"As of June 8, 2026, the user does not currently live in Warsaw; current
city is unknown."` was stored, and MINUTES LATER `"...currently lives in
Tampa..."` arrived. The negation fact is now permanently stale noise: it says
"unknown" while the answer is known. v0.5.0's `_NEGATION_RE` only PREVENTS
promoting such text to a keyed value — nothing ever retires the fact itself.

**Design (archive-only, two hooks):**
1. `src/pmb/reasoning/attributes.py`: add
   `detect_negated_state(content) -> Optional[str]` returning the canonical
   attribute when content matches negation/unknown phrasing about the user
   (reuse `_NEGATION_RE` + add patterns for `"current <attr> is unknown"`,
   `"no longer <verb>"`, `"больше не"`, `"уже не"` + attribute aliases from
   `_ALIAS_GROUPS`). Conservative: if attribute can't be confidently mapped,
   return None.
2. **Write-time hook** in `src/pmb/core/engine/write.py`: after a successful
   keyed upsert of `user::X` (both `record_keyed_fact` and
   `_maybe_promote_current_state` paths), scan ACTIVE plain facts where
   `detect_negated_state(content) == X` and `timestamp < new fact's` →
   archive them with `metadata["superseded_by"] = <new ulid>` and
   `metadata["superseded_reason"] = "negation_obsoleted_by_value"`.
   Skip pinned events and `kind=lesson` (lessons are instructions, not state).
   Bound the scan (e.g. last 2000 active facts) to keep the write path fast.
3. **Repair pass:** extend `repair_keyed_facts` (or add Pass 3 to
   `pmb repair-keyed`) doing the same retroactively: for every attribute with
   a CURRENT keyed value, archive older active negation facts for that
   attribute. Dry-run default, report a table.

**Config:** `keyed.archive_obsolete_negations` (bool, default **True** — this
is a bug-class fix, and archive-only is reversible).

**Tests (`tests/test_negation_tombstones.py`):** the Warsaw case verbatim
(negation fact → newer Tampa keyed fact → negation archived with
superseded_by); lesson with same phrasing NOT archived; pinned NOT archived;
config off → nothing archived; repair pass collapses a pre-existing case.

**Verification on data:** on a COPY of the real workspace run the repair
dry-run and show the user that exactly the
"does not currently live in Warsaw; current city is unknown" fact is the
candidate. Applying to the REAL workspace requires the user's explicit OK.

---

## TASK 6 (P1) — `pmb decay --archive-cold` (time-based forgetting)

Today `pmb decay` (`src/pmb/cli/commands/maintenance.py:582-585`) only lowers
importance; junk lingers forever. Add an opt-in archival mode:

`pmb decay --archive-cold [--days N] [--max-importance F] [--apply]`
- Candidates: ACTIVE events with `importance <= F` (default 0.25) AND
  `access_count == 0` AND `age > N days` (default 90) AND not pinned AND not
  a current keyed fact AND `event_type in {"fact", "activity"}` (never
  lessons, goals, preferences, milestones).
- Dry-run is the default; prints a rich table (ulid, age, importance, head of
  content) + total. `--apply` archives with
  `metadata["archived_reason"]="decay_cold"`.
- Engine method `archive_cold(days, max_importance, dry_run=True) -> dict` in
  the engine (near decay logic), CLI is a thin wrapper.
- Config (additive): `decay.archive_cold_days` (90), `decay.archive_cold_max_importance` (0.25).

**Tests:** seeded old/cold facts archived; accessed/pinned/keyed/lesson/goal
survivors; dry-run mutates nothing; defaults respected from config.

---

## TASK 7 (P2) — `pmb declutter` (junk sweeper, heuristic + optional LLM judge)

New command `pmb declutter [--apply] [--llm]`, new module
`src/pmb/maintenance/declutter.py`.

**Heuristic candidates (no LLM):**
- test artifacts: keyed keys matching `r"test_attr|tmp|dummy|placeholder"`,
  content matching obvious test patterns (`final_value`, `lorem`, …);
- near-empty content (< 8 chars after strip) or pure stopwords;
- negation tombstones already obsoleted (delegate to Task 5's detector);
- exact-duplicate active content (keep newest, archive rest).

**Optional `--llm` judge:** for borderline candidates ONLY (cap: 50 per run),
resolve via `pmb.health.consolidate.resolve_llm_client` with timeout clamped
to ≤ 15s per batch and the existing circuit breaker
(`pmb.core.circuit_breaker`) — same pattern as `_recall_with_decomposition`
in `recall.py`. Prompt: "Is this memory junk (test data / placeholder /
meaningless)? Answer JSON {ulid, junk: bool, reason}." Never on any hot path;
this is a manual maintenance command.

Dry-run default; `--apply` archives with `archived_reason="declutter"` +
reason. Output: rich table grouped by reason.

**One-off after this lands (ONLY with the user's explicit OK in chat):** run
it on the real workspace; it must catch `user::test_attr_39da34=final_value`
(known junk currently ranked #1 in the corpus).

**Tests:** each heuristic class; LLM judge mocked (never a real subprocess in
tests); breaker-open → heuristics still work, LLM skipped; archive-only.

---

## TASK 8 (P2) — Write-time quality gate (default OFF)

Config: `write.quality_gate` (bool, default **False**).

When ON, in the shared write path (`write.py`, before embedding): score the
incoming content with cheap heuristics (length < 8, placeholder/test
patterns from Task 7, pure-stopword content). Suspected junk is **NOT
rejected** (a memory system must not silently drop user input):
- cap `importance` at 0.2,
- set `metadata["quality_flag"] = "suspect_junk"`,
- exclude `quality_flag` events from keyed promotion and from
  `_maybe_promote_current_state`.
`pmb declutter` then treats flagged events as first-class candidates.

**Tests:** gate off (default) → byte-identical behavior to today; gate on →
flag+cap applied, good content untouched, flagged content never promoted.

---

## TASK 9 (P1) — Plans: "запомни, что дальше делаем X" must become a goal, not a fact

**Infra already exists — this is a routing gap.** Confirmed:
- `src/pmb/core/engine/goals.py` — `record_goal(title, status, parent_goal_ulid, due_at, ...)`,
  `event_type="goal"`, statuses pending/in_progress/done/cancelled, RU/EN dedup.
- `record_batch` already accepts `{"type": "goal", ...}` (`src/pmb/core/engine/batch.py:172`).
- MCP `record_goal` exposed (`src/pmb/mcp/tools.py:863`); prepare/auto-context
  already surfaces open goals.

**What to do:**
1. **Docstrings are the agent's instructions — fix the routing there.** In
   `src/pmb/mcp/tools.py` update `record_fact`, `record_batch`, `record_goal`
   docstrings with an explicit rule block:
   "FUTURE INTENT → goal, NOT fact. If the user says 'запомни, что будем
   делать X / remember we'll do X next / план такой / next steps are…',
   record `{"type": "goal", "title": ..., "status": "pending"}` (with
   `parent_goal_ulid` for sub-steps), never `{"type": "fact"}`."
   Mirror the same rule in the agent-instructions template in
   `src/pmb/cli/connect.py` (the WRITE-triggers table: add a "будем делать
   дальше / next we'll" row → goal).
2. **Batch alias:** in `batch.py` accept `{"type": "plan", ...}` as an alias
   for goal with `metadata["kind"]="plan"`, `status="pending"` default.
   Echo `"type": "plan"` back in results.
3. **Write-time safety net** (cheap, like `_maybe_promote_current_state`):
   in `record_fact`, if content matches a conservative future-intent pattern
   (`r"^(?:будем|планируем|надо будет|next(?: steps?)?[:,]|plan[:,]|we(?:'ll| will)\b)"`,
   case-insensitive, first 40 chars) → don't move the content; just add
   `metadata["suggest_goal"]=True` so the dashboard/`pmb doctor` can show
   "N facts look like plans — consider `pmb goals promote`". NO automatic
   conversion (too risky for false positives).
4. **CLI:** check whether a goals listing command exists
   (`rg -n "goal" src/pmb/cli/commands/`); if not, add `pmb goals`
   (list open goals with status/due, `--all` includes done) and
   `pmb goals done <ulid>`.

**Tests:** batch `plan` alias → `event_type="goal"` + kind=plan; docstring
contains the routing rule (string assertion keeps it from regressing);
suggest_goal flag set only on clear future-intent phrasings, NOT on
"we decided X yesterday".

---

## TASK 10 (P2) — Reference data out of Python into config

Move stable word-lists to `PMB_HOME/reference.yaml` (one file, namespaced
keys), loaded once with the current in-code values as fallback defaults —
missing file = identical behavior, file present = per-deployment extension.
- `_ALIAS_GROUPS` (`src/pmb/reasoning/attributes.py`) — merge semantics:
  yaml EXTENDS code defaults, can't shrink them.
- `KNOWN_TECHS` (`src/pmb/graph/entities.py`) — extend-only.
- `text_match` stopwords (`src/pmb/core/text_match.py`) — extend-only.
- `_NOT_PROPER` (`src/pmb/reasoning/pamvr.py`) — extend-only.
- `_KIND_PRIORITY` (`src/pmb/core/engine/types.py`) — override-allowed.
Add a loader module `src/pmb/reference_data.py` (cached, tolerant of missing/
malformed yaml → warning + defaults). Document the schema in README.

**Tests:** no file → exact current behavior; file with extras → merged;
malformed file → warning + defaults, no crash.

---

## TASK 11 (P3, design + minimal slice) — Offline LLM tier for open-ended understanding

Principle (recorded project decision): hardcoded regex/keyword lists are the
wrong tool for OPEN-ENDED language understanding. Keep regexes only as cheap
fast-paths; open-ended detection belongs to the OFFLINE LLM tier
(consolidate/reflect), batched, bounded, breaker-protected, **never on the
synchronous hot path**.

**Minimal slice now:** extend the existing consolidation/reflection batch
(find it: `rg -n "reflect|consolidat" src/pmb/health/`) so each batch also
emits keyed-state suggestions:
`{subject, attribute, value, negation: bool, confidence}` for facts the regex
fast-path missed. Apply suggestions with `confidence >= 0.8` through the SAME
canonical upsert (`record_keyed_fact`) and Task-5 tombstone logic; below
threshold → store as `metadata["suggested_key"]` for the dashboard. Gate with
config `consolidate.suggest_keyed` (default True — it runs offline anyway).
Clamp LLM timeout, respect the circuit breaker, cap batch size.

**Tests:** with a mocked LLM client, suggestions above/below threshold;
negation suggestions create tombstones; breaker-open → batch skipped quietly.

---

## Recommended order, versioning, finish line

1. Branch `chore/hardcode-ux-memory` off fresh `main`.
2. Tasks **1, 2** (hardcode, small) → full suite.
3. Tasks **3, 4** (CLI/UX) → full suite.
4. Tasks **5, 9** (memory correctness + plans) → full suite.
5. Task **6** → suite; Tasks **7, 8, 10** → suite; Task **11** last.
6. Update `CHANGELOG.md` ([0.6.0]: Fixed/Added/Changed per task), bump
   version `0.5.0 → 0.6.0` in `pyproject.toml` AND `src/pmb/__init__.py`.
7. Run Ruff. Run the FULL suite. Compare to baseline (~914 passed,
   3-4 known flaky).
8. Print for the user (do not run): `git add -A`, `git commit -m ...`,
   `git push -u origin chore/hardcode-ux-memory`, PR command via `gh`.
9. Real-workspace actions (Task 5 repair apply, Task 7 declutter apply,
   forgetting `user::test_attr_39da34`) — ONLY after explicit user OK,
   with a backup copy taken first and a before/after diff shown.

## Definition of done (checklist)
- [ ] No personal-name literals in `src/` (rg check: `alex`, `carol`, `adyen` …)
- [ ] `rg "\p{Cyrillic}" src/pmb/cli` → only connect.py trigger templates
- [ ] Bare `pmb` → status panel; `pmb workspace use personal` works
- [ ] Warsaw-"unknown" tombstone case covered by tests + dry-run shown on copy
- [ ] `pmb decay --archive-cold`, `pmb declutter` exist, dry-run default
- [ ] `{"type": "plan"}` lands as goal; MCP docstrings route future intent to goals
- [ ] Reference yaml loader with fallback defaults
- [ ] Full suite at baseline, CHANGELOG + version 0.6.0, push commands printed
