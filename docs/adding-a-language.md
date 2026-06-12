# Adding a language to PMB

**Short version: you probably don't have to do anything.** As of v0.9.0 PMB has
no per-language packs in its core. A language works because the multilingual
embedder already knows it — not because someone hand-wrote a list for it.

## Why most languages just work

PMB's language understanding comes from three layers, none of which is a
per-language file:

1. **Recall** — the embedder (`paraphrase-multilingual-MiniLM-L12-v2`, 50+
   languages) maps same-meaning text to nearby vectors across languages. A
   Russian query finds an English fact with no translation. This was always
   language-agnostic.
2. **Intents + keyed extraction (warm)** — classified by **English semantic
   anchors**. The anchors carry English exemplars only; the embedder projects
   your language next to them, so "was sind meine Ziele" lands on the same
   `goals_query` anchor as "what are my open goals". One mechanism, every
   language the model knows — no German/Spanish/… data to write.
3. **Cold lexical path (self-compiling)** — when the warm anchor tier
   classifies your messages, it logs which n-grams co-fired with which anchor.
   The maintenance tick distils the high-precision ones into
   `$PMB_HOME/lang/auto.yaml` (**anchor→lexicon distillation, ALD**). After you
   have used a language a little, its common phrasings classify *cold* — at
   regex speed, no model — generated from your own traffic, not from a pack.

The old `ru.yaml` / `uk.yaml` packs were **deleted** in v0.9.0: the embedder +
anchors + ALD cover their job. English remains as a tiny inline bootstrap floor
for a brand-new, empty workspace.

## Check that your language works

```bash
pmb warmup                                   # load the embedder once
pmb recall "<a query in your language>"      # should surface the right facts
```

Recall is the part that works on day one. Intents/extraction work as soon as the
**daemon** is running (the anchors are warm-only). The cold lexical path fills in
over the next few days of real use as ALD distils your phrasings — check progress
any time:

```bash
cat "$PMB_HOME/lang/auto.yaml"               # grows from your traffic
```

### Honest limits

- **Needs the warm daemon.** ALD only learns while the anchor tier is classifying
  (warm). Pure cold-stdio use with no daemon never populates `auto.yaml`.
- **Needs traffic.** A language you have barely used has nothing distilled yet;
  recall still works, cold intents don't until the support threshold is crossed.
- **Space-delimited languages.** ALD tokenises on word boundaries, so CJK
  (no spaces) stays warm-anchor-only — it is not distilled to the cold path.
- **Embedder ceiling.** Recall quality is the embedder's. CJK and a few
  lower-resource languages are weaker on exact top-1 (still strong in top-3).

## If your language is weak: swap the embedder

This is the real lever now — not writing a pack. If recall is poor for your
language, use a stronger multilingual model:

```bash
pmb config set embedding.model BAAI/bge-m3   # heavier, broader language coverage
pmb reindex                                  # re-embed under the new model
pmb daemon restart
```

The anchor calibration is keyed by model id, so it rebuilds itself for the new
embedder automatically — no other changes needed.

## Optional: hand-seed a pack (escape hatch)

The file-based pack mechanism still exists for power users who want to bootstrap
a language's **cold** path *before* ALD has seen enough traffic, or to pin
domain vocabulary. It is opt-in and additive — with no pack files PMB behaves
exactly as shipped.

```bash
pmb lang list              # shipped templates (de, es) + what's enabled
pmb lang detect            # sample your workspace, suggest a template (never auto-enables)
pmb lang enable de         # copy the German template to $PMB_HOME/lang/de.yaml
pmb lang enable fr         # no template for fr → scaffolds an empty file to edit
pmb daemon restart         # warm daemon picks it up
pmb reindex                # align the BM25 index with the extended tokenizer
```

A pack is **active** when its file exists in `$PMB_HOME/lang/<code>.yaml`. This
is the same format ALD writes to `auto.yaml`, so a hand pack and the distilled
one merge cleanly.

### Pack schema (all keys optional — include only what you have)

```yaml
code: de
name: German

# Function words dropped from lexical matching / not treated as proper nouns
# when they open a sentence.
stopwords: [der, die, das, und, ist, ich, nicht, ...]

# Sentence-initial words that look capitalised but aren't names.
not_proper: [wann, warum, wo, wer, was, wie]

# First-person markers — lets PAMVR recognise "this fact is about the user".
first_person: [ich, mein, meine, mir, mich]

# Verb synonym groups. Canonical keys are PMB's (live / work / use / own /
# decide / deploy / migrate / fix / name / …); add this language's stems so an
# English query like "where do I live" matches a fact written in this language.
verb_synonyms:
  live: [wohne, wohnt, lebe, lebt]
  work: [arbeite, arbeitet]

# Keyed-fact attribute aliases. Canonical keys are PMB's (city / country /
# employer / job_title / email / phone / hometown / relationship_status / …);
# add the labels this language uses so "Stadt" maps to the same key as "city".
attribute_aliases:
  city: [stadt, wohnort]
  employer: [arbeitgeber, firma]
```

After editing a pack, restart any running `pmb daemon` and run `pmb reindex` so
the BM25 index is rebuilt with the extended tokenizer.

## Contributing a template upstream

Built-in templates live in `src/pmb/lang/packs/` (`de.yaml`, `es.yaml` are the
reference examples). To contribute a new template, add a file there and open a
PR. Note that templates are a convenience for the cold-start escape hatch above —
the core no longer depends on them, and the project does **not** ship
default-active packs.
