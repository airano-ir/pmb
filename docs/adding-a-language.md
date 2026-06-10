# Adding a language to PMB

PMB's lexical fast-paths (stopwords, function-words, verb synonyms, attribute
aliases, first-person markers) ship covering **English, Russian and Ukrainian**.
Everything else — the embedding model, vector recall, the graph — is already
language-agnostic. To make the *lexical* layer work for another language you add
a **language pack**: one YAML file. No code changes.

## Quick start

```bash
pmb lang list              # see built-in templates (de, es) + what's enabled
pmb lang detect            # sample your workspace, suggest packs (never auto-enables)
pmb lang enable de         # copy the German template into $PMB_HOME/lang/de.yaml
pmb daemon restart         # so the warm daemon picks it up
pmb reindex                # align the BM25 index with the extended tokenizer
```

To add a language with **no** built-in template, create the file yourself:

```bash
pmb lang enable fr         # scaffolds an empty $PMB_HOME/lang/fr.yaml
# then edit that file (see the schema below)
```

## How it works

- The EN/RU/UK lists live in code as the **floor** and never change.
- A pack is **active** when its file exists in `$PMB_HOME/lang/<code>.yaml`.
- Active packs **extend** the floor (union) — they never remove anything. So
  with no pack files PMB behaves byte-for-byte as before.
- Activation is **opt-in**, not automatic by script: German and English share
  the Latin alphabet, so auto-enabling German on any Latin corpus would pollute
  an English workspace's stopwords. `pmb lang detect` suggests; you decide.

## Pack schema

All keys are optional; include only what you have.

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

# Verb synonym groups. The canonical keys are PMB's (live / work / use / own /
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

## Contributing a pack upstream

Built-in templates live in `src/pmb/lang/packs/`. To contribute German/Spanish
improvements or a new language, add or edit a file there and open a PR — the
`de.yaml` / `es.yaml` files are the reference examples.
