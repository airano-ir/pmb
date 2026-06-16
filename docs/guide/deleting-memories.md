# Deleting memories

PMB gives you two ways to remove a memory, and two levels of removal. You can do
it from the web dashboard or from the command line, and both offer the same
choice between a reversible archive and a permanent delete.

## Soft vs hard: pick the right level

- **Archive (soft, reversible).** The memory is hidden from recall and from the
  active list, but the record stays on disk. You can bring it back at any time.
  This is the safe default and what you want most of the time.
- **Delete forever (hard, permanent).** The memory is purged completely: the
  database row, its search vector, and its links in the entity graph are all
  removed. Recall can never surface it again, and it cannot be restored.

If you are not sure, archive. You can always purge later, but you cannot undo a
permanent delete.

## From the dashboard

1. Start the dashboard: `pmb dashboard` (it opens on
   `http://127.0.0.1:8765`). If the shared daemon already uses that port, run it
   elsewhere, for example `pmb dashboard --port 18888`.
2. Open any memory by clicking it in the Timeline, the Map, or another view. A
   detail panel appears.
3. At the bottom of the panel you have three actions:
   - **Pin** keeps the memory at maximum importance and protects it from
     automatic cleanup.
   - **Archive** hides it reversibly. You can restore it later from the command
     line with `pmb restore`.
   - **Delete** (the red button) removes it permanently. The dashboard asks you
     to confirm first, because this cannot be undone.

After you act, the panel closes and the current view refreshes so the memory you
removed disappears right away.

## From the command line

### Delete one or more memories

```bash
pmb delete <ulid>                 # archive (reversible) - the default
pmb delete <ulid> --hard          # purge permanently
pmb delete <ulid_a> <ulid_b> ...  # several at once
pmb delete <ulid> --yes           # skip the confirmation prompt
```

`pmb delete` always shows a short preview of what it is about to remove and asks
you to confirm, unless you pass `--yes`. For a permanent delete the prompt
defaults to "no", so a stray Enter never destroys anything.

### Restore an archived memory

```bash
pmb restore <ulid>
```

This undoes a soft delete (an archive). A permanent `--hard` delete cannot be
restored.

### Other ways to remove memory

```bash
pmb forget <ulid>            # quick single archive (same as a soft delete)
pmb forget-topic <topic>     # archive everything that mentions a topic
pmb prune-expired            # archive memories whose TTL has passed
```

`pmb forget-topic` is handy for clearing a whole project or subject in one step.
Add `--dry-run` to preview what it would archive, and `--yes` to skip the
prompt.

## Finding the ULID

Every memory has a short id (a ULID). You can find it in a few places:

- In the dashboard event panel, and in `pmb timeline` and `pmb list`, where the
  first part of the ULID is shown.
- In `pmb recall "<query>"` output next to each hit.

You only need the leading part of the ULID that the views display.

## What a permanent delete removes

A hard delete is thorough on purpose. For the target memory it removes:

- the event row in SQLite,
- the embedding vector in the local vector index, so it leaves search entirely,
- the rows that linked it to entities and to other events in the graph, so
  nothing points at a memory that no longer exists.

Archived memories, by contrast, keep all of this and are simply marked as
archived until you restore them or purge them later.
