"""Per-area PMB CLI command modules.

Each module defines either a sub-Typer (e.g. `graph_app`) or registers root
commands onto the shared `pmb.cli._common.app`. `pmb.cli.main` imports them so
their command registrations run; nothing here imports `pmb.cli.main` (that
would be a cycle).
"""
