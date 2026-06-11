"""X4 — Service Level Objectives as code.

The quality / latency / durability targets PMB commits to, each tied to the
test that ENFORCES it. Keeping them here (not only in prose) means an SLO can't
silently lose its gate: `tests/test_slo.py` checks every `enforced_by` points
at a real, present test.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SLO:
    name: str
    target: str
    enforced_by: str   # "tests/<file>.py::<test_name>"


SLOS: list[SLO] = [
    SLO("recall.quality.en_top1", ">= 0.80 top-1 on the English eval bucket",
        "tests/test_memory_eval.py::test_memory_quality_floors"),
    SLO("recall.quality.ru_top1", ">= 0.66 top-1 on the Russian eval bucket",
        "tests/test_memory_eval.py::test_memory_quality_floors"),
    SLO("recall.quality.uk_top1", ">= 0.50 top-1 on the Ukrainian eval bucket",
        "tests/test_memory_eval.py::test_memory_quality_floors"),
    SLO("recall.quality.xl_top3", ">= 0.66 top-3 cross-lingual (EN query -> RU fact)",
        "tests/test_memory_eval.py::test_memory_quality_floors"),
    SLO("recall.quality.overall_top3", ">= 0.85 overall top-3",
        "tests/test_memory_eval.py::test_memory_quality_floors"),
    SLO("hook.latency.p95", "< 300 ms thin-client overhead p95 (warm daemon)",
        "tests/test_hook_client.py::test_prepare_context_client_overhead_p95"),
    SLO("durability.backup_round_trip", "a file-copy backup restores every event",
        "tests/test_backup_restore.py::test_file_copy_backup_round_trips"),
    SLO("durability.schema_self_heal", "a missing index is re-applied on Engine open",
        "tests/test_backup_restore.py::test_schema_migration_self_heals_missing_index"),
    SLO("api.tool_surface_stable", "the public MCP tool surface never silently shrinks",
        "tests/test_api_contract.py::test_mcp_tool_surface_is_stable"),
]
