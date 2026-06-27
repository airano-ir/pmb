"""A warm engine must pick up out-of-band writes (the daemon-staleness fix)."""
from __future__ import annotations

import time

from pmb.core.engine import Engine


def test_warm_engine_picks_up_out_of_band_writes(tmp_pmb_home, tmp_workspace_dir):
    """Regression (bug 3): a long-running engine (the warm daemon holds one in
    memory) must see facts written out-of-band by ANOTHER process - e.g. a CLI
    `pmb fact` / `index` / `track` while the daemon is warm - instead of serving
    a stale in-RAM search index until it restarts."""
    daemon = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    daemon.remember("seed", "alpha baseline content")
    # Warm the search index (first search loads it and records the cache mtime).
    daemon.recall("alpha", top_k=3)

    # A SEPARATE engine (the CLI process) writes a fact to the SAME workspace.
    cli = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    time.sleep(0.05)  # ensure the on-disk cache mtime strictly advances
    cli.remember("q", "zzdeployregion canonical deploy region is eu-central-7")

    # The warm engine must now find it WITHOUT a restart.
    res = daemon.recall("zzdeployregion canonical deploy region", top_k=5)
    assert any("eu-central-7" in r.content for r in res.results), (
        "warm engine served a stale index and missed the out-of-band write"
    )
