"""Phase 2 / issue #7 (additive slice): recall_scoped filters the one user
memory by project, WITHOUT changing default recall behaviour or touching the
recall hot-path/cache."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.core.engine import Engine


@pytest.fixture
def tmp_pmb_home():
    import gc
    import shutil
    import time as _t
    tmp = tempfile.mkdtemp()
    home = Path(tmp) / "pmb_home"
    os.environ["PMB_HOME"] = str(home)
    try:
        yield home
    finally:
        os.environ.pop("PMB_HOME", None)
        gc.collect()
        for _ in range(3):
            try:
                shutil.rmtree(tmp, ignore_errors=False)
                break
            except (OSError, PermissionError):
                _t.sleep(0.2)
                gc.collect()
        else:
            shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def tmp_workspace_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def _engine(ws, home, **over):
    cfg = {"recall.cache_size": 0}
    cfg.update(over)
    return Engine(cwd=ws, pmb_home=home, config_overrides=cfg)


def test_recall_scoped_filters_by_project(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    # Both facts share the query terms so both surface in the wide recall;
    # the project filter must then keep only alpha.
    eng.record_fact("alpha service uses Postgres database",
                    metadata={"project_name": "alpha"})
    eng.record_fact("beta service uses Postgres database",
                    metadata={"project_name": "beta"})

    # sanity: unscoped recall surfaces both
    both = eng.recall("service uses Postgres database", top_k=10)
    assert len(both.results) >= 2, [r.content for r in both.results]

    res = eng.recall_scoped("service uses Postgres database",
                            project="alpha", top_k=5)
    contents = " ".join(r.content for r in res.results)
    assert "alpha service" in contents      # alpha fact kept
    assert "beta service" not in contents   # beta fact filtered out
    assert res.escalation["stopped"] == "project_filtered"
    assert res.escalation["project"] == "alpha"


def test_recall_scoped_no_project_equals_plain_recall(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact("decided to use Postgres for storage",
                    metadata={"project_name": "alpha"})
    r_scoped = eng.recall_scoped("Postgres", top_k=5)   # project None → plain
    r_plain = eng.recall("Postgres", top_k=5)
    assert [x.ulid for x in r_scoped.results] == [x.ulid for x in r_plain.results]


def test_recall_scoped_unknown_project_returns_empty(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact("decided to use Postgres", metadata={"project_name": "alpha"})
    res = eng.recall_scoped("Postgres", project="nonexistent", top_k=5)
    assert res.results == []
    assert res.escalation["n_after_filter"] == 0
