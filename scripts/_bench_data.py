from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(
    os.environ.get("PMB_BENCH_DATA") or Path(__file__).resolve().parent / "data"
)


def data_path(name: str) -> str:
    """Absolute path to ``name`` inside the benchmark data dir (created lazily)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(DATA_DIR / name)


# Convenience constant — the one input dataset every benchmark reads.
LOCOMO_JSON = data_path("locomo10.json")


def download_locomo() -> str:
    """Fetch the public LoCoMo dataset (Snap Research, arXiv 2402.17753).

    Pulled from the public HF mirror ``Percena/locomo-mc10`` (file
    ``raw/locomo10.json``). No-op if it is already present.
    """
    import shutil

    dst = Path(data_path("locomo10.json"))
    if dst.exists():
        print(f"locomo10.json already present: {dst} ({dst.stat().st_size} bytes)")
        return str(dst)

    from huggingface_hub import hf_hub_download

    tmp = DATA_DIR / "_hf_tmp"
    src = hf_hub_download(
        "Percena/locomo-mc10",
        "raw/locomo10.json",
        repo_type="dataset",
        local_dir=str(tmp),
    )
    shutil.copy(src, dst)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Downloaded locomo10.json → {dst} ({dst.stat().st_size} bytes)")
    return str(dst)


if __name__ == "__main__":
    download_locomo()
