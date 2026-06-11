"""
OS scheduler config helper.

Generates a config file/command for:
- Linux/macOS: cron entries
- Windows: schtasks commands

Examples:
- daily decay at 04:00
- weekly self-test on Sunday at 05:00
- weekly compact on Sunday at 06:00
"""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path


def _pmb_command() -> str:
    """Resolve the pmb executable. Prefer the venv-internal one we're running
    inside, since shutil.which("pmb") only succeeds when the venv is on PATH —
    and scheduled tasks don't have the user's interactive PATH."""
    py = Path(sys.executable)
    for candidate in (py.parent / "pmb.exe", py.parent / "pmb"):
        if candidate.exists():
            return str(candidate)
    found = shutil.which("pmb")
    if found:
        return found
    # Fallback — `python -m pmb.cli` works because pmb/cli/__main__.py exists
    return f"{sys.executable} -m pmb.cli"


def generate_scheduler_config() -> dict:
    """
    Returns config snippets for the current OS.

    {"os": "linux"|"darwin"|"windows", "cron"|"crontab"|"schtasks": [...]}
    """
    pmb = _pmb_command()
    system = platform.system().lower()

    if system in ("linux", "darwin"):
        return _unix_config(pmb, system)
    elif system == "windows":
        return _windows_config(pmb)
    else:
        return {"os": system, "supported": False}


def _unix_config(pmb: str, system: str) -> dict:
    crontab = [
        "# PMB maintenance — auto-generated",
        f"0 4 * * *   {pmb} decay >> ~/.pmb/cron.log 2>&1     # daily decay 4 AM",
        f"0 5 * * 0   {pmb} health run >> ~/.pmb/cron.log 2>&1  # weekly self-test Sun 5 AM",
        f"0 6 * * 0   {pmb} compact >> ~/.pmb/cron.log 2>&1   # weekly compaction Sun 6 AM",
    ]
    install_steps = [
        "# 1. Open crontab editor:",
        "crontab -e",
        "",
        "# 2. Add lines below:",
        *crontab,
        "",
        "# 3. Save and verify:",
        "crontab -l",
    ]
    return {
        "os": system,
        "supported": True,
        "type": "cron",
        "crontab_lines": crontab,
        "install_steps": install_steps,
    }


def _windows_config(pmb: str) -> dict:
    # schtasks /TR is itself wrapped in double quotes; inner double quotes
    # inside the command must be escaped as \" — and a path with spaces still
    # needs them. So we build the inner string with escaped quotes around the
    # executable path.
    pmb_quoted = f'\\"{pmb}\\"' if " " in pmb else pmb
    cmds = [
        f'schtasks /Create /SC DAILY /ST 04:00 /TN "PMB Daily Decay" /TR "{pmb_quoted} decay" /F',
        f'schtasks /Create /SC WEEKLY /D SUN /ST 05:00 /TN "PMB Weekly Health" /TR "{pmb_quoted} health run" /F',
        f'schtasks /Create /SC WEEKLY /D SUN /ST 06:00 /TN "PMB Weekly Compact" /TR "{pmb_quoted} compact" /F',
    ]
    install_steps = [
        "# Run in PowerShell as admin or as your current user.",
        "# Tasks run as the current user; ensure the venv path is reachable.",
        *cmds,
        "",
        "# Verify:",
        'schtasks /Query /TN "PMB Daily Decay"',
        "",
        "# Remove later:",
        'schtasks /Delete /TN "PMB Daily Decay" /F',
        'schtasks /Delete /TN "PMB Weekly Health" /F',
        'schtasks /Delete /TN "PMB Weekly Compact" /F',
    ]
    return {
        "os": "windows",
        "supported": True,
        "type": "schtasks",
        "commands": cmds,
        "install_steps": install_steps,
    }
