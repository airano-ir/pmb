"""Hook commands must be cross-shell safe (PowerShell / cmd / bash)."""
from __future__ import annotations

from pmb.cli.hooks import _claude_hook_specs, _pmb_hook_entry


def test_hook_entry_executable_token_is_space_free():
    # The command runs UNQUOTED through PowerShell/cmd/bash, so its FIRST token
    # (the executable) must have no spaces - a space there would force quoting,
    # which PowerShell cannot parse without the & call operator. (The entry as a
    # whole may now contain spaces: `<python> -m pmb.hookclient`.)
    entry = _pmb_hook_entry()
    first = entry.split(" ", 1)[0]
    assert " " not in first
    assert not first.startswith('"')


def test_hook_entry_prefers_signed_python_over_unsigned_shim(monkeypatch):
    """Windows Smart App Control blocks the UNSIGNED pip `pmb-hook.exe`, which
    silently kills every hook. Prefer the SIGNED interpreter via
    `<python> -m pmb.hookclient` when its path is space-free; only fall back to
    the bare console-script when python's own path has spaces."""
    import pmb.cli.hooks as H
    monkeypatch.setattr(H.sys, "executable", r"C:\Py\Python312\python.exe")
    assert H._pmb_hook_entry() == r"C:\Py\Python312\python.exe -m pmb.hookclient"
    # a python path WITH spaces can't be used unquoted cross-shell → fall back
    monkeypatch.setattr(H.sys, "executable", r"C:\Program Files\Py\python.exe")
    assert H._pmb_hook_entry() == "pmb-hook"


def test_hook_commands_have_no_leading_quote():
    """Regression: headless `claude -p` on Windows runs hooks through
    PowerShell, which ParserErrors on a quoted command (`"pmb-hook" ...`)
    that has no `&` call operator - this silently broke SessionStart / Stop /
    auto-recall hooks in headless agents. The command must be unquoted (a bare
    pmb-hook on PATH, or a space-free absolute path)."""
    h = _pmb_hook_entry()
    specs = _claude_hook_specs()
    assert specs
    for spec in specs:
        cmd = spec["command"]
        assert not cmd.startswith('"'), f"PowerShell-unsafe (leading quote): {cmd}"
        assert cmd.startswith(h + " "), cmd
        assert "--quiet" in cmd
