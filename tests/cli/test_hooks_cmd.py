"""Hook commands must be cross-shell safe (PowerShell / cmd / bash)."""
from __future__ import annotations

from pmb.cli.hooks import _claude_hook_specs, _pmb_hook_entry


def test_hook_entry_is_space_free():
    # A space in the invocation would force quoting, which PowerShell cannot
    # parse without the & call operator.
    assert " " not in _pmb_hook_entry()


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
