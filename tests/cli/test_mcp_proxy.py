"""`pmb mcp proxy` - the lightweight stdio<->daemon bridge codex uses so it
shares the ONE warm daemon instead of loading its own Engine+model.

The happy path (a live daemon present) is an integration concern - here we lock
the two no-daemon branches deterministically (no model, no real daemon, no
network) and the entry shape `pmb connect codex --daemon` writes.
"""
from __future__ import annotations

import pytest
import typer


def test_make_proxy_entry_shape(tmp_path):
    from pmb.cli.connect import make_proxy_entry
    e = make_proxy_entry(tmp_path, workspace_id="team", pmb_home=tmp_path / "h")
    assert "command" in e and e["command"]            # stdio-shaped
    assert e["args"][-2:] == ["mcp", "proxy"]          # the bridge, not pmb-mcp
    assert e["env"]["PMB_CWD"] == str(tmp_path)
    assert e["env"]["PMB_WORKSPACE"] == "team"
    assert e["env"]["PMB_HOME"].endswith("h")


def test_proxy_no_daemon_no_fallback_exits(monkeypatch):
    # No daemon, --no-autostart, --no-fallback → exit 1 (don't silently load a
    # heavy in-process model when the caller asked for bridge-only).
    import pmb.mcp.registry as reg
    monkeypatch.setattr(reg, "find_live_daemon", lambda *a, **k: None)
    from pmb.cli.commands.hooks import mcp_proxy_cmd
    with pytest.raises(typer.Exit):
        mcp_proxy_cmd(host="127.0.0.1", port=8765, path="/mcp",
                      autostart=False, fallback=False)


def test_proxy_falls_back_to_in_process_server(monkeypatch):
    # No daemon, --no-autostart, fallback ON → run the in-process stdio server so
    # the host still gets memory.
    import pmb.mcp.registry as reg
    monkeypatch.setattr(reg, "find_live_daemon", lambda *a, **k: None)
    import pmb.mcp.server as srv
    called = {}
    monkeypatch.setattr(srv, "main", lambda: called.setdefault("ran", True))
    from pmb.cli.commands.hooks import mcp_proxy_cmd
    mcp_proxy_cmd(host="127.0.0.1", port=8765, path="/mcp",
                  autostart=False, fallback=True)
    assert called.get("ran") is True
