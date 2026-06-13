"""PMB MCP package.

S1 (perf): this package init is LAZY. Importing a light submodule such as
`pmb.mcp.registry` (stdlib-only, by contract) or `pmb.mcp.daemon` must NOT drag
in `pmb.mcp.server` - which pulls fastmcp + the mcp SDK + pydantic + the whole
engine (measured 3-6 s on first import). The hook client reaches the daemon via
`pmb.mcp.registry` on EVERY user message, so that eager import was the single
biggest latency cost in the system.

`build_server` / `main` stay importable as `pmb.mcp.build_server` etc. via PEP
562 - resolved only the first time the attribute is accessed (i.e. when someone
actually wants to BUILD a server), mirroring `pmb/__init__.py`.
"""
from __future__ import annotations

__all__ = ["build_server", "main"]


_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # attribute name -> (module, attr in module)
    "build_server": ("pmb.mcp.server", "build_server"),
    "main":         ("pmb.mcp.server", "main"),
}


def __getattr__(name: str):
    """PEP 562 lazy attribute access - import the heavy server on demand."""
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module 'pmb.mcp' has no attribute {name!r}")
    module_name, attr = target
    import importlib
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value  # cache: subsequent accesses skip __getattr__
    return value
