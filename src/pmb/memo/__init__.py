"""Exploration memo cache - incremental cognition for the agent.

Memoize the conclusions an agent reaches after expensive codebase exploration,
keyed to the content hashes of the files it relied on, and replay them in a
future session with a freshness check - so the agent reuses a conclusion
instead of re-deriving it, and re-reads only the files that actually changed.
"""
