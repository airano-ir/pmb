"""`python -m pmb.agent_wrapper` - start a chat session backed by PMB memory."""
from __future__ import annotations

import argparse

from pmb.agent_wrapper.loop import AgentConfig, AgentLoop


def main() -> None:
    p = argparse.ArgumentParser(prog="pmb-chat")
    p.add_argument("--model", default="haiku",
                   help="Model alias: haiku | sonnet | opus, or full ID")
    p.add_argument("--transport", default="auto",
                   choices=("auto", "claude", "anthropic", "ollama"),
                   help="Which way to call the model. `claude` = CLI subprocess (no key). `anthropic` = ANTHROPIC_API_KEY. `ollama` = local or remote Ollama (set PMB_OLLAMA_URL). `auto` picks claude → anthropic → ollama in that order.")
    p.add_argument("--ollama-url", default=None,
                   help="Override Ollama base URL (else PMB_OLLAMA_URL / OLLAMA_HOST / http://localhost:11434)")
    p.add_argument("--window", type=int, default=200_000)
    p.add_argument("--target-max", type=float, default=0.75)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-persist", action="store_true",
                   help="Do not write turns to PMB memory")
    p.add_argument("--no-selective", action="store_true",
                   help="Disable selective compression (use baseline drop-oldest)")
    p.add_argument("--consolidate-on-exit", action="store_true",
                   help="Run pmb consolidate after the session ends")
    args = p.parse_args()

    cfg = AgentConfig(
        model=args.model,
        transport=args.transport,
        ollama_url=args.ollama_url,
        window=args.window,
        target_max=args.target_max,
        recall_top_k=args.top_k,
        persist_turns=not args.no_persist,
        consolidate_on_exit=args.consolidate_on_exit,
        selective_compression=not args.no_selective,
    )
    AgentLoop(config=cfg).repl()


if __name__ == "__main__":
    main()
