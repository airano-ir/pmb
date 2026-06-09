"""`pmb health ...` — self-test, trends, conflicts, user-feedback summary."""

from __future__ import annotations

import typer
from rich.panel import Panel
from rich.table import Table

from pmb.cli._common import console
from pmb.core.engine import Engine

health_app = typer.Typer(help="Health checks: self-test, trends, conflicts, user feedback.")


@health_app.command("feedback")
def health_feedback():
    """Show summary of recorded recall feedback (the real-user metric)."""
    eng = Engine()
    s = eng.feedback_summary()
    if s["verdict"] == "no_data":
        console.print("[yellow]No feedback recorded yet. Use `pmb feedback ULID useful|wrong|irrelevant`.[/]")
        return

    color = {"healthy": "green", "mixed": "yellow", "poor": "red"}[s["verdict"]]
    rate = s["useful_rate"]
    rate_str = f"{rate:.1%}" if rate is not None else "-"

    console.print(Panel.fit(
        f"Verdict: [{color}]{s['verdict']}[/]\n"
        f"Total feedback entries: {s['total']}\n"
        f"  useful: [green]{s['useful']}[/]\n"
        f"  wrong: [yellow]{s['wrong']}[/]\n"
        f"  irrelevant: [red]{s['irrelevant']}[/]\n"
        f"useful_rate: [{color}]{rate_str}[/]\n"
        f"unique queries: {s['n_unique_queries']}",
        title="User Recall Feedback",
    ))

    if s["most_wrong_events"]:
        t = Table(show_header=True, header_style="bold magenta", title="Most-flagged-wrong events")
        t.add_column("ULID")
        t.add_column("Wrong count", justify="right")
        for ulid, n in s["most_wrong_events"]:
            t.add_row(ulid, str(n))
        console.print(t)


@health_app.command("run")
def health_run(
    n: int = typer.Option(20, "-n", "--n-samples"),
    min_age_days: float = typer.Option(1.0, "--min-age", help="Skip events younger than this"),
    no_adaptive: bool = typer.Option(False, "--no-adaptive",
                                     help="Skip adaptive boost"),
):
    """Run a self-test: the system quizzes itself with questions from old memory."""
    eng = Engine()
    result = eng.run_self_test(
        n_samples=n, min_age_days=min_age_days,
        apply_adaptive=not no_adaptive,
    )
    st = result["self_test"]

    if st["n_tested"] == 0:
        if st.get("empty_reason") == "all_events_younger_than_min_age":
            console.print(
                f"[yellow]Skipped self-test:[/] "
                f"all {st['n_too_recent']} content events are younger than "
                f"{st['eligible_min_age_days']} days. "
                f"Lower with `--min-age 0` for a fresh workspace."
            )
        else:
            console.print(
                f"[yellow]Skipped self-test:[/] "
                f"no qa/fact/git events in workspace ({st['n_total_active']} total)."
            )
        return

    table = Table(show_header=True, header_style="bold magenta", title="Self-Test Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Tested", str(st["n_tested"]))
    table.add_row("Total active in workspace", str(st["n_total_active"]))
    table.add_row("accuracy@1", f"[green]{st['accuracy_at_1']:.1%}[/]")
    table.add_row("accuracy@3", f"[green]{st['accuracy_at_3']:.1%}[/]")
    table.add_row("accuracy@5", f"[green]{st['accuracy_at_5']:.1%}[/]")
    if st["avg_rank"] is not None:
        table.add_row("avg rank when found", f"{st['avg_rank']:.2f}")
    console.print(table)

    # Session-aware loose metric (closer to real-world recall)
    if st.get("session_accuracy_at_5") is not None:
        console.print(
            f"\n[dim]session-aware acc@5 (any same-session hit counts):[/] "
            f"[cyan]{st['session_accuracy_at_5']:.1%}[/] "
            f"(coverage {st.get('session_coverage', 0):.0%})"
        )

    if result.get("adaptive"):
        a = result["adaptive"]
        console.print(
            f"[cyan]Adaptive boost:[/] {a['n_boosted']} events boosted "
            f"({a['n_superboosted']} super-boosted after repeated failures)"
        )

    if result.get("feedback_adaptive"):
        fa = result["feedback_adaptive"]
        if fa["n_feedback_entries"] > 0:
            console.print(
                f"[cyan]User feedback ({fa['n_feedback_entries']} entries):[/] "
                f"promoted {fa['n_promoted_useful']} useful + "
                f"{fa['n_promoted_expected']} expected, "
                f"demoted {fa['n_demoted_wrong']} wrong"
            )

    if st["failed_queries"]:
        console.print(f"\n[yellow]Failed queries (saved {len(st['failed_queries'])}):[/]")
        for f in st["failed_queries"][:3]:
            console.print(f"  Q: '{f['query']}'")
            console.print(f"     expected: {f['expected_content_preview'][:80]}")


@health_app.command("trend")
def health_trend():
    """Show the self-test accuracy trend over time."""
    eng = Engine()
    t = eng.health_trend()
    if t["verdict"] == "insufficient":
        console.print(f"[yellow]Need more self-test runs (have {t['n_runs']}).[/]")
        return

    color = {"stable": "green", "degrading": "red", "improving": "cyan"}[t["verdict"]]
    console.print(Panel.fit(
        f"Verdict: [{color}]{t['verdict']}[/]\n"
        f"Runs: {t['n_runs']}\n"
        f"First acc@5: {t['first_run_acc5']:.1%}\n"
        f"Last acc@5:  {t['last_run_acc5']:.1%}\n"
        f"Δ: {t['delta_pp']:+.1f}pp",
        title="Health Trend",
    ))


@health_app.command("conflicts")
def health_conflicts(
    resolve: bool = typer.Option(False, "--resolve",
                                  help="Auto-archive obvious supersede conflicts"),
):
    """Find contradictions between facts recorded at different times."""
    eng = Engine()
    conflicts = eng.detect_conflicts()
    if not conflicts:
        console.print("[green]No conflicts detected.[/]")
        return

    table = Table(show_header=True, header_style="bold magenta", title="Conflicts")
    table.add_column("Key")
    table.add_column("Older")
    table.add_column("Newer")
    table.add_column("Resolution")
    table.add_column("Confidence")
    for c in conflicts[:20]:
        table.add_row(
            c["key"],
            c["older_value"][:40],
            c["newer_value"][:40],
            c["suggested_resolution"],
            f"{c['confidence']:.2f}",
        )
    console.print(table)

    if resolve:
        action = eng.auto_resolve_conflicts(dry_run=False)
        console.print(
            f"\n[yellow]Auto-resolved:[/] {action['n_archived']} conflicts archived "
            f"({action['n_supersede_candidates']} candidates total)"
        )
