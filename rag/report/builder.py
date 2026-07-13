"""Build and render benchmark reports."""

from __future__ import annotations

import json
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_retrieval_metrics(metrics: dict) -> None:
    """Print a rich table of retrieval benchmark metrics."""
    table = Table(title="Retrieval Benchmark", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold")

    def color(val: float, good: float = 0.7) -> str:
        c = "green" if val >= good else "yellow" if val >= 0.4 else "red"
        return f"[{c}]{val:.4f}[/{c}]"

    table.add_row("Questions evaluated", str(metrics.get("n_questions", 0)))
    table.add_row("Hit@1", color(metrics.get("hit_at_1", 0)))
    table.add_row("Hit@3", color(metrics.get("hit_at_3", 0)))
    table.add_row("Hit@5", color(metrics.get("hit_at_5", 0)))
    table.add_row("MRR", color(metrics.get("mrr", 0)))
    console.print(table)


def print_generation_metrics(metrics: dict) -> None:
    """Print a rich table of generation benchmark metrics."""
    table = Table(
        title=f"Generation Benchmark — model: {metrics.get('model', 'unknown')}", box=box.ROUNDED
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold")

    def color(val: float, good: float = 0.6) -> str:
        c = "green" if val >= good else "yellow" if val >= 0.3 else "red"
        return f"[{c}]{val:.4f}[/{c}]"

    table.add_row("Questions evaluated", str(metrics.get("n_questions", 0)))
    table.add_row("Errors", str(metrics.get("errors", 0)))
    table.add_row("Avg faithfulness score", color(metrics.get("avg_faithfulness_score", 0)))
    table.add_row("Min faithfulness score", color(metrics.get("min_faithfulness_score", 0)))
    table.add_row("Max faithfulness score", color(metrics.get("max_faithfulness_score", 0)))
    if metrics.get("n_unanswerable"):
        crr = metrics.get("correct_refusal_rate", 0)
        table.add_row(
            f"Correct-refusal rate ({metrics['n_unanswerable']} unanswerable)", color(crr)
        )
    if metrics.get("n_answerable"):
        frr = metrics.get("false_refusal_rate", 0)
        # false refusals: lower is better, so invert the color scale
        c = "green" if frr <= 0.05 else "yellow" if frr <= 0.2 else "red"
        table.add_row("False-refusal rate (answerable)", f"[{c}]{frr:.4f}[/{c}]")
    console.print(table)


def print_judge_metrics(metrics: dict) -> None:
    """Print a rich table of LLM-as-judge metrics."""
    table = Table(
        title=(
            f"LLM-as-Judge — judge: {metrics.get('judge_model', 'unknown')} "
            f"(rubric {metrics.get('prompt_version', '?')})"
        ),
        box=box.ROUNDED,
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold")

    def color(val: float, good: float = 0.7) -> str:
        c = "green" if val >= good else "yellow" if val >= 0.4 else "red"
        return f"[{c}]{val:.4f}[/{c}]"

    table.add_row("Answers judged", str(metrics.get("n_judged", 0)))
    table.add_row("Judge errors", str(metrics.get("errors", 0)))
    table.add_row("Correct", color(metrics.get("correct_rate", 0)))
    table.add_row("Partial", f"{metrics.get('partial_rate', 0):.4f}")
    table.add_row("Incorrect / hallucinated", f"{metrics.get('incorrect_rate', 0):.4f}")
    table.add_row("Grounded", color(metrics.get("grounded_rate", 0)))
    console.print(table)
    console.print(
        "[dim]Judge scores are only as good as their agreement with a human — "
        "run `rag-eval agreement` on a filled-in grade sheet before quoting these.[/dim]"
    )


def print_agreement(stats: dict) -> None:
    """Print human/judge agreement study results."""
    n = stats.get("n_graded", 0)
    agreement = stats.get("agreement", 0)
    kappa = stats.get("cohens_kappa", 0)
    c_a = "green" if agreement >= 0.8 else "yellow" if agreement >= 0.6 else "red"
    c_k = "green" if kappa >= 0.6 else "yellow" if kappa >= 0.4 else "red"
    lines = [
        f"[yellow]Hand-graded answers:[/yellow]  {n}",
        f"[yellow]Raw agreement:      [/yellow]  [{c_a}]{agreement:.1%}[/{c_a}]",
        f"[yellow]Cohen's kappa:      [/yellow]  [{c_k}]{kappa:.3f}[/{c_k}]",
    ]
    console.print(
        Panel("\n".join(lines), title="[bold]Human/Judge Agreement[/bold]", border_style="blue")
    )
    disagreements = stats.get("disagreements", [])
    if disagreements:
        table = Table(title="Disagreements", box=box.SIMPLE)
        table.add_column("qa_id", style="cyan")
        table.add_column("Judge said")
        table.add_column("Human said")
        for d in disagreements:
            table.add_row(d["qa_id"], d["judge"], d["human"])
        console.print(table)


def print_summary(retrieval: dict, generation: dict | None = None) -> None:
    """Print a summary panel."""
    lines = []
    if retrieval:
        h5 = retrieval.get("hit_at_5", 0)
        mrr = retrieval.get("mrr", 0)
        c_h5 = "green" if h5 >= 0.7 else "yellow" if h5 >= 0.4 else "red"
        c_mrr = "green" if mrr >= 0.7 else "yellow" if mrr >= 0.4 else "red"
        lines.append(f"[yellow]Hit@5:[/yellow]  [{c_h5}]{h5:.1%}[/{c_h5}]")
        lines.append(f"[yellow]MRR:  [/yellow]  [{c_mrr}]{mrr:.1%}[/{c_mrr}]")
    if generation:
        score = generation.get("avg_faithfulness_score", 0)
        c = "green" if score >= 0.6 else "yellow" if score >= 0.3 else "red"
        lines.append(f"[yellow]Faithfulness:[/yellow]  [{c}]{score:.1%}[/{c}]")
    console.print(
        Panel("\n".join(lines), title="[bold]Benchmark Summary[/bold]", border_style="blue")
    )


def save_report(report: dict, path: Path) -> None:
    """Write report as formatted JSON."""
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    console.print(f"[green]✓ Report saved → {path}[/green]")
