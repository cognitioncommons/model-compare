"""Terminal display with rich."""

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .diff import ComparisonSummary, DiffResult, compare_all_outputs
from .metrics import AggregateMetrics, calculate_aggregate_metrics, rank_models
from .runner import ModelResult

console = Console()


def display_result(result: ModelResult) -> None:
    """Display a single model result."""
    if result.error:
        panel = Panel(
            f"[red]Error: {result.error}[/red]",
            title=f"[bold red]{result.model}[/bold red]",
            border_style="red",
        )
    else:
        content = result.output
        # Detect if output contains code
        if "```" in content:
            panel = Panel(
                content,
                title=f"[bold green]{result.model}[/bold green]",
                subtitle=f"[dim]{result.latency_ms:.0f}ms | {result.output_tokens} tokens | ${result.cost_usd:.6f}[/dim]",
                border_style="green",
            )
        else:
            panel = Panel(
                content,
                title=f"[bold cyan]{result.model}[/bold cyan]",
                subtitle=f"[dim]{result.latency_ms:.0f}ms | {result.output_tokens} tokens | ${result.cost_usd:.6f}[/dim]",
                border_style="cyan",
            )
    console.print(panel)


def display_results_side_by_side(results: list[ModelResult]) -> None:
    """Display multiple results side by side."""
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    # Create a table with one column per model
    table = Table(show_header=True, header_style="bold magenta", expand=True)

    for result in results:
        status = "[red]ERROR[/red]" if result.error else "[green]OK[/green]"
        table.add_column(f"{result.model}\n{status}", overflow="fold")

    # Add outputs as a single row
    row = []
    for result in results:
        if result.error:
            row.append(f"[red]{result.error}[/red]")
        else:
            # Truncate long outputs
            output = result.output
            if len(output) > 1000:
                output = output[:1000] + "\n[dim]...(truncated)[/dim]"
            row.append(output)

    table.add_row(*row)
    console.print(table)


def display_metrics_table(results: list[ModelResult]) -> None:
    """Display a metrics comparison table."""
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    table = Table(title="Model Comparison Metrics", show_header=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Latency", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Tokens/sec", justify="right")
    table.add_column("Status", justify="center")

    # Find best values for highlighting
    min_latency = min(r.latency_ms for r in results)
    min_cost = min(r.cost_usd for r in results)
    max_words = max(len(r.output.split()) for r in results)

    for result in results:
        latency_str = f"{result.latency_ms:.0f}ms"
        if result.latency_ms == min_latency:
            latency_str = f"[bold green]{latency_str}[/bold green]"

        words = len(result.output.split())
        words_str = str(words)
        if words == max_words:
            words_str = f"[bold green]{words_str}[/bold green]"

        cost_str = f"${result.cost_usd:.6f}"
        if result.cost_usd == min_cost:
            cost_str = f"[bold green]{cost_str}[/bold green]"

        latency_sec = result.latency_ms / 1000.0
        tokens_per_sec = result.output_tokens / latency_sec if latency_sec > 0 else 0

        status = "[red]ERROR[/red]" if result.error else "[green]OK[/green]"

        table.add_row(
            result.model,
            latency_str,
            words_str,
            str(result.output_tokens),
            cost_str,
            f"{tokens_per_sec:.1f}",
            status,
        )

    console.print(table)


def display_aggregate_metrics(metrics: AggregateMetrics) -> None:
    """Display aggregate metrics summary."""
    table = Table(title="Aggregate Metrics", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Models Compared", str(len(metrics.models)))
    table.add_row("", "")
    table.add_row("[bold]Latency[/bold]", "")
    table.add_row("  Fastest", f"{metrics.fastest_model} ({metrics.latency_min:.0f}ms)")
    table.add_row("  Slowest", f"{metrics.slowest_model} ({metrics.latency_max:.0f}ms)")
    table.add_row("  Mean", f"{metrics.latency_mean:.0f}ms")
    table.add_row("  Median", f"{metrics.latency_median:.0f}ms")
    table.add_row("", "")
    table.add_row("[bold]Cost[/bold]", "")
    table.add_row("  Total", f"${metrics.total_cost:.6f}")
    table.add_row(
        "  Cheapest", f"{metrics.cheapest_model} (${metrics.cost_min:.6f})"
    )
    table.add_row(
        "  Most Expensive",
        f"{metrics.most_expensive_model} (${metrics.cost_max:.6f})",
    )
    table.add_row("", "")
    table.add_row("[bold]Output[/bold]", "")
    table.add_row("  Shortest", metrics.shortest_output_model)
    table.add_row("  Longest", metrics.longest_output_model)
    table.add_row("  Avg Words", f"{metrics.word_count_mean:.0f}")
    table.add_row("", "")
    table.add_row("[bold]Tokens[/bold]", "")
    table.add_row("  Total Used", str(metrics.total_tokens_used))
    table.add_row(
        "  Per Dollar",
        f"{metrics.tokens_per_dollar:.0f}"
        if metrics.tokens_per_dollar != float("inf")
        else "N/A",
    )
    table.add_row("", "")
    table.add_row("[bold]Quality[/bold]", "")
    table.add_row("  Success Rate", f"{metrics.success_rate:.0%}")
    table.add_row("  Errors", str(metrics.error_count))

    console.print(table)


def display_diff(diff_result: DiffResult) -> None:
    """Display diff between two model outputs."""
    console.print(
        f"\n[bold]Comparing {diff_result.model_a} vs {diff_result.model_b}[/bold]"
    )
    console.print(f"Similarity: [cyan]{diff_result.similarity_ratio:.1%}[/cyan]")
    console.print(f"Common lines: {diff_result.common_lines}")
    console.print(f"Different lines: {diff_result.different_lines}")

    if diff_result.unified_diff:
        console.print("\n[bold]Unified Diff:[/bold]")
        syntax = Syntax(diff_result.unified_diff, "diff", theme="monokai")
        console.print(syntax)


def display_comparison_summary(summary: ComparisonSummary) -> None:
    """Display summary of comparing all model outputs."""
    console.print("\n[bold]Comparison Summary[/bold]")
    console.print(f"Models: {', '.join(summary.models)}")
    console.print(f"Average Similarity: [cyan]{summary.average_similarity:.1%}[/cyan]")

    if summary.most_similar_pair:
        console.print(f"Most Similar: {summary.most_similar_pair}")
    if summary.least_similar_pair:
        console.print(f"Least Similar: {summary.least_similar_pair}")

    if summary.pairwise_similarities:
        console.print("\n[bold]Pairwise Similarities:[/bold]")
        table = Table(show_header=True)
        table.add_column("Model A")
        table.add_column("Model B")
        table.add_column("Similarity", justify="right")

        for (model_a, model_b), sim in sorted(
            summary.pairwise_similarities.items(), key=lambda x: x[1], reverse=True
        ):
            color = "green" if sim > 0.8 else "yellow" if sim > 0.5 else "red"
            table.add_row(model_a, model_b, f"[{color}]{sim:.1%}[/{color}]")

        console.print(table)

    if summary.consensus_lines:
        console.print(f"\n[bold]Consensus Lines ({len(summary.consensus_lines)}):[/bold]")
        for line in summary.consensus_lines[:10]:
            console.print(f"  [dim]{line}[/dim]")
        if len(summary.consensus_lines) > 10:
            console.print(f"  [dim]...and {len(summary.consensus_lines) - 10} more[/dim]")


def display_rankings(results: list[ModelResult]) -> None:
    """Display model rankings."""
    rankings = rank_models(results)

    table = Table(title="Model Rankings", show_header=True)
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Score", justify="right")

    for i, (model, score) in enumerate(rankings, 1):
        medal = ""
        if i == 1:
            medal = "[gold1]1st[/gold1]"
        elif i == 2:
            medal = "[silver]2nd[/silver]"
        elif i == 3:
            medal = "[orange3]3rd[/orange3]"
        else:
            medal = f"{i}th"

        table.add_row(medal, model, f"{score:.3f}")

    console.print(table)


def display_full_comparison(results: list[ModelResult]) -> None:
    """Display comprehensive comparison of all results."""
    console.print("\n" + "=" * 60)
    console.print("[bold]MODEL COMPARISON RESULTS[/bold]")
    console.print("=" * 60 + "\n")

    # Show individual outputs
    console.print("[bold]Individual Outputs:[/bold]\n")
    for result in results:
        display_result(result)
        console.print()

    # Show metrics table
    display_metrics_table(results)
    console.print()

    # Show aggregate metrics
    metrics = calculate_aggregate_metrics(results)
    display_aggregate_metrics(metrics)
    console.print()

    # Show comparison summary
    summary = compare_all_outputs(results)
    display_comparison_summary(summary)
    console.print()

    # Show rankings
    display_rankings(results)


def display_batch_summary(
    batch_results: dict[str, list[ModelResult]],
) -> None:
    """Display summary of batch run."""
    console.print("\n[bold]BATCH SUMMARY[/bold]\n")

    table = Table(show_header=True)
    table.add_column("Prompt", style="cyan", max_width=40)
    table.add_column("Models", justify="center")
    table.add_column("Errors", justify="center")
    table.add_column("Total Cost", justify="right")
    table.add_column("Avg Latency", justify="right")

    total_cost = 0.0
    total_prompts = 0
    total_errors = 0

    for prompt, results in batch_results.items():
        # Truncate prompt for display
        prompt_display = prompt[:40] + "..." if len(prompt) > 40 else prompt

        errors = sum(1 for r in results if r.error)
        cost = sum(r.cost_usd for r in results)
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

        total_cost += cost
        total_prompts += 1
        total_errors += errors

        error_str = f"[red]{errors}[/red]" if errors > 0 else "[green]0[/green]"

        table.add_row(
            prompt_display,
            str(len(results)),
            error_str,
            f"${cost:.6f}",
            f"{avg_latency:.0f}ms",
        )

    console.print(table)
    console.print()
    console.print(f"[bold]Total Prompts:[/bold] {total_prompts}")
    console.print(f"[bold]Total Cost:[/bold] ${total_cost:.6f}")
    console.print(f"[bold]Total Errors:[/bold] {total_errors}")
