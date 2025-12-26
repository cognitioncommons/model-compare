"""Click CLI for model-compare."""

import json
import os
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from . import __version__
from .diff import compare_all_outputs, diff_outputs, semantic_diff
from .display import (
    console,
    display_batch_summary,
    display_comparison_summary,
    display_diff,
    display_full_comparison,
    display_metrics_table,
    display_rankings,
    display_result,
)
from .metrics import (
    calculate_aggregate_metrics,
    compare_models_table,
    cost_breakdown,
)
from .runner import ModelResult, run_batch, run_prompt


@click.group()
@click.version_option(version=__version__)
def cli():
    """Model Compare - Side-by-side comparison of LLM model outputs."""
    pass


@cli.command()
@click.argument("prompt_file", type=click.Path(exists=True))
@click.option(
    "--models",
    "-m",
    required=True,
    help="Comma-separated list of models to compare (e.g., gpt-4,claude-3-opus)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Save results to JSON file",
)
@click.option(
    "--diff",
    "-d",
    is_flag=True,
    help="Show unified diff between outputs",
)
@click.option(
    "--side-by-side",
    "-s",
    is_flag=True,
    help="Show outputs side by side",
)
@click.option(
    "--metrics-only",
    is_flag=True,
    help="Only show metrics table, not full outputs",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Minimal output (just the outputs)",
)
def run(
    prompt_file: str,
    models: str,
    output: str | None,
    diff: bool,
    side_by_side: bool,
    metrics_only: bool,
    quiet: bool,
):
    """Run a prompt against multiple models.

    Examples:

        model-compare run prompt.txt --models gpt-4,claude-3-opus

        model-compare run prompt.txt -m gpt-4,gpt-3.5-turbo --diff

        model-compare run prompt.txt -m gpt-4o,claude-3-5-sonnet -o results.json
    """
    # Read prompt file
    prompt_path = Path(prompt_file)
    prompt = prompt_path.read_text().strip()

    # Parse models
    model_list = [m.strip() for m in models.split(",")]

    if not quiet:
        console.print(f"[bold]Running prompt against {len(model_list)} models...[/bold]")
        console.print(f"Models: {', '.join(model_list)}")
        console.print()

    # Run comparison
    results = run_prompt(model_list, prompt)

    # Display results
    if quiet:
        for result in results:
            click.echo(f"=== {result.model} ===")
            if result.error:
                click.echo(f"ERROR: {result.error}")
            else:
                click.echo(result.output)
            click.echo()
    elif metrics_only:
        display_metrics_table(results)
        display_rankings(results)
    elif side_by_side:
        from .display import display_results_side_by_side

        display_results_side_by_side(results)
        display_metrics_table(results)
    else:
        display_full_comparison(results)

    # Show diff if requested
    if diff and len(results) >= 2:
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                diff_result = diff_outputs(results[i], results[j])
                display_diff(diff_result)

    # Save to file if requested
    if output:
        save_results(results, output, prompt)
        console.print(f"\n[green]Results saved to {output}[/green]")


@cli.command()
@click.argument("prompts_dir", type=click.Path(exists=True))
@click.option(
    "--models",
    "-m",
    required=True,
    help="Comma-separated list of models to compare",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output directory for results",
)
@click.option(
    "--pattern",
    "-p",
    default="*.txt",
    help="Glob pattern for prompt files (default: *.txt)",
)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Only show summary, not individual results",
)
def batch(
    prompts_dir: str,
    models: str,
    output: str | None,
    pattern: str,
    summary_only: bool,
):
    """Run multiple prompts against multiple models.

    Examples:

        model-compare batch prompts/ --models gpt-4,gpt-3.5-turbo

        model-compare batch prompts/ -m gpt-4,claude-3-opus -o results/

        model-compare batch prompts/ -m gpt-4o -p "*.md" --summary-only
    """
    # Find all prompt files
    prompts_path = Path(prompts_dir)
    prompt_files = sorted(prompts_path.glob(pattern))

    if not prompt_files:
        console.print(f"[red]No files matching '{pattern}' found in {prompts_dir}[/red]")
        return

    # Parse models
    model_list = [m.strip() for m in models.split(",")]

    console.print(f"[bold]Running batch comparison...[/bold]")
    console.print(f"Prompts: {len(prompt_files)} files")
    console.print(f"Models: {', '.join(model_list)}")
    console.print()

    # Read all prompts
    prompts = []
    prompt_names = []
    for pf in prompt_files:
        prompts.append(pf.read_text().strip())
        prompt_names.append(pf.name)

    # Run batch
    batch_results = run_batch(model_list, prompts)

    # Display results
    if not summary_only:
        for prompt, results in batch_results.items():
            prompt_idx = prompts.index(prompt)
            console.print(f"\n[bold]Prompt: {prompt_names[prompt_idx]}[/bold]")
            console.print("-" * 40)
            display_metrics_table(results)
            display_rankings(results)

    # Display summary
    display_batch_summary(batch_results)

    # Save results if output directory specified
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save individual results
        for prompt, results in batch_results.items():
            prompt_idx = prompts.index(prompt)
            result_file = output_path / f"{prompt_names[prompt_idx]}.json"
            save_results(results, str(result_file), prompt)

        # Save summary
        summary_file = output_path / "summary.json"
        save_batch_summary(batch_results, prompt_names, str(summary_file))

        console.print(f"\n[green]Results saved to {output}/[/green]")


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["html", "json", "markdown", "csv"]),
    default="html",
    help="Output format (default: html)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
def report(results_dir: str, output_format: str, output: str | None):
    """Generate a report from saved results.

    Examples:

        model-compare report results/ --format html

        model-compare report results/ -f markdown -o report.md

        model-compare report results/ -f csv -o metrics.csv
    """
    results_path = Path(results_dir)
    result_files = list(results_path.glob("*.json"))

    if not result_files:
        console.print(f"[red]No JSON result files found in {results_dir}[/red]")
        return

    # Load all results
    all_results: list[dict] = []
    for rf in result_files:
        if rf.name == "summary.json":
            continue
        with open(rf) as f:
            all_results.append(json.load(f))

    # Generate report based on format
    if output_format == "html":
        report_content = generate_html_report(all_results)
        extension = ".html"
    elif output_format == "json":
        report_content = json.dumps(all_results, indent=2)
        extension = ".json"
    elif output_format == "markdown":
        report_content = generate_markdown_report(all_results)
        extension = ".md"
    elif output_format == "csv":
        report_content = generate_csv_report(all_results)
        extension = ".csv"

    # Output
    if output:
        output_file = Path(output)
    else:
        output_file = results_path / f"report{extension}"

    output_file.write_text(report_content)
    console.print(f"[green]Report generated: {output_file}[/green]")


def save_results(results: list[ModelResult], filepath: str, prompt: str) -> None:
    """Save results to JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "results": [
            {
                "model": r.model,
                "output": r.output,
                "latency_ms": r.latency_ms,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "total_tokens": r.total_tokens,
                "cost_usd": r.cost_usd,
                "error": r.error,
                "metadata": r.metadata,
            }
            for r in results
        ],
        "metrics": {
            "comparison_table": compare_models_table(results),
            "cost_breakdown": cost_breakdown(results),
        },
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def save_batch_summary(
    batch_results: dict[str, list[ModelResult]],
    prompt_names: list[str],
    filepath: str,
) -> None:
    """Save batch summary to JSON file."""
    # Get list of prompts in order
    prompts = list(batch_results.keys())

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_prompts": len(batch_results),
        "models": list({r.model for results in batch_results.values() for r in results}),
        "prompts": [],
        "totals": {
            "total_cost": sum(
                r.cost_usd for results in batch_results.values() for r in results
            ),
            "total_tokens": sum(
                r.total_tokens for results in batch_results.values() for r in results
            ),
            "error_count": sum(
                1 for results in batch_results.values() for r in results if r.error
            ),
        },
    }

    for prompt, results in batch_results.items():
        prompt_idx = prompts.index(prompt)
        prompt_data = {
            "name": prompt_names[prompt_idx] if prompt_idx < len(prompt_names) else f"prompt_{prompt_idx}",
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "results_count": len(results),
            "total_cost": sum(r.cost_usd for r in results),
            "avg_latency": sum(r.latency_ms for r in results) / len(results) if results else 0,
            "errors": sum(1 for r in results if r.error),
        }
        summary["prompts"].append(prompt_data)

    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)


def generate_html_report(all_results: list[dict]) -> str:
    """Generate HTML report from results."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }
        h1 { color: #333; }
        h2 { color: #666; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f4f4f4; }
        tr:nth-child(even) { background-color: #fafafa; }
        .output { background-color: #f8f8f8; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; }
        .error { color: #d32f2f; }
        .success { color: #388e3c; }
        .metric { font-weight: bold; }
        .prompt { background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Model Comparison Report</h1>
"""

    for result_data in all_results:
        timestamp = result_data.get("timestamp", "Unknown")
        prompt = result_data.get("prompt", "")
        results = result_data.get("results", [])

        html += f"""
    <h2>Comparison Results</h2>
    <p><strong>Timestamp:</strong> {timestamp}</p>
    <div class="prompt"><strong>Prompt:</strong> {prompt[:500]}{'...' if len(prompt) > 500 else ''}</div>

    <h3>Metrics</h3>
    <table>
        <tr>
            <th>Model</th>
            <th>Latency (ms)</th>
            <th>Output Tokens</th>
            <th>Cost (USD)</th>
            <th>Status</th>
        </tr>
"""

        for r in results:
            status = f'<span class="error">ERROR</span>' if r.get("error") else '<span class="success">OK</span>'
            html += f"""        <tr>
            <td>{r.get('model', 'Unknown')}</td>
            <td>{r.get('latency_ms', 0):.0f}</td>
            <td>{r.get('output_tokens', 0)}</td>
            <td>${r.get('cost_usd', 0):.6f}</td>
            <td>{status}</td>
        </tr>
"""

        html += """    </table>

    <h3>Outputs</h3>
"""

        for r in results:
            model = r.get("model", "Unknown")
            output = r.get("output", "")
            error = r.get("error")

            if error:
                html += f"""    <h4>{model}</h4>
    <div class="output error">{error}</div>
"""
            else:
                # Escape HTML in output
                output_escaped = (
                    output.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                html += f"""    <h4>{model}</h4>
    <div class="output">{output_escaped}</div>
"""

    html += """
</body>
</html>
"""
    return html


def generate_markdown_report(all_results: list[dict]) -> str:
    """Generate Markdown report from results."""
    md = "# Model Comparison Report\n\n"

    for result_data in all_results:
        timestamp = result_data.get("timestamp", "Unknown")
        prompt = result_data.get("prompt", "")
        results = result_data.get("results", [])

        md += f"## Comparison Results\n\n"
        md += f"**Timestamp:** {timestamp}\n\n"
        md += f"**Prompt:**\n\n```\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}\n```\n\n"

        md += "### Metrics\n\n"
        md += "| Model | Latency (ms) | Output Tokens | Cost (USD) | Status |\n"
        md += "|-------|--------------|---------------|------------|--------|\n"

        for r in results:
            status = "ERROR" if r.get("error") else "OK"
            md += f"| {r.get('model', 'Unknown')} | {r.get('latency_ms', 0):.0f} | {r.get('output_tokens', 0)} | ${r.get('cost_usd', 0):.6f} | {status} |\n"

        md += "\n### Outputs\n\n"

        for r in results:
            model = r.get("model", "Unknown")
            output = r.get("output", "")
            error = r.get("error")

            md += f"#### {model}\n\n"
            if error:
                md += f"**Error:** {error}\n\n"
            else:
                md += f"```\n{output}\n```\n\n"

    return md


def generate_csv_report(all_results: list[dict]) -> str:
    """Generate CSV report from results."""
    lines = ["timestamp,prompt,model,latency_ms,input_tokens,output_tokens,total_tokens,cost_usd,status"]

    for result_data in all_results:
        timestamp = result_data.get("timestamp", "")
        prompt = result_data.get("prompt", "").replace('"', '""')[:100]
        results = result_data.get("results", [])

        for r in results:
            status = "error" if r.get("error") else "ok"
            line = f'"{timestamp}","{prompt}","{r.get("model", "")}",{r.get("latency_ms", 0):.0f},{r.get("input_tokens", 0)},{r.get("output_tokens", 0)},{r.get("total_tokens", 0)},{r.get("cost_usd", 0):.6f},{status}'
            lines.append(line)

    return "\n".join(lines)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
