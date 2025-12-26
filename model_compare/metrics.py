"""Quality metrics for model outputs."""

import statistics
from dataclasses import dataclass, field
from typing import Any

from .runner import ModelResult


@dataclass
class OutputMetrics:
    """Metrics for a single model output."""

    model: str
    # Length metrics
    char_count: int
    word_count: int
    line_count: int
    paragraph_count: int
    # Performance metrics
    latency_ms: float
    # Token metrics
    input_tokens: int
    output_tokens: int
    total_tokens: int
    # Cost metrics
    cost_usd: float
    cost_per_1k_output_tokens: float
    # Quality indicators
    has_error: bool
    # Efficiency metrics
    chars_per_second: float
    tokens_per_second: float


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple model outputs."""

    models: list[str]
    # Latency stats
    latency_min: float
    latency_max: float
    latency_mean: float
    latency_median: float
    latency_stddev: float
    fastest_model: str
    slowest_model: str
    # Cost stats
    total_cost: float
    cost_min: float
    cost_max: float
    cheapest_model: str
    most_expensive_model: str
    # Output stats
    shortest_output_model: str
    longest_output_model: str
    word_count_mean: float
    # Token stats
    total_tokens_used: int
    tokens_per_dollar: float
    # Quality stats
    error_count: int
    success_rate: float
    # Per-model breakdown
    per_model: dict[str, OutputMetrics] = field(default_factory=dict)


def calculate_output_metrics(result: ModelResult) -> OutputMetrics:
    """Calculate metrics for a single model output."""
    text = result.output

    # Length metrics
    char_count = len(text)
    word_count = len(text.split())
    lines = text.splitlines()
    line_count = len(lines)
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    paragraph_count = len(paragraphs)

    # Efficiency metrics
    latency_sec = result.latency_ms / 1000.0
    chars_per_second = char_count / latency_sec if latency_sec > 0 else 0
    tokens_per_second = (
        result.output_tokens / latency_sec if latency_sec > 0 else 0
    )

    # Cost per 1k output tokens
    cost_per_1k = (
        (result.cost_usd / result.output_tokens * 1000)
        if result.output_tokens > 0
        else 0
    )

    return OutputMetrics(
        model=result.model,
        char_count=char_count,
        word_count=word_count,
        line_count=line_count,
        paragraph_count=paragraph_count,
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        cost_usd=result.cost_usd,
        cost_per_1k_output_tokens=cost_per_1k,
        has_error=result.error is not None,
        chars_per_second=chars_per_second,
        tokens_per_second=tokens_per_second,
    )


def calculate_aggregate_metrics(results: list[ModelResult]) -> AggregateMetrics:
    """Calculate aggregate metrics across multiple model outputs."""
    if not results:
        return AggregateMetrics(
            models=[],
            latency_min=0,
            latency_max=0,
            latency_mean=0,
            latency_median=0,
            latency_stddev=0,
            fastest_model="",
            slowest_model="",
            total_cost=0,
            cost_min=0,
            cost_max=0,
            cheapest_model="",
            most_expensive_model="",
            shortest_output_model="",
            longest_output_model="",
            word_count_mean=0,
            total_tokens_used=0,
            tokens_per_dollar=0,
            error_count=0,
            success_rate=0,
        )

    # Calculate per-model metrics
    per_model = {r.model: calculate_output_metrics(r) for r in results}

    # Latency stats
    latencies = [r.latency_ms for r in results]
    latency_min = min(latencies)
    latency_max = max(latencies)
    latency_mean = statistics.mean(latencies)
    latency_median = statistics.median(latencies)
    latency_stddev = statistics.stdev(latencies) if len(latencies) > 1 else 0

    fastest = min(results, key=lambda r: r.latency_ms)
    slowest = max(results, key=lambda r: r.latency_ms)

    # Cost stats
    costs = [r.cost_usd for r in results]
    total_cost = sum(costs)
    cost_min = min(costs)
    cost_max = max(costs)

    cheapest = min(results, key=lambda r: r.cost_usd)
    most_expensive = max(results, key=lambda r: r.cost_usd)

    # Output stats
    word_counts = [len(r.output.split()) for r in results]
    shortest = min(results, key=lambda r: len(r.output))
    longest = max(results, key=lambda r: len(r.output))

    # Token stats
    total_tokens = sum(r.total_tokens for r in results)
    tokens_per_dollar = total_tokens / total_cost if total_cost > 0 else float("inf")

    # Quality stats
    error_count = sum(1 for r in results if r.error is not None)
    success_rate = (len(results) - error_count) / len(results)

    return AggregateMetrics(
        models=[r.model for r in results],
        latency_min=latency_min,
        latency_max=latency_max,
        latency_mean=latency_mean,
        latency_median=latency_median,
        latency_stddev=latency_stddev,
        fastest_model=fastest.model,
        slowest_model=slowest.model,
        total_cost=total_cost,
        cost_min=cost_min,
        cost_max=cost_max,
        cheapest_model=cheapest.model,
        most_expensive_model=most_expensive.model,
        shortest_output_model=shortest.model,
        longest_output_model=longest.model,
        word_count_mean=statistics.mean(word_counts),
        total_tokens_used=total_tokens,
        tokens_per_dollar=tokens_per_dollar,
        error_count=error_count,
        success_rate=success_rate,
        per_model=per_model,
    )


def compare_models_table(results: list[ModelResult]) -> list[dict[str, Any]]:
    """Generate a comparison table for all models."""
    table = []
    for result in results:
        metrics = calculate_output_metrics(result)
        table.append(
            {
                "model": result.model,
                "latency_ms": round(metrics.latency_ms, 1),
                "words": metrics.word_count,
                "tokens_out": metrics.output_tokens,
                "cost_usd": f"${metrics.cost_usd:.6f}",
                "tokens_per_sec": round(metrics.tokens_per_second, 1),
                "status": "error" if metrics.has_error else "ok",
            }
        )
    return table


def rank_models(
    results: list[ModelResult],
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Rank models by weighted score.

    Higher is better. Weights should sum to 1.0.
    Default weights: latency=0.3, cost=0.3, output_length=0.2, success=0.2
    """
    if not results:
        return []

    default_weights = {
        "latency": 0.3,
        "cost": 0.3,
        "output_length": 0.2,
        "success": 0.2,
    }
    weights = weights or default_weights

    # Normalize metrics (0-1 scale, lower is better for latency/cost)
    latencies = [r.latency_ms for r in results]
    costs = [r.cost_usd for r in results]
    lengths = [len(r.output) for r in results]

    max_latency = max(latencies) if latencies else 1
    max_cost = max(costs) if max(costs) > 0 else 1
    max_length = max(lengths) if lengths else 1

    scores: list[tuple[str, float]] = []
    for result in results:
        # Lower latency is better (invert)
        latency_score = 1 - (result.latency_ms / max_latency) if max_latency > 0 else 0
        # Lower cost is better (invert)
        cost_score = 1 - (result.cost_usd / max_cost) if max_cost > 0 else 1
        # Longer output often means more thorough (up to a point)
        length_score = len(result.output) / max_length if max_length > 0 else 0
        # Success is binary
        success_score = 0 if result.error else 1

        total_score = (
            weights.get("latency", 0) * latency_score
            + weights.get("cost", 0) * cost_score
            + weights.get("output_length", 0) * length_score
            + weights.get("success", 0) * success_score
        )
        scores.append((result.model, total_score))

    # Sort by score (higher is better)
    return sorted(scores, key=lambda x: x[1], reverse=True)


def cost_breakdown(results: list[ModelResult]) -> dict[str, Any]:
    """Generate detailed cost breakdown."""
    breakdown: dict[str, Any] = {
        "total_cost_usd": sum(r.cost_usd for r in results),
        "by_model": {},
        "by_provider": {"openai": 0.0, "anthropic": 0.0, "google": 0.0},
    }

    for result in results:
        breakdown["by_model"][result.model] = {
            "cost_usd": result.cost_usd,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_per_1k_input": (
                result.cost_usd / result.input_tokens * 1000
                if result.input_tokens > 0
                else 0
            ),
            "cost_per_1k_output": (
                result.cost_usd / result.output_tokens * 1000
                if result.output_tokens > 0
                else 0
            ),
        }

        # Aggregate by provider
        if result.model.startswith(("gpt-", "o1")):
            breakdown["by_provider"]["openai"] += result.cost_usd
        elif result.model.startswith("claude"):
            breakdown["by_provider"]["anthropic"] += result.cost_usd
        elif result.model.startswith("gemini"):
            breakdown["by_provider"]["google"] += result.cost_usd

    return breakdown
