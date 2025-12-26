"""Diff and compare model outputs."""

import difflib
from dataclasses import dataclass
from typing import Any

from .runner import ModelResult


@dataclass
class DiffResult:
    """Result of comparing two model outputs."""

    model_a: str
    model_b: str
    similarity_ratio: float
    unified_diff: str
    side_by_side: list[tuple[str, str]]
    common_lines: int
    different_lines: int
    only_in_a: int
    only_in_b: int


@dataclass
class ComparisonSummary:
    """Summary of comparing multiple model outputs."""

    models: list[str]
    pairwise_similarities: dict[tuple[str, str], float]
    average_similarity: float
    most_similar_pair: tuple[str, str] | None
    least_similar_pair: tuple[str, str] | None
    consensus_lines: list[str]  # Lines that appear in all outputs


def get_unified_diff(
    text_a: str, text_b: str, label_a: str = "Model A", label_b: str = "Model B"
) -> str:
    """Generate a unified diff between two texts."""
    lines_a = text_a.splitlines(keepends=True)
    lines_b = text_b.splitlines(keepends=True)

    diff = difflib.unified_diff(lines_a, lines_b, fromfile=label_a, tofile=label_b)
    return "".join(diff)


def get_side_by_side(text_a: str, text_b: str) -> list[tuple[str, str]]:
    """Generate side-by-side comparison of two texts."""
    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()

    # Use SequenceMatcher to align lines
    matcher = difflib.SequenceMatcher(None, lines_a, lines_b)
    result: list[tuple[str, str]] = []

    for opcode, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if opcode == "equal":
            for i in range(a_end - a_start):
                result.append((lines_a[a_start + i], lines_b[b_start + i]))
        elif opcode == "replace":
            max_len = max(a_end - a_start, b_end - b_start)
            for i in range(max_len):
                a_line = lines_a[a_start + i] if a_start + i < a_end else ""
                b_line = lines_b[b_start + i] if b_start + i < b_end else ""
                result.append((a_line, b_line))
        elif opcode == "delete":
            for i in range(a_end - a_start):
                result.append((lines_a[a_start + i], ""))
        elif opcode == "insert":
            for i in range(b_end - b_start):
                result.append(("", lines_b[b_start + i]))

    return result


def calculate_similarity(text_a: str, text_b: str) -> float:
    """Calculate similarity ratio between two texts (0.0 to 1.0)."""
    return difflib.SequenceMatcher(None, text_a, text_b).ratio()


def diff_outputs(result_a: ModelResult, result_b: ModelResult) -> DiffResult:
    """Compare outputs from two model results."""
    text_a = result_a.output
    text_b = result_b.output

    lines_a = set(text_a.splitlines())
    lines_b = set(text_b.splitlines())

    common = lines_a & lines_b
    only_a = lines_a - lines_b
    only_b = lines_b - lines_a

    return DiffResult(
        model_a=result_a.model,
        model_b=result_b.model,
        similarity_ratio=calculate_similarity(text_a, text_b),
        unified_diff=get_unified_diff(text_a, text_b, result_a.model, result_b.model),
        side_by_side=get_side_by_side(text_a, text_b),
        common_lines=len(common),
        different_lines=len(only_a) + len(only_b),
        only_in_a=len(only_a),
        only_in_b=len(only_b),
    )


def compare_all_outputs(results: list[ModelResult]) -> ComparisonSummary:
    """Compare outputs from all model results."""
    if len(results) < 2:
        return ComparisonSummary(
            models=[r.model for r in results],
            pairwise_similarities={},
            average_similarity=1.0 if results else 0.0,
            most_similar_pair=None,
            least_similar_pair=None,
            consensus_lines=[],
        )

    # Calculate pairwise similarities
    similarities: dict[tuple[str, str], float] = {}
    for i, result_a in enumerate(results):
        for result_b in results[i + 1 :]:
            sim = calculate_similarity(result_a.output, result_b.output)
            similarities[(result_a.model, result_b.model)] = sim

    # Find most and least similar pairs
    if similarities:
        most_similar = max(similarities.items(), key=lambda x: x[1])
        least_similar = min(similarities.items(), key=lambda x: x[1])
        avg_similarity = sum(similarities.values()) / len(similarities)
    else:
        most_similar = (None, 0.0)
        least_similar = (None, 0.0)
        avg_similarity = 0.0

    # Find consensus lines (appear in all outputs)
    if results:
        line_sets = [set(r.output.splitlines()) for r in results]
        consensus = line_sets[0]
        for line_set in line_sets[1:]:
            consensus = consensus & line_set
    else:
        consensus = set()

    return ComparisonSummary(
        models=[r.model for r in results],
        pairwise_similarities=similarities,
        average_similarity=avg_similarity,
        most_similar_pair=most_similar[0] if most_similar[0] else None,
        least_similar_pair=least_similar[0] if least_similar[0] else None,
        consensus_lines=sorted(consensus),
    )


def highlight_differences(text_a: str, text_b: str) -> tuple[list[str], list[str]]:
    """Return lists of words that are different in each text.

    Useful for highlighting specific differences.
    """
    words_a = text_a.split()
    words_b = text_b.split()

    matcher = difflib.SequenceMatcher(None, words_a, words_b)

    diff_a: list[str] = []
    diff_b: list[str] = []

    for opcode, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if opcode in ("replace", "delete"):
            diff_a.extend(words_a[a_start:a_end])
        if opcode in ("replace", "insert"):
            diff_b.extend(words_b[b_start:b_end])

    return diff_a, diff_b


def semantic_diff(results: list[ModelResult]) -> dict[str, Any]:
    """Analyze semantic differences between model outputs.

    Returns structural analysis of differences.
    """
    analysis: dict[str, Any] = {
        "models": [r.model for r in results],
        "outputs": {},
        "structure": {},
    }

    for result in results:
        text = result.output
        analysis["outputs"][result.model] = {
            "length": len(text),
            "word_count": len(text.split()),
            "line_count": len(text.splitlines()),
            "paragraph_count": len([p for p in text.split("\n\n") if p.strip()]),
            "has_code_blocks": "```" in text,
            "has_bullet_points": any(
                line.strip().startswith(("- ", "* ", "1. "))
                for line in text.splitlines()
            ),
            "has_headers": any(
                line.strip().startswith("#") for line in text.splitlines()
            ),
        }

    # Analyze structural consistency
    structures = list(analysis["outputs"].values())
    if structures:
        analysis["structure"]["code_block_consensus"] = all(
            s["has_code_blocks"] == structures[0]["has_code_blocks"] for s in structures
        )
        analysis["structure"]["bullet_consensus"] = all(
            s["has_bullet_points"] == structures[0]["has_bullet_points"]
            for s in structures
        )
        analysis["structure"]["header_consensus"] = all(
            s["has_headers"] == structures[0]["has_headers"] for s in structures
        )

    return analysis
