"""Run prompts against multiple LLM models."""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class ModelResult:
    """Result from running a prompt against a model."""

    model: str
    prompt: str
    output: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Pricing per 1M tokens (input, output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4": (30.0, 60.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
    # Anthropic
    "claude-3-opus": (15.0, 75.0),
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-3-sonnet": (3.0, 15.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-5-haiku": (0.80, 4.0),
    # Google
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash-exp": (0.0, 0.0),  # Free during experimental
}


def calculate_cost(
    model: str, input_tokens: int, output_tokens: int
) -> float:
    """Calculate cost in USD for a model run."""
    pricing = MODEL_PRICING.get(model, (0.0, 0.0))
    input_cost = (input_tokens / 1_000_000) * pricing[0]
    output_cost = (output_tokens / 1_000_000) * pricing[1]
    return input_cost + output_cost


def get_provider(model: str) -> str:
    """Determine the API provider from model name."""
    if model.startswith(("gpt-", "o1")):
        return "openai"
    elif model.startswith("claude"):
        return "anthropic"
    elif model.startswith("gemini"):
        return "google"
    else:
        return "openai"  # Default to OpenAI-compatible


class ModelRunner:
    """Run prompts against multiple LLM models."""

    def __init__(
        self,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        google_api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.timeout = timeout

    async def run_openai(
        self, model: str, prompt: str, client: httpx.AsyncClient
    ) -> ModelResult:
        """Run prompt against OpenAI model."""
        start_time = time.perf_counter()

        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.perf_counter() - start_time) * 1000
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            return ModelResult(
                model=model,
                prompt=prompt,
                output=data["choices"][0]["message"]["content"],
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=calculate_cost(model, input_tokens, output_tokens),
                metadata={"finish_reason": data["choices"][0].get("finish_reason")},
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ModelResult(
                model=model,
                prompt=prompt,
                output="",
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                error=str(e),
            )

    async def run_anthropic(
        self, model: str, prompt: str, client: httpx.AsyncClient
    ) -> ModelResult:
        """Run prompt against Anthropic model."""
        start_time = time.perf_counter()

        try:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.perf_counter() - start_time) * 1000
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = input_tokens + output_tokens

            # Extract text from content blocks
            content = data.get("content", [])
            output = ""
            for block in content:
                if block.get("type") == "text":
                    output += block.get("text", "")

            return ModelResult(
                model=model,
                prompt=prompt,
                output=output,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=calculate_cost(model, input_tokens, output_tokens),
                metadata={"stop_reason": data.get("stop_reason")},
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ModelResult(
                model=model,
                prompt=prompt,
                output="",
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                error=str(e),
            )

    async def run_google(
        self, model: str, prompt: str, client: httpx.AsyncClient
    ) -> ModelResult:
        """Run prompt against Google Gemini model."""
        start_time = time.perf_counter()

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            response = await client.post(
                url,
                params={"key": self.google_api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.perf_counter() - start_time) * 1000
            usage = data.get("usageMetadata", {})
            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            total_tokens = usage.get("totalTokenCount", 0)

            # Extract text from candidates
            candidates = data.get("candidates", [])
            output = ""
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    output += part.get("text", "")

            return ModelResult(
                model=model,
                prompt=prompt,
                output=output,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=calculate_cost(model, input_tokens, output_tokens),
                metadata={
                    "finish_reason": (
                        candidates[0].get("finishReason") if candidates else None
                    )
                },
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ModelResult(
                model=model,
                prompt=prompt,
                output="",
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                error=str(e),
            )

    async def run_model(self, model: str, prompt: str) -> ModelResult:
        """Run a prompt against a single model."""
        provider = get_provider(model)

        async with httpx.AsyncClient() as client:
            if provider == "openai":
                return await self.run_openai(model, prompt, client)
            elif provider == "anthropic":
                return await self.run_anthropic(model, prompt, client)
            elif provider == "google":
                return await self.run_google(model, prompt, client)
            else:
                return ModelResult(
                    model=model,
                    prompt=prompt,
                    output="",
                    latency_ms=0,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    cost_usd=0.0,
                    error=f"Unknown provider for model: {model}",
                )

    async def run_models(
        self, models: list[str], prompt: str
    ) -> list[ModelResult]:
        """Run a prompt against multiple models concurrently."""
        tasks = [self.run_model(model, prompt) for model in models]
        return await asyncio.gather(*tasks)

    async def run_batch(
        self, models: list[str], prompts: list[str]
    ) -> dict[str, list[ModelResult]]:
        """Run multiple prompts against multiple models.

        Returns a dict mapping prompt -> list of results for each model.
        """
        results: dict[str, list[ModelResult]] = {}
        for prompt in prompts:
            results[prompt] = await self.run_models(models, prompt)
        return results


def run_prompt(models: list[str], prompt: str) -> list[ModelResult]:
    """Synchronous wrapper to run a prompt against multiple models."""
    runner = ModelRunner()
    return asyncio.run(runner.run_models(models, prompt))


def run_batch(
    models: list[str], prompts: list[str]
) -> dict[str, list[ModelResult]]:
    """Synchronous wrapper to run multiple prompts against multiple models."""
    runner = ModelRunner()
    return asyncio.run(runner.run_batch(models, prompts))
