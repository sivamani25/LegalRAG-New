"""OpenRouter chat-completion client with configurable model fallback."""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter generation fails for all configured models."""


class OpenRouterClient:
    """Minimal OpenRouter client with primary and fallback model support."""

    def __init__(
        self,
        config: dict[str, Any],
        api_key: str | None = None,
        http_client: Any | None = None,
        timeout: float = 60.0,
    ) -> None:
        load_dotenv()
        generation = config.get("generation", {})
        if generation.get("provider", "openrouter") != "openrouter":
            raise ValueError("OpenRouterClient requires provider='openrouter'.")

        self.primary_model = generation.get("model", "")
        self.fallback_model = generation.get("fallback_model", "")
        self.temperature = float(generation.get("temperature", 0.2))
        self.max_tokens = int(generation.get("max_tokens", 1024))
        self.api_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY", "")
        self._owns_client = http_client is None
        self.http_client = http_client or httpx.Client(timeout=timeout)

        if not self.primary_model:
            raise ValueError("generation.model must be configured.")

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text, trying the fallback model if the primary fails."""
        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is missing. Set it as an environment variable."
            )
        if not messages:
            raise ValueError("messages must contain at least one message.")

        models = [self.primary_model]
        if self.fallback_model and self.fallback_model != self.primary_model:
            models.append(self.fallback_model)

        errors: list[str] = []
        for model in models:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": (
                    self.temperature if temperature is None else temperature
                ),
                "max_completion_tokens": self.max_tokens if max_tokens is None else max_tokens,
                "reasoning": {
                    "max_tokens": 512,
                    "exclude": True,
                },
            }
            try:
                response = self.http_client.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices")

                if not isinstance(choices, list) or not choices:
                    raise ValueError("OpenRouter returned no choices.")

                choice = choices[0]
                if not isinstance(choice, dict):
                    raise ValueError("OpenRouter returned an invalid choice.")

                message = choice.get("message")
                if not isinstance(message, dict):
                    raise ValueError("OpenRouter returned an invalid message.")

                content = message.get("content")

                
                if not isinstance(content, str) or not content.strip():
                    finish_reason = choice.get("finish_reason") or "not provided"
                    reasoning = message.get("reasoning")
                    reasoning_chars = (
                        len(reasoning) if isinstance(reasoning, str) else 0
                    )
                    raise ValueError(
                        "Empty or non-text content "
                        f"(finish_reason={finish_reason}, "
                        f"reasoning_chars={reasoning_chars})."
                    )

                return content.strip()

            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                errors.append(f"{model}: {type(exc).__name__}: {exc}")

        raise OpenRouterError(
            "Generation failed for all configured models. " + " | ".join(errors)
        )

    def close(self) -> None:
        """Close the HTTP client if this instance created it."""
        if self._owns_client:
            self.http_client.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

        




        
