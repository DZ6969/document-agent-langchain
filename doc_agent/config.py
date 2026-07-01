"""Configuration and the shared Claude chat model.

The model is created through ``langchain-anthropic`` (Anthropic's official
LangChain integration, which wraps the official ``anthropic`` SDK).
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

# Default to the most capable Opus model. Override with the MODEL env var
# (e.g. claude-sonnet-5 or claude-haiku-4-5) to trade quality for cost.
DEFAULT_MODEL = "claude-opus-4-8"


def get_llm(model: str | None = None, max_tokens: int = 4096) -> ChatAnthropic:
    """Return a configured Claude chat model.

    Raises a clear error if the API key is missing so the failure is obvious
    instead of surfacing deep inside an API call.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatAnthropic(
        model=model or os.getenv("MODEL", DEFAULT_MODEL),
        max_tokens=max_tokens,
    )
