"""LLM client — DeepSeek via its Anthropic-compatible Messages API.

Config resolution (first hit wins):
  1. env: DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL, or the
     ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL family.
  2. a JSON file with an `env` block (default ~/.claude/deepseek.json) holding
     ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY, ANTHROPIC_DEFAULT_SONNET_MODEL (=pro),
     ANTHROPIC_DEFAULT_HAIKU_MODEL (=flash).

The key is read at call time and never logged or persisted. `available()` lets
callers degrade gracefully (e.g. emit metrics-only reports) when no key is present.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

CONFIG_PATHS = ["~/.claude/deepseek.json", "~/.claude-deepseek/deepseek.json"]


class LLMUnavailable(RuntimeError):
    """Raised when no LLM credentials can be resolved."""


def _from_file() -> dict | None:
    for p in CONFIG_PATHS:
        fp = Path(p).expanduser()
        if fp.exists():
            try:
                env = json.loads(fp.read_text()).get("env", {})
            except (json.JSONDecodeError, OSError):
                continue
            if env.get("ANTHROPIC_API_KEY"):
                return {
                    "base": env.get("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic"),
                    "key": env["ANTHROPIC_API_KEY"],
                    "pro": env.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "deepseek-v4-pro"),
                    "flash": env.get("ANTHROPIC_DEFAULT_HAIKU_MODEL", "deepseek-v4-flash"),
                }
    return None


def config() -> dict | None:
    """Resolve LLM config from env then config file, or None if unavailable."""
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return {
            "base": os.environ.get("DEEPSEEK_BASE_URL")
            or os.environ.get("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic"),
            "key": key,
            "pro": os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            "flash": os.environ.get("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
        }
    return _from_file()


def available() -> bool:
    return config() is not None


def _build(prompt: str, system: str | None, model: str, max_tokens: int,
           temperature: float) -> dict:
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]}
    if system:
        body["system"] = system
    return body


def _parse(data: dict) -> str:
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()


def _resolve_model(model: str | None, cfg: dict) -> str:
    if model in (None, "pro"):
        return cfg["pro"]
    if model == "flash":
        return cfg["flash"]
    return model


def complete(prompt: str, *, system: str | None = None, model: str | None = None,
             max_tokens: int = 1800, temperature: float = 0.3, retries: int = 3) -> str:
    """Send one message and return the model's text. `model` may be 'pro'/'flash'/id."""
    cfg = config()
    if not cfg:
        raise LLMUnavailable(
            "no DeepSeek credentials (set DEEPSEEK_API_KEY or ~/.claude/deepseek.json)")
    body = json.dumps(_build(prompt, system, _resolve_model(model, cfg),
                             max_tokens, temperature)).encode()
    req = urllib.request.Request(
        f"{cfg['base']}/v1/messages", data=body,
        headers={"x-api-key": cfg["key"], "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return _parse(json.loads(r.read()))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"LLM request failed after {retries} tries") from last
