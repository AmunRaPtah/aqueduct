"""LLM client config resolution + request/response shaping (offline, no API calls)."""

from __future__ import annotations

import json

from aqueduct import llm


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k-123")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    cfg = llm.config()
    assert cfg["key"] == "k-123" and cfg["pro"] == "deepseek-v4-pro"
    assert llm.available() is True


def test_config_from_file(monkeypatch, tmp_path):
    for v in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    f = tmp_path / "deepseek.json"
    f.write_text(json.dumps({"env": {
        "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
        "ANTHROPIC_API_KEY": "file-key",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4-pro",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash"}}))
    monkeypatch.setattr(llm, "CONFIG_PATHS", [str(f)])
    cfg = llm.config()
    assert cfg["key"] == "file-key" and cfg["flash"] == "deepseek-v4-flash"


def test_unavailable_when_no_config(monkeypatch):
    for v in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(llm, "CONFIG_PATHS", [])
    assert llm.available() is False


def test_model_alias_resolution():
    cfg = {"pro": "PRO", "flash": "FLASH"}
    assert llm._resolve_model(None, cfg) == "PRO"
    assert llm._resolve_model("flash", cfg) == "FLASH"
    assert llm._resolve_model("custom-id", cfg) == "custom-id"


def test_build_and_parse():
    body = llm._build("hi", system="be brief", model="m", max_tokens=10, temperature=0.2)
    assert body["system"] == "be brief" and body["messages"][0]["content"] == "hi"
    assert llm._parse({"content": [{"type": "text", "text": "a "}, {"type": "text", "text": "b"}]}) == "a b"
