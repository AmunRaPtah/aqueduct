"""Mailer: markdown->HTML + send payload (offline; HTTP mocked, no email sent)."""

from __future__ import annotations

import json

from aqueduct import mailer


def test_available_reflects_env(monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    assert mailer.available() is False
    monkeypatch.setenv("RESEND_API_KEY", "re_x")
    assert mailer.available() is True


def test_md_to_html_renders_structure():
    md = "# Title\n\n**bold** text\n\n- a\n- b\n\n| x | y |\n| --- | --- |\n| 1 | 2 |"
    html = mailer.md_to_html(md)
    assert "<h1>Title</h1>" in html
    assert "<strong>bold</strong>" in html
    assert "<ul><li>a</li><li>b</li></ul>" in html
    assert "<table" in html and "<td>1</td>" in html
    assert "<" in html and "&lt;script&gt;" not in html  # plain text, nothing to escape here


def test_md_to_html_escapes_html():
    assert "&lt;script&gt;" in mailer.md_to_html("a <script> tag")


def test_send_builds_payload_and_posts(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test")
    monkeypatch.setenv("AQUEDUCT_EMAIL_TO", "me@example.com")
    captured = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"id": "email-123"}).encode()

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        captured["body"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr(mailer.urllib.request, "urlopen", fake_urlopen)
    mid = mailer.send("Subject", "# Hi\n\nbody", to=None)

    assert mid == "email-123"
    assert captured["url"] == mailer.RESEND_URL
    assert captured["headers"]["authorization"] == "Bearer re_test"
    assert captured["headers"]["user-agent"]            # Cloudflare needs it
    assert captured["body"]["to"] == ["me@example.com"]
    assert captured["body"]["subject"] == "Subject"
    assert "<h1>Hi</h1>" in captured["body"]["html"]


def test_send_requires_key(monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    try:
        mailer.send("s", "m", to="x@y.com")
        raise AssertionError("expected MailUnavailable")
    except mailer.MailUnavailable:
        pass
