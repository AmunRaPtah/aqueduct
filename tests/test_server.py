"""HTTP API dispatch + a real round-trip over a socket (offline)."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import seed

from aqueduct import corpus, embeddings, server


def _seed(con):
    seed.seed_document("PMC1", title="Naloxone reversal",
                       abstract="Naloxone is an opioid antagonist.",
                       sections=[("Body", "The antagonist reverses overdose. " * 6)])
    corpus.build(con)
    embeddings.build_index(con, backend="lsa", dims=8)


# ---- pure dispatch (no sockets) ----
def test_health_is_public():
    status, payload = server.route("GET", "/health", {}, {}, authed=False)
    assert status == 200 and payload["status"] == "ok" and "index" in payload


def test_auth_required_when_not_authed():
    status, _ = server.route("GET", "/retrieve", {"q": ["opioid"]}, {}, authed=False)
    assert status == 401


def test_retrieve_dispatch(con, env):
    _seed(con)
    status, payload = server.route("GET", "/retrieve",
                                   {"q": ["opioid antagonist"], "k": ["3"]}, {}, authed=True)
    assert status == 200 and payload["n"] >= 1
    assert payload["chunks"][0]["id"] == "PMC1"


def test_missing_query_is_400():
    status, payload = server.route("GET", "/retrieve", {}, {}, authed=True)
    assert status == 400


def test_unknown_route_404():
    status, _ = server.route("GET", "/nope", {}, {}, authed=True)
    assert status == 404


# ---- real socket round-trip ----
def test_http_roundtrip(con, env):
    _seed(con)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server._Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/retrieve?q=opioid+antagonist&k=2", timeout=10) as r:
            data = json.load(r)
        assert data["n"] >= 1 and data["chunks"][0]["id"] == "PMC1"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
            assert json.load(r)["status"] == "ok"
    finally:
        httpd.shutdown()
