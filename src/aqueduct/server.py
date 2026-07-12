"""HTTP retrieval API — exposes Aqueduct's RAG to remote consumers (e.g. the Pardalos bot).

Stdlib only (no web framework). Endpoints return JSON:
  GET  /health                         -> {status, index}
  GET  /retrieve?q=..&k=..&min_score=..&source=..&section=..   -> rag.retrieve()
  POST /retrieve   {query,k,min_score,sources,sec_types}        -> rag.retrieve()
  GET  /facts                          -> analysis.facts()
  GET  /discover?drug=NAME | ?gene=SYM -> discover.ranked_papers[_protein]()

Auth: if AQUEDUCT_API_KEY is set, requests must send `Authorization: Bearer <key>`
(open when unset, for local use). Bind to 127.0.0.1 unless you front it with a tunnel.
"""

from __future__ import annotations

import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import analysis, discover, embeddings, rag


def _one(params: dict, key: str, default=None):
    v = params.get(key)
    return v[0] if v else default


def route(method: str, path: str, params: dict, body: dict, authed: bool) -> tuple[int, dict]:
    """Pure request dispatch (no sockets) — returns (status, payload). Unit-testable."""
    if path == "/health":
        return 200, {"status": "ok", "index": embeddings.index_info()}
    if not authed:
        return 401, {"error": "unauthorized"}

    if path == "/retrieve":
        q = (_one(params, "q") or body.get("query") or "").strip()
        if not q:
            return 400, {"error": "missing query (q / query)"}
        k = int(_one(params, "k", body.get("k", 8)))
        min_score = float(_one(params, "min_score", body.get("min_score", 0.0)))
        sources = params.get("source") or body.get("sources")
        sections = params.get("section") or body.get("sec_types")
        graph = _one(params, "graph", "1") not in ("0", "false") and body.get("graph", True)
        return 200, rag.retrieve(q, k=k, min_score=min_score, sources=sources,
                                 sec_types=sections, graph=bool(graph))

    if path == "/facts":
        return 200, analysis.facts()

    if path == "/discover":
        drug, gene = _one(params, "drug"), _one(params, "gene")
        k = int(_one(params, "k", 8))
        if drug:
            prof, ranked = discover.ranked_papers(drug, k=k)
            return 200, {"drug": drug, "profile": prof,
                         "papers": [{"id": p, "score": round(s, 4), "direct": d}
                                    for p, s, d in ranked]}
        if gene:
            prof, ranked = discover.ranked_papers_protein(gene, k=k)
            return 200, {"gene": gene, "profile": prof,
                         "papers": [{"id": p, "score": round(s, 4), "direct": d}
                                    for p, s, d in ranked]}
        return 400, {"error": "provide ?drug= or ?gene="}

    return 404, {"error": "not found", "path": path}


def _authed(headers) -> bool:
    key = os.environ.get("AQUEDUCT_API_KEY")
    if not key:
        return True
    return hmac.compare_digest(headers.get("Authorization", ""), f"Bearer {key}")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle(self, method: str):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        body = {}
        if method == "POST":
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n:
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    return self._send(400, {"error": "invalid JSON body"})
        try:
            status, payload = route(method, parsed.path, params, body, _authed(self.headers))
        except Exception as e:  # noqa: BLE001 - never leak a stack trace to clients
            status, payload = 500, {"error": "internal error", "detail": str(e)[:200]}
        self._send(status, payload)

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def log_message(self, *a):  # quiet by default
        pass


def serve(host: str = "127.0.0.1", port: int = 8800) -> None:
    """Run the retrieval API (blocking)."""
    httpd = ThreadingHTTPServer((host, port), _Handler)
    auth = "on" if os.environ.get("AQUEDUCT_API_KEY") else "OFF (set AQUEDUCT_API_KEY)"
    print(f"[serve]   Aqueduct API on http://{host}:{port}  (auth: {auth})")
    print(f"[serve]   try: curl 'http://{host}:{port}/retrieve?q=opioid+receptor&k=3'")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
