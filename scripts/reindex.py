"""Rebuild the warehouse from the existing landing zone — no network fetch.

Useful after a schema or chunking change (it re-derives bronze→silver→gold, the link
graph, and the semantic index from already-landed data). The default backend follows
`embeddings.default_backend()`; pass one explicitly to override.

    python scripts/reindex.py                 # auto backend (st if installed, else lsa)
    python scripts/reindex.py --backend lsa   # force the keyless LSA index (fast)
"""

from __future__ import annotations

import argparse
import time

from aqueduct import corpus, datasets, embeddings, links, validate
from aqueduct.storage import connect


def _timed(label, fn):
    start = time.time()
    result = fn()
    print(f">>> {label}: {time.time() - start:.1f}s", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["auto", "lsa", "st"], default="auto")
    ap.add_argument("--dims", type=int, default=128, help="LSA dimensions")
    args = ap.parse_args(argv)

    con = connect()
    try:
        _timed("corpus", lambda: corpus.build(con))
        _timed("datasets", lambda: datasets.build(con))
        _timed("links", lambda: links.build(con))
        _timed("index", lambda: embeddings.build_index(
            con, backend=args.backend, dims=args.dims, force=True))
        _timed("validate", lambda: validate.validate(con))
    finally:
        con.close()
    print(">>> done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
