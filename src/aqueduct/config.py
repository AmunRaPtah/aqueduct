"""Central configuration: filesystem paths for each pipeline layer."""

from __future__ import annotations

from pathlib import Path

# Project root = three levels up from this file (src/aqueduct/config.py -> project).
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # landing zone: ingested files live here
WAREHOUSE = DATA_DIR / "warehouse.duckdb"  # the DuckDB database file


def ensure_dirs() -> None:
    """Create the data directories if they don't exist yet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def raw_source_dir(source: str) -> Path:
    """Landing-zone subdirectory for a named source (e.g. 'europepmc')."""
    d = RAW_DIR / source
    d.mkdir(parents=True, exist_ok=True)
    return d
