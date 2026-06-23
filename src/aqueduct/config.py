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


def rel_data_path(p: Path | str) -> str:
    """Serialize a data-file path portably: relative to DATA_DIR.

    Manifests are migrated between machines (VPS -> Lightning studio), so paths
    must not be absolute. Falls back to an absolute string only if the file lives
    outside DATA_DIR (shouldn't happen for landing-zone files).
    """
    rp = Path(p).resolve()
    try:
        return str(rp.relative_to(DATA_DIR))
    except ValueError:
        return str(rp)


def resolve_data_path(stored: str) -> Path:
    """Resolve a manifest path against the local DATA_DIR.

    Tolerates absolute paths written on another machine (pre-migration
    manifests). Absolute paths are remapped under the local DATA_DIR by their
    segment after the last 'raw/' WITHOUT stat'ing the original — a foreign path
    like /root/... is not just non-existent but unreadable (EACCES) on the
    target, so probing it would raise. A local path under DATA_DIR remaps to
    itself, so this is a no-op there.
    """
    p = Path(stored)
    if not p.is_absolute():
        return DATA_DIR / p
    parts = p.parts
    if "raw" in parts:  # remap <other>/data/raw/... -> local DATA_DIR/raw/...
        i = len(parts) - 1 - parts[::-1].index("raw")
        return DATA_DIR.joinpath(*parts[i:])
    return DATA_DIR / "raw" / p.name


def raw_source_dir(source: str) -> Path:
    """Landing-zone subdirectory for a named source (e.g. 'europepmc')."""
    d = RAW_DIR / source
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir(name: str) -> Path:
    """A derived-artifact cache subdirectory (e.g. downloaded PDFs). Not the warehouse."""
    d = DATA_DIR / "cache" / name
    d.mkdir(parents=True, exist_ok=True)
    return d
