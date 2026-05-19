"""
app/memory/importer.py

CLI tool for importing external memories into the database.

Supports two formats:

  JSON  — array of memory objects
  CSV   — one memory per row

JSON schema (each object):
  {
    "content":    "...",               required
    "role":       "user|assistant",    optional, default "assistant"
    "source":     "import",            optional, default "import"
    "type":       "episode|fact",      optional, default "fact"
    "confidence": 0.0-1.0,            optional (facts only), default 1.0
    "created_at": 1234567890.0,        optional Unix timestamp
    "tags":       ["tag1", "tag2"]     optional
  }

CSV columns (header row required):
  content, role, source, type, confidence

Usage
-----
  python -m app.memory.importer \\
      --db   ./data/memory.db \\
      --file ./my_memories.json \\
      --dry-run

  python -m app.memory.importer \\
      --db   ./data/memory.db \\
      --file ./my_memories.csv
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterator

from app.memory.db import MemoryDB
from app.memory.writer import MemoryWriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Record normalisation
# ---------------------------------------------------------------------------

def _normalise(raw: dict) -> dict:
    """Normalise a raw imported record to a consistent shape."""
    return {
        "content":    str(raw.get("content", "")).strip(),
        "role":       str(raw.get("role",    "assistant")),
        "source":     str(raw.get("source",  "import")),
        "type":       str(raw.get("type",    "fact")),
        "confidence": float(raw.get("confidence", 1.0)),
        "created_at": float(raw.get("created_at", time.time())),
        "tags":       raw.get("tags", []),
    }


def _validate(record: dict) -> list[str]:
    """Return list of validation error messages. Empty = valid."""
    errors = []
    if not record["content"]:
        errors.append("content is empty")
    if record["role"] not in ("user", "assistant", "system", "event"):
        errors.append(f"unknown role: {record['role']!r}")
    if record["type"] not in ("episode", "fact"):
        errors.append(f"type must be 'episode' or 'fact', got {record['type']!r}")
    if not 0.0 <= record["confidence"] <= 1.0:
        errors.append(f"confidence out of range: {record['confidence']}")
    return errors


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Iterator[dict]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a top-level array")
    yield from data


def _read_csv(path: Path) -> Iterator[dict]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        yield from reader


def _read_file(path: Path) -> Iterator[dict]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        yield from _read_json(path)
    elif suffix == ".csv":
        yield from _read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .json or .csv")


# ---------------------------------------------------------------------------
# Import logic
# ---------------------------------------------------------------------------

def run_import(
    db_path:  str,
    file_path: str,
    dry_run:  bool = False,
    verbose:  bool = False,
) -> dict:
    """
    Import memories from a file into the database.

    Returns a summary dict with counts of imported, skipped, and errored records.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    counts = {"imported": 0, "skipped": 0, "errors": 0, "total": 0}

    with MemoryDB(db_path) as db:
        writer = MemoryWriter(db, embed_fn=None)  # no embeddings during import

        # Create a dedicated import session
        if not dry_run:
            session_id = writer.create_session(tags=["import", path.stem])

        for raw in _read_file(path):
            counts["total"] += 1
            record = _normalise(raw)
            errors = _validate(record)

            if errors:
                counts["errors"] += 1
                if verbose:
                    print(f"  [SKIP] record {counts['total']}: {'; '.join(errors)}")
                continue

            if verbose:
                print(f"  [{'DRY' if dry_run else 'OK'}] {record['type']:7s} | "
                      f"{record['role']:9s} | {record['content'][:60]!r}")

            if dry_run:
                counts["imported"] += 1
                continue

            try:
                if record["type"] == "fact":
                    writer.write_fact(
                        fact       = record["content"],
                        confidence = record["confidence"],
                        source     = "import",
                    )
                else:
                    writer.write_episode(
                        session_id      = session_id,
                        role            = record["role"],
                        content         = record["content"],
                        source          = "import",
                        salience_weight = record["confidence"],
                    )
                counts["imported"] += 1
            except Exception as exc:
                counts["errors"] += 1
                logger.error("Import error on record %d: %s", counts["total"], exc)

        if not dry_run and counts["imported"] > 0:
            writer.close_session(
                session_id,
                summary=f"Import from {path.name}: {counts['imported']} records"
            )

    return counts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Import memories into the Eyes and Ears memory database."
    )
    parser.add_argument(
        "--db",    required=True,
        help="Path to memory.db (created if it doesn't exist)"
    )
    parser.add_argument(
        "--file",  required=True,
        help="Path to .json or .csv memory file"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and preview without writing to the database"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print each record as it is processed"
    )
    args = parser.parse_args()

    print(f"\nImporting from: {args.file}")
    print(f"Target DB:      {args.db}")
    if args.dry_run:
        print("Mode:           DRY RUN (no changes will be written)\n")
    else:
        print()

    try:
        counts = run_import(
            db_path   = args.db,
            file_path = args.file,
            dry_run   = args.dry_run,
            verbose   = args.verbose,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone.")
    print(f"  Total records : {counts['total']}")
    print(f"  Imported      : {counts['imported']}")
    print(f"  Skipped/errors: {counts['errors']}")
    if args.dry_run:
        print("\n(Dry run — nothing was written to the database.)")


if __name__ == "__main__":
    main()
