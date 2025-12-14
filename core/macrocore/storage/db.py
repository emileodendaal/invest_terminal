from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db_path(project_root: Path) -> Path:
    # DB will be created in your project root as: invest_terminal/macro_tracker.sqlite
    return project_root / "macro_tracker.sqlite"


def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect(db_path) as con:
        con.executescript(schema)


def set_meta(db_path: Path, key: str, value: str) -> None:
    with connect(db_path) as con:
        con.execute(
            """
            INSERT INTO meta(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, value),
        )


def get_meta(db_path: Path, key: str) -> str | None:
    with connect(db_path) as con:
        row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def count_rows(db_path: Path, table: str) -> int:
    # Table names are fixed by us (not user input), so this is safe.
    with connect(db_path) as con:
        row = con.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    return int(row["n"])


def data_health(db_path: Path) -> dict:
    return {
        "macro_series": count_rows(db_path, "macro_observations"),
        "events": count_rows(db_path, "events"),
        "prices": count_rows(db_path, "prices"),
        "last_update": get_meta(db_path, "last_update_utc"),
    }
