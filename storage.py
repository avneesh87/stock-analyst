from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

import pandas as pd


DATABASE_PATH = Path(__file__).with_name("stock_analyst.db")


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _database():
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database() -> None:
    with _database() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS watchlist (
                symbol TEXT PRIMARY KEY,
                added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS portfolio (
                symbol TEXT PRIMARY KEY,
                quantity REAL NOT NULL CHECK (quantity > 0),
                average_price REAL NOT NULL CHECK (average_price >= 0),
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS quality_screen_cache (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                generated_at TEXT NOT NULL,
                source TEXT NOT NULL,
                stats_json TEXT NOT NULL,
                rows_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS quality_screen_refresh (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                error TEXT
            );
            """
        )


def list_watchlist() -> list[str]:
    with _database() as connection:
        rows = connection.execute(
            "SELECT symbol FROM watchlist ORDER BY symbol"
        ).fetchall()
    return [str(row["symbol"]) for row in rows]


def add_to_watchlist(symbol: str) -> None:
    with _database() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO watchlist (symbol) VALUES (?)",
            (symbol,),
        )


def remove_from_watchlist(symbol: str) -> None:
    with _database() as connection:
        connection.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))


def list_holdings() -> list[dict[str, object]]:
    with _database() as connection:
        rows = connection.execute(
            """
            SELECT symbol, quantity, average_price
            FROM portfolio
            ORDER BY symbol
            """
        ).fetchall()
    return [dict(row) for row in rows]


def upsert_holding(symbol: str, quantity: float, average_price: float) -> None:
    with _database() as connection:
        connection.execute(
            """
            INSERT INTO portfolio (symbol, quantity, average_price)
            VALUES (?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                quantity = excluded.quantity,
                average_price = excluded.average_price,
                updated_at = CURRENT_TIMESTAMP
            """,
            (symbol, quantity, average_price),
        )


def remove_holding(symbol: str) -> None:
    with _database() as connection:
        connection.execute("DELETE FROM portfolio WHERE symbol = ?", (symbol,))


def save_quality_screen_cache(
    companies: pd.DataFrame,
    source: str,
    stats: dict[str, int],
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    rows = companies.to_dict("records")
    with _database() as connection:
        connection.execute(
            """
            INSERT INTO quality_screen_cache (
                id, generated_at, source, stats_json, rows_json
            )
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                generated_at = excluded.generated_at,
                source = excluded.source,
                stats_json = excluded.stats_json,
                rows_json = excluded.rows_json
            """,
            (
                generated_at,
                source,
                json.dumps(stats),
                json.dumps(rows),
            ),
        )
    return generated_at


def load_quality_screen_cache(
    max_age_hours: int = 24,
) -> tuple[pd.DataFrame, str, dict[str, int], str] | None:
    with _database() as connection:
        row = connection.execute(
            """
            SELECT generated_at, source, stats_json, rows_json
            FROM quality_screen_cache
            WHERE id = 1
            """
        ).fetchone()
    if row is None:
        return None

    generated_at = datetime.fromisoformat(str(row["generated_at"]))
    if datetime.now(timezone.utc) - generated_at > timedelta(hours=max_age_hours):
        return None

    stats = json.loads(str(row["stats_json"]))
    rows = json.loads(str(row["rows_json"]))
    return (
        pd.DataFrame(rows),
        str(row["source"]),
        {str(key): int(value) for key, value in stats.items()},
        generated_at.isoformat(),
    )


def begin_quality_screen_refresh() -> bool:
    now = datetime.now(timezone.utc)
    with _database() as connection:
        current = connection.execute(
            """
            SELECT status, started_at
            FROM quality_screen_refresh
            WHERE id = 1
            """
        ).fetchone()
        if current is not None and current["status"] == "running":
            started_at = datetime.fromisoformat(str(current["started_at"]))
            if now - started_at < timedelta(minutes=30):
                return False
        connection.execute(
            """
            INSERT INTO quality_screen_refresh (
                id, status, started_at, completed_at, error
            )
            VALUES (1, 'running', ?, NULL, NULL)
            ON CONFLICT(id) DO UPDATE SET
                status = 'running',
                started_at = excluded.started_at,
                completed_at = NULL,
                error = NULL
            """,
            (now.isoformat(),),
        )
    return True


def complete_quality_screen_refresh() -> None:
    with _database() as connection:
        connection.execute(
            """
            UPDATE quality_screen_refresh
            SET status = 'completed', completed_at = ?, error = NULL
            WHERE id = 1
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )


def fail_quality_screen_refresh(error: str) -> None:
    with _database() as connection:
        connection.execute(
            """
            UPDATE quality_screen_refresh
            SET status = 'failed', completed_at = ?, error = ?
            WHERE id = 1
            """,
            (datetime.now(timezone.utc).isoformat(), error),
        )


def get_quality_screen_refresh() -> dict[str, str | None] | None:
    with _database() as connection:
        row = connection.execute(
            """
            SELECT status, started_at, completed_at, error
            FROM quality_screen_refresh
            WHERE id = 1
            """
        ).fetchone()
    return dict(row) if row is not None else None
