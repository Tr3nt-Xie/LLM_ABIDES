#!/usr/bin/env python3
"""
LOB Viewer
==========

Print recent top-of-book snapshots (best bid/ask, spread, mid) from an
enhanced order book SQLite database.

Auto-detects table name: orderbook_snapshots or lob_snapshots.
"""

from __future__ import annotations

import argparse
import sqlite3
from typing import Optional

import pandas as pd


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    try:
        return pd.read_sql(q, conn, params=(name,)).shape[0] > 0
    except Exception:
        return False


def _detect_snapshot_table(conn: sqlite3.Connection) -> Optional[str]:
    for name in ("orderbook_snapshots", "lob_snapshots"):
        if _table_exists(conn, name):
            return name
    return None


def load_snapshots(db_path: str, symbol: str, limit: int) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    table = _detect_snapshot_table(conn)
    if not table:
        conn.close()
        raise SystemExit("No snapshots table found (orderbook_snapshots or lob_snapshots)")

    # Columns to try
    cols = [
        "timestamp",
        "symbol",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread",
    ]
    # Build query defensively
    # Prefer exact cols; if not present, compute spread/mid when possible
    q = f"SELECT * FROM {table} WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?"
    df = pd.read_sql(q, conn, params=(symbol, limit))
    conn.close()

    if df.empty:
        return df

    # Normalize timestamp and derive fields if needed
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if "mid_price" not in df.columns:
        if {"best_bid", "best_ask"}.issubset(df.columns):
            df["mid_price"] = (df["best_bid"].astype(float) + df["best_ask"].astype(float)) / 2.0
    if "spread" not in df.columns:
        if {"best_bid", "best_ask"}.issubset(df.columns):
            df["spread"] = df["best_ask"].astype(float) - df["best_bid"].astype(float)

    return df[[c for c in cols if c in df.columns]]


def main() -> int:
    p = argparse.ArgumentParser(description="Print recent top-of-book snapshots from an order book DB")
    p.add_argument("--db", required=True, help="Path to SQLite DB")
    p.add_argument("--symbol", required=True, help="Symbol (e.g., AAPL)")
    p.add_argument("--limit", type=int, default=20, help="Number of rows to show (default 20)")
    args = p.parse_args()

    df = load_snapshots(args.db, args.symbol, args.limit)
    if df.empty:
        print("(No snapshots found for symbol in DB)")
        return 0

    # Show in chronological order (oldest first)
    df = df.sort_values("timestamp")
    # Print a compact view
    cols = [c for c in ["timestamp", "best_bid", "best_ask", "spread", "mid_price"] if c in df.columns]
    print(df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

