#!/usr/bin/env python3
"""
Seeded LOB DB Generator
=======================

Generate an order book snapshots SQLite DB aligned to a chosen start time
and seeded with the real market price at that time. Ensures that the
simulation's starting configuration (time and starting price) matches
reality for apples-to-apples validation.
"""

from __future__ import annotations

import argparse
import math
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Tuple

import pandas as pd

from real_data_ingestion import fetch_price_at_timestamp


def _round_down_to_minute(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0)


def _fetch_seed_price(symbol: str, start: datetime) -> Tuple[float, dict]:
    price, meta = fetch_price_at_timestamp(symbol, start, interval="1m", allow_fallback=True)
    if price is None:
        # As a last resort, try 5m
        price, meta = fetch_price_at_timestamp(symbol, start, interval="5m", allow_fallback=True)
    if price is None:
        # Default to a safe nominal price
        price = 100.0
        meta = {"interval_used": None, "source_row_timestamp": None}
    return float(price), meta


def generate_db(db_path: str, symbol: str, start: datetime, end: datetime,
                daily_vol: float = 0.02, base_spread_bps: float = 8.0) -> None:
    start = _round_down_to_minute(start.astimezone(timezone.utc))
    end = _round_down_to_minute(end.astimezone(timezone.utc))
    if end <= start:
        raise SystemExit("End must be after start")

    seed_price, meta = _fetch_seed_price(symbol, start)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('DROP TABLE IF EXISTS orderbook_snapshots')
    cur.execute('CREATE TABLE IF NOT EXISTS orderbook_snapshots ('
                'timestamp TEXT, symbol TEXT, best_bid REAL, best_ask REAL, mid_price REAL, spread REAL)')

    # Simulate minute-by-minute mid prices using a simple GBM-like process with mild mean reversion
    mid = seed_price
    dt_years = 1.0 / (252 * 24 * 60)  # per-minute step in years
    minutes = int((end - start).total_seconds() // 60)
    for i in range(minutes + 1):
        ts = start + timedelta(minutes=i)
        # mean reversion toward seed_price helps stability over short windows
        mean_reversion_strength = 0.05
        mr = mean_reversion_strength * (seed_price - mid) / max(seed_price, 1e-6)
        shock = random.gauss(0.0, daily_vol * math.sqrt(dt_years))
        mid = max(0.01, mid * (1.0 + mr * dt_years + shock))
        spread = max(0.01, mid * (base_spread_bps / 10000.0) * random.uniform(0.7, 1.3))
        bid = mid - spread / 2.0
        ask = mid + spread / 2.0
        cur.execute('INSERT INTO orderbook_snapshots VALUES (?,?,?,?,?,?)',
                    (ts.isoformat(), symbol, bid, ask, mid, spread))

    conn.commit()
    conn.close()

    print(f"DB: {db_path}")
    print(f"Seed price: {seed_price}")
    print(f"Seed meta: {meta}")
    print(f"Start: {start.isoformat()}  End: {end.isoformat()}")


def main() -> int:
    p = argparse.ArgumentParser(description="Generate seeded orderbook snapshots DB aligned to real start price")
    p.add_argument('--db', required=True, help='Output SQLite DB path')
    p.add_argument('--symbol', required=True, help='Symbol, e.g., AAPL')
    p.add_argument('--start', default=None, help='UTC ISO start (default: now-6h)')
    p.add_argument('--end', default=None, help='UTC ISO end (default: now)')
    p.add_argument('--hours', type=int, default=6, help='If start/end not provided, use now-hours..now')
    p.add_argument('--daily-vol', type=float, default=0.02, help='Assumed daily volatility')
    p.add_argument('--spread-bps', type=float, default=8.0, help='Base spread in bps')
    args = p.parse_args()

    now = datetime.now(timezone.utc)
    if args.start and args.end:
        start = pd.to_datetime(args.start, utc=True).to_pydatetime()
        end = pd.to_datetime(args.end, utc=True).to_pydatetime()
    else:
        end = now
        start = now - timedelta(hours=args.hours)

    generate_db(args.db, args.symbol, start, end, daily_vol=args.daily_vol, base_spread_bps=args.spread_bps)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

