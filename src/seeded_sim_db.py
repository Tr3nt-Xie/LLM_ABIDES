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
from typing import Tuple, Optional, List, Dict

import pandas as pd

from real_data_ingestion import fetch_price_at_timestamp, fetch_intraday_ohlcv, MarketFetchConfig


def _load_events_csv(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        else:
            df['timestamp'] = pd.NaT
        # Coerce fields
        for c in ('sentiment_score', 'confidence'):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['timestamp'])
    except Exception:
        return pd.DataFrame()


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


def _calibrate_per_minute_vol(symbol: str, start: datetime, end: datetime) -> tuple[float, dict]:
    cfg = MarketFetchConfig(symbol=symbol, start=start, end=end, interval="1m", allow_fallback=True)
    real = fetch_intraday_ohlcv(cfg)
    if real is None or real.empty or 'close' not in real.columns:
        return 0.0, {"interval_used": None}
    r = pd.Series(pd.to_numeric(real['close'], errors='coerce')).dropna()
    if len(r) < 3:
        return 0.0, {"interval_used": real.attrs.get('interval_used')}
    ret = (r.astype(float).apply(lambda x: float(x))).pipe(lambda s: (s.astype(float).apply(lambda x: x))).pct_change()
    # Use log returns if safer; but for small deltas, pct_change is fine
    try:
        lr = (r.astype(float)).apply(lambda x: float(x)).pipe(lambda s: (s.astype(float))).apply(lambda x: x)
        import numpy as _np
        ret = (pd.Series(_np.log(r)).diff())
    except Exception:
        pass
    std = float(ret.dropna().std()) if len(ret.dropna()) else 0.0
    # Intraday shape: hour-of-day volatility multipliers (normalized to mean 1)
    real['timestamp'] = pd.to_datetime(real['timestamp'], utc=True, errors='coerce')
    real_ret = pd.Series((pd.Series(pd.to_numeric(real['close'], errors='coerce')).dropna()).pipe(lambda s: pd.Series(_np.log(s)).diff())) if 'close' in real.columns else pd.Series(dtype=float)
    try:
        real['ret'] = pd.Series(_np.log(pd.to_numeric(real['close'], errors='coerce'))).diff()
        real['hour'] = real['timestamp'].dt.hour
        g = real.dropna(subset=['ret']).groupby('hour')['ret'].std()
        if not g.empty and g.mean() and g.mean() > 0:
            shape = (g / g.mean()).to_dict()
        else:
            shape = {}
    except Exception:
        shape = {}
    return std, {"interval_used": real.attrs.get('interval_used'), "shape": shape}


def generate_db(db_path: str, symbol: str, start: datetime, end: datetime,
                daily_vol: float = 0.02, base_spread_bps: float = 8.0,
                calibrate_vol: bool = False, intraday_shape: bool = False,
                events_csv: Optional[str] = None, impact_strength: float = 0.05,
                impact_decay_minutes: int = 60) -> None:
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

    # Derive per-minute volatility from real series if requested
    per_minute_std = None
    vol_meta = {}
    if calibrate_vol:
        per_minute_std, vol_meta = _calibrate_per_minute_vol(symbol, start, end)

    # Prepare news impact events (optional)
    events_df = _load_events_csv(events_csv)

    # Simulate minute-by-minute mid prices using a process with mild mean reversion
    mid = seed_price
    # Use trading minutes per day if not calibrating
    trading_minutes_per_day = 390.0
    dt_years = 1.0 / (252 * trading_minutes_per_day)  # per-minute step in years over trading time
    minutes = int((end - start).total_seconds() // 60)
    for i in range(minutes + 1):
        ts = start + timedelta(minutes=i)
        # mean reversion toward seed_price helps stability over short windows
        mean_reversion_strength = 0.05
        mr = mean_reversion_strength * (seed_price - mid) / max(seed_price, 1e-6)
        if per_minute_std and per_minute_std > 0:
            # Optionally modulate by intraday shape
            mult = 1.0
            if intraday_shape and 'shape' in vol_meta and isinstance(vol_meta['shape'], dict):
                mult = float(vol_meta['shape'].get(ts.hour, 1.0))
            shock = random.gauss(0.0, per_minute_std * mult)
        else:
            # Fall back to daily_vol parameter mapped to per-minute using trading minutes
            shock = random.gauss(0.0, daily_vol / math.sqrt(trading_minutes_per_day))
        # News-driven drift (optional): sum decayed influences of events around current ts
        drift = 0.0
        if not events_df.empty:
            window_start = ts - timedelta(minutes=impact_decay_minutes * 3)
            sub = events_df[(events_df['timestamp'] <= ts) & (events_df['timestamp'] >= window_start)]
            if not sub.empty:
                # exponential decay by minutes since event
                dtm = (ts - sub['timestamp']).dt.total_seconds() / 60.0
                weight = (-(dtm / max(1.0, float(impact_decay_minutes)))).apply(lambda x: math.exp(x))
                sent = pd.to_numeric(sub.get('sentiment_score', 0.0), errors='coerce').fillna(0.0)
                conf = pd.to_numeric(sub.get('confidence', 0.5), errors='coerce').fillna(0.5)
                influence = (sent * conf * weight).sum()
                drift = impact_strength * float(influence)
        # Apply combined update
        mid = max(0.01, mid * (1.0 + mr * dt_years + drift + shock))
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
    if calibrate_vol:
        print(f"Per-minute std (real): {per_minute_std}")
        print(f"Vol meta: {vol_meta}")


def main() -> int:
    p = argparse.ArgumentParser(description="Generate seeded orderbook snapshots DB aligned to real start price")
    p.add_argument('--db', required=True, help='Output SQLite DB path')
    p.add_argument('--symbol', required=True, help='Symbol, e.g., AAPL')
    p.add_argument('--start', default=None, help='UTC ISO start (default: now-6h)')
    p.add_argument('--end', default=None, help='UTC ISO end (default: now)')
    p.add_argument('--hours', type=int, default=6, help='If start/end not provided, use now-hours..now')
    p.add_argument('--daily-vol', type=float, default=0.02, help='Assumed daily volatility (used if not calibrating)')
    p.add_argument('--spread-bps', type=float, default=8.0, help='Base spread in bps')
    p.add_argument('--calibrate-vol', action='store_true', help='Calibrate per-minute volatility to real OHLCV')
    p.add_argument('--intraday-shape', action='store_true', help='Apply intraday volatility U-shape based on real data')
    p.add_argument('--events-csv', default=None, help='Path to Yahoo/LLM events CSV to drive news impact')
    p.add_argument('--impact-strength', type=float, default=0.05, help='Scale for news drift contribution')
    p.add_argument('--impact-decay-minutes', type=int, default=60, help='Half-life scale for news impact decay')
    args = p.parse_args()

    now = datetime.now(timezone.utc)
    if args.start and args.end:
        start = pd.to_datetime(args.start, utc=True).to_pydatetime()
        end = pd.to_datetime(args.end, utc=True).to_pydatetime()
    else:
        end = now
        start = now - timedelta(hours=args.hours)

    generate_db(
        args.db,
        args.symbol,
        start,
        end,
        daily_vol=args.daily_vol,
        base_spread_bps=args.spread_bps,
        calibrate_vol=args.calibrate_vol,
        intraday_shape=args.intraday_shape,
        events_csv=args.events_csv,
        impact_strength=args.impact_strength,
        impact_decay_minutes=args.impact_decay_minutes,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

