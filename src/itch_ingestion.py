#!/usr/bin/env python3
"""
NASDAQ ITCH Ingestion Utilities
===============================

Lightweight helpers to load NASDAQ ITCH trade data and aggregate it to
per-second price series for comparison with simulated executions.

Notes
-----
- This module prioritizes CSV ingestion of pre-parsed ITCH datasets to avoid
  heavy binary parsers and external dependencies. If you have raw ITCH binary
  files, convert them to CSV first (e.g., using vendor tools) or implement the
  stubbed binary loader below.
- Expected CSV columns (flexible):
  - timestamp (or: time, ts, datetime, date/time parts)
  - symbol (case-sensitive ticker)
  - price (float, in dollars)
  - shares or size (optional; for VWAP)
  - type/event_type (optional; if present, rows should include trade events)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class ITCHLoadConfig:
    path: str
    symbol: Optional[str] = None
    fmt: str = "csv"  # 'csv' | 'binary' (binary is stubbed)
    tz_local: str = "America/New_York"  # ITCH timestamps are in exchange time
    assume_local_naive: bool = True      # If timestamps have no tz, localize then convert to UTC
    price_col: str = "price"
    size_col: str = "size"
    timestamp_col: Optional[str] = None  # If None, auto-detect
    type_col: Optional[str] = None       # e.g., 'type' / 'event_type'
    trade_type_values: Optional[set[str]] = None  # e.g., {'P','Q','TRADE'}; if None, do not filter


def _coerce_timestamp_series(
    s: pd.Series,
    tz_local: str,
    assume_local_naive: bool,
) -> pd.Series:
    """Coerce a timestamp-like Series into UTC-aware pandas Timestamps.

    Handles common cases:
    - ISO8601 strings
    - Integer/float epoch in ns/us/ms/s
    - Naive datetimes interpreted as tz_local if assume_local_naive
    """
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        # Try to infer epoch unit by magnitude (do not force Int64 for floats)
        numeric = s.dropna().astype(float)
        maxv = float(numeric.max()) if not numeric.empty else 0.0
        if maxv > 1e14:
            # nanoseconds
            ts = pd.to_datetime(s, unit="ns", utc=True, errors="coerce")
        elif maxv > 1e11:
            # milliseconds
            ts = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
        elif maxv > 1e9:
            # seconds (with possible fractional part)
            ts = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
        else:
            # Likely seconds; if offset-from-midnight, this will be NaT
            ts = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
        return ts

    # String or datetime-like
    ts = pd.to_datetime(s, utc=False, errors="coerce")
    # If tz-naive and assume_local_naive, localize to exchange tz then convert to UTC
    if assume_local_naive:
        # If series is tz-naive, localize to exchange tz then convert to UTC
        if getattr(ts.dt, "tz", None) is None:
            try:
                ts = ts.dt.tz_localize(tz_local).dt.tz_convert("UTC")
            except Exception:
                # If localization fails, try parsing with utc=True as fallback
                ts = pd.to_datetime(s, utc=True, errors="coerce")
        else:
            # Series already has tz; convert to UTC
            try:
                ts = ts.dt.tz_convert("UTC")
            except Exception:
                ts = pd.to_datetime(s, utc=True, errors="coerce")
    else:
        # Assume incoming has tz info or is already UTC; if naive, coerce to UTC
        try:
            if getattr(ts.dt, "tz", None) is None:
                ts = ts.dt.tz_localize("UTC")
            else:
                ts = ts.dt.tz_convert("UTC")
        except Exception:
            ts = pd.to_datetime(s, utc=True, errors="coerce")
    return ts


def _detect_timestamp_column(df: pd.DataFrame, hint: Optional[str]) -> Optional[str]:
    if hint and hint in df.columns:
        return hint
    candidates = [
        "timestamp", "ts", "time", "datetime", "date_time",
        "Timestamp", "Time", "Datetime", "DateTime",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Try split date/time columns
    if "date" in df.columns and ("time" in df.columns or "nanoseconds" in df.columns or "ns" in df.columns):
        return None  # will be constructed
    return None


def _construct_timestamp_from_parts(df: pd.DataFrame, tz_local: str) -> Optional[pd.Series]:
    """Construct timestamp from date + time/nanoseconds parts if present."""
    if "date" in df.columns and "time" in df.columns:
        try:
            dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
            dt = dt.dt.tz_localize(tz_local).dt.tz_convert("UTC")
            return dt
        except Exception:
            pass
    # ITCH often uses ns offset from midnight with a trading date
    ns_col = None
    for c in ("nanoseconds", "ns", "time_ns", "offset_ns"):
        if c in df.columns:
            ns_col = c
            break
    if ns_col is not None and "date" in df.columns:
        try:
            base = pd.to_datetime(df["date"].astype(str), errors="coerce")
            base = base.dt.tz_localize(tz_local)
            delta = pd.to_timedelta(df[ns_col].astype("Int64"), unit="ns")
            dt = (base + delta).dt.tz_convert("UTC")
            return dt
        except Exception:
            return None
    return None


def load_itch_trades_csv(cfg: ITCHLoadConfig) -> pd.DataFrame:
    """Load ITCH trades from a CSV file and return a normalized DataFrame.

    Columns in the returned frame:
    - timestamp (UTC, tz-aware)
    - symbol
    - price (float)
    - size (optional; float)
    """
    df = pd.read_csv(cfg.path)

    # Normalize column names for flexible matching
    original_columns: Dict[str, str] = {c: c for c in df.columns}
    cols_lower = {c.lower(): c for c in df.columns}

    # Determine timestamp
    ts_col = _detect_timestamp_column(df, cfg.timestamp_col)
    if ts_col is None:
        ts_series = _construct_timestamp_from_parts(df, cfg.tz_local)
        if ts_series is None:
            # Try any numeric time-like column as epoch
            for candidate in ("epoch", "ts", "time", "timestamp", "epochns", "epoch_ns"):
                if candidate in df.columns:
                    ts_series = _coerce_timestamp_series(df[candidate], cfg.tz_local, cfg.assume_local_naive)
                    break
        if ts_series is None:
            raise ValueError("Could not detect/construct timestamp from CSV. Provide timestamp_col in config.")
        df["timestamp"] = ts_series
    else:
        df["timestamp"] = _coerce_timestamp_series(df[ts_col], cfg.tz_local, cfg.assume_local_naive)

    # Price column
    price_col = cfg.price_col if cfg.price_col in df.columns else None
    if price_col is None:
        for c in ("price", "trade_price", "px", "Price", "PRICE"):
            if c in df.columns:
                price_col = c
                break
    if price_col is None:
        raise ValueError("No price column found in ITCH CSV")
    df["price"] = df[price_col].astype(float)

    # Size/volume
    size_col = cfg.size_col if cfg.size_col in df.columns else None
    if size_col is None:
        for c in ("size", "shares", "quantity", "vol", "volume", "Size", "Shares"):
            if c in df.columns:
                size_col = c
                break
    if size_col is not None:
        df["size"] = df[size_col].astype(float)
    else:
        df["size"] = np.nan

    # Symbol
    sym_col = None
    for c in ("symbol", "Symbol", "ticker", "Ticker"):
        if c in df.columns:
            sym_col = c
            break
    if sym_col is None:
        raise ValueError("No symbol column found in ITCH CSV")
    df["symbol"] = df[sym_col].astype(str)

    # Optional filtering by event type
    if cfg.type_col and cfg.type_col in df.columns and cfg.trade_type_values:
        df = df[df[cfg.type_col].astype(str).str.upper().isin({v.upper() for v in cfg.trade_type_values})]

    # Filter symbol
    if cfg.symbol is not None:
        df = df[df["symbol"] == cfg.symbol]

    # Keep only needed columns
    out = df[["timestamp", "symbol", "price", "size"]].copy()
    out = out.dropna(subset=["timestamp", "price"]).sort_values("timestamp")
    return out


def load_itch_trades_binary(cfg: ITCHLoadConfig) -> pd.DataFrame:
    """Stub for raw ITCH binary loading.

    For now, raise an informative error to encourage CSV usage.
    """
    raise NotImplementedError(
        "Binary ITCH parsing is not implemented. Convert to CSV first or add a parser."
    )


def load_itch_trades(cfg: ITCHLoadConfig) -> pd.DataFrame:
    if cfg.fmt.lower() == "csv":
        return load_itch_trades_csv(cfg)
    elif cfg.fmt.lower() in ("bin", "binary", "itch"):
        return load_itch_trades_binary(cfg)
    else:
        raise ValueError(f"Unknown ITCH format: {cfg.fmt}")


def aggregate_trades_to_seconds(
    trades: pd.DataFrame,
    agg: str = "last",  # 'last' | 'vwap' | 'median'
    price_col: str = "price",
    size_col: str = "size",
) -> pd.DataFrame:
    """Aggregate tick trades to per-second price series.

    Returns a DataFrame with columns: timestamp, price
    """
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["timestamp", "price"]) 
    df = trades.copy()
    if "timestamp" not in df.columns:
        raise ValueError("trades DataFrame must include 'timestamp'")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df["second"] = df["timestamp"].dt.floor("1s")

    if agg == "last":
        # Last trade price within each second
        sec = df.groupby("second")[price_col].last().rename("price").reset_index()
    elif agg == "vwap":
        # Volume-weighted average price within each second
        # Fallback to simple mean if size is missing
        has_size = size_col in df.columns and df[size_col].notna().any()
        if has_size:
            tmp = df[["second", price_col, size_col]].copy()
            tmp["wx"] = tmp[price_col] * tmp[size_col].astype(float)
            agg_df = tmp.groupby("second")[ ["wx", size_col] ].sum(min_count=1)
            sec = (agg_df["wx"] / agg_df[size_col]).rename("price").reset_index()
        else:
            sec = df.groupby("second")[price_col].mean().rename("price").reset_index()
    elif agg == "median":
        sec = df.groupby("second")[price_col].median().rename("price").reset_index()
    else:
        raise ValueError("Unsupported agg; use 'last' | 'vwap' | 'median'")

    sec = sec.rename(columns={"second": "timestamp"})
    sec["timestamp"] = pd.to_datetime(sec["timestamp"], utc=True)
    return sec[["timestamp", "price"]]

