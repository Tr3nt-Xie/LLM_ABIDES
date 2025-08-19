#!/usr/bin/env python3
"""
Real Data Ingestion Utilities
=============================

Fetch real-world market OHLCV and recent news for validation of simulations.

Sources:
- yfinance: intraday OHLCV and recent news headlines per symbol
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import yfinance as yf


@dataclass
class MarketFetchConfig:
    symbol: str
    start: datetime
    end: datetime
    interval: str = "1m"


def fetch_intraday_ohlcv(cfg: MarketFetchConfig) -> pd.DataFrame:
    ticker = yf.Ticker(cfg.symbol)
    df = ticker.history(start=cfg.start, end=cfg.end, interval=cfg.interval, actions=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.lower)
    # Normalize to expected columns
    keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep_cols]
    # Move index to timestamp column, ensure tz-aware
    df = df.reset_index()
    ts_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def fetch_recent_news(symbol: str, max_items: int = 50) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    news_items = ticker.news or []
    rows: List[Dict[str, Any]] = []
    for item in news_items[:max_items]:
        ts = item.get("providerPublishTime")
        ts_dt = pd.to_datetime(ts, unit="s", utc=True) if ts is not None else pd.NaT
        rows.append(
            {
                "timestamp": ts_dt,
                "title": item.get("title"),
                "publisher": item.get("publisher"),
                "link": item.get("link"),
                "type": item.get("type"),
            }
        )
    return pd.DataFrame(rows)


def align_simulation_with_real_market(
    sim_trades: pd.DataFrame,
    real_ohlcv: pd.DataFrame,
    price_col: str = "close",
    on: str = "timestamp",
    tolerance: str = "1min",
) -> pd.DataFrame:
    if sim_trades is None or sim_trades.empty or real_ohlcv is None or real_ohlcv.empty:
        return pd.DataFrame()
    st = sim_trades.copy()
    rt = real_ohlcv.copy()
    st[on] = pd.to_datetime(st[on], utc=True, errors="coerce")
    rt[on] = pd.to_datetime(rt[on], utc=True, errors="coerce")
    st["_rounded_ts"] = st[on].dt.floor(tolerance)
    rt["_rounded_ts"] = rt[on].dt.floor(tolerance)
    merged = pd.merge(
        st,
        rt[["_rounded_ts", price_col]].rename(columns={price_col: "real_" + price_col}),
        on="_rounded_ts",
        how="left",
    )
    if "price" in merged.columns and "real_" + price_col in merged.columns:
        merged["price_error_bps"] = (merged["price"] - merged["real_" + price_col]) / merged["real_" + price_col] * 10000
    return merged


def basic_validation_report(merged: pd.DataFrame) -> Dict[str, Any]:
    if merged is None or merged.empty or "price_error_bps" not in merged.columns:
        return {"error": "No comparable data"}
    err = merged["price_error_bps"].dropna()
    if err.empty:
        return {"error": "No comparable data"}
    return {
        "num_points": int(len(err)),
        "mean_abs_error_bps": float(err.abs().mean()),
        "median_abs_error_bps": float(err.abs().median()),
        "p95_abs_error_bps": float(err.abs().quantile(0.95)),
        "bias_bps": float(err.mean()),
    }


def fetch_and_compare(
    symbol: str,
    sim_trades: pd.DataFrame,
    start: datetime,
    end: datetime,
    interval: str = "1m",
):
    cfg = MarketFetchConfig(symbol=symbol, start=start, end=end, interval=interval)
    ohlcv = fetch_intraday_ohlcv(cfg)
    merged = align_simulation_with_real_market(sim_trades, ohlcv)
    summary = basic_validation_report(merged)
    return {"ohlcv": ohlcv, "merged": merged, "summary": summary}

