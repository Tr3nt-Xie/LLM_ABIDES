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
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import pandas as pd
import yfinance as yf
import re
from urllib.parse import urlparse


@dataclass
class MarketFetchConfig:
	symbol: str
	start: datetime
	end: datetime
	interval: str = "1m"
	allow_fallback: bool = True  # fall back to coarser intervals if needed
	prepost: bool = True  # include pre/post market data when available



def _yf_history(symbol: str, start: datetime, end: datetime, interval: str, prepost: bool) -> pd.DataFrame:
	t = yf.Ticker(symbol)
	df = t.history(start=start, end=end, interval=interval, actions=False, prepost=prepost)
	if df is None:
		return pd.DataFrame()
	return df


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
	if df is None or df.empty:
		return pd.DataFrame()
	df = df.rename(columns=str.lower)
	keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
	df = df[keep_cols]
	df = df.reset_index()
	ts_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])
	df = df.rename(columns={ts_col: "timestamp"})
	df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
	return df


def fetch_intraday_ohlcv(cfg: MarketFetchConfig) -> pd.DataFrame:
	"""Fetch OHLCV with optional interval fallback: 1m -> 5m -> 1d."""
	intervals = [cfg.interval]
	if cfg.allow_fallback:
		for alt in ("5m", "15m", "1d"):
			if alt not in intervals:
				intervals.append(alt)
	for itv in intervals:
		raw = _yf_history(cfg.symbol, cfg.start, cfg.end, itv, cfg.prepost)
		df = _normalize_history(raw)
		if not df.empty:
			# Annotate interval used
			df.attrs["interval_used"] = itv
			return df
	return pd.DataFrame()


def fetch_recent_news(symbol: str, max_items: int = 50) -> pd.DataFrame:
	ticker = yf.Ticker(symbol)
	news_items = ticker.news or []
	rows: List[Dict[str, Any]] = []
	for item in news_items[:max_items]:
		ts = item.get("providerPublishTime")
		ts_dt = pd.to_datetime(ts, unit="s", utc=True) if ts is not None else pd.NaT
		# Fallback: infer timestamp from link if providerPublishTime missing
		if pd.isna(ts_dt):
			inferred = _infer_timestamp_from_link(item.get("link"))
			ts_dt = inferred if inferred is not None else pd.NaT
		rows.append(
			{
				"timestamp": ts_dt,
				"title": item.get("title"),
				"publisher": item.get("publisher"),
				"link": item.get("link"),
				"type": item.get("type"),
			}
		)
	df = pd.DataFrame(rows)
	# As a last resort, assign synthetic, ordered timestamps to any remaining NaT to preserve ordering
	if not df.empty and "timestamp" in df.columns:
		na_idx = df.index[df["timestamp"].isna()].tolist()
		if na_idx:
			now = pd.Timestamp.utcnow()
			for i, idx in enumerate(na_idx):
				# Space them 5 minutes apart backwards to maintain order stability
				df.at[idx, "timestamp"] = now - pd.Timedelta(minutes=5 * i)
	return df


def _infer_timestamp_from_link(link: Optional[str]) -> Optional[pd.Timestamp]:
	"""Attempt to infer a publish date from a news link.

	Supports common patterns like /YYYY/MM/DD/ or -YYYY-MM-DD- in paths.
	Returns a UTC pd.Timestamp at 16:00:00 for the parsed date.
	"""
	if not link:
		return None
	try:
		u = urlparse(link)
		path = u.path or ""
		# Pattern 1: /YYYY/MM/DD/
		m = re.search(r"/(20\d{2})/(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/", path)
		if m:
			y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
			return pd.Timestamp(year=y, month=mth, day=d, hour=16, tz="UTC")
		# Pattern 2: -YYYY-MM-DD or _YYYY-MM-DD in path
		m = re.search(r"[-_](20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])", path)
		if m:
			y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
			return pd.Timestamp(year=y, month=mth, day=d, hour=16, tz="UTC")
		# Pattern 3: YYYYMMDD in path segments
		m = re.search(r"/(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(/|\b)", path)
		if m:
			y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
			return pd.Timestamp(year=y, month=mth, day=d, hour=16, tz="UTC")
		return None
	except Exception:
		return None


def fetch_price_at_timestamp(symbol: str, at_utc: datetime, interval: str = "1m", allow_fallback: bool = True, prepost: bool = True) -> Tuple[Optional[float], Dict[str, Any]]:
	"""Fetch the real-world price nearest to a UTC timestamp.
	Returns (price, metadata) where price may be None if unavailable.
	Metadata includes: interval_used, source_row_timestamp.
	"""
	# Build a small window around the timestamp
	at = pd.to_datetime(at_utc, utc=True)
	window_start = (at - pd.Timedelta(days=2)).to_pydatetime()
	window_end = (at + pd.Timedelta(days=1)).to_pydatetime()
	cfg = MarketFetchConfig(symbol=symbol, start=window_start, end=window_end, interval=interval, allow_fallback=allow_fallback, prepost=prepost)
	df = fetch_intraday_ohlcv(cfg)
	if df is None or df.empty:
		return None, {"interval_used": None, "source_row_timestamp": None}
	# Find the last bar at or before the timestamp; if none, take the earliest after
	df = df.sort_values('timestamp')
	df_before = df[df['timestamp'] <= at]
	row = df_before.iloc[-1] if not df_before.empty else (df.iloc[0] if len(df) else None)
	if row is None:
		return None, {"interval_used": df.attrs.get('interval_used'), "source_row_timestamp": None}
	price = float(row['close']) if 'close' in row else None
	meta = {"interval_used": df.attrs.get('interval_used'), "source_row_timestamp": pd.to_datetime(row['timestamp']).isoformat()}
	return price, meta


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
	allow_fallback: bool = True,
) -> Tuple[Dict[str, Any], str]:
	cfg = MarketFetchConfig(symbol=symbol, start=start, end=end, interval=interval, allow_fallback=allow_fallback)
	ohlcv = fetch_intraday_ohlcv(cfg)
	merged = align_simulation_with_real_market(sim_trades, ohlcv)
	summary = basic_validation_report(merged)
	used = ohlcv.attrs.get("interval_used", interval) if isinstance(ohlcv, pd.DataFrame) else interval
	return {"ohlcv": ohlcv, "merged": merged, "summary": summary}, used

