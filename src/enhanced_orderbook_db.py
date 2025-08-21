#!/usr/bin/env python3
"""
Enhanced Order Book DB (Minimal Implementation)
==============================================

Provides:
- EnhancedOrderBookConfig: configuration for generation and storage
- EnhancedOrderBookDB: generates synthetic order book data, writes to SQLite, and exports DataFrames

This minimal version is designed to satisfy the interface used by
`enhanced_orderbook_main.py` without requiring ABIDES.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class EnhancedOrderBookConfig:
	"""Configuration for the enhanced order book generator."""
	db_path: str = "orderbook.db"
	num_agents: int = 1000
	simulation_days: int = 1
	symbols: List[str] = field(default_factory=lambda: ["AAPL", "GOOGL"])
	base_orders_per_minute: int = 100
	use_in_memory: bool = False
	# Optional features (not used extensively in this minimal version)
	enable_momentum_trading: bool = True
	enable_mean_reversion: bool = True
	enable_news_impact: bool = True
	enable_intraday_patterns: bool = True
	# Trading hours and timing
	trading_hours_start: int = 9
	trading_hours_end: int = 16
	# Optional explicit start time in UTC (ISO-8601 string)
	simulation_start_utc: Optional[str] = None
	# Random seed for reproducibility
	seed: int = 42


class EnhancedOrderBookDB:
	"""Synthetic order book generator with SQLite persistence."""

	def __init__(self, config: EnhancedOrderBookConfig):
		self.config = config
		np.random.seed(config.seed)

		# Determine database path
		self._db_path = ":memory:" if config.use_in_memory else config.db_path
		self._conn = sqlite3.connect(self._db_path)
		self._ensure_schema()

		# Internal cache of generated dataframes to avoid re-reading
		self._cache: Dict[str, pd.DataFrame] = {}

	def _ensure_schema(self) -> None:
		"""Create tables if they do not exist."""
		cursor = self._conn.cursor()
		cursor.execute(
			"""
			CREATE TABLE IF NOT EXISTS orders (
				order_id TEXT PRIMARY KEY,
				timestamp TEXT,
				agent_type TEXT,
				symbol TEXT,
				side TEXT,
				order_type TEXT,
				price REAL,
				quantity INTEGER
			);
			"""
		)
		cursor.execute(
			"""
			CREATE TABLE IF NOT EXISTS trades (
				trade_id TEXT PRIMARY KEY,
				timestamp TEXT,
				symbol TEXT,
				price REAL,
				quantity INTEGER,
				market_impact REAL
			);
			"""
		)
		cursor.execute(
			"""
			CREATE TABLE IF NOT EXISTS snapshots (
				timestamp TEXT,
				symbol TEXT,
				best_bid REAL,
				best_ask REAL,
				mid_price REAL,
				last_trade_price REAL,
				spread REAL
			);
			"""
		)
		self._conn.commit()

	def _generate_time_index(self) -> pd.DatetimeIndex:
		"""Generate a 1-minute time index across simulation days in local naive time."""
		# Start reference in UTC or now
		if self.config.simulation_start_utc:
			start = pd.to_datetime(self.config.simulation_start_utc)
		else:
			start = pd.Timestamp.utcnow().floor("min")
		# Use only trading hours window per day
		minutes_per_day = (self.config.trading_hours_end - self.config.trading_hours_start) * 60
		full_range = pd.date_range(start=start, periods=self.config.simulation_days * minutes_per_day, freq="1min")
		return full_range

	def _simulate_for_symbol(self, symbol: str, time_index: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
		"""Create synthetic orders, trades, and snapshots for a single symbol."""
		base_price = 100.0 + np.random.uniform(-2.0, 2.0)
		vol_per_min = 0.0008  # per-minute volatility

		# Simple drift-decay process
		noise = np.random.normal(0.0, vol_per_min, size=len(time_index)).cumsum()
		prices = base_price * (1.0 + noise)
		prices = np.maximum(1e-4, prices)

		# Bid/ask from spread in bps (typical 5 bps with variability)
		spread_bps = np.clip(np.random.normal(5.0, 2.0, size=len(time_index)), 1.0, 25.0)
		spreads = prices * (spread_bps / 10000.0)
		best_bid = prices - spreads / 2.0
		best_ask = prices + spreads / 2.0

		# Orders per minute (Poisson around base_orders_per_minute / number of symbols)
		orders_per_minute_mean = max(1, int(self.config.base_orders_per_minute / max(1, len(self.config.symbols))))
		orders_records = []
		trades_records = []
		agent_types = ["retail", "institutional", "hft", "market_maker"]

		order_counter = 0
		trade_counter = 0
		last_trade_price = None

		for idx, ts in enumerate(time_index):
			# Burstiness: opening/closing higher activity if enabled
			hour = pd.Timestamp(ts).hour
			activity_multiplier = 1.0
			if self.config.enable_intraday_patterns:
				if hour in (self.config.trading_hours_start, self.config.trading_hours_end - 1):
					activity_multiplier = 2.0
				elif hour == (self.config.trading_hours_start + self.config.trading_hours_end) // 2:
					activity_multiplier = 0.6

			n_orders = np.random.poisson(orders_per_minute_mean * activity_multiplier)
			if n_orders == 0:
				continue

			# Generate orders
			for _ in range(n_orders):
				order_counter += 1
				order_id = f"{symbol}_ORD_{order_counter:08d}"
				side = np.random.choice(["BUY", "SELL"])  # order side
				order_type = np.random.choice(["LIMIT", "MARKET"], p=[0.7, 0.3])
				quantity = int(np.random.lognormal(mean=5.5, sigma=0.8))  # heavy tailed
				quantity = max(1, min(quantity, 100000))
				# Limit prices around book, market orders cross at current mid
				if order_type == "LIMIT":
					# place on respective side within a few ticks
					price_offset = np.random.uniform(0, spreads[idx])
					price = best_bid[idx] - price_offset if side == "BUY" else best_ask[idx] + price_offset
				else:
					price = prices[idx]

				agent_type = np.random.choice(agent_types, p=[0.6, 0.25, 0.1, 0.05])
				orders_records.append(
					(
						order_id,
						pd.Timestamp(ts).isoformat(),
						agent_type,
						symbol,
						side,
						order_type,
						float(price),
						int(quantity),
					)
				)

				# Convert a fraction of orders to trades to simulate execution
				if order_type == "MARKET" or np.random.rand() < 0.05:
					trade_counter += 1
					trade_id = f"{symbol}_TRD_{trade_counter:08d}"
					trade_price = float(prices[idx] + np.random.normal(0, spreads[idx] / 6.0))
					trade_qty = int(max(1, np.random.normal(quantity * 0.5, quantity * 0.2)))
					market_impact = float(abs(np.random.exponential(scale=0.0001)))  # in fractional returns
					last_trade_price = trade_price
					trades_records.append(
						(
							trade_id,
							pd.Timestamp(ts).isoformat(),
							symbol,
							trade_price,
							trade_qty,
							market_impact,
						)
					)

		# Snapshots: once per 5 minutes to keep size reasonable
		snapshot_every = 5
		snap_indices = np.arange(0, len(time_index), snapshot_every)
		snapshots_records = []
		for i in snap_indices:
			mid = prices[i]
			bb = best_bid[i]
			ba = best_ask[i]
			ltp = last_trade_price if last_trade_price is not None else mid
			spread_val = ba - bb
			snapshots_records.append(
				(
					pd.Timestamp(time_index[i]).isoformat(),
					symbol,
					float(bb),
					float(ba),
					float(mid),
					float(ltp),
					float(spread_val),
				)
			)

		orders_df = pd.DataFrame(
			orders_records,
			columns=[
				"order_id",
				"timestamp",
				"agent_type",
				"symbol",
				"side",
				"order_type",
				"price",
				"quantity",
			],
		)
		trades_df = pd.DataFrame(
			trades_records,
			columns=["trade_id", "timestamp", "symbol", "price", "quantity", "market_impact"],
		)
		snapshots_df = pd.DataFrame(
			snapshots_records,
			columns=["timestamp", "symbol", "best_bid", "best_ask", "mid_price", "last_trade_price", "spread"],
		)

		return {"orders": orders_df, "trades": trades_df, "snapshots": snapshots_df}

	def generate_simulation_data(self) -> Dict[str, object]:
		"""Generate synthetic data, persist to SQLite, and return a summary."""
		start_time = time.time()

		# Generate time index for trading minutes across days
		time_index = self._generate_time_index()

		# Generate per symbol and concatenate
		all_orders = []
		all_trades = []
		all_snaps = []
		for symbol in self.config.symbols:
			data = self._simulate_for_symbol(symbol, time_index)
			all_orders.append(data["orders"]) if not data["orders"].empty else None
			all_trades.append(data["trades"]) if not data["trades"].empty else None
			all_snaps.append(data["snapshots"]) if not data["snapshots"].empty else None

		orders_df = pd.concat(all_orders, ignore_index=True) if all_orders else pd.DataFrame()
		trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
		snapshots_df = pd.concat(all_snaps, ignore_index=True) if all_snaps else pd.DataFrame()

		# Persist
		orders_df.to_sql("orders", self._conn, if_exists="replace", index=False)
		trades_df.to_sql("trades", self._conn, if_exists="replace", index=False)
		snapshots_df.to_sql("snapshots", self._conn, if_exists="replace", index=False)
		self._conn.commit()

		# Cache for quick export
		self._cache = {"orders": orders_df, "trades": trades_df, "snapshots": snapshots_df}

		elapsed = max(1e-6, time.time() - start_time)
		summary = {
			"total_orders": int(len(orders_df)),
			"total_trades": int(len(trades_df)),
			"total_snapshots": int(len(snapshots_df)),
			"symbols": list(self.config.symbols),
			"orders_per_second": float(len(orders_df) / elapsed),
			"db_path": self._db_path,
		}
		return summary

	def export_data_analysis(self) -> Dict[str, pd.DataFrame]:
		"""Return DataFrames for analysis. Loads from DB if not cached."""
		if self._cache:
			return self._cache

		orders_df = pd.read_sql_query("SELECT * FROM orders", self._conn)
		trades_df = pd.read_sql_query("SELECT * FROM trades", self._conn)
		snapshots_df = pd.read_sql_query("SELECT * FROM snapshots", self._conn)
		self._cache = {"orders": orders_df, "trades": trades_df, "snapshots": snapshots_df}
		return self._cache

	def close(self) -> None:
		try:
			self._conn.close()
		except Exception:
			pass