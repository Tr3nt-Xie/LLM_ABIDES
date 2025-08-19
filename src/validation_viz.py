#!/usr/bin/env python3
"""
Validation Visualization
========================

Generate comparative plots between simulated market data (from the enhanced
orderbook SQLite DB) and real market OHLCV (via yfinance) to validate the
simulation quality against stylized facts and basic benchmarks.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
except Exception:
    sns = None

from real_data_ingestion import (
    MarketFetchConfig,
    fetch_intraday_ohlcv,
)


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_simulated_frames(db_path: str, symbol: str) -> Dict[str, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    orders = pd.read_sql(
        f"SELECT * FROM orders WHERE symbol = '{symbol}'", conn
    ) if _table_exists(conn, 'orders') else pd.DataFrame()
    trades = pd.read_sql(
        f"SELECT * FROM trades WHERE symbol = '{symbol}'", conn
    ) if _table_exists(conn, 'trades') else pd.DataFrame()
    snapshots = pd.read_sql(
        f"SELECT * FROM orderbook_snapshots WHERE symbol = '{symbol}'", conn
    ) if _table_exists(conn, 'orderbook_snapshots') else pd.DataFrame()
    conn.close()
    return {"orders": orders, "trades": trades, "snapshots": snapshots}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return pd.read_sql(q, conn, params=(name,)).shape[0] > 0


def plot_price_timeseries(sim_snap: pd.DataFrame, real_ohlcv: pd.DataFrame, out: Path, symbol: str) -> None:
    if sim_snap.empty or real_ohlcv.empty:
        return
    sim = sim_snap.copy()
    sim['timestamp'] = pd.to_datetime(sim['timestamp'], utc=True, errors='coerce')
    sim = sim.dropna(subset=['timestamp'])
    # Prefer mid_price if available, else best bid/ask mid
    if 'mid_price' in sim.columns:
        sim_price = sim[['timestamp', 'mid_price']].rename(columns={'mid_price': 'price'})
    else:
        sim['price'] = (sim.get('best_bid') + sim.get('best_ask')) / 2.0
        sim_price = sim[['timestamp', 'price']]

    real = real_ohlcv[['timestamp', 'close']].copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sim_price['timestamp'], sim_price['price'], label=f'Sim {symbol} (mid)', alpha=0.7)
    ax.plot(real['timestamp'], real['close'], label=f'Real {symbol} (close)', alpha=0.7)
    ax.set_title(f"Price Timeseries: {symbol} (Sim vs Real)")
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Price')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"price_timeseries_{symbol}.png", dpi=120)
    plt.close(fig)


def plot_return_distributions(sim_snap: pd.DataFrame, real_ohlcv: pd.DataFrame, out: Path, symbol: str) -> None:
    if sim_snap.empty or real_ohlcv.empty:
        return
    sim = sim_snap.copy()
    sim['timestamp'] = pd.to_datetime(sim['timestamp'], utc=True, errors='coerce')
    sim = sim.dropna(subset=['timestamp'])
    if 'mid_price' in sim.columns:
        s = sim['mid_price'].astype(float)
    else:
        s = ((sim.get('best_bid') + sim.get('best_ask')) / 2.0).astype(float)
    sim_ret = np.log(s).diff().dropna()

    real = real_ohlcv['close'].astype(float)
    real_ret = np.log(real).diff().dropna()

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(sim_ret, bins=50, alpha=0.8)
    ax[0].set_title(f'Sim Returns (log) {symbol}')
    ax[1].hist(real_ret, bins=50, alpha=0.8, color='orange')
    ax[1].set_title(f'Real Returns (log) {symbol}')
    for a in ax:
        a.set_yscale('log')
        a.set_xlabel('Return')
        a.set_ylabel('Frequency (log)')
    fig.tight_layout()
    fig.savefig(out / f"return_distributions_{symbol}.png", dpi=120)
    plt.close(fig)


def plot_autocorrelations(sim_snap: pd.DataFrame, out: Path, symbol: str) -> None:
    if sim_snap.empty:
        return
    sim = sim_snap.copy()
    sim['timestamp'] = pd.to_datetime(sim['timestamp'], utc=True, errors='coerce')
    sim = sim.dropna(subset=['timestamp'])
    if 'mid_price' in sim.columns:
        s = sim['mid_price'].astype(float)
    else:
        s = ((sim.get('best_bid') + sim.get('best_ask')) / 2.0).astype(float)
    ret = np.log(s).diff().dropna()
    sq = (ret ** 2)

    def acf(series: pd.Series, lags: int = 20) -> np.ndarray:
        series = series.dropna().values
        m = series.mean()
        v = series.var()
        res = []
        for k in range(1, lags + 1):
            if len(series) > k:
                res.append(np.corrcoef(series[:-k], series[k:])[0, 1])
            else:
                res.append(np.nan)
        return np.array(res)

    lags = 20
    acf_ret = acf(ret, lags)
    acf_sq = acf(sq, lags)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(range(1, lags + 1), acf_ret)
    ax[0].set_title('Autocorr(Returns)')
    ax[0].set_xlabel('Lag')
    ax[0].set_ylabel('ACF')

    ax[1].bar(range(1, lags + 1), acf_sq, color='orange')
    ax[1].set_title('Autocorr(Squared Returns)')
    ax[1].set_xlabel('Lag')
    ax[1].set_ylabel('ACF')

    fig.suptitle(f"Autocorrelations: {symbol}")
    fig.tight_layout()
    fig.savefig(out / f"autocorr_{symbol}.png", dpi=120)
    plt.close(fig)


def plot_intraday_volume(sim_trades: pd.DataFrame, real_ohlcv: pd.DataFrame, out: Path, symbol: str) -> None:
    if sim_trades.empty or real_ohlcv.empty:
        return
    tr = sim_trades.copy()
    tr['timestamp'] = pd.to_datetime(tr['timestamp'], utc=True, errors='coerce')
    tr = tr.dropna(subset=['timestamp'])
    tr['hour'] = tr['timestamp'].dt.hour
    sim_vol_by_hour = tr.groupby('hour')['quantity'].sum()
    sim_vol_norm = sim_vol_by_hour / sim_vol_by_hour.mean() if sim_vol_by_hour.mean() else sim_vol_by_hour

    ro = real_ohlcv.copy()
    ro['hour'] = pd.to_datetime(ro['timestamp'], utc=True).dt.hour
    real_vol_by_hour = ro.groupby('hour')['volume'].sum()
    real_vol_norm = real_vol_by_hour / real_vol_by_hour.mean() if real_vol_by_hour.mean() else real_vol_by_hour

    hours = sorted(set(sim_vol_norm.index).union(real_vol_norm.index))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hours, [sim_vol_norm.get(h, np.nan) for h in hours], label='Sim (norm)', marker='o')
    ax.plot(hours, [real_vol_norm.get(h, np.nan) for h in hours], label='Real (norm)', marker='o')
    ax.set_title(f"Intraday Volume Pattern (normalized): {symbol}")
    ax.set_xlabel('Hour (UTC)')
    ax.set_ylabel('Volume / mean')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"intraday_volume_{symbol}.png", dpi=120)
    plt.close(fig)


def generate_plots(db_path: str, symbol: str, start: datetime, end: datetime, outdir: str) -> Dict[str, Any]:
    out = Path(outdir)
    _ensure_outdir(out)
    frames = load_simulated_frames(db_path, symbol)
    cfg = MarketFetchConfig(symbol=symbol, start=start, end=end, interval='1m')
    real = fetch_intraday_ohlcv(cfg)

    # Plots
    plot_price_timeseries(frames['snapshots'], real, out, symbol)
    plot_return_distributions(frames['snapshots'], real, out, symbol)
    plot_autocorrelations(frames['snapshots'], out, symbol)
    plot_intraday_volume(frames['trades'], real, out, symbol)
    return {"sim": frames, "real": real}


def main():
    p = argparse.ArgumentParser(description='Generate comparative validation plots (Sim vs Real)')
    p.add_argument('--db', required=True, help='Path to SQLite DB (enhanced orderbook)')
    p.add_argument('--symbol', required=True, help='Symbol to analyze (e.g., AAPL)')
    p.add_argument('--outdir', default='validation_plots', help='Directory to save plots')
    p.add_argument('--val-start', default=None, help='UTC ISO start (default: now-6h)')
    p.add_argument('--val-end', default=None, help='UTC ISO end (default: now)')
    args = p.parse_args()

    if args.val_start and args.val_end:
        start = pd.to_datetime(args.val_start, utc=True)
        end = pd.to_datetime(args.val_end, utc=True)
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=6)

    generate_plots(args.db, args.symbol, start, end, args.outdir)
    print(f"✅ Plots saved to: {args.outdir}")


if __name__ == '__main__':
    main()

