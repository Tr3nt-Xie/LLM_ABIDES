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
from typing import Dict, Any, Tuple, Optional

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

from scipy import stats


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _filter_rth(df: pd.DataFrame, ts_col: str = 'timestamp') -> pd.DataFrame:
    """Filter to regular trading hours using America/New_York (DST-aware)."""
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    # Convert to US/Eastern for correct RTH handling across DST
    df[ts_col] = df[ts_col].dt.tz_convert('America/New_York')
    df = df.set_index(ts_col)
    df = df.between_time('09:30', '16:00')
    # Convert back to UTC
    df = df.reset_index()
    df[ts_col] = df[ts_col].dt.tz_convert('UTC')
    return df


def _resample_mid(df: pd.DataFrame, price_col: str, ts_col: str = 'timestamp', rule: str = '1min') -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    out = (df.set_index(ts_col)[price_col].resample(rule).last().dropna().to_frame('mid'))
    out = out.reset_index().rename(columns={ts_col: 'timestamp'})
    return out


def _ks_emd_bootstrap(sim_series: pd.Series, real_series: pd.Series, n_boot: int = 200, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    # KS statistic
    ks_stat, ks_p = stats.ks_2samp(sim_series.dropna(), real_series.dropna())
    # Approx EMD on 1D via sorted sample L1 distance (quantile matching)
    m = min(len(sim_series), len(real_series))
    if m == 0:
        return {'ks_stat': np.nan, 'ks_p': np.nan, 'emd': np.nan}
    s_sim = np.sort(sim_series.dropna().values)[:m]
    s_real = np.sort(real_series.dropna().values)[:m]
    emd = np.mean(np.abs(s_sim - s_real))
    # Bootstrap CIs
    ks_boot = []
    emd_boot = []
    for _ in range(n_boot):
        idx_s = rng.integers(0, m, size=m)
        idx_r = rng.integers(0, m, size=m)
        ks_b, _ = stats.ks_2samp(s_sim[idx_s], s_real[idx_r])
        emd_b = np.mean(np.abs(np.sort(s_sim[idx_s]) - np.sort(s_real[idx_r])))
        ks_boot.append(ks_b)
        emd_boot.append(emd_b)
    return {
        'ks_stat': float(ks_stat),
        'ks_p': float(ks_p),
        'ks_ci_95': (float(np.quantile(ks_boot, 0.025)), float(np.quantile(ks_boot, 0.975))),
        'emd': float(emd),
        'emd_ci_95': (float(np.quantile(emd_boot, 0.025)), float(np.quantile(emd_boot, 0.975))),
    }


def load_simulated_frames(db_path: str, symbol: str) -> Dict[str, pd.DataFrame]:
	conn = sqlite3.connect(db_path)
	# Enhanced schema
	orders = pd.read_sql(
		f"SELECT * FROM orders WHERE symbol = '{symbol}'", conn
	) if _table_exists(conn, 'orders') else pd.DataFrame()
	trades = pd.read_sql(
		f"SELECT * FROM trades WHERE symbol = '{symbol}'", conn
	) if _table_exists(conn, 'trades') else pd.DataFrame()
	snapshots = pd.read_sql(
		f"SELECT * FROM orderbook_snapshots WHERE symbol = '{symbol}'", conn
	) if _table_exists(conn, 'orderbook_snapshots') else pd.DataFrame()
	# Scaled schema fallbacks
	if trades.empty and _table_exists(conn, 'detailed_trades'):
		trades = pd.read_sql(
			f"SELECT timestamp, symbol, price, quantity, buy_order_id, sell_order_id, buy_agent_id, sell_agent_id FROM detailed_trades WHERE symbol = '{symbol}'",
			conn
		)
	if snapshots.empty and _table_exists(conn, 'lob_snapshots'):
		snapshots = pd.read_sql(
			f"SELECT timestamp, symbol, best_bid, best_ask, mid_price, bid_depth_json, ask_depth_json, spread FROM lob_snapshots WHERE symbol = '{symbol}'",
			conn
		)
	conn.close()
	return {"orders": orders, "trades": trades, "snapshots": snapshots}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return pd.read_sql(q, conn, params=(name,)).shape[0] > 0


def _infer_time_window_from_db(db_path: str, symbol: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
	"""Infer min/max timestamps for a symbol from snapshots or trades in the DB."""
	try:
		conn = sqlite3.connect(db_path)
		q1 = pd.read_sql(f"SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM orderbook_snapshots WHERE symbol='{symbol}'", conn)
		min_ts = pd.to_datetime(q1['min_ts'].iloc[0], utc=True) if not q1.empty else None
		max_ts = pd.to_datetime(q1['max_ts'].iloc[0], utc=True) if not q1.empty else None
		if min_ts is None or pd.isna(min_ts):
			q2 = pd.read_sql(f"SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM trades WHERE symbol='{symbol}'", conn)
			min_ts = pd.to_datetime(q2['min_ts'].iloc[0], utc=True) if not q2.empty else None
			max_ts = pd.to_datetime(q2['max_ts'].iloc[0], utc=True) if not q2.empty else None
		conn.close()
		return (min_ts, max_ts)
	except Exception:
		return (None, None)


def plot_price_timeseries(sim_snap: pd.DataFrame, real_ohlcv: pd.DataFrame, out: Path, symbol: str) -> None:
    if sim_snap.empty or real_ohlcv.empty:
        return
    sim = sim_snap.copy()
    sim['timestamp'] = pd.to_datetime(sim['timestamp'], utc=True, errors='coerce')
    sim = sim.dropna(subset=['timestamp'])
    # Prefer mid_price if available, else best bid/ask mid, else 'price' column
    if 'mid_price' in sim.columns:
        sim_price = sim[['timestamp', 'mid_price']].rename(columns={'mid_price': 'price'})
    elif {'best_bid', 'best_ask'}.issubset(sim.columns):
        sim['price'] = (sim.get('best_bid') + sim.get('best_ask')) / 2.0
        sim_price = sim[['timestamp', 'price']]
    elif 'price' in sim.columns:
        sim_price = sim[['timestamp', 'price']]
    else:
        return

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


def plot_price_timeseries_aligned(sim_snap: pd.DataFrame, real_ohlcv: pd.DataFrame, out: Path, symbol: str, resample_rule: str, apply_rth: bool) -> None:
	"""Plot sim vs real price aligned to the same cadence and RTH handling."""
	if sim_snap.empty or real_ohlcv.empty:
		return
	# Prepare sim mid series
	sim = sim_snap.copy()
	sim['timestamp'] = pd.to_datetime(sim['timestamp'], utc=True, errors='coerce')
	if apply_rth:
		sim = _filter_rth(sim, 'timestamp')
	if 'mid_price' in sim.columns:
		sim_mid = sim[['timestamp', 'mid_price']].rename(columns={'mid_price': 'mid'})
	elif {'best_bid', 'best_ask'}.issubset(sim.columns):
		sim['mid'] = (sim.get('best_bid') + sim.get('best_ask')) / 2.0
		sim_mid = sim[['timestamp', 'mid']]
	elif 'price' in sim.columns:
		sim_mid = sim[['timestamp', 'price']].rename(columns={'price': 'mid'})
	else:
		return
	sim_rs = _resample_mid(sim_mid, 'mid', 'timestamp', resample_rule)
	# Prepare real series
	real = real_ohlcv.copy()
	if apply_rth:
		real = _filter_rth(real, 'timestamp')
	real_mid = real[['timestamp', 'close']].rename(columns={'close': 'mid'})
	real_rs = _resample_mid(real_mid, 'mid', 'timestamp', resample_rule)
	if sim_rs.empty or real_rs.empty:
		return
	fig, ax = plt.subplots(figsize=(12, 5))
	ax.plot(sim_rs['timestamp'], sim_rs['mid'], label=f'Sim {symbol} (mid, {resample_rule})', alpha=0.7)
	ax.plot(real_rs['timestamp'], real_rs['mid'], label=f'Real {symbol} (close, {resample_rule})', alpha=0.7)
	ax.set_title(f"Price Timeseries (Aligned): {symbol}")
	ax.set_xlabel('Time (UTC)')
	ax.set_ylabel('Price')
	ax.legend()
	fig.tight_layout()
	fig.savefig(out / f"price_timeseries_aligned_{symbol}.png", dpi=120)
	plt.close(fig)


def plot_spread_distribution(sim_snap: pd.DataFrame, out: Path, symbol: str) -> None:
    if sim_snap.empty:
        return
    df = sim_snap.copy()
    # prefer relative spread in bps if available
    if 'relative_spread_bps' in df.columns:
        x = df['relative_spread_bps'].dropna()
        label = 'Relative Spread (bps)'
    elif {'best_bid', 'best_ask'}.issubset(df.columns):
        mid = (df['best_bid'] + df['best_ask']) / 2.0
        x = ((df['best_ask'] - df['best_bid']) / mid * 10000).replace([np.inf, -np.inf], np.nan).dropna()
        label = 'Relative Spread (bps)'
    else:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(x, bins=50, alpha=0.85)
    ax.set_title(f'Spread Distribution: {symbol}')
    ax.set_xlabel(label)
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(out / f"spread_distribution_{symbol}.png", dpi=120)
    plt.close(fig)


def plot_order_sign_acf(sim_trades: pd.DataFrame, out: Path, symbol: str) -> None:
    if sim_trades.empty:
        return
    tr = sim_trades.copy()
    tr['timestamp'] = pd.to_datetime(tr['timestamp'], utc=True, errors='coerce')
    tr = tr.dropna(subset=['timestamp'])
    # derive sign: +1 for buy aggressor, -1 for sell
    if 'aggressor_side' in tr.columns:
        tr['sign'] = tr['aggressor_side'].map({'BUY': 1, 'SELL': -1}).fillna(0)
    elif 'side' in tr.columns:
        tr['sign'] = tr['side'].map({'BUY': 1, 'SELL': -1}).fillna(0)
    else:
        return
    sign = tr.sort_values('timestamp')['sign']
    # simple ACF of order signs (first 50 lags)
    lags = 50
    s = sign.values.astype(float)
    res = []
    for k in range(1, lags + 1):
        if len(s) > k:
            res.append(np.corrcoef(s[:-k], s[k:])[0, 1])
        else:
            res.append(np.nan)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, lags + 1), res)
    ax.set_title(f'Order Sign Autocorrelation: {symbol}')
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    fig.tight_layout()
    fig.savefig(out / f"order_sign_acf_{symbol}.png", dpi=120)
    plt.close(fig)


def plot_market_impact_curve(sim_trades: pd.DataFrame, out: Path, symbol: str) -> None:
    if sim_trades.empty:
        return
    tr = sim_trades.copy()
    # bucket by trade size (quantity)
    bins = [0, 100, 500, 1000, 5000, np.inf]
    labels = ['<100', '100-500', '500-1k', '1k-5k', '>5k']
    tr['size_bucket'] = pd.cut(tr['quantity'].astype(float), bins=bins, labels=labels)
    # estimate temporary impact proxy: price move vs previous trade mid
    tr = tr.sort_values('timestamp')
    if 'price' not in tr.columns:
        return
    tr['prev_price'] = tr['price'].shift(1)
    tr['impact_bps'] = ((tr['price'] - tr['prev_price']) / tr['prev_price'] * 10000).replace([np.inf, -np.inf], np.nan)
    impact = tr.groupby('size_bucket', observed=True)['impact_bps'].median().dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    impact.plot(kind='bar', ax=ax)
    ax.set_title(f'Market Impact vs Trade Size: {symbol}')
    ax.set_xlabel('Size bucket (shares)')
    ax.set_ylabel('Median impact (bps)')
    fig.tight_layout()
    fig.savefig(out / f"market_impact_curve_{symbol}.png", dpi=120)
    plt.close(fig)


def plot_intraday_volatility(sim_snap: pd.DataFrame, real_ohlcv: pd.DataFrame, out: Path, symbol: str) -> None:
    if sim_snap.empty or real_ohlcv.empty:
        return
    sim = sim_snap.copy()
    sim['timestamp'] = pd.to_datetime(sim['timestamp'], utc=True, errors='coerce')
    sim = sim.dropna(subset=['timestamp'])
    if 'mid_price' in sim.columns:
        s = sim['mid_price'].astype(float)
    else:
        s = ((sim.get('best_bid') + sim.get('best_ask')) / 2.0).astype(float)
    sim_ret = np.log(s).diff().abs()
    sim['hour'] = sim['timestamp'].dt.hour
    sim_vol = sim.join(sim_ret.rename('abs_ret'))[['hour', 'abs_ret']].groupby('hour')['abs_ret'].mean()
    sim_vol = sim_vol / sim_vol.mean() if sim_vol.mean() else sim_vol

    ro = real_ohlcv.copy()
    ro['timestamp'] = pd.to_datetime(ro['timestamp'], utc=True, errors='coerce')
    ro = ro.dropna(subset=['timestamp'])
    rr = np.log(ro['close'].astype(float)).diff().abs()
    ro['hour'] = ro['timestamp'].dt.hour
    real_vol = ro.join(rr.rename('abs_ret'))[['hour', 'abs_ret']].groupby('hour')['abs_ret'].mean()
    real_vol = real_vol / real_vol.mean() if real_vol.mean() else real_vol

    hours = sorted(set(sim_vol.index).union(real_vol.index))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hours, [sim_vol.get(h, np.nan) for h in hours], label='Sim vol (norm)', marker='o')
    ax.plot(hours, [real_vol.get(h, np.nan) for h in hours], label='Real vol (norm)', marker='o')
    ax.set_title(f"Intraday Volatility U-shape (normalized): {symbol}")
    ax.set_xlabel('Hour (UTC)')
    ax.set_ylabel('Abs return / mean')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"intraday_volatility_{symbol}.png", dpi=120)
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


def _interval_to_rule(interval_used: str) -> Tuple[str, bool]:
	"""Map yfinance interval to pandas resample rule and whether to apply RTH filter."""
	m = interval_used.lower() if interval_used else '1m'
	if m in ('1m', '2m'):
		return ('1min', True)
	if m in ('5m', '15m', '30m', '60m', '90m', '1h'):
		# approximate to nearest
		return (m.replace('m', 'min').replace('h', 'H'), True)
	if m in ('1d', '1wk', '1mo'):
		return ('1D' if m == '1d' else ('1W' if m == '1wk' else '1M'), False)
	return ('1min', True)


def generate_plots(db_path: str, symbol: str, start: datetime, end: datetime, outdir: str) -> Dict[str, Any]:
	out = Path(outdir)
	_ensure_outdir(out)
	frames = load_simulated_frames(db_path, symbol)
	cfg = MarketFetchConfig(symbol=symbol, start=start, end=end, interval='1m')
	real = fetch_intraday_ohlcv(cfg)
	interval_used = real.attrs.get('interval_used', '1m') if isinstance(real, pd.DataFrame) else '1m'
	resample_rule, apply_rth = _interval_to_rule(interval_used)
	# Align to RTH and cadence
	snap = frames['snapshots']
	if not snap.empty:
		snap_use = _filter_rth(snap, 'timestamp') if apply_rth else snap
		# Ensure a 'mid' column exists for resampling
		if 'mid_price' in snap_use.columns:
			sim_mid_input = snap_use[['timestamp', 'mid_price']].rename(columns={'mid_price': 'mid'})
		elif {'best_bid', 'best_ask'}.issubset(snap_use.columns):
			tmp = snap_use.copy()
			tmp['mid'] = (tmp['best_bid'] + tmp['best_ask']) / 2.0
			sim_mid_input = tmp[['timestamp', 'mid']]
		else:
			sim_mid_input = pd.DataFrame(columns=['timestamp', 'mid'])
		sim_mid_series = _resample_mid(sim_mid_input, 'mid', 'timestamp', resample_rule) if not sim_mid_input.empty else pd.DataFrame(columns=['timestamp', 'mid'])
	else:
		sim_mid_series = pd.DataFrame(columns=['timestamp', 'mid'])
	real_use = _filter_rth(real, 'timestamp') if (not real.empty and apply_rth) else real
	# Choose input for plotting: snapshots if available, otherwise trades (price series)
	sim_for_plots = frames['snapshots'] if not frames['snapshots'].empty else frames['trades']
	# Plots
	plot_price_timeseries(sim_for_plots, real_use, out, symbol)
	# Also save an aligned cadence plot for visual correctness
	plot_price_timeseries_aligned(sim_for_plots, real, out, symbol, resample_rule, apply_rth)
	plot_return_distributions(frames['snapshots'], real_use, out, symbol)
	plot_autocorrelations(frames['snapshots'], out, symbol)
	# Only generate intraday U-shape plots when we actually have intraday data
	if apply_rth:
		plot_intraday_volume(frames['trades'], real_use, out, symbol)
		plot_intraday_volatility(frames['snapshots'], real_use, out, symbol)
	plot_spread_distribution(frames['snapshots'], out, symbol)
	plot_order_sign_acf(frames['trades'], out, symbol)
	plot_market_impact_curve(frames['trades'], out, symbol)
	# Metrics at cadence
	metrics = {}
	try:
		if not sim_mid_series.empty and not real_use.empty:
			real_mid = real_use[['timestamp', 'close']].rename(columns={'close': 'mid'})
			real_mid_rs = _resample_mid(real_mid, 'mid', 'timestamp', resample_rule)
			merged = pd.merge(sim_mid_series, real_mid_rs, on='timestamp', suffixes=('_sim', '_real')).dropna()
			if not merged.empty:
				ret_sim = np.log(merged['mid_sim']).diff().dropna()
				ret_real = np.log(merged['mid_real']).diff().dropna()
				metrics = _ks_emd_bootstrap(ret_sim, ret_real, n_boot=300)
				metrics['cadence'] = resample_rule
				metrics['interval_used'] = interval_used
	except Exception as e:
		metrics = {'error': str(e), 'interval_used': interval_used, 'cadence': resample_rule}
	with open(out / f"metrics_{symbol}.txt", 'w') as f:
		f.write(f"Interval used: {interval_used} | Cadence: {resample_rule}\n")
		f.write(f"KS: {metrics.get('ks_stat')} (p={metrics.get('ks_p')})\n")
		if 'ks_ci_95' in metrics:
			f.write(f"KS 95% CI: {metrics['ks_ci_95']}\n")
		f.write(f"EMD: {metrics.get('emd')}\n")
		if 'emd_ci_95' in metrics:
			f.write(f"EMD 95% CI: {metrics['emd_ci_95']}\n")
	return {"sim": frames, "real": real_use, "metrics": metrics}


def main():
    p = argparse.ArgumentParser(description='Generate comparative validation plots (Sim vs Real)')
    p.add_argument('--db', required=True, help='Path to SQLite DB (enhanced orderbook)')
    p.add_argument('--symbol', required=True, help='Symbol to analyze (e.g., AAPL)')
    p.add_argument('--outdir', default='validation_plots', help='Directory to save plots')
    p.add_argument('--val-start', default=None, help='UTC ISO start (default: infer from DB)')
    p.add_argument('--val-end', default=None, help='UTC ISO end (default: infer from DB)')
    args = p.parse_args()

    if args.val_start and args.val_end:
        start = pd.to_datetime(args.val_start, utc=True)
        end = pd.to_datetime(args.val_end, utc=True)
    else:
        min_ts, max_ts = _infer_time_window_from_db(args.db, args.symbol)
        if min_ts is None or max_ts is None or pd.isna(min_ts) or pd.isna(max_ts):
            # Fallback: last 6h
            from datetime import timezone
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=6)
        else:
            # Pad by 30 minutes on each side
            start = (min_ts - pd.Timedelta(minutes=30)).to_pydatetime()
            end = (max_ts + pd.Timedelta(minutes=30)).to_pydatetime()
            # Clamp to recent if the inferred window is too old for Yahoo 1m/5m
            from datetime import timezone as _tz
            now = datetime.now(_tz.utc)
            if (now - pd.to_datetime(end, utc=True)).total_seconds() > 10 * 24 * 3600:
                # Use a recent 6h window
                end = now
                start = end - timedelta(hours=6)

    generate_plots(args.db, args.symbol, start, end, args.outdir)
    print(f"✅ Plots saved to: {args.outdir}")


if __name__ == '__main__':
    main()

