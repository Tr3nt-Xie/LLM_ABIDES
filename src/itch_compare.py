#!/usr/bin/env python3
"""
ITCH vs Simulation Execution Comparison (Per-Second)
====================================================

Compare simulated execution prices against NASDAQ ITCH trade data aggregated to
per-second windows. Outputs a CSV of aligned executions and summary metrics,
and optionally a plot overlaying ITCH per-second prices with simulation fills.

Usage
-----
python src/itch_compare.py \
  --db path/to/sim.db \
  --symbol AAPL \
  --itch path/to/itch_trades.csv \
  --outdir out/itch_compare/AAPL \
  --agg last  # or vwap, median

Note: Ensure your PYTHONPATH includes 'src' if importing from modules here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itch_ingestion import ITCHLoadConfig, load_itch_trades, aggregate_trades_to_seconds

# Reuse DB loaders from validation_viz to keep schema compatibility
from validation_viz import load_simulated_frames


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_sim_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["timestamp", "price", "quantity", "side", "aggressor_side"]) 
    df = trades.copy()
    # Standardize timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])  # keep executed rows only
    # Ensure price present
    if "price" not in df.columns:
        raise ValueError("Simulation trades must include 'price'")
    # Quantity optional but helpful
    if "quantity" not in df.columns:
        df["quantity"] = np.nan
    # Side mapping
    if "aggressor_side" not in df.columns:
        if "side" not in df.columns:
            df["aggressor_side"] = "UNK"
        else:
            df["aggressor_side"] = df["side"].astype(str)
    else:
        df["aggressor_side"] = df["aggressor_side"].astype(str)
    return df[["timestamp", "price", "quantity", "aggressor_side"]].rename(columns={"aggressor_side": "side"})


def _compute_metrics(aligned: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if aligned is None or aligned.empty:
        return {"error": "No comparable data"}
    if "price_error_bps" not in aligned.columns:
        return {"error": "No error column"}
    err = aligned["price_error_bps"].replace([np.inf, -np.inf], np.nan).dropna()
    if err.empty:
        return {"error": "No comparable data"}
    out["count"] = int(len(err))
    out["coverage_ratio"] = float(len(err) / len(aligned)) if len(aligned) else 0.0
    out["mean_abs_error_bps"] = float(err.abs().mean())
    out["median_abs_error_bps"] = float(err.abs().median())
    out["p95_abs_error_bps"] = float(err.abs().quantile(0.95))
    out["bias_bps"] = float(err.mean())
    # Optional side-conditioned metrics
    if "side" in aligned.columns:
        for side_val, grp in aligned.dropna(subset=["price_error_bps"]).groupby("side"):
            gerr = grp["price_error_bps"].dropna()
            if not gerr.empty:
                out[f"{side_val}_mean_abs_bps"] = float(gerr.abs().mean())
                out[f"{side_val}_bias_bps"] = float(gerr.mean())
    return out


def compare_sim_to_itch(
    db_path: str,
    symbol: str,
    itch_path: str,
    outdir: str,
    agg: str = "last",
    include_plot: bool = True,
) -> Dict[str, Any]:
    out_path = Path(outdir)
    _ensure_outdir(out_path)

    # Load simulation frames
    frames = load_simulated_frames(db_path, symbol)
    sim_trades = _normalize_sim_trades(frames.get("trades", pd.DataFrame()))

    # Load and aggregate ITCH trades
    cfg = ITCHLoadConfig(path=itch_path, symbol=symbol, fmt="csv")
    itch_trades = load_itch_trades(cfg)
    itch_sec = aggregate_trades_to_seconds(itch_trades, agg=agg)

    if sim_trades.empty or itch_sec.empty:
        aligned = pd.DataFrame()
        metrics = {"error": "Missing data", "sim_trades": len(sim_trades), "itch_seconds": len(itch_sec)}
    else:
        # Round to second for matching
        st = sim_trades.copy()
        st["sec"] = st["timestamp"].dt.floor("1s")
        it = itch_sec.rename(columns={"timestamp": "sec", "price": "itch_price"})
        # Align simulated executions to ITCH per second (left join to keep all sim fills)
        aligned = pd.merge(
            st,
            it,
            on="sec",
            how="left",
        )
        # Compute error in bps
        if "itch_price" in aligned.columns:
            aligned["price_error_bps"] = (
                (aligned["price"] - aligned["itch_price"]) / aligned["itch_price"] * 10000.0
            )
        metrics = _compute_metrics(aligned)

    # Save outputs
    comp_csv = out_path / f"itch_compare_{symbol}.csv"
    aligned_cols = [
        c for c in ["timestamp", "sec", "price", "quantity", "side", "itch_price", "price_error_bps"]
        if c in (aligned.columns if not aligned.empty else [])
    ]
    if not aligned.empty:
        aligned[aligned_cols].to_csv(comp_csv, index=False)

    # Plot overlay: ITCH per-second series vs sim execution scatter
    if include_plot and not aligned.empty and not itch_sec.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        # ITCH line
        ax.plot(itch_sec["timestamp"], itch_sec["price"], label=f"ITCH {symbol} (per-sec {agg})", alpha=0.8)
        # Sim executions
        colors = aligned["side"].map({"BUY": "tab:green", "SELL": "tab:red"}).fillna("tab:blue")
        sizes = aligned.get("quantity", pd.Series([100] * len(aligned))).astype(float).pow(0.5)
        sizes = (sizes / (sizes.max() if sizes.max() else 1.0) * 40.0) + 12.0
        ax.scatter(aligned["timestamp"], aligned["price"], c=colors, s=sizes, alpha=0.6, label="Sim executions")
        ax.set_title(f"{symbol}: ITCH per-second vs Sim Executions")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Price")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path / f"itch_vs_sim_{symbol}.png", dpi=120)
        plt.close(fig)

    # Save metrics
    import json
    with open(out_path / f"metrics_{symbol}.json", "w") as f:
        json.dump({"symbol": symbol, "agg": agg, **metrics}, f, indent=2)

    return {"aligned": aligned, "metrics": metrics, "itch_seconds": itch_sec}


def main():
    p = argparse.ArgumentParser(description="Compare sim executions vs ITCH per-second prices")
    p.add_argument("--db", required=True, help="Path to simulation SQLite DB")
    p.add_argument("--symbol", required=True, help="Symbol to analyze (e.g., AAPL)")
    p.add_argument("--itch", required=True, help="Path to ITCH trades CSV")
    p.add_argument("--outdir", default="itch_compare_out", help="Output directory")
    p.add_argument("--agg", default="last", choices=["last", "vwap", "median"], help="Second-level aggregation for ITCH")
    p.add_argument("--no-plot", action="store_true", help="Disable plots")
    args = p.parse_args()

    result = compare_sim_to_itch(
        db_path=args.db,
        symbol=args.symbol,
        itch_path=args.itch,
        outdir=args.outdir,
        agg=args.agg,
        include_plot=(not args.no_plot),
    )
    print("Comparison metrics:")
    print(result["metrics"])  # simple stdout summary


if __name__ == "__main__":
    main()

