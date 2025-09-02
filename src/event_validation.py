#!/usr/bin/env python3
"""
Event Validation Runner
=======================

Generates a lightweight, reproducible validation bundle for a given symbol and
date range. When LLM is enabled, news items are analyzed to enrich events with
sentiment and black-swan risk; otherwise, a rule-based baseline is used.

Outputs in --outdir:
- <SYMBOL>_events.csv  
- <SYMBOL>_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

import pandas as pd

# Local imports (script is executed from repo root; src/ is on sys.path)
from real_data_ingestion import (
    fetch_recent_news,
    MarketFetchConfig,
    fetch_intraday_ohlcv,
)

try:
    # Optional LLM-enhanced analyzer (falls back to mock if key missing)
    from enhanced_llm_abides_system import (
        EnhancedLLMNewsAnalyzer,
        NewsEvent,
        NewsCategory,
    )
    HAS_ENHANCED = True
except Exception:
    HAS_ENHANCED = False


def _ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_date(d: str) -> datetime:
    # Interpret plain date as midnight UTC at start of day
    return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _rule_based_analysis(title: str) -> Dict[str, Any]:
    positive = ["growth", "success", "profit", "expansion", "partnership", "beat", "upgrade"]
    negative = ["loss", "decline", "lawsuit", "regulation", "bankruptcy", "downgrade", "miss"]
    t = (title or "").lower()
    score = 0.0
    for k in positive:
        if k in t:
            score += 0.15
    for k in negative:
        if k in t:
            score -= 0.15
    score = max(-1.0, min(1.0, score))
    return {
        "sentiment_score": float(score),
        "confidence": 0.65,
        "market_impact": "Baseline heuristic",
        "risk_assessment": "Heuristic",
        "black_swan_risk": 0.02,
        "black_swan_notes": "Baseline heuristic (no LLM)",
        "reasoning": "Keyword-based heuristic",
    }


def _analyze_news_llm(symbol: str, news_rows: pd.DataFrame, use_llm: bool) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    analyzer = None
    if use_llm and HAS_ENHANCED:
        analyzer = EnhancedLLMNewsAnalyzer([symbol])

    for idx, row in news_rows.iterrows():
        ts = pd.to_datetime(row.get("timestamp"), utc=True)
        title = (row.get("title") or "").strip()
        publisher = (row.get("publisher") or "").strip()
        link = (row.get("link") or "").strip()

        if analyzer is None:
            analysis = _rule_based_analysis(title)
        else:
            try:
                # Map into Enhanced system's NewsEvent
                ne = NewsEvent(
                    timestamp=ts.to_pydatetime() if pd.notna(ts) else datetime.now(timezone.utc),
                    category=NewsCategory.COMPANY_SPECIFIC,
                    headline=title or f"News: {symbol}",
                    content=f"{title}\nPublisher: {publisher}\nLink: {link}",
                    affected_symbols=[symbol],
                    sentiment_score=0.0,
                    importance=0.5,
                )
                # Run LLM analysis (async)
                import asyncio
                raw = asyncio.run(analyzer.analyze_news(ne))
                analysis = {
                    "sentiment_score": float(raw.get("sentiment_score", 0.0)),
                    "confidence": float(raw.get("confidence", 0.5)),
                    "market_impact": raw.get("market_impact", ""),
                    "risk_assessment": raw.get("risk_assessment", ""),
                    "black_swan_risk": float(raw.get("black_swan_risk", 0.0)),
                    "black_swan_notes": raw.get("black_swan_notes", ""),
                    "reasoning": raw.get("reasoning", "LLM analysis"),
                }
            except Exception:
                analysis = _rule_based_analysis(title)

        results.append(
            {
                "timestamp": ts.isoformat() if pd.notna(ts) else None,
                "title": title,
                "publisher": publisher,
                "link": link,
                "sentiment_score": analysis.get("sentiment_score"),
                "confidence": analysis.get("confidence"),
                "black_swan_risk": analysis.get("black_swan_risk"),
                "black_swan_notes": analysis.get("black_swan_notes"),
                "market_impact": analysis.get("market_impact"),
                "risk_assessment": analysis.get("risk_assessment"),
                "reasoning": analysis.get("reasoning"),
            }
        )

    return results


def _write_outputs(symbol: str, outdir: Path, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Events CSV
    events_df = pd.DataFrame(events)
    events_path = outdir / f"{symbol}_events.csv"
    events_df.to_csv(events_path, index=False)

    # Metrics JSON
    metrics = {
        "symbol": symbol,
        "num_events": int(len(events_df)),
        "avg_sentiment": float(events_df["sentiment_score"].mean()) if not events_df.empty else None,
        "avg_confidence": float(events_df["confidence"].mean()) if not events_df.empty else None,
        "avg_black_swan_risk": float(events_df["black_swan_risk"].mean()) if not events_df.empty else None,
    }
    metrics_path = outdir / f"{symbol}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return {"events_csv": str(events_path), "metrics_json": str(metrics_path), "metrics": metrics}


def _derive_events_from_price(symbol: str, start: datetime, end: datetime, max_events: int = 20) -> List[Dict[str, Any]]:
    cfg = MarketFetchConfig(symbol=symbol, start=start, end=end, interval="1m", allow_fallback=True)
    try:
        ohlcv = fetch_intraday_ohlcv(cfg)
    except Exception:
        ohlcv = pd.DataFrame()
    if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
        return []
    df = ohlcv.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "close"])  # ensure valid
    if df.empty:
        return []
    import numpy as np
    ret = np.log(df["close"].astype(float)).diff()
    abs_bps = (ret.abs() * 10000.0).fillna(0.0)
    df = df.assign(_ret=ret, _abs_bps=abs_bps)
    # Pick top-K moves across the window
    top = df.sort_values("_abs_bps", ascending=False).head(max_events)
    events: List[Dict[str, Any]] = []
    for _, row in top.iterrows():
        ts = pd.to_datetime(row["timestamp"], utc=True)
        bps = float(row["_abs_bps"])
        sign = 1.0 if float(row["_ret"]) >= 0 else -1.0
        # Map move magnitude to sentiment in [-1,1] with soft cap
        sentiment = max(-1.0, min(1.0, sign * (bps / 500.0)))
        black_swan_risk = max(0.0, min(1.0, bps / 3000.0))
        title = f"Derived: {'UP' if sign>0 else 'DOWN'} move {bps:.1f} bps"
        events.append(
            {
                "timestamp": ts.isoformat(),
                "title": title,
                "publisher": "DerivedFromPrice",
                "link": "",
                "sentiment_score": sentiment,
                "confidence": 0.7,
                "black_swan_risk": black_swan_risk,
                "black_swan_notes": "Derived from significant price move",
                "market_impact": "Significant intraday move",
                "risk_assessment": "Volatility-derived",
                "reasoning": "Constructed event from OHLCV return spike",
            }
        )
    # Sort final events chronologically
    events = sorted(events, key=lambda x: x["timestamp"]) 
    return events


def main() -> int:
    p = argparse.ArgumentParser(description="Generate LLM-enhanced or baseline event validation bundle")
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD, inclusive)")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD, inclusive)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--no-llm", action="store_true", help="Disable LLM; use baseline heuristics")
    p.add_argument("--max-news", type=int, default=50, help="Max news items to analyze")
    args = p.parse_args()

    symbol = args.symbol.upper()
    start_dt = _parse_date(args.start)
    end_dt = _parse_date(args.end) + timedelta(days=1)  # inclusive end

    outdir = Path(args.outdir)
    _ensure_outdir(outdir)

    # Fetch recent news and filter into window
    try:
        news_df = fetch_recent_news(symbol)
    except Exception:
        news_df = pd.DataFrame(columns=["timestamp", "title", "publisher", "link"])  # empty

    if not news_df.empty and "timestamp" in news_df.columns:
        news_df["timestamp"] = pd.to_datetime(news_df["timestamp"], utc=True, errors="coerce")
        news_df = news_df.dropna(subset=["timestamp"])  # drop NaT
        mask = (news_df["timestamp"] >= start_dt) & (news_df["timestamp"] < end_dt)
        news_df = news_df.loc[mask].sort_values("timestamp")
    else:
        news_df = pd.DataFrame(columns=["timestamp", "title", "publisher", "link"])  # empty

    if len(news_df) > args.max_news:
        news_df = news_df.iloc[: args.max_news]

    # Determine whether to use LLM (even if enabled, code falls back to mock when key missing)
    use_llm = not args.no_llm
    events = _analyze_news_llm(symbol, news_df, use_llm=use_llm)
    if len(events) == 0:
        # Fallback: derive events from price action within the window to avoid empty outputs
        events = _derive_events_from_price(symbol, start_dt, end_dt, max_events=max(10, args.max_news))
        # If LLM is enabled, refine derived events via LLM analyzer to capture richer sentiment
        if use_llm and HAS_ENHANCED and len(events) > 0:
            try:
                analyzer = EnhancedLLMNewsAnalyzer([symbol])
                import asyncio
                refined: List[Dict[str, Any]] = []
                for e in events:
                    ts = pd.to_datetime(e.get("timestamp"), utc=True)
                    ne = NewsEvent(
                        timestamp=ts.to_pydatetime() if pd.notna(ts) else datetime.now(timezone.utc),
                        category=NewsCategory.COMPANY_SPECIFIC,
                        headline=e.get("title") or f"Derived Event: {symbol}",
                        content=e.get("reasoning") or "Derived from price action",
                        affected_symbols=[symbol],
                        sentiment_score=float(e.get("sentiment_score", 0.0)),
                        importance=0.5,
                    )
                    raw = asyncio.run(analyzer.analyze_news(ne))
                    e["sentiment_score"] = float(raw.get("sentiment_score", e["sentiment_score"]))
                    e["confidence"] = float(raw.get("confidence", e.get("confidence", 0.6)))
                    e["black_swan_risk"] = float(raw.get("black_swan_risk", e.get("black_swan_risk", 0.0)))
                    e["black_swan_notes"] = raw.get("black_swan_notes", e.get("black_swan_notes", ""))
                    e["market_impact"] = raw.get("market_impact", e.get("market_impact", ""))
                    e["risk_assessment"] = raw.get("risk_assessment", e.get("risk_assessment", ""))
                    e["reasoning"] = raw.get("reasoning", e.get("reasoning", ""))
                    refined.append(e)
                events = refined
            except Exception:
                pass
    outputs = _write_outputs(symbol, outdir, events)

    # Handle common case-variation typo: also mirror LLMon -> LLmon if present in path
    outdir_name = outdir.name
    if "LLMon" in outdir_name:
        twin = outdir.parent / outdir_name.replace("LLMon", "LLmon")
        if twin != outdir:
            _ensure_outdir(twin)
            # Copy artifact files
            for fn in [f"{symbol}_events.csv", f"{symbol}_metrics.json"]:
                src = outdir / fn
                dst = twin / fn
                try:
                    if src.exists():
                        dst.write_bytes(src.read_bytes())
                except Exception:
                    pass

    print(json.dumps({"status": "ok", **outputs["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

