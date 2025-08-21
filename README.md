# ABIDES-LLM Integration Project

A complete integration of Large Language Models (LLM) with ABIDES-inspired market simulation, focused on high-fidelity microstructure, real-world validation, and scalable limit order book generation.

## 🚀 Highlights

- LLM-enhanced agents and news coupling
- Realistic market microstructure with a live price–time priority book
- Real-market validation: fetch OHLCV/news, align to sim, compute KS/EMD with bootstrap CIs
- SQLite-backed storage for orders, trades, and snapshots
- Scalable multi-symbol generation with detailed microstructure

### 🆕 Enhanced Features

- 📊 Complete order/trade/snapshot recording
- 🧪 Real-market validation and metrics (KS, EMD; bootstrap CIs)
- 📈 Microstructure realism: spreads, market impact, order-sign ACF, intraday U-shapes
- 🤖 Inventory-based market-maker quoting; Hawkes-like sign memory; lognormal order sizes; cancellations
- 🗞️ LLM/news coupling: sentiment drives drift/vol with decay

## 📁 Project Structure

```
.
├── enhanced_orderbook_main.py        # End-to-end enhanced OB simulation + optional validation
├── scaled_lob_main.py                # Scaled LOB generation with live book
├── requirements.txt
├── README.md
└── src/
    ├── abides_llm_agents.py          # LLM-enhanced (ABIDES-style) agents (mock-compatible)
    ├── abides_llm_config.py          # ABIDES configuration scaffolding
    ├── enhanced_orderbook_db.py      # Enhanced OB + DB; real-price seeding
    ├── llm_analysis_system.py        # LLM analysis + real-market validation entry point
    ├── real_data_ingestion.py        # yfinance OHLCV + news; interval fallback
    ├── scaled_lob_generator.py       # Live price–time priority book, agents, news coupling
    ├── validation_viz.py             # Plots + KS/EMD with bootstrap CIs; RTH filtering
    └── mock_abides_core.py           # Lightweight ABIDES Core mock (if real ABIDES unavailable)
```

## 🛠️ Quick Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
# Optional for real LLM: echo "OPENAI_API_KEY=..." > .env
```

## 🔄 End-to-End Workflow

### 1) Run the Enhanced Order Book (multi-symbol, recent-day seeding)

```bash
# Quick test (2 symbols), with validation enabled
python enhanced_orderbook_main.py --config quick_test --validate-real --val-symbol AAPL

# Custom multi-symbol run (seeds starting price from real data at start time)
python enhanced_orderbook_main.py --custom \
  --agents 1000 --days 1 --symbols AAPL GOOGL MSFT TSLA AMZN \
  --db-path multi_symbols_orderbook.db \
  --start-utc 2025-08-18T13:30:00Z
```
Outputs:
- Database: quick_test_orderbook.db (or your custom path)
- Reports: enhanced_orderbook_output/reports/
- Data summaries: enhanced_orderbook_output/data_summaries/

Notes:
- Starting prices are seeded from real-world data via yfinance at the simulation start time (with interval fallback).
- Use recent dates for 1-minute validation (yfinance limit: ~last 30 days).

### 2) Generate Realism Plots and Metrics (RTH, cadence-aligned)

```bash
# Infers the correct time window from the DB automatically
python src/validation_viz.py --db multi_symbols_orderbook.db --symbol AAPL --outdir validation_plots_rth/AAPL
```
This produces:
- Price series (sim mid vs real close), return distributions
- Return and squared-return autocorrelations (volatility clustering)
- Intraday volume and volatility U-shapes (normalized)
- Spread distribution, order-sign ACF, market impact vs trade size
- KS/EMD with 95% bootstrap CIs saved to metrics_<SYMBOL>.txt (includes interval_used and cadence)

Cadence and RTH alignment:
- Real OHLCV cadence is inferred from yfinance interval_used (1m/5m/15m/1d via fallback)
- RTH filter is applied for minute-level data using America/New_York (DST-aware)

### 3) Scaled LOB Live-Book Generation

```bash
python scaled_lob_main.py --scale small
# or custom
python scaled_lob_main.py --custom --scale-factor 100 --days 1 \
  --symbols AAPL GOOGL MSFT --db-path scaled_lob.db
```
Implements: live price–time priority book, inventory-based MM quoting, Hawkes-like sign memory,
lognormal sizes, cancellations, detailed snapshots.

### 4) Couple LLM News to the Market (optional)

```python
from scaled_lob_generator import ScaledLOBGenerator, ScaledLOBConfig
cfg = ScaledLOBConfig(scale_factor=10, simulation_days=1, symbols=['AAPL'], db_path='shock.db')
engine = ScaledLOBGenerator(cfg)
# Positive news shock with high confidence
engine.inject_news_signal('AAPL', sentiment=0.8, confidence=0.9)
summary = engine.generate_scaled_data()
```
Then generate plots using step (2).

## 🌐 Real-World Data Validation and News Ingestion

The framework ingests real market data and news to validate simulations against actual markets.

### What it does
- Fetches intraday OHLCV bars (1m preferred) using yfinance with interval fallback (1m → 5m → 15m → 1d)
- Optionally fetches recent news headlines for the symbol
- Aligns simulated timestamps with real bars; computes error metrics and distributional distances

### Metrics reported
- Mean/median/P95 absolute price error (bps); bias (bps)
- KS statistic and 1D Earth Mover’s Distance (EMD) with 95% bootstrap CIs

### How to run (programmatic + CLI)
- Enhanced Order Book optional validation: pass --validate-real --val-symbol <TICKER>
- For standalone plots and metrics: use src/validation_viz.py as in step (2)

### Modules
- src/real_data_ingestion.py
  - fetch_intraday_ohlcv(cfg): OHLCV via yfinance with fallback; annotates interval_used
  - fetch_price_at_timestamp(symbol, ts, interval='1m', allow_fallback=True): nearest price for seeding
  - fetch_recent_news(symbol): recent news items
- src/llm_analysis_system.py
  - LLMOrderBookAnalyzer.validate_against_real_market(...): orchestrates fetch-and-compare
- src/validation_viz.py
  - generate_plots(...): RTH filtering, cadence-aligned comparisons, KS/EMD with bootstrap

## ✅ What’s Calibrated and Why It Matters

- Order sizes: lognormal per agent type → realistic heavy tails
- Order arrivals: Hawkes-like sign memory → order-sign autocorrelation
- Cancellations: agent-type rates → realistic top-of-book churn
- Market makers: inventory-based quoting → emergent spreads
- News coupling: sentiment affects drift/vol with exponential decay
- Validation: RTH-only (minute-level), mid-to-mid, KS/EMD with CIs

## 🧪 Tips

- yfinance 1-min data only covers ~last 30 days. For older dates, use 5m/15m/daily fallback or set a recent --start-utc.
- validation_viz.py infers cadence from interval_used and applies RTH only for minute-level data.
- To specify a comparison window explicitly, use --val-start/--val-end (UTC ISO-8601) in enhanced_orderbook_main.py or pass start/end to validation_viz.py.
- Set OPENAI_API_KEY in your environment or .env for real LLM calls.

## 🤖 LLM Integration

- LLM usage is enabled automatically when OPENAI_API_KEY is set.
- The system can run in LLM-disabled mode for cost-free testing (use --no-llm).

## 🐛 Troubleshooting

- Module import issues:
  ```bash
  export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
  ```
- yfinance 1m unavailable for older dates: the system falls back to 5m/15m/1d and records interval_used; plots and metrics adjust accordingly.
- Plots misaligned: ensure recent start times, verify outdir, confirm interval_used and cadence in metrics_<SYMBOL>.txt.

## 🤝 Contributing

1. Create a feature branch
2. Make changes and add tests if applicable
3. Open a pull request

## 📜 License and Acknowledgments

- ABIDES Framework: JPMorgan Chase & Co.
- OpenAI: GPT models for LLM integration
- Python Scientific Stack

Happy Trading with AI! 🤖📈