# ABIDES-LLM Integration Project

A complete integration of **Large Language Models (LLM)** with **ABIDES** (Agent-Based Interactive Discrete Event Simulation) for realistic market simulation and algorithmic trading research.

## 🚀 Features

- **LLM-Enhanced Trading Agents**: AI-powered agents that analyze news and make trading decisions
- **Market Sentiment Analysis**: Real-time analysis of market news and events
- **Multi-Agent System**: Momentum, contrarian, and neutral trading strategies
- **Realistic Market Simulation**: Complete market microstructure with order books
- **News-Driven Trading**: Agents react to market news with sophisticated reasoning
- **Comprehensive Analytics**: Detailed performance tracking and visualization

### 🆕 Enhanced Features

- **📊 Complete Order Book Recording**: Full order flow tracking and trade execution logging
- **🧪 ABIDES-Style Experiments**: Market impact studies, co-location analysis, agent validation
- **📈 Market Microstructure Analysis**: Spread analysis, volume-price relationships, stylized facts
- **💾 Comprehensive Data Export**: CSV/JSON export for external analysis and research
- **⚡ Real-time Performance Tracking**: Live P&L, portfolio valuation, and risk metrics
- **🔬 Experimental Framework**: Reproduce key experiments from the ABIDES research paper
- **🧭 Real-Market Validation**: Fetch real OHLCV/news via public APIs and compare simulation vs reality

## 📁 Project Structure

```
clean_project/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/                   # Core library modules
│   ├── abides_llm_agents.py          # LLM-enhanced trading agents
│   ├── abides_llm_config.py          # ABIDES configuration system
│   ├── enhanced_llm_abides_system.py # Enhanced LLM integration
│   ├── validation_viz.py             # Realism plots + KS/EMD metrics
│   └── real_data_ingestion.py        # yfinance OHLCV + news
└── examples/
```

## 🛠️ Quick Setup

```bash
# If venv is not available, you can install directly with --break-system-packages on this environment
pip install -r requirements.txt --break-system-packages

# Configure your OpenAI key (optional; mock mode if absent)
export OPENAI_API_KEY=sk-...
# Or create a .env file:
echo "OPENAI_API_KEY=sk-..." > .env
```

## 🔄 End-to-End Workflow

### 1) Run the Simple Demo (no ABIDES required)
```bash
python examples/simple_abides_llm_demo.py
```

### 2) Generate Realism Plots and Metrics (RTH, mid-to-mid)
```bash
# Infers the correct time window from the DB automatically
python src/validation_viz.py --db quick_test_orderbook.db --symbol AAPL --outdir validation_plots_rth/AAPL
```
This produces:
- Price series (sim mid vs real close), return distributions
- Return and squared-return autocorrelations (volatility clustering)
- Intraday volume and volatility U-shapes (normalized)
- Spread distribution, order-sign ACF, market impact vs trade size
- KS/EMD with 95% bootstrap CIs saved to `metrics_<SYMBOL>.txt`

### 3) ABIDES configuration (optional, when ABIDES is installed)
```bash
python -c "import sys; sys.path.insert(0, 'src'); import abides_llm_config as c; c.build_config()"
```

### 4) Couple LLM News to the Market (optional)
```python
from scaled_lob_generator import ScaledLOBGenerator, ScaledLOBConfig
cfg = ScaledLOBConfig(scale_factor=10, simulation_days=1, symbols=['AAPL'], db_path='shock.db')
engine = ScaledLOBGenerator(cfg)
# Positive news shock with high confidence
engine.inject_news_signal('AAPL', sentiment=0.8, confidence=0.9)
summary = engine.generate_scaled_data()
```
Then generate plots with step (2).

### 5) ITCH comparison (per-second execution alignment)

```bash
# Compare simulated executions with NASDAQ ITCH trade prices aggregated per second
python src/itch_compare.py \
  --db quick_test_orderbook.db \
  --symbol AAPL \
  --itch /path/to/itch_trades_AAPL.csv \
  --outdir itch_compare_out/AAPL \
  --agg last   # options: last, vwap, median
```

Outputs in `--outdir`:
- `itch_compare_<SYMBOL>.csv`: per-execution alignment with ITCH per-second price and error in bps
- `metrics_<SYMBOL>.json`: summary metrics (mean/median abs error bps, p95, bias, coverage)
- `itch_vs_sim_<SYMBOL>.png`: ITCH per-second series overlaid with sim executions

Notes:
- ITCH loader expects a CSV with columns including `timestamp`, `symbol`, `price` (and optional `size`). Timestamps are converted to UTC; naive timestamps are localized to `America/New_York` by default.
- For raw binary ITCH, convert to CSV first or implement the stub in `src/itch_ingestion.py`.

## ✅ What’s Calibrated and Why It Matters

- **Order sizes**: Lognormal per agent type → realistic heavy tails
- **Order arrivals**: Hawkes-like sign memory → order-sign autocorrelation
- **Cancellations**: Agent-type rates → realistic top-of-book churn
- **Market makers**: Inventory-based quoting → spreads emerge from quoting pressure
- **News coupling**: LLM sentiment affects drift/vol with decay → realistic event responses
- **Validation**: RTH-only, mid-to-mid 1-min bars; KS/EMD with 95% CIs

## 🧪 Tips
- For yfinance 1-min data, only the last ~30 days are available. If your simulation dates are older, use daily/5-min intervals or adjust the simulation date to a recent day.
- Use `--val-start/--val-end` in `enhanced_orderbook_main.py` when you want explicit windows.
- To compare multiple symbols, run `validation_viz.py` per symbol into separate outdirs.

## 📁 Project Structure

```
clean_project/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/                   # Core library modules
│   ├── abides_llm_agents.py          # LLM-enhanced trading agents
│   ├── abides_llm_config.py          # ABIDES configuration system
│   └── enhanced_llm_abides_system.py # Enhanced LLM integration
├── examples/              # Demo and example scripts
│   └── simple_abides_llm_demo.py     # Main demonstration
├── tests/                 # Test suites (to be added)
└── docs/                  # Documentation (to be added)
```

## 🛠️ Quick Setup

### 1. Clone and Setup Environment

```bash
# Navigate to project directory
cd clean_project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

Create a `.env` file for LLM integration:

```bash
# Create environment file
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```

**Note**: The system works in demo mode without API keys using mock LLM responses.

### 3. Run the Application

```bash
# Basic demo
python main.py --demo

# Enhanced demo with order book recording
python main.py --enhanced

# Full ABIDES experiments suite
python main.py --experiments

# Test configuration
python main.py --config

# Interactive mode (shows all options)
python main.py
```

### Non-Interactive Scaling
```bash
# Medium preset without prompts
python main.py --scale-data --scale-preset medium --yes

# Custom parameters without prompts
python main.py --scale-data --scale-preset custom \
  --scale-factor 200 --num-symbols 8 --days 3 \
  --orders-per-minute 120 --batch-size 4000 --yes
```

## 🎯 Usage Examples

### Basic Demo
```bash
python main.py --demo
```

### Enhanced Demo with Order Book Recording
```bash
python main.py --enhanced
```

### ABIDES Experiments Suite
```bash
python main.py --experiments
```

### Interactive Mode
```bash
python main.py
# Follow the interactive menu to select options
```

### Direct Example Execution
```bash
python examples/simple_abides_llm_demo.py
```

### Demo Artifacts
Running the simple demo saves artifacts under `artifacts/demo_run_<timestamp>`:
- summary.json
- news_analysis.json
- trades_<Trader>.csv

## 🧪 Available Experiments

The framework includes implementations of key ABIDES paper experiments:

### Market Impact Study
- Analyze how large orders affect market prices
- Study relationship between order size and price impact
- Generate event studies around impact trades

### Co-location Benefits Analysis  
- Examine trading advantages of low-latency connections
- Compare performance across different latency configurations
- Quantify the value of co-location in trading

### Background Agent Validation
- Validate that simulated agents produce realistic market behavior
- Compare simulated vs. historical price movements
- Analyze stylized facts (volatility clustering, fat tails, etc.)

### LLM vs Traditional Agent Comparison
- Compare performance of LLM-enhanced vs. algorithmic agents
- Analyze strategy adaptation and news response capabilities
- Study market efficiency implications

## 📊 Data Export and Analysis

The enhanced system provides comprehensive data export capabilities:

### Order Book Data
- Complete trade execution logs with timestamps
- Order flow records with agent attribution
- Market snapshots at regular intervals
- Price impact analysis around significant trades

### Agent Performance Data
- Portfolio valuation over time
- P&L tracking and risk metrics
- Trading signal generation and execution
- Strategy performance comparison

### Market Microstructure Data
- Bid-ask spread evolution
- Volume-price relationships
- Order book depth analysis
- Stylized facts validation

### Export Formats
```bash
# Data is automatically exported to CSV files:
# - *_trades.csv: All executed trades
# - *_orders.csv: Order history and status
# - *_snapshots.csv: Order book snapshots
# - *_agent_activity.csv: Agent decisions and performance
# - *_news_analysis.csv: News sentiment and impact
```

## 📈 Large-Scale Data Generation

The system includes powerful scaling capabilities to generate massive order book datasets:

### Scaling Presets
- **Light**: 10x scale, 5 symbols, 1 day → ~24,000 orders
- **Medium**: 100x scale, 10 symbols, 7 days → ~3.4M orders  
- **Heavy**: 500x scale, 20 symbols, 30 days → ~72M orders
- **Custom**: Configure your own parameters

### Example Performance
```bash
# Medium preset generated:
Total Orders: 33,605,484
Duration: 5 minutes
Orders/sec: 109,945
Output: Compressed CSV files
```

### Usage
```bash
# Run large-scale data generation
python main.py --scale-data

# Select from presets or configure custom scaling
# Outputs compressed CSV files with:
# - Realistic order flow patterns
# - Multiple agent types (retail, institutional, HFT, market makers)
# - Intraday trading patterns
# - Price movements with volatility clustering
```

## 🤖 LLM Integration

The system integrates LLMs in multiple ways:

### 1. News Analysis
- Parses market news and events
- Analyzes sentiment and market impact
- Generates trading signals based on news

### 2. Trading Strategy
- LLM agents reason about market conditions
- Generate sophisticated trading decisions
- Adapt strategies based on market feedback

### 3. Multi-Agent Interaction
- Multiple LLM agents with different strategies
- Momentum traders, contrarian traders, neutral traders
- Collaborative and competitive market dynamics

## 🌐 Real-World Data Validation and News Ingestion

The framework can ingest real market data and recent news to validate simulation quality against actual markets.

### What it does
- Fetches intraday OHLCV bars (e.g., 1-minute) for a symbol using `yfinance`
- Optionally fetches recent news headlines for the symbol (publisher, title, link)
- Aligns simulated trade timestamps to real market bars and computes error metrics

### Metrics reported
- Mean/median absolute price error in basis points (bps)
- 95th percentile absolute error (bps)
- Price bias (bps)

### How to run (validation plots)
```bash
python src/validation_viz.py --db quick_test_orderbook.db --symbol AAPL --outdir validation_plots_rth/AAPL
```

### Modules added
- `src/real_data_ingestion.py`
  - `fetch_intraday_ohlcv(cfg)`: fetches OHLCV bars via `yfinance`
  - `fetch_recent_news(symbol)`: fetches recent news items for a symbol
  - `align_simulation_with_real_market(sim_trades, real_ohlcv)`: aligns timestamps, computes price error (bps)
  - `basic_validation_report(merged)`: summarizes error metrics
  - `fetch_and_compare(symbol, sim_trades, start, end, interval)`: convenience wrapper
  

Notes:
- Internet access is required for fetching real data
- `yfinance` is installed via `requirements.txt`

## 📊 Sample Output

```
🚀 ABIDES-LLM Integration Demo
===========================
⚠️  No OpenAI API key found (using mock LLM)
LLM Enhancement: Mock Mode

✓ NewsAnalyzer initialized for symbols: ['ABM']
✓ MomentumTrader initialized: momentum strategy, risk=0.8
✓ ContrarianTrader initialized: contrarian strategy, risk=0.6
✓ NeutralTrader initialized: neutral strategy, risk=0.4

--- Event 1/3 ---
📰 NEWS: Partnership announced for ambitious initiative
   Category: product_launch
   Sentiment: -0.72
   Symbols: ['ABM']
[MomentumTrader] Generated signal: SELL 0.37
[MomentumTrader] EXECUTED: SELL 1872 shares at $104.35
[ContrarianTrader] Generated signal: BUY 0.22
[ContrarianTrader] EXECUTED: BUY 1191 shares at $98.38

📊 SIMULATION RESULTS
============================================================
Events Processed: 3
Total Signals Generated: 9
LLM Enhancement: Disabled

Trader Performance:
  MomentumTrader:
    Cash: $9,919,473.82
    Shares: 5,800
    Total Value: $10,499,473.82
    P&L: $-526.18 (-0.01%)
    Trades: 3
  ContrarianTrader:
    Cash: $10,048,415.76
    Shares: 4,569
    Total Value: $10,505,315.76
    P&L: $5,315.76 (+0.05%)
    Trades: 3
```

## 🔧 Configuration

### System Requirements
- Python 3.8+
- 4GB+ RAM recommended
- Internet connection (for LLM API calls)

### Core Dependencies
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.5.0` - Data manipulation
- `matplotlib>=3.5.0` - Visualization
- `openai>=1.0.0` - LLM integration
- `python-dotenv>=0.20.0` - Environment management
- `yfinance>=0.2.40` - Real market data (OHLCV/news) ingestion

### Optional Dependencies
- None

## 🎨 Architecture

### Core Components

1. **LLM Agents** (`src/abides_llm_agents.py`)
   - `ABIDESLLMNewsAnalyzer`: Analyzes market news
   - `ABIDESLLMTradingAgent`: Makes trading decisions
   - `ABIDESLLMMarketMaker`: Provides market liquidity

2. **Configuration System** (`src/abides_llm_config.py`)
   - ABIDES-compatible configuration
   - Agent setup and initialization
   - Market structure definition

3. **Enhanced System** (`src/enhanced_llm_abides_system.py`)
   - Advanced LLM reasoning
   - Market sentiment analysis
   - News event processing

### Data Flow

```
Market News → LLM Analysis → Trading Signals → ABIDES Simulation → Results
```

## 🚀 Advanced Usage

### Custom Trading Strategies

```python
from src.abides_llm_agents import ABIDESLLMTradingAgent

# Create custom trading agent
agent = ABIDESLLMTradingAgent(
    id=1,
    name="CustomTrader",
    strategy="momentum",
    risk_tolerance=0.7,
    llm_config={
        "model": "gpt-3.5-turbo",
        "temperature": 0.8
    }
)
```

### Custom Market Scenarios

```python
from src.abides_llm_config import build_config

# Create custom market configuration
config = build_config(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    num_llm_traders=5,
    end_time="16:00:00",
    llm_enabled=True
)
```

## 🔬 Research Applications

This framework is designed for:

- **Algorithmic Trading Research**: Test LLM-based trading strategies
- **Market Microstructure Studies**: Analyze agent interactions
- **News Impact Analysis**: Study how news affects trading behavior
- **Multi-Agent Systems**: Research collaborative AI trading
- **Risk Management**: Test risk models with AI agents

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📖 Documentation

- **Quick Start**: This README
- **API Reference**: See docstrings in source files
- **Examples**: Check `examples/` directory
- **Configuration**: See `src/abides_llm_config.py`

## 🐛 Troubleshooting

### Common Issues

**"Module not found" errors**:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**OpenAI API errors**:
- Check your API key in `.env`
- The system works in demo mode without API keys

**Memory issues**:
- Reduce number of agents in configuration
- Use smaller time windows for simulation

### Getting Help

1. Check this README
2. Review error messages carefully
3. Try running `python main.py --config` to test setup
4. Use demo mode if LLM APIs are unavailable

## 📜 License

This project is open source. See individual file headers for specific license information.

## 🙏 Acknowledgments

- **ABIDES Framework**: JPMorgan Chase & Co.
- **OpenAI**: GPT models for LLM integration
- **Python Community**: Scientific computing libraries

## 📞 Contact

For questions or issues:
- Check the documentation
- Review example code
- Test with demo mode first

---

**Happy Trading with AI! 🤖📈**