# Enhanced ABIDES-LLM Framework

## Overview

This enhanced framework integrates **real LLM-powered trading agents** with **realistic order book mechanics** and **ABIDES-style experiments**. It addresses all the key requirements:

✅ **Fixed bugs** - Updated autogen imports and dependencies  
✅ **Real LLM integration** - Uses OpenAI API for actual reasoning (no mock functions)  
✅ **Proper order book recording** - Full order execution tracking with SQLite database  
✅ **ABIDES-style experiments** - Market impact studies and systematic testing  

## Key Features

### 🤖 Real LLM Integration
- **No mock functions** - Uses OpenAI GPT-4 for actual trading decisions
- **News sentiment analysis** - LLM processes market news and generates sentiment scores
- **Strategy-based reasoning** - Agents use momentum, value, volatility, and arbitrage strategies
- **Risk management** - LLM considers risk tolerance and position sizing

### 📊 Realistic Order Book System
- **Full order matching** - Price/time priority with partial fills
- **Order types** - Market, limit, stop orders with proper execution
- **Real-time tracking** - All orders and trades stored in SQLite database
- **Market microstructure** - Bid-ask spreads, market impact, slippage modeling

### 🔬 ABIDES-Style Experiments
- **Market impact studies** - Test how large orders affect prices (like in ABIDES papers)
- **Strategy comparison** - Compare LLM trading strategies systematically
- **Background agents** - Realistic market makers providing liquidity
- **Data collection** - Comprehensive logging for analysis

## Installation & Setup

### 1. Install Dependencies
```bash
pip install --user --break-system-packages numpy pandas scipy matplotlib seaborn openai python-dateutil dataclasses-json pydantic attrs pyyaml requests httpx aiofiles ujson pyautogen
```

### 2. Set OpenAI API Key
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Run the Framework
```bash
python3 run_enhanced_abides_llm.py
```

## Usage Examples

### Quick Demo (30 minutes)
```bash
python3 run_enhanced_abides_llm.py --mode demo
```

### Market Impact Experiment
```bash
python3 run_enhanced_abides_llm.py --mode impact
```

### Strategy Comparison
```bash
python3 run_enhanced_abides_llm.py --mode strategy
```

### Interactive Mode
```bash
python3 run_enhanced_abides_llm.py --mode interactive
```

## Framework Architecture

```
Enhanced ABIDES-LLM Framework
├── enhanced_llm_abides_system.py     # Real LLM integration
├── realistic_order_book_system.py    # Order book with execution tracking
├── abides_experiments.py             # ABIDES-style experiment framework
└── run_enhanced_abides_llm.py        # Main runner script
```

### Core Components

1. **LLMInterface** - Handles OpenAI API calls with fallback to mock responses
2. **EnhancedLLMNewsAnalyzer** - Processes news events using GPT-4
3. **AdvancedLLMTradingAgent** - LLM-powered trading agents with different strategies
4. **Exchange** - Central exchange managing order books for multiple symbols
5. **OrderBook** - Price/time priority matching with comprehensive recording
6. **MarketSimulation** - Main simulation engine orchestrating all components

## Key Improvements Made

### 🔧 Bug Fixes
- **Fixed autogen import** - Replaced with direct OpenAI API integration
- **Updated dependencies** - Compatible with latest package versions
- **Error handling** - Comprehensive try/catch blocks with fallbacks

### 🚀 Real LLM Integration
- **OpenAI GPT-4 API** - Actual reasoning instead of random responses
- **Structured prompts** - System prompts for financial analysis and trading
- **JSON responses** - Structured outputs for trading signals and analysis
- **Fallback system** - Mock responses if API fails or key missing

### 📈 Advanced Order Book
- **Price/time priority** - Standard exchange matching rules
- **Partial fills** - Orders can be partially executed
- **Order types** - Market, limit, stop orders
- **Trade recording** - Every trade stored with complete details
- **Market data** - Real-time bid/ask prices and spreads
- **SQLite database** - Persistent storage for all order book data

### 🧪 ABIDES-Style Experiments
- **Market impact studies** - Similar to ABIDES paper experiments
- **Multiple order sizes** - Test impact of 5K, 15K, 30K share orders
- **Price recovery analysis** - Track how quickly markets recover
- **Strategy performance** - Compare momentum vs value vs volatility strategies
- **Background agents** - Realistic market makers for liquidity

## Experiment Types

### 1. Market Impact Experiment
Studies how large orders affect market prices:
- Tests multiple order sizes (5,000 to 50,000 shares)
- Measures immediate price impact
- Tracks recovery time to baseline prices
- Compares with baseline simulation without impact orders

### 2. Strategy Comparison Experiment  
Compares different LLM trading strategies:
- **Momentum agents** - Buy rising stocks, sell falling ones
- **Value agents** - Look for undervalued opportunities
- **Volatility agents** - Trade on market uncertainty
- **Arbitrage agents** - Exploit price discrepancies

### 3. News Impact Analysis
Tests how LLM agents respond to news:
- Generates realistic market news events
- LLM analyzes sentiment and market impact
- Tracks agent decisions based on news
- Measures trading volume and price changes

## Data Collection & Analysis

### Order Book Data
- **SQLite database** - All orders and trades stored permanently
- **Order book snapshots** - Periodic depth of market data
- **Trade history** - Complete execution records with timestamps
- **Market statistics** - Volume, value traded, average trade size

### Agent Performance
- **Portfolio tracking** - Real-time mark-to-market values
- **Trade history** - Every trade with signal strength and confidence
- **Performance metrics** - Returns, Sharpe ratio, win rate
- **Strategy analysis** - Performance by strategy type

### Reports Generated
- **HTML reports** - Comprehensive experiment summaries
- **CSV files** - Trading data for further analysis
- **JSON results** - Complete simulation data
- **Order book analysis** - Market microstructure metrics

## Research Applications

This framework enables research similar to the ABIDES papers:

1. **Agent behavior studies** - How LLM agents respond to market conditions
2. **Market impact analysis** - Effect of large orders on price discovery
3. **Strategy evaluation** - Performance of different AI trading approaches
4. **News impact modeling** - How information affects market dynamics
5. **Market microstructure** - Order flow and execution analysis

## Example Output

```
SIMPLE DEMO RESULTS SUMMARY
============================================================

Agent Performance:
  LLM_AGENT_1 (momentum): 2.34% return, 12 trades
  LLM_AGENT_2 (value): -0.87% return, 8 trades  
  LLM_AGENT_3 (volatility): 1.56% return, 15 trades

Market Statistics:
  AAPL: 145 trades, $1,423,567.00 total value
  MSFT: 132 trades, $1,287,432.00 total value

News Events Generated: 6
Total Agent Decisions: 89
```

## Files Generated

- `simulation_results/` - JSON results and HTML reports
- `experiment_reports/` - Detailed experiment analysis
- `market_simulation.db` - SQLite database with all order book data
- Trading CSV files with complete trade history

## Advanced Usage

### Custom Experiments
```python
from abides_experiments import ExperimentConfig, MarketSimulation

config = ExperimentConfig(
    name="my_experiment",
    symbols=["AAPL", "GOOGL", "TSLA"],
    duration_minutes=240,
    num_llm_agents=10,
    num_background_agents=50,
    news_frequency=0.15
)

simulation = MarketSimulation(config)
results = await simulation.run_simulation()
```

### Order Book Analysis
```python
from realistic_order_book_system import Exchange

exchange = Exchange(["AAPL", "MSFT"])
# ... run simulation ...

# Get detailed order book data
snapshot = exchange.get_market_data("AAPL")
trading_history = exchange.get_trading_history("AAPL")
market_report = exchange.generate_market_report()
```

## Next Steps for Further Development

1. **More agent strategies** - Add mean reversion, pairs trading, etc.
2. **Real market data** - Integrate with actual market data feeds
3. **Advanced order types** - Iceberg orders, TWAP, VWAP algorithms
4. **Risk management** - Portfolio-level risk controls and limits
5. **Multi-asset trading** - Correlations and portfolio optimization
6. **High-frequency features** - Microsecond timing and latency modeling

## Troubleshooting

### Common Issues

1. **"No OpenAI API key found"**
   - Set the environment variable: `export OPENAI_API_KEY="your-key"`
   - Or use the command line: `--api-key your-key`

2. **"Module not found" errors**
   - Install missing packages: `pip install package-name`
   - Check virtual environment activation

3. **Database errors**
   - Delete `market_simulation.db` to reset
   - Check file permissions in the directory

4. **Memory issues with long simulations**
   - Reduce simulation duration or number of agents
   - Clear results directories periodically

### Getting Help

The framework includes comprehensive logging that shows:
- LLM API calls and responses
- Order submissions and executions  
- News events and agent decisions
- Performance metrics and errors

Check the log output for detailed debugging information.

---

**This enhanced framework transforms your basic ABIDES-LLM integration into a comprehensive research platform with real AI reasoning, proper market mechanics, and systematic experimentation capabilities.**