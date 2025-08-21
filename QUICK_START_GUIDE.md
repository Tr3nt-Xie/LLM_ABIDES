# ABIDES-LLM Market Simulator - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Environment Setup (Already Done!)
```bash
# Virtual environment is already created and dependencies installed
source venv/bin/activate
```

### 2. Test Basic Functionality
```bash
# Run the basic demo to verify everything works
python main.py --demo
```

**Expected Output**: You should see LLM agents trading based on news events with performance metrics.

### 3. Generate Your First Dataset
```bash
# Quick test with enhanced order book (2 minutes)
python enhanced_orderbook_main.py --config quick_test

# Or try scaled LOB generation (6 minutes)
python scaled_lob_main.py --scale small
```

### 4. Analyze Results
```bash
# View generated data
ls -la *.db
ls -la enhanced_orderbook_output/
ls -la scaled_lob_output/

# Generate validation plots
python src/validation_viz.py --db quick_test_orderbook.db --symbol AAPL
```

## 📊 What You Get

### Enhanced Order Book System
- **Database**: `quick_test_orderbook.db` (35K+ orders)
- **Reports**: `enhanced_orderbook_output/reports/`
- **Data**: `enhanced_orderbook_output/data_summaries/`

### Scaled LOB System  
- **Database**: `small_scale_lob.db` (250K+ orders, 217 MB)
- **Reports**: `scaled_lob_output/reports/`
- **Performance**: 678 orders/second

### Validation Plots
- **Location**: `validation_plots_rth/AAPL/`
- **Files**: 9 plots showing market realism metrics

## 🔍 Explore Your Data

### SQLite Database Access
```python
import sqlite3
import pandas as pd

# Connect to your generated database
conn = sqlite3.connect('quick_test_orderbook.db')

# View available tables
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print(tables)

# Query orders
orders = pd.read_sql("SELECT * FROM orders LIMIT 10", conn)
print(orders.head())

# Query trades
trades = pd.read_sql("SELECT * FROM trades LIMIT 10", conn)
print(trades.head())
```

### Key Tables Available
- `orders` - All order submissions
- `trades` - Executed trades
- `snapshots` - Order book snapshots
- `market_stats` - Market statistics

## 🎯 Common Use Cases

### 1. Market Impact Analysis
```python
# Analyze how large orders affect prices
trades = pd.read_sql("""
    SELECT symbol, quantity, price_impact_bps, timestamp 
    FROM trades 
    WHERE quantity > 1000
    ORDER BY timestamp
""", conn)
```

### 2. Agent Performance Comparison
```python
# Compare different agent types
agent_perf = pd.read_sql("""
    SELECT agent_type, 
           COUNT(*) as orders,
           AVG(quantity) as avg_size,
           AVG(price_impact_bps) as avg_impact
    FROM orders 
    GROUP BY agent_type
""", conn)
```

### 3. Order Book Dynamics
```python
# Study spread evolution
spreads = pd.read_sql("""
    SELECT timestamp, symbol, spread_bps
    FROM snapshots 
    WHERE spread_bps IS NOT NULL
    ORDER BY timestamp
""", conn)
```

## ⚡ Quick Experiments

### Experiment 1: Different Market Conditions
```bash
# Test with more agents
python enhanced_orderbook_main.py --custom --agents 1000 --days 1 --symbols AAPL GOOGL

# Test with different symbols
python enhanced_orderbook_main.py --custom --agents 500 --days 1 --symbols TSLA MSFT AMZN
```

### Experiment 2: Scale Up
```bash
# Medium scale (100x)
python scaled_lob_main.py --scale medium

# Custom large scale
python scaled_lob_main.py --custom --scale-factor 500 --days 3 --symbols AAPL GOOGL MSFT
```

### Experiment 3: Real Market Comparison
```bash
# Compare with real AAPL data
python enhanced_orderbook_main.py --config quick_test --validate-real --val-symbol AAPL
```

## 🔧 Customization

### Modify Agent Behavior
Edit `src/abides_llm_agents.py` to change:
- Trading strategies
- Risk tolerance
- News sensitivity
- Order sizing

### Adjust Market Parameters
Edit `src/enhanced_orderbook_db.py` to modify:
- Spread generation
- Order arrival rates
- Price volatility
- Agent distribution

### Scale Configuration
Edit `src/scaled_lob_generator.py` to adjust:
- Number of agents
- Order frequency
- Market maker behavior
- Order book depth

## 📈 Performance Tips

### For Large Simulations
```bash
# Use smaller time windows for testing
python enhanced_orderbook_main.py --custom --agents 100 --days 0.1

# Monitor system resources
htop  # or top
```

### Database Optimization
```python
# For very large datasets, consider PostgreSQL
# Current SQLite is good for datasets up to ~1GB
```

## 🐛 Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
```

**"API key not found" warnings:**
```bash
# Check .env file exists and has your OpenAI key
cat .env
```

**Database errors:**
```bash
# Remove old database and regenerate
rm *.db
python enhanced_orderbook_main.py --config quick_test
```

**Memory issues:**
```bash
# Reduce scale for testing
python scaled_lob_main.py --scale small  # instead of medium/large
```

## 🎉 Next Steps

1. **Run your first simulation** (5 minutes)
2. **Explore the generated data** (10 minutes)  
3. **Try different configurations** (15 minutes)
4. **Analyze results with validation plots** (10 minutes)
5. **Scale up for research** (30+ minutes)

## 📞 Need Help?

- Check `README.md` for detailed documentation
- Review `PROJECT_STATUS_REPORT.md` for current status
- Examine generated reports in output directories
- Use the examples in `examples/` directory

---

**Your ABIDES-LLM market simulator is ready to use!** 🚀

Start with `python main.py --demo` and explore from there.