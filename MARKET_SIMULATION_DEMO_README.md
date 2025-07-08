# Multi-Round Market Simulation Test Demo

A comprehensive test demonstration of the ABIDES-LLM market simulation system featuring multiple simulation rounds, different market scenarios, and detailed performance analysis.

## 🎯 Overview

This demo includes two main components:

1. **`multi_round_market_simulation_demo.py`** - Full-featured simulation using the complete ABIDES-LLM framework
2. **`quick_multi_round_test.py`** - Lightweight mock simulation for quick testing and demonstration

Both systems demonstrate:
- Multiple market simulation rounds with different configurations
- Various market scenarios (bull, bear, volatile, stable)
- Agent performance comparison across different strategies
- Comprehensive data collection and analysis
- Detailed reporting and visualization

## 🚀 Quick Start

### Option 1: Quick Demo (Recommended for Testing)

Run the lightweight mock simulation that doesn't require full ABIDES setup:

```bash
python quick_multi_round_test.py
```

This will run 5 different test scenarios in about 10-15 seconds and generate:
- Detailed performance reports
- CSV summaries
- JSON data files

### Option 2: Full Simulation

If you have the complete ABIDES-LLM system set up:

```bash
python multi_round_market_simulation_demo.py
```

This runs comprehensive simulations using the full ABIDES framework with real LLM integration.

## 📊 Test Scenarios

### Quick Test Scenarios

| Scenario | Description | Duration | Agents | Symbols | Market Type |
|----------|-------------|----------|---------|---------|-------------|
| `quick_stable_test` | Basic functionality validation | 1 min | 5 | AAPL, MSFT | Stable |
| `bull_market_test` | Rising market with momentum | 1.5 min | 8 | AAPL, MSFT, GOOGL | Bull |
| `bear_market_test` | Declining market, defensive strategies | 1.5 min | 8 | AAPL, MSFT, GOOGL | Bear |
| `volatile_market_test` | High volatility, mixed signals | 2 min | 10 | AAPL, MSFT, AMZN, TSLA | Volatile |
| `large_scale_test` | Scalability test | 2.5 min | 15 | 6 symbols | Stable |

### Full Simulation Scenarios

| Scenario | Description | Duration | Agents | Features |
|----------|-------------|----------|---------|-----------|
| `quick_smoke_test` | Fast validation | ~36 sec | 7 | Basic functionality |
| `bull_market_momentum` | Momentum strategies in rising market | 6 min | 13 | LLM news analysis |
| `bear_market_defensive` | Defensive strategies in declining market | 5 min | 11 | High news frequency |
| `volatile_market_chaos` | Mixed signals, high volatility | 7 min | 19 | Complex scenarios |
| `large_scale_stress_test` | Scalability and performance | 9 min | 31 | 8 symbols |
| `llm_effectiveness_test` | LLM vs traditional comparison | 6 min | 10 | Performance analysis |

## 🏗️ Architecture

### Core Components

1. **Market Data Generator**
   - Realistic price movements
   - Market regime simulation (bull/bear/volatile/stable)
   - Bid/ask spreads and volume simulation

2. **News Event System**
   - Realistic news headlines
   - Sentiment analysis
   - Market impact simulation

3. **Agent Types**
   - **Momentum Agents**: Follow price trends
   - **Contrarian Agents**: Trade against sentiment
   - **Mean Reversion Agents**: Exploit price reversions
   - **Trend Following Agents**: Long-term trend strategies
   - **Arbitrage Agents**: Exploit price inefficiencies

4. **Performance Analytics**
   - PnL tracking
   - Trade execution analysis
   - Risk metrics
   - Market regime performance

## 📈 Output Files

Both demos generate comprehensive output files:

### Quick Test Output (`quick_test_results/`)
- `quick_test_results.json` - Detailed analysis of all test runs
- `test_summary.csv` - Tabular summary for easy analysis
- Console output with real-time progress

### Full Simulation Output (`simulation_results/`)
- `multi_round_simulation_report.json` - Complete analysis
- `simulation_summary.csv` - Performance summary
- `simulation_analysis.png` - Visualization charts
- `test_scenarios.json` - Configuration details
- Individual scenario outputs in subdirectories

## 📊 Sample Output

### Console Output
```
🚀 Starting Quick Multi-Round Market Simulation Tests
======================================================================

📊 Running Test 1/5: quick_stable_test
Market Regime: Stable
Symbols: 2, Agents: 5
✅ Test completed successfully
   Total PnL: $12,450.75
   Profitable Agents: 3/5
   Total Trades: 23

📊 Running Test 2/5: bull_market_test
Market Regime: Bull
Symbols: 3, Agents: 8
✅ Test completed successfully
   Total PnL: $28,912.34
   Profitable Agents: 6/8
   Total Trades: 45

...

🎯 QUICK MULTI-ROUND TEST RESULTS
======================================================================
📊 Total Tests: 5
✅ Successful Tests: 5
📈 Success Rate: 100.0%
⏱️  Total Duration: 12.3 seconds
⚡ Avg Test Duration: 2.5 seconds

💹 OVERALL PERFORMANCE
📊 Total Trades: 156
💰 Total PnL: $45,234.67
🤖 Total Agents: 46
📈 Avg PnL per Agent: $983.36
⚡ Trades per Second: 12.68
```

### Performance Analysis

The system analyzes:
- **Success Rate**: Percentage of scenarios that completed successfully
- **Agent Performance**: Comparison between different trading strategies
- **Market Regime Impact**: How different market conditions affect performance
- **Scalability**: Performance as system scale increases
- **Trading Efficiency**: Trades per second and execution quality

## 🔧 Configuration

### Customizing Test Scenarios

You can modify the test configurations in either file:

```python
# In quick_multi_round_test.py
QuickTestConfig(
    name="custom_test",
    symbols=["AAPL", "MSFT", "GOOGL"],
    agents_count=10,
    duration_minutes=2.0,
    news_frequency=2.0,  # Events per minute
    market_regime="volatile",  # "bull", "bear", "volatile", "stable"
    volatility_level=0.6  # 0.1 to 1.0
)

# In multi_round_market_simulation_demo.py
TestScenario(
    name="custom_scenario",
    description="Custom market scenario",
    symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
    duration_hours=0.2,  # 12 minutes real time
    market_regime="bull",
    news_frequency=0.3,
    llm_agents_count=4,
    abides_agents_count=12,
    real_time_factor=150.0,  # Simulation speed multiplier
    expected_outcomes={"positive_pnl_ratio": 0.6}
)
```

### Agent Strategies

Available agent strategies:
- `momentum` - Follows price and sentiment trends
- `contrarian` - Trades against market sentiment
- `mean_reversion` - Exploits price reversions to mean
- `trend_following` - Long-term trend strategies
- `arbitrage` - Exploits price inefficiencies
- `adaptive` - Adapts strategy based on market conditions

## 🔍 Key Metrics

### Performance Metrics
- **Total PnL**: Net profit/loss across all agents
- **Win Rate**: Percentage of profitable agents
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest portfolio decline
- **Trade Frequency**: Average trades per agent

### Market Metrics
- **Volatility**: Price movement variability
- **Liquidity Score**: Market depth and efficiency
- **Spread Analysis**: Bid-ask spread statistics
- **Price Impact**: Order impact on market prices

### System Metrics
- **Execution Speed**: Trades processed per second
- **Success Rate**: Percentage of successful scenarios
- **Scalability**: Performance vs. system complexity
- **Error Rate**: Frequency of execution failures

## 🚀 Advanced Usage

### Running Specific Scenarios

```python
# Run only specific test scenarios
demo = MultiRoundSimulationDemo()
demo.test_scenarios = [s for s in demo.test_scenarios if 'bull' in s.name]
results = demo.run_all_simulations()
```

### Custom Analysis

```python
# Add custom analysis
def custom_analysis(results):
    # Your custom analysis logic here
    return analysis_results

# Integrate into demo
demo = QuickMultiRoundTester()
results = demo.run_all_tests()
my_analysis = custom_analysis(results)
```

## 📋 Requirements

### Quick Test (Minimal Requirements)
```
Python 3.7+
numpy
pandas
```

### Full Simulation
```
Python 3.8+
All packages in requirements_abides.txt
OpenAI API key (for LLM integration)
ABIDES framework
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: The quick test falls back to mock mode if full ABIDES is not available
2. **Memory Usage**: Large-scale tests may require significant RAM
3. **Performance**: Adjust `real_time_factor` to speed up/slow down simulations
4. **API Limits**: LLM integration may hit rate limits with high news frequency

### Error Handling

Both demos include comprehensive error handling:
- Failed scenarios are logged but don't stop the entire test suite
- Mock fallbacks when dependencies are unavailable
- Detailed error reporting in output files

## 🎯 Use Cases

### Development & Testing
- Validate system functionality before deployment
- Performance regression testing
- Load testing and scalability analysis

### Research & Analysis
- Compare trading strategies across market regimes
- Analyze LLM effectiveness in financial markets
- Study market microstructure effects

### Demonstration & Education
- Show market simulation capabilities
- Demonstrate agent-based modeling concepts
- Illustrate market dynamics and agent interactions

## 📚 Next Steps

1. **Run the quick test** to see the system in action
2. **Analyze the output files** to understand performance characteristics
3. **Customize scenarios** for your specific use cases
4. **Set up full ABIDES** for production-level simulations
5. **Integrate with your own trading strategies** and analysis tools

## 📞 Support

For issues or questions:
1. Check the console output for error messages
2. Review the generated log files
3. Examine the detailed JSON output for debugging information
4. Ensure all dependencies are properly installed

---

**Note**: This demo is designed to showcase the capabilities of the ABIDES-LLM market simulation system. The quick test provides a fast way to evaluate functionality, while the full simulation offers comprehensive market modeling with LLM integration.