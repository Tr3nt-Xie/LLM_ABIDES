# ABIDES-LLM Verification Framework

A comprehensive verification system for testing LLM-ABIDES simulators against the original ABIDES paper experiments and generating order book visualizations similar to Figure 3.

## 🎯 Overview

This verification framework allows you to:
- **Reproduce ABIDES paper experiments** with your LLM-enhanced agents
- **Generate Figure 3 style visualizations** of order book dynamics around high-impact events
- **Verify stylized facts** of financial markets in your simulations
- **Compare LLM agents** with traditional ABIDES agents
- **Certify compliance** with the original ABIDES research standards

## 📋 Reference

**Paper**: "ABIDES: Towards High-Fidelity Market Simulation for AI Research"  
**Figure 3**: "Example of order book visualization around the time of a high impact trade"

## 🚀 Quick Start

### 1. Basic Demo (No Dependencies)
```bash
python3 simple_verification_demo.py
```

### 2. Full Framework (Requires Dependencies)
```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scipy

# Run integrated verification
python3 integrated_verification_demo.py
```

### 3. Integration with Your LLM-ABIDES System
```python
from abides_verification_framework import ABIDESVerificationFramework, ABIDESVerificationConfig

# Configure verification
config = ABIDESVerificationConfig(
    num_llm_agents=10,
    num_value_agents=100,
    analyze_order_book=True,
    capture_high_impact_events=True
)

# Run verification
framework = ABIDESVerificationFramework(config)
results = framework.run_full_verification_suite()
```

## 📦 Framework Components

### Core Classes

| Component | Description |
|-----------|-------------|
| `ABIDESVerificationFramework` | Main verification coordinator |
| `ABIDESPaperExperiments` | Reproduces original ABIDES experiments |
| `OrderBookVisualizer` | Creates Figure 3 style visualizations |
| `MarketDataAnalyzer` | Analyzes stylized facts and market properties |
| `IntegratedABIDESVerification` | Integrates with your LLM-ABIDES system |

### Data Classes

| Class | Purpose |
|-------|---------|
| `ABIDESVerificationConfig` | Configuration for experiments |
| `OrderBookSnapshot` | Order book state at specific time |
| `HighImpactEvent` | High-impact trading events for analysis |

## 🧪 Verification Experiments

### Experiment 1: Stylized Facts Verification

Tests if your LLM-ABIDES system reproduces key financial market stylized facts:

- **Volatility Clustering**: Periods of high volatility followed by high volatility
- **Fat Tails**: Non-normal return distributions with extreme events
- **Autocorrelation**: Weak-form market efficiency in returns
- **Long Memory**: Persistent effects in market data

```python
experiment_config = {
    "agents": {
        "value_agents": 100,
        "momentum_agents": 25, 
        "noise_agents": 5000,
        "market_makers": 1,
        "llm_agents": 10
    },
    "metrics": ["volatility_clustering", "fat_tails", "autocorrelation", "long_memory"]
}
```

### Experiment 2: Market Impact Analysis

Analyzes how large orders and LLM decisions affect price formation:

- **Large Institutional Orders**: >0.5% price impact
- **LLM Coordinated Trading**: >1.0% price impact  
- **News-Driven Trading**: >2.0% price impact

Creates visualizations around ±30 minutes of high-impact events.

### Experiment 3: Agent Behavior Comparison

Compares LLM agents with traditional ABIDES agents:

- Trading frequency patterns
- Order size distributions
- Reaction speeds to news events
- Risk-adjusted profitability metrics

## 📊 Order Book Visualization (Figure 3 Style)

The framework generates comprehensive four-panel visualizations:

### Panel 1: Order Book Depth Evolution
- Bid and ask volume over time
- Shows liquidity changes around events
- Highlights market stress periods

### Panel 2: Price Impact Timeline  
- Price changes relative to baseline
- Event timing and magnitude
- Impact persistence analysis

### Panel 3: Spread and Volume Analysis
- Bid-ask spread dynamics
- Volume relationships
- Market quality metrics

### Panel 4: Order Book Heatmap
- Depth visualization across price levels
- Time evolution of order book structure
- Visual impact of large orders

## 📈 Example Order Book Snapshot

```
Time: 10:30:15 | Mid-Price: $100.25 | Spread: $0.02

ASKS (Sell Orders)    |    BIDS (Buy Orders)
Price   Volume        |    Price   Volume  
----------------      |    ----------------
$100.28   250         |    $100.24   180
$100.27   420         |    $100.23   310  
$100.26   150   ←ASK  |  BID→  $100.22   275
================      |    ================

⚡ HIGH IMPACT EVENT: Large LLM buy order executed
   Impact: +0.75% price increase
```

## 🔧 Configuration Options

### Basic Configuration
```python
config = ABIDESVerificationConfig(
    # Agent populations (from ABIDES paper)
    num_value_agents=100,
    num_momentum_agents=25,
    num_noise_agents=5000,
    num_market_makers=1,
    
    # LLM agent parameters
    num_llm_agents=10,
    
    # Analysis settings
    analyze_order_book=True,
    capture_high_impact_events=True,
    impact_threshold=0.01,  # 1% price movement
    
    # Output settings
    output_dir="verification_results",
    save_plots=True,
    save_data=True
)
```

### Advanced Configuration
```python
# Custom experiment parameters
config.simulation_start_time = "09:30:00"
config.simulation_end_time = "16:00:00"
config.seed = 42

# Analysis parameters
config.impact_threshold = 0.005  # 0.5% threshold
config.time_window_minutes = 30  # Analysis window around events
```

## 📊 Results and Output

### Generated Files

| File | Description |
|------|-------------|
| `verification_results.json` | Detailed experimental results |
| `verification_summary.md` | Human-readable summary report |
| `integrated_verification_report.md` | Full compliance report |
| `order_book_analysis_*.png` | Figure 3 style visualizations |
| `llm_performance_analysis.png` | LLM vs traditional comparison |
| `market_impact_comparison.png` | Impact analysis charts |

### Sample Results
```json
{
  "overall_assessment": {
    "abides_compliance": "85%",
    "stylized_facts_score": "8.2/10", 
    "llm_enhancement": "Significant",
    "certification": "VERIFIED - Meets ABIDES standards"
  },
  "experiments": {
    "stylized_facts": {
      "volatility_clustering": "✓ Detected",
      "fat_tails": "✓ Present", 
      "autocorrelation": "✓ Weak-form efficiency"
    }
  }
}
```

## 🚀 Integration Guide

### Step 1: Import Framework
```python
from abides_verification_framework import (
    ABIDESVerificationFramework,
    ABIDESVerificationConfig,
    OrderBookSnapshot,
    HighImpactEvent
)
```

### Step 2: Configure Experiments
```python
config = ABIDESVerificationConfig(
    output_dir="my_verification",
    num_llm_agents=15,
    capture_high_impact_events=True
)
```

### Step 3: Run Your Simulation
```python
# Your existing LLM-ABIDES simulation code
simulation_results = run_llm_abides_simulation()
```

### Step 4: Generate Order Book Data
```python
# Convert your simulation data to order book snapshots
order_book_snapshots = convert_to_snapshots(simulation_results)
high_impact_events = identify_impact_events(simulation_results)
```

### Step 5: Run Verification
```python
framework = ABIDESVerificationFramework(config)
results = framework.run_full_verification_suite()
```

### Step 6: Create Visualizations
```python
# Generate Figure 3 style plots
for event in high_impact_events:
    framework.visualizer.create_figure_3_style_visualization(
        event, order_book_snapshots
    )
```

## 📈 Performance Metrics

The framework analyzes several key performance indicators:

### Market Quality Metrics
- **Bid-Ask Spread**: Average spread width
- **Market Depth**: Liquidity at best bid/ask
- **Price Efficiency**: Correlation with fundamentals
- **Volatility**: Annualized price volatility

### LLM Agent Performance
- **Decision Latency**: Time to make trading decisions
- **Success Rate**: Percentage of profitable trades
- **Reaction Time**: Speed of response to news events
- **Risk-Adjusted Returns**: Sharpe ratio and similar metrics

### Simulation Performance
- **Execution Time**: Total simulation runtime
- **Throughput**: Trades processed per second
- **Memory Usage**: Peak memory consumption
- **CPU Utilization**: Average processor usage

## 🎯 Compliance Scoring

The framework provides quantitative compliance scores:

### Stylized Facts Score (0-10)
- Based on presence of key market properties
- Weighted by importance in financial literature
- Compared against empirical benchmarks

### ABIDES Paper Compliance (0-100%)
- Measures adherence to original ABIDES experiments
- Includes agent behavior and market dynamics
- References specific paper sections and figures

### Overall Certification
- **VERIFIED**: Meets ABIDES paper standards (>80%)
- **PARTIAL**: Some compliance issues (60-80%)
- **NEEDS_WORK**: Significant deviations (<60%)

## 🔍 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scipy
   ```

2. **Memory Issues with Large Simulations**
   ```python
   # Reduce agent populations for testing
   config.num_noise_agents = 1000  # Instead of 5000
   ```

3. **Visualization Errors**
   ```python
   # Ensure sufficient order book data
   assert len(order_book_snapshots) > 10
   ```

4. **Integration Problems**
   ```python
   # Check data format compatibility
   assert isinstance(snapshot, OrderBookSnapshot)
   ```

## 📚 Research Applications

This verification framework is suitable for:

- **Academic Research**: Validating LLM agent behavior in financial markets
- **Industry Applications**: Testing trading algorithms and strategies  
- **Regulatory Studies**: Analyzing market impact and stability
- **Educational Use**: Teaching market microstructure concepts

## 🤝 Contributing

To extend the framework:

1. **Add New Experiments**: Extend `ABIDESPaperExperiments`
2. **Custom Visualizations**: Enhance `OrderBookVisualizer`  
3. **Additional Metrics**: Expand `MarketDataAnalyzer`
4. **New Agent Types**: Update verification configurations

## 📄 License

This verification framework is provided under the same license as your LLM-ABIDES system.

## 📞 Support

For questions about the verification framework:
1. Check the demo outputs in `simple_verification_output/`
2. Review the generated reports for diagnostic information
3. Ensure your LLM-ABIDES system produces compatible data formats

---

**Next Steps:**
1. Run `python3 simple_verification_demo.py` to see the framework overview
2. Install dependencies and try the full integrated demo
3. Adapt the framework to your specific LLM-ABIDES implementation
4. Generate Figure 3 style visualizations for your research

**Happy Simulating! 🚀📊**