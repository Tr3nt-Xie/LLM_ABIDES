# Agent-Based Market Simulation with LLM Enhancement

## Abstract
This project implements an agent-based market simulator integrating Large Language Models (LLMs) with the ABIDES framework to study the impact of AI-enhanced trading agents on market dynamics. We compare three conditions: LLMON (LLM-enhanced agents), LLMOFF (traditional agents), and Baseline (noise traders) against real NASDAQ ITCH data.

## Requirements
```bash
python >= 3.8
numpy
pandas
matplotlib
sqlite3
openai
python-dotenv
```

## Installation
```bash
git clone [repository]
cd [repository]
pip install -r requirements.txt
```

## Configuration
Create `.env` file with OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Data Preparation
1. Download LOBSTER sample data (AMZN_2012-06-21):
```bash
wget https://lobsterdata.com/info/sample/AMZN_2012-06-21_34200000_57600000_message_1.csv
wget https://lobsterdata.com/info/sample/AMZN_2012-06-21_34200000_57600000_orderbook_1.csv
```

2. Place files in workspace root directory.

## Reproduction Steps

### 1. Generate Calibrated LOB Databases
```bash
python3 src/calibrated_lob_generator.py
```
Generates three databases with ~11,400 trades each (matching real market volume):
- `AMZN_2012-06-21_LLMON_calibrated.db`
- `AMZN_2012-06-21_LLMOFF_calibrated.db`
- `AMZN_2012-06-21_Baseline_calibrated.db`

### 2. Create Visualization
```bash
python3 create_plot_with_news.py
```
Produces Figure 4-style plot comparing all conditions with news injection markers.

### 3. Statistical Validation (Optional)
```bash
python3 validate_calibrated_lobs.py
```
Computes market microstructure statistics and KS tests.

## Key Components

### Agent Types
- **LLMON**: 30% smart momentum (LLM), 15% contrarian, 35% simple momentum, 20% stabilizers
- **LLMOFF**: 50% simple momentum, 15% contrarian, 35% stabilizers  
- **Baseline**: 40% noise traders, 25% mean reversion, 35% stabilizers

### News Events (2012-06-21)
| Hour | Event | Sentiment |
|------|-------|-----------|
| 1 | Fed Cautious | -0.3 |
| 2 | Spain Crisis | -0.4 |
| 3 | Tech Weakness | -0.2 |
| 4 | Failed Recovery | -0.1 |
| 5 | Buying Interest | +0.2 |

### Parameters
```python
initial_price = 223.56
duration_seconds = 23400  # 6.5 hours
target_trades = 11419  # From real NASDAQ
news_decay_rate = 0.0008
news_impact_multiplier = {
    "LLMON": 1.5,     # Intelligent response to news
    "LLMOFF": 0.12,   # Minimal awareness
    "Baseline": 0.03  # Almost no response
}
```

## Results

### Price Changes
- Real NASDAQ: -1.34%
- LLMON: -2.09%
- LLMOFF: -0.96%
- Baseline: +1.77%

### Trade Statistics
All conditions generate ~11,400 trades with 2.4 unique prices per second average.

## File Structure
```
/workspace/
├── src/
│   ├── calibrated_lob_generator.py    # Main generator
│   ├── itch_data_parser.py           # NASDAQ data parser
│   └── lob_comparison_experiment.py   # Comparison tools
├── lob_databases_calibrated_volume/   # Generated databases
├── create_plot_with_news.py          # Visualization script
└── README.md
```

## Citation
```bibtex
@article{abides_llm_2024,
  title={Agent-Based Market Simulation with LLM Enhancement},
  author={[Authors]},
  year={2024}
}
```

## License
MIT