# ABIDES-LLM Integration Project

A complete integration of **Large Language Models (LLM)** with **ABIDES** (Agent-Based Interactive Discrete Event Simulation) for realistic market simulation and algorithmic trading research.

## 🚀 Features

- **LLM-Enhanced Trading Agents**: AI-powered agents that analyze news and make trading decisions
- **Market Sentiment Analysis**: Real-time analysis of market news and events
- **Multi-Agent System**: Momentum, contrarian, and neutral trading strategies
- **Realistic Market Simulation**: Complete market microstructure with order books
- **News-Driven Trading**: Agents react to market news with sophisticated reasoning
- **Comprehensive Analytics**: Detailed performance tracking and visualization

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
# Run interactive demo
python main.py

# Run automated demo
python main.py --demo

# Test configuration
python main.py --config
```

## 🎯 Usage Examples

### Basic Demo
```bash
python main.py --demo
```

### Interactive Mode
```bash
python main.py
# Follow the interactive menu
```

### Direct Example Execution
```bash
python examples/simple_abides_llm_demo.py
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

## 📊 Sample Output

```
🚀 ABIDES-LLM Integration Demo
=================================
⚠️  No OpenAI API key found (using mock LLM)
LLM Enhancement: Mock Mode

✓ NewsAnalyzer initialized for symbols: ['ABM']
✓ MomentumTrader initialized: momentum strategy, risk=0.8
✓ ContrarianTrader initialized: contrarian strategy, risk=0.6
✓ NeutralTrader initialized: neutral strategy, risk=0.4

--- Event 1/3 ---
📰 NEWS: Market update affects ABM
   Category: mergers
   Sentiment: -0.26
   Symbols: ['ABM']

📊 SIMULATION RESULTS
=====================
Events Processed: 3
Total Signals Generated: 0
LLM Enhancement: Disabled

Trader Performance:
  MomentumTrader: $10,000,000.00 (0 trades)
  ContrarianTrader: $10,000,000.00 (0 trades)
  NeutralTrader: $10,000,000.00 (0 trades)
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

### Optional Dependencies
- `plotly>=5.10.0` - Advanced visualization
- `sqlalchemy>=1.4.0` - Database integration
- `tiktoken>=0.5.0` - Token counting for LLM

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
