"""
ABIDES Configuration File: LLM-Enhanced Market Simulation
=========================================================

This configuration file follows the official ABIDES experimental framework 
structure and can be used with the ABIDES command line interface:

    abides abides_llm_config.py --end_time "12:00:00"

Or programmatically:
    from abides_core import abides
    config_state = build_config(seed=0, end_time='12:00:00')
    end_state = abides.run(config_state)

Based on rmsc04 configuration with LLM agent enhancements.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Official ABIDES imports
    from abides_core.utils import subdict
    from abides_markets.agents import ExchangeAgent
    from abides_markets.agents import NoiseAgent, ValueAgent, MomentumAgent, MarketMakerAgent
    from abides_markets.orders import Side
    ABIDES_AVAILABLE = True
except ImportError:
    # Fallback when ABIDES is not installed
    print("Warning: Official ABIDES not found. Using mock configuration.")
    ABIDES_AVAILABLE = False
    
    class ExchangeAgent:
        pass
    class NoiseAgent:
        pass
    class ValueAgent:
        pass
    class MomentumAgent:
        pass
    class MarketMakerAgent:
        pass
    
    def subdict(d, keys):
        return {k: d[k] for k in keys if k in d}

# Import our LLM agents
from abides_llm_agents import (
    ABIDESLLMNewsAnalyzer, ABIDESLLMTradingAgent, ABIDESLLMMarketMaker
)


def build_config(
    seed=None,
    end_time="16:00:00",
    log_dir=None,
    
    # LLM Configuration
    llm_enabled=True,
    llm_config=None,
    num_llm_traders=5,
    num_llm_market_makers=1,
    
    # Traditional Agents Configuration
    num_noise=100,
    num_value=25,
    num_momentum=10,
    num_market_makers=2,
    
    # Market Configuration
    symbols=None,
    starting_cash=10_000_000,
    
    # Simulation Configuration
    book_logging=True,
    book_log_depth=10,
    stream_history=500,
    
    **kwargs
):
    """
    Build ABIDES configuration with LLM-enhanced agents
    
    This function creates a complete ABIDES configuration that includes:
    - LLM-enhanced news analysis and trading agents
    - Traditional ABIDES agents (noise, value, momentum)
    - Exchange agent with proper market structure
    - Complete logging and data collection setup
    
    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    end_time : str
        Simulation end time (e.g., "16:00:00")
    llm_enabled : bool
        Whether to include LLM agents
    llm_config : dict
        LLM configuration for API access
    num_llm_traders : int
        Number of LLM trading agents
    num_llm_market_makers : int
        Number of LLM market makers
    
    Returns:
    --------
    dict
        Complete ABIDES configuration
    """
    
    # Set defaults
    if seed is None:
        seed = np.random.randint(0, 2**31)
    
    if symbols is None:
        symbols = ["ABM"]
    
    if log_dir is None:
        log_dir = "./abides_logs"
    
    # Default LLM configuration
    if llm_enabled and llm_config is None:
        llm_config = {
            "config_list": [
                {
                    "model": "gpt-4o-mini",  # More cost-effective default
                    "api_key": os.getenv("OPENAI_API_KEY", "your-api-key-here"),
                    "temperature": 0.3
                }
            ],
            "timeout": 60
        }
        
        # Warn if no API key is set
        if llm_config["config_list"][0]["api_key"] == "your-api-key-here":
            print("Warning: No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
            print("LLM features will be disabled for this run.")
            llm_enabled = False
    
    # Base configuration
    config = {
        'seed': seed,
        'start_time': "09:30:00",
        'end_time': end_time,
        'agents': [],
        'agent_latency_model': 'deterministic',  # or 'gaussian'
        'default_computation_delay': 1_000_000,  # 1ms in nanoseconds
        'custom_properties': {},
        'random_state': np.random.RandomState(seed),
    }
    
    # Exchange configuration
    exchange_config = {
        'log_orders': True,
        'book_logging': book_logging,
        'book_log_depth': book_log_depth,
        'stream_history': stream_history,
        'log_dir': log_dir
    }
    
    agents = []
    agent_id = 0
    
    # =============================================================================
    # EXCHANGE AGENT - Core market infrastructure
    # =============================================================================
    
    # Create exchange configuration
    mkt_open = pd.to_datetime("09:30:00").time()
    mkt_close = pd.to_datetime(end_time).time()
    
    agents.append({
        'agent_id': agent_id,
        'agent_class': ExchangeAgent if ABIDES_AVAILABLE else 'ExchangeAgent',
        'args': {
            'id': agent_id,
            'name': "EXCHANGE_AGENT",
            'type': "ExchangeAgent",
            'mkt_open': mkt_open,
            'mkt_close': mkt_close,
            'symbols': symbols,
            'book_logging': book_logging,
            'book_log_depth': book_log_depth,
            'log_orders': True,
            'pipeline_delay': 40_000,  # 40 microseconds
            'computation_delay': 1_000_000,  # 1 millisecond
            'stream_history_length': stream_history,
            'random_state': np.random.RandomState(seed=seed)
        }
    })
    agent_id += 1
    
    # =============================================================================
    # LLM-ENHANCED AGENTS - Advanced reasoning and market analysis
    # =============================================================================
    
    if llm_enabled:
        
        # LLM News Analyzer Agent
        agents.append({
            'agent_id': agent_id,
            'agent_class': ABIDESLLMNewsAnalyzer,
            'args': {
                'id': agent_id,
                'name': f"LLM_NEWS_ANALYZER_{agent_id}",
                'type': "ABIDESLLMNewsAnalyzer",
                'symbols': symbols,
                'llm_config': llm_config,
                'random_state': np.random.RandomState(seed=seed + agent_id),
                'log_orders': False
            }
        })
        agent_id += 1
        
        # LLM Trading Agents with different strategies
        strategies = ["momentum", "value", "volatility"]
        risk_tolerances = [0.3, 0.5, 0.7, 0.9]
        
        for i in range(num_llm_traders):
            strategy = strategies[i % len(strategies)]
            risk_tolerance = risk_tolerances[i % len(risk_tolerances)]
            
            agents.append({
                'agent_id': agent_id,
                'agent_class': ABIDESLLMTradingAgent,
                'args': {
                    'id': agent_id,
                    'name': f"LLM_TRADER_{strategy.upper()}_{agent_id}",
                    'type': "ABIDESLLMTradingAgent",
                    'symbol': symbols[0],
                    'strategy': strategy,
                    'risk_tolerance': risk_tolerance,
                    'llm_config': llm_config,
                    'initial_cash': starting_cash,
                    'random_state': np.random.RandomState(seed=seed + agent_id),
                    'log_orders': True
                }
            })
            agent_id += 1
        
        # LLM Market Makers
        for i in range(num_llm_market_makers):
            spread_bps = 10 + (i * 5)  # Varying spreads
            max_inventory = 1000 + (i * 500)
            
            agents.append({
                'agent_id': agent_id,
                'agent_class': ABIDESLLMMarketMaker,
                'args': {
                    'id': agent_id,
                    'name': f"LLM_MARKET_MAKER_{agent_id}",
                    'type': "ABIDESLLMMarketMaker",
                    'symbol': symbols[0],
                    'spread_bps': spread_bps,
                    'max_inventory': max_inventory,
                    'llm_config': llm_config,
                    'initial_cash': starting_cash,
                    'random_state': np.random.RandomState(seed=seed + agent_id),
                    'log_orders': True
                }
            })
            agent_id += 1
    
    # =============================================================================
    # TRADITIONAL ABIDES AGENTS - Baseline market participants
    # =============================================================================
    
    if ABIDES_AVAILABLE:
        
        # Traditional Market Makers
        for i in range(num_market_makers):
            agents.append({
                'agent_id': agent_id,
                'agent_class': MarketMakerAgent,
                'args': {
                    'id': agent_id,
                    'name': f"MARKET_MAKER_{agent_id}",
                    'type': "MarketMakerAgent",
                    'symbol': symbols[0],
                    'starting_cash': starting_cash,
                    'pov': 0.025,  # 2.5% participation rate
                    'min_order_size': 20,
                    'max_order_size': 50,
                    'wake_up_freq': '1min',
                    'random_state': np.random.RandomState(seed=seed + agent_id),
                    'log_orders': True
                }
            })
            agent_id += 1
        
        # Value Agents - Fundamental analysis traders
        for i in range(num_value):
            agents.append({
                'agent_id': agent_id,
                'agent_class': ValueAgent,
                'args': {
                    'id': agent_id,
                    'name': f"VALUE_AGENT_{agent_id}",
                    'type': "ValueAgent",
                    'symbol': symbols[0],
                    'starting_cash': starting_cash // 10,  # Smaller positions
                    'sigma_n': 1000000,  # Fundamental value noise
                    'r_bar': 100000,  # Base fundamental value
                    'kappa': 1.67e-12,  # Mean reversion strength
                    'sigma_s': 0,  # No fundamental shock
                    'lambda_a': 7e-11,  # Arrival rate of orders
                    'log_orders': True,
                    'random_state': np.random.RandomState(seed=seed + agent_id)
                }
            })
            agent_id += 1
        
        # Momentum Agents - Technical analysis traders
        for i in range(num_momentum):
            agents.append({
                'agent_id': agent_id,
                'agent_class': MomentumAgent,
                'args': {
                    'id': agent_id,
                    'name': f"MOMENTUM_AGENT_{agent_id}",
                    'type': "MomentumAgent",
                    'symbol': symbols[0],
                    'starting_cash': starting_cash // 5,
                    'min_order_size': 1,
                    'max_order_size': 10,
                    'wake_up_freq': '20s',
                    'log_orders': True,
                    'random_state': np.random.RandomState(seed=seed + agent_id)
                }
            })
            agent_id += 1
        
        # Noise Agents - Random traders providing market noise
        for i in range(num_noise):
            agents.append({
                'agent_id': agent_id,
                'agent_class': NoiseAgent,
                'args': {
                    'id': agent_id,
                    'name': f"NOISE_AGENT_{agent_id}",
                    'type': "NoiseAgent",
                    'symbol': symbols[0],
                    'starting_cash': starting_cash // 100,  # Very small positions
                    'sigma_n': 100000,
                    'r_bar': 100000,
                    'kappa': 1.67e-15,
                    'sigma_s': 0,
                    'lambda_a': 7e-11,
                    'log_orders': False,  # Don't log noise agent orders
                    'random_state': np.random.RandomState(seed=seed + agent_id)
                }
            })
            agent_id += 1
    
    # Add agents to configuration
    config['agents'] = agents
    
    # =============================================================================
    # FINAL CONFIGURATION SETTINGS
    # =============================================================================
    
    # Logging configuration
    config.update({
        'log_dir': log_dir,
        'book_logging': book_logging,
        'book_log_depth': book_log_depth,
        'stream_history': stream_history,
        'exchange_log_orders': True,
        'log_events': True
    })
    
    # Market structure configuration
    config.update({
        'symbols': symbols,
        'fundamental_value': {symbols[0]: 100000},  # $100 fundamental value in cents
        'oracle_type': 'MeanRevertingOracle',
        'oracle_config': {
            'symbols': symbols,
            'r_bar': 100000,
            'kappa': 1.67e-12,
            'sigma_s': 0,
            'megashock_lambda_a': 2.77778e-13,
            'megashock_mean': 0,
            'megashock_var': 1e10,
            'random_state': np.random.RandomState(seed=seed)
        }
    })
    
    # Custom properties for analysis
    config['custom_properties'].update({
        'llm_enabled': llm_enabled,
        'num_llm_agents': (num_llm_traders + num_llm_market_makers + (1 if llm_enabled else 0)),
        'num_traditional_agents': (num_noise + num_value + num_momentum + num_market_makers),
        'total_agents': len(agents),
        'configuration_name': 'LLM_Enhanced_ABIDES',
        'configuration_version': '1.0.0'
    })
    
    print(f"ABIDES Configuration Created:")
    print(f"  - Total Agents: {len(agents)}")
    print(f"  - LLM Agents: {config['custom_properties']['num_llm_agents']}")
    print(f"  - Traditional Agents: {config['custom_properties']['num_traditional_agents']}")
    print(f"  - Symbols: {symbols}")
    print(f"  - End Time: {end_time}")
    print(f"  - Log Directory: {log_dir}")
    
    return config


def build_rmsc_llm_config(**kwargs):
    """
    Build RMSC-style configuration with LLM enhancements
    
    This creates a configuration similar to the official RMSC04 but with
    LLM agents added for enhanced market simulation capabilities.
    """
    
    defaults = {
        'num_noise': 1000,
        'num_value': 102,
        'num_momentum': 12,
        'num_market_makers': 2,
        'num_llm_traders': 5,
        'num_llm_market_makers': 1,
        'end_time': "16:00:00",
        'book_logging': True,
        'book_log_depth': 10
    }
    
    # Merge with user-provided kwargs
    config_args = {**defaults, **kwargs}
    
    return build_config(**config_args)


def quick_llm_demo_config(**kwargs):
    """
    Quick demo configuration for testing LLM integration
    
    Smaller scale simulation for development and testing.
    """
    
    demo_defaults = {
        'num_noise': 50,
        'num_value': 10,
        'num_momentum': 5,
        'num_market_makers': 1,
        'num_llm_traders': 3,
        'num_llm_market_makers': 1,
        'end_time': "11:00:00",  # Shorter simulation
        'book_logging': True,
        'stream_history': 100
    }
    
    config_args = {**demo_defaults, **kwargs}
    
    return build_config(**config_args)


# =============================================================================
# EXAMPLE USAGE FUNCTIONS
# =============================================================================

def run_with_abides_framework():
    """
    Example of how to run with the official ABIDES framework
    """
    
    try:
        # This would work with official ABIDES installation
        from abides_core import abides
        
        # Build configuration
        config_state = build_config(
            seed=42,
            end_time='12:00:00',
            llm_enabled=True,
            num_llm_traders=3
        )
        
        # Run simulation
        print("Starting ABIDES simulation with LLM agents...")
        end_state = abides.run(config_state)
        
        print("Simulation completed successfully!")
        return end_state
        
    except ImportError:
        print("Official ABIDES not available. Install ABIDES to run simulations.")
        print("\nTo install ABIDES:")
        print("git clone https://github.com/jpmorganchase/abides-jpmc-public")
        print("cd abides-jpmc-public")
        print("pip install -e .")
        
        # Return configuration for inspection
        return build_config(seed=42, end_time='12:00:00')


def command_line_usage():
    """
    Show how to use this configuration with ABIDES command line
    """
    
    print("ABIDES Command Line Usage:")
    print("=" * 50)
    print()
    print("# Basic usage:")
    print("abides abides_llm_config.py --end_time '12:00:00'")
    print()
    print("# With custom parameters:")
    print("abides abides_llm_config.py --end_time '16:00:00' --seed 42")
    print()
    print("# Multiple configurations:")
    print("# Edit the build_config() call at the bottom of this file")
    print("# to change default parameters")
    print()
    print("Available configuration functions:")
    print("- build_config(): Full customizable configuration")
    print("- build_rmsc_llm_config(): RMSC-style with LLM agents")
    print("- quick_llm_demo_config(): Quick demo configuration")


if __name__ == "__main__":
    # Default configuration when run as a script
    # This allows ABIDES command line to import and use this config
    
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM-Enhanced ABIDES Configuration')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--end_time', type=str, default='12:00:00', help='Simulation end time')
    parser.add_argument('--demo', action='store_true', help='Run quick demo configuration')
    parser.add_argument('--show_usage', action='store_true', help='Show command line usage')
    
    args = parser.parse_args()
    
    if args.show_usage:
        command_line_usage()
    elif args.demo:
        config = quick_llm_demo_config(seed=args.seed, end_time=args.end_time)
        print(f"Demo configuration created with {len(config['agents'])} agents")
    else:
        # This is the configuration that ABIDES will use
        config = build_config(seed=args.seed, end_time=args.end_time)
        
        # Try to run if ABIDES is available
        if ABIDES_AVAILABLE:
            try:
                result = run_with_abides_framework()
                print("ABIDES simulation completed!")
            except Exception as e:
                print(f"Error running ABIDES: {e}")
                print("Configuration created successfully though.")
        else:
            print("Configuration created. Install ABIDES to run simulations.")