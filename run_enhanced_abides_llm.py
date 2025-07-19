#!/usr/bin/env python3
"""
Enhanced ABIDES-LLM Framework Runner
===================================

Complete framework runner that integrates:
- Real LLM-powered trading agents (no mock functions)  
- Realistic order book with proper execution and recording
- ABIDES-style experiments including market impact studies
- Comprehensive data collection and analysis

This script demonstrates how to run the framework with real OpenAI API calls
and collect order book data similar to the ABIDES research platform.
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import our enhanced modules
from enhanced_llm_abides_system import (
    LLMInterface, EnhancedLLMNewsAnalyzer, AdvancedLLMTradingAgent,
    RealisticNewsGenerator, NewsEvent, MarketSignal
)
from realistic_order_book_system import Exchange, Order, OrderType, OrderSide
from abides_experiments import (
    MarketSimulation, MarketImpactExperiment, ExperimentConfig, 
    MarketImpactConfig, generate_experiment_report,
    create_market_impact_experiment, create_strategy_comparison_experiment
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check environment setup and dependencies"""
    logger.info("🔍 Checking environment setup...")
    
    # Check Python version
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
    else:
        logger.warning(f"⚠️  Python {version.major}.{version.minor}.{version.micro} (May have issues)")
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your-openai-api-key-here":
        logger.info("✅ OpenAI API Key Found - Real LLM features enabled")
        llm_available = True
    else:
        logger.warning("⚠️  No OpenAI API Key - Using mock LLM responses")
        llm_available = False
    
    # Check required packages
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'openai', 
        'sqlite3', 'asyncio', 'dataclasses', 'typing'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        logger.error("Please install missing packages with: pip install <package_name>")
        return False
    else:
        logger.info("✅ All required packages found")
    
    # Create output directories
    output_dirs = ['simulation_results', 'order_book_data', 'experiment_reports']
    for dir_name in output_dirs:
        Path(dir_name).mkdir(exist_ok=True)
        logger.info(f"📁 Created directory: {dir_name}")
    
    return True


async def run_simple_demo():
    """Run a simple demonstration of the enhanced framework"""
    logger.info("🚀 Starting Simple Demo with Real LLM Integration")
    
    # Create a simple experiment configuration
    config = ExperimentConfig(
        name="simple_demo",
        description="Simple demonstration of enhanced ABIDES-LLM framework",
        symbols=["AAPL", "MSFT"],
        duration_minutes=30,  # 30 minutes for quick demo
        num_llm_agents=3,
        num_background_agents=10,
        initial_capital=1000000,
        news_frequency=0.2,  # More frequent for demo
        random_seed=42
    )
    
    # Run simulation
    simulation = MarketSimulation(config)
    results = await simulation.run_simulation()
    
    # Generate report
    report_path = "simulation_results/simple_demo_report.html"
    generate_experiment_report(results, report_path)
    
    logger.info(f"✅ Simple demo completed! Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SIMPLE DEMO RESULTS SUMMARY")
    print("="*60)
    
    # Agent performance
    print("\nAgent Performance:")
    for agent_id, perf in results['agent_performance'].items():
        print(f"  {agent_id} ({perf['strategy']}): "
              f"{perf['total_return']*100:.2f}% return, "
              f"{perf['num_trades']} trades")
    
    # Market statistics
    print("\nMarket Statistics:")
    for symbol, data in results['market_report']['symbols'].items():
        stats = data['stats']
        print(f"  {symbol}: {stats['trade_count']} trades, "
              f"${stats['value_traded']:,.2f} total value")
    
    print(f"\nNews Events Generated: {len(results['news_events'])}")
    print(f"Total Agent Decisions: {len(results['agent_decisions'])}")
    
    return results


async def run_market_impact_experiment():
    """Run a comprehensive market impact experiment"""
    logger.info("📊 Starting Market Impact Experiment")
    
    # Create experiment configurations
    base_config, impact_config = create_market_impact_experiment()
    
    # Reduce duration for demo purposes
    base_config.duration_minutes = 120  # 2 hours
    impact_config.impact_order_sizes = [5000, 15000, 30000]  # Fewer sizes for speed
    
    # Run experiment
    experiment = MarketImpactExperiment(base_config, impact_config)
    results = await experiment.run_experiment()
    
    # Save detailed results
    results_path = "simulation_results/market_impact_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ Market impact experiment completed! Results saved to: {results_path}")
    
    # Print impact analysis
    print("\n" + "="*60)
    print("MARKET IMPACT EXPERIMENT RESULTS")
    print("="*60)
    
    impact_analysis = results.get('impact_analysis', {})
    price_impacts = impact_analysis.get('price_impact', {})
    
    if price_impacts:
        print("\nPrice Impact by Order Size:")
        for order_size, impact in price_impacts.items():
            print(f"  {order_size:,} shares: {impact*100:.3f}% price impact")
    
    recovery_times = impact_analysis.get('recovery_time', {})
    if recovery_times:
        print("\nRecovery Times:")
        for order_size, time in recovery_times.items():
            print(f"  {order_size:,} shares: {time} minutes to recover")
    
    return results


async def run_strategy_comparison_experiment():
    """Run strategy comparison experiment"""
    logger.info("🔬 Starting Strategy Comparison Experiment")
    
    config = create_strategy_comparison_experiment()
    config.duration_minutes = 120  # 2 hours for demo
    
    simulation = MarketSimulation(config)
    results = await simulation.run_simulation()
    
    # Generate report
    report_path = "experiment_reports/strategy_comparison_report.html"
    generate_experiment_report(results, report_path)
    
    logger.info(f"✅ Strategy comparison completed! Report saved to: {report_path}")
    
    # Analyze strategy performance
    print("\n" + "="*60)
    print("STRATEGY COMPARISON RESULTS")
    print("="*60)
    
    strategy_performance = {}
    for agent_id, perf in results['agent_performance'].items():
        strategy = perf['strategy']
        if strategy not in strategy_performance:
            strategy_performance[strategy] = []
        strategy_performance[strategy].append(perf['total_return'])
    
    print("\nAverage Returns by Strategy:")
    for strategy, returns in strategy_performance.items():
        avg_return = sum(returns) / len(returns)
        print(f"  {strategy}: {avg_return*100:.2f}% average return")
    
    return results


def save_order_book_analysis(exchange: Exchange, output_path: str = "order_book_analysis.json"):
    """Save comprehensive order book analysis"""
    logger.info("💾 Saving order book analysis...")
    
    analysis = {}
    
    for symbol in exchange.symbols:
        book = exchange.order_books[symbol]
        
        # Get current snapshot
        snapshot = book.get_book_snapshot(depth=10)
        stats = book.get_market_stats()
        
        # Analyze order book depth
        bid_depth = sum(qty for price, qty in snapshot['bids'])
        ask_depth = sum(qty for price, qty in snapshot['asks'])
        
        # Analyze trade patterns
        trade_sizes = [trade.quantity for trade in book.trades]
        trade_prices = [trade.price for trade in book.trades]
        
        analysis[symbol] = {
            'snapshot': snapshot,
            'stats': stats,
            'order_book_depth': {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': bid_depth + ask_depth
            },
            'trade_analysis': {
                'avg_trade_size': sum(trade_sizes) / len(trade_sizes) if trade_sizes else 0,
                'trade_size_std': pd.Series(trade_sizes).std() if trade_sizes else 0,
                'price_volatility': pd.Series(trade_prices).std() if trade_prices else 0,
                'total_trades': len(book.trades)
            }
        }
    
    # Save analysis
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    logger.info(f"📊 Order book analysis saved to: {output_path}")


async def run_interactive_mode():
    """Run interactive mode for exploring the framework"""
    print("\n" + "="*60)
    print("ENHANCED ABIDES-LLM INTERACTIVE MODE")
    print("="*60)
    print("Choose an experiment to run:")
    print("1. Simple Demo (30 minutes)")
    print("2. Market Impact Experiment (2 hours)")
    print("3. Strategy Comparison (2 hours)")
    print("4. Custom Experiment")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                await run_simple_demo()
                break
            elif choice == "2":
                await run_market_impact_experiment()
                break
            elif choice == "3":
                await run_strategy_comparison_experiment()
                break
            elif choice == "4":
                await run_custom_experiment()
                break
            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")


async def run_custom_experiment():
    """Run a custom user-defined experiment"""
    print("\n" + "="*40)
    print("CUSTOM EXPERIMENT CONFIGURATION")
    print("="*40)
    
    try:
        # Get user inputs
        name = input("Experiment name: ").strip() or "custom_experiment"
        
        symbols_input = input("Symbols (comma-separated, default: AAPL,MSFT): ").strip()
        symbols = [s.strip().upper() for s in symbols_input.split(",")] if symbols_input else ["AAPL", "MSFT"]
        
        duration = int(input("Duration in minutes (default: 60): ") or "60")
        
        num_llm_agents = int(input("Number of LLM agents (default: 5): ") or "5")
        num_bg_agents = int(input("Number of background agents (default: 20): ") or "20")
        
        news_freq = float(input("News frequency (events per minute, default: 0.1): ") or "0.1")
        
        # Create configuration
        config = ExperimentConfig(
            name=name,
            description="Custom user-defined experiment",
            symbols=symbols,
            duration_minutes=duration,
            num_llm_agents=num_llm_agents,
            num_background_agents=num_bg_agents,
            news_frequency=news_freq,
            initial_capital=1000000,
            random_seed=42
        )
        
        logger.info(f"🎯 Running custom experiment: {name}")
        
        # Run simulation
        simulation = MarketSimulation(config)
        results = await simulation.run_simulation()
        
        # Generate report
        report_path = f"experiment_reports/{name}_report.html"
        generate_experiment_report(results, report_path)
        
        logger.info(f"✅ Custom experiment completed! Report saved to: {report_path}")
        
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
    except Exception as e:
        logger.error(f"Error in custom experiment: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced ABIDES-LLM Framework with Real API Integration"
    )
    parser.add_argument(
        "--mode", 
        choices=["demo", "impact", "strategy", "custom", "interactive"],
        default="interactive",
        help="Experiment mode to run"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check environment setup"
    )
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    # Check environment
    if not check_environment():
        logger.error("❌ Environment check failed!")
        sys.exit(1)
    
    if args.check_only:
        logger.info("✅ Environment check completed successfully!")
        return
    
    print("\n" + "="*80)
    print("🤖 ENHANCED ABIDES-LLM FRAMEWORK")
    print("Real LLM Integration + Realistic Order Books + ABIDES-Style Experiments")
    print("="*80)
    
    # Run selected mode
    try:
        if args.mode == "demo":
            asyncio.run(run_simple_demo())
        elif args.mode == "impact":
            asyncio.run(run_market_impact_experiment())
        elif args.mode == "strategy":
            asyncio.run(run_strategy_comparison_experiment())
        elif args.mode == "custom":
            asyncio.run(run_custom_experiment())
        else:  # interactive
            asyncio.run(run_interactive_mode())
            
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error running experiment: {e}")
        raise
    
    print("\n" + "="*80)
    print("🎉 Framework execution completed!")
    print("Check the generated reports in simulation_results/ and experiment_reports/")
    print("Order book data is stored in the SQLite database: market_simulation.db")
    print("="*80)


if __name__ == "__main__":
    main()