"""
Complete Example: LLM-Based Agents with ABIDES Integration
=========================================================

This example demonstrates how to integrate LLM-based trading agents with ABIDES
to create a realistic market simulation that responds to news events and influences
HFT trading behavior.

Key Features:
1. LLM agents analyze news and generate market signals
2. ABIDES agents are influenced by LLM signals
3. Realistic market response to news events
4. Integration with existing ABIDES HFT strategies
"""

import json
import time
import random
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

from llm_abides_integration import NewsCategory
from abides_bridge import ABIDESLLMBridge, demo_abides_integration


class MarketScenarioGenerator:
    """Generate realistic market scenarios for testing"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.scenario_templates = self._load_scenarios()
    
    def _load_scenarios(self) -> Dict[str, Dict]:
        """Load predefined market scenarios"""
        return {
            'earnings_season': {
                'description': 'Quarterly earnings announcements',
                'news_categories': [NewsCategory.EARNINGS, NewsCategory.COMPANY_SPECIFIC],
                'frequency': 0.3,  # Higher news frequency
                'volatility_multiplier': 1.5
            },
            'fed_announcement': {
                'description': 'Federal Reserve policy announcement',
                'news_categories': [NewsCategory.MACRO_ECONOMIC],
                'frequency': 0.1,
                'volatility_multiplier': 2.0
            },
            'merger_activity': {
                'description': 'M&A announcements and rumors',
                'news_categories': [NewsCategory.MERGERS, NewsCategory.REGULATORY],
                'frequency': 0.2,
                'volatility_multiplier': 1.8
            },
            'normal_trading': {
                'description': 'Normal market conditions',
                'news_categories': [NewsCategory.TECHNICAL, NewsCategory.COMPANY_SPECIFIC],
                'frequency': 0.05,
                'volatility_multiplier': 1.0
            }
        }
    
    def generate_scenario_events(self, scenario_name: str, duration_steps: int) -> List[Dict]:
        """Generate events for a specific scenario"""
        scenario = self.scenario_templates.get(scenario_name, self.scenario_templates['normal_trading'])
        events = []
        
        for step in range(duration_steps):
            if random.random() < scenario['frequency']:
                category = random.choice(scenario['news_categories'])
                symbol = random.choice(self.symbols)
                
                event = {
                    'step': step,
                    'type': 'news',
                    'category': category.value,
                    'symbol': symbol,
                    'headline': self._generate_headline(category, symbol),
                    'volatility_impact': scenario['volatility_multiplier']
                }
                events.append(event)
        
        return events
    
    def _generate_headline(self, category: NewsCategory, symbol: str) -> str:
        """Generate realistic headlines for different categories"""
        headlines = {
            NewsCategory.EARNINGS: [
                f"{symbol} reports better-than-expected Q3 earnings",
                f"{symbol} misses revenue estimates, stock slides",
                f"{symbol} raises full-year guidance after strong quarter"
            ],
            NewsCategory.MERGERS: [
                f"Rumors swirl around potential {symbol} takeover",
                f"{symbol} announces acquisition of competitor",
                f"Regulatory approval expected for {symbol} merger"
            ],
            NewsCategory.MACRO_ECONOMIC: [
                "Federal Reserve hints at interest rate changes",
                "Inflation data impacts market sentiment",
                "Trade negotiations show progress"
            ],
            NewsCategory.REGULATORY: [
                f"New regulations may impact {symbol} operations",
                f"SEC investigation into {symbol} disclosed",
                f"{symbol} receives regulatory approval for new product"
            ]
        }
        
        return random.choice(headlines.get(category, [f"Market update affects {symbol}"]))


class PerformanceAnalyzer:
    """Analyze performance of LLM-ABIDES integration"""
    
    def __init__(self):
        self.metrics_history = []
        self.order_flow_analysis = []
        self.signal_effectiveness = []
    
    def record_step_metrics(self, step_data: Dict, market_data: Dict):
        """Record metrics for each simulation step"""
        metrics = {
            'timestamp': step_data['timestamp'],
            'step': len(self.metrics_history),
            'total_orders': step_data['num_orders_generated'],
            'active_signals': step_data['total_active_signals'],
            'market_volatility': self._calculate_volatility(market_data),
            'signal_diversity': self._calculate_signal_diversity(step_data.get('llm_result', {}))
        }
        self.metrics_history.append(metrics)
    
    def _calculate_volatility(self, market_data: Dict) -> float:
        """Calculate market volatility measure"""
        if not market_data:
            return 0.0
        
        spreads = []
        for symbol_data in market_data.values():
            if 'bid' in symbol_data and 'ask' in symbol_data:
                spread = symbol_data['ask'] - symbol_data['bid']
                spreads.append(spread / symbol_data.get('price', 1))
        
        return sum(spreads) / len(spreads) if spreads else 0.0
    
    def _calculate_signal_diversity(self, llm_result: Dict) -> float:
        """Calculate diversity of LLM signals"""
        agent_decisions = llm_result.get('agent_decisions', [])
        if not agent_decisions:
            return 0.0
        
        total_decisions = sum(len(agent['decisions'].get('decisions', [])) for agent in agent_decisions)
        return min(total_decisions / 10.0, 1.0)  # Normalize to 0-1
    
    def analyze_signal_effectiveness(self, bridge: ABIDESLLMBridge) -> Dict:
        """Analyze effectiveness of LLM signals"""
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        
        # Calculate correlations
        signal_order_corr = df['active_signals'].corr(df['total_orders'])
        volatility_signal_corr = df['market_volatility'].corr(df['active_signals'])
        
        # Calculate signal-to-order conversion rate
        total_signals = df['active_signals'].sum()
        total_orders = df['total_orders'].sum()
        conversion_rate = total_orders / total_signals if total_signals > 0 else 0
        
        return {
            'signal_order_correlation': signal_order_corr,
            'volatility_signal_correlation': volatility_signal_corr,
            'signal_to_order_conversion_rate': conversion_rate,
            'average_orders_per_step': df['total_orders'].mean(),
            'average_signals_per_step': df['active_signals'].mean(),
            'max_concurrent_signals': df['active_signals'].max()
        }
    
    def generate_performance_report(self, filepath: str = "performance_report.json"):
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return
        
        df = pd.DataFrame(self.metrics_history)
        
        report = {
            'simulation_summary': {
                'total_steps': len(self.metrics_history),
                'duration': self.metrics_history[-1]['timestamp'],
                'total_orders_generated': df['total_orders'].sum(),
                'total_signals_processed': df['active_signals'].sum()
            },
            'performance_metrics': self.analyze_signal_effectiveness(None),
            'statistical_summary': {
                'orders_per_step': {
                    'mean': df['total_orders'].mean(),
                    'std': df['total_orders'].std(),
                    'min': df['total_orders'].min(),
                    'max': df['total_orders'].max()
                },
                'signals_per_step': {
                    'mean': df['active_signals'].mean(),
                    'std': df['active_signals'].std(),
                    'min': df['active_signals'].min(),
                    'max': df['active_signals'].max()
                }
            },
            'time_series_data': self.metrics_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Performance report saved to {filepath}")


def run_comprehensive_simulation():
    """Run a comprehensive simulation with multiple scenarios"""
    
    print("=== Comprehensive LLM-ABIDES Integration Simulation ===")
    
    # Setup
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    bridge = ABIDESLLMBridge(symbols, num_influenced_agents=15)
    scenario_gen = MarketScenarioGenerator(symbols)
    analyzer = PerformanceAnalyzer()
    
    print(f"Initialized simulation with {len(symbols)} symbols")
    print(f"Created {len(bridge.influenced_agents)} LLM-influenced agents")
    
    # Test different scenarios
    scenarios = ['normal_trading', 'earnings_season', 'fed_announcement', 'merger_activity']
    
    for scenario_name in scenarios:
        print(f"\n--- Testing Scenario: {scenario_name.upper()} ---")
        
        # Generate scenario events
        events = scenario_gen.generate_scenario_events(scenario_name, 10)
        print(f"Generated {len(events)} events for this scenario")
        
        # Run simulation steps for this scenario
        for step in range(10):
            # Generate realistic market data
            market_data = generate_realistic_market_data(symbols, step, scenario_name)
            
            # Process any scheduled events
            step_events = [e for e in events if e['step'] == step]
            for event in step_events:
                if event['type'] == 'news':
                    news_event = bridge.process_news_event(
                        headline=event['headline'],
                        category=event['category']
                    )
                    print(f"  Step {step}: {news_event.headline}")
            
            # Run simulation step
            step_result = bridge.run_simulation_step(market_data)
            
            # Record metrics
            analyzer.record_step_metrics(step_result, market_data)
            
            # Brief progress update
            if step_result['num_orders_generated'] > 0:
                print(f"  Step {step}: {step_result['num_orders_generated']} orders, "
                      f"{step_result['total_active_signals']} active signals")
            
            # Small delay to simulate real-time
            time.sleep(0.1)
    
    # Final analysis
    print("\n=== Simulation Complete ===")
    
    # Generate performance report
    analyzer.generate_performance_report()
    
    # Export final configuration
    bridge.export_abides_config("final_llm_abides_config.json")
    
    # Get and display summary metrics
    performance_metrics = bridge.get_performance_metrics()
    print(f"\nFinal Metrics:")
    print(f"Total simulation steps: {performance_metrics['total_simulation_steps']}")
    print(f"Total orders generated: {performance_metrics['total_orders_generated']}")
    print(f"Average orders per step: {performance_metrics['avg_orders_per_step']:.1f}")
    print(f"Average signals per step: {performance_metrics['avg_signals_per_step']:.1f}")
    
    # Signal effectiveness analysis
    effectiveness = analyzer.analyze_signal_effectiveness(bridge)
    print(f"\nSignal Effectiveness:")
    print(f"Signal-to-order conversion rate: {effectiveness['signal_to_order_conversion_rate']:.2%}")
    print(f"Signal-order correlation: {effectiveness['signal_order_correlation']:.3f}")
    
    return bridge, analyzer


def generate_realistic_market_data(symbols: List[str], step: int, scenario: str) -> Dict:
    """Generate realistic market data based on scenario"""
    
    # Base volatility adjustment based on scenario
    volatility_multipliers = {
        'normal_trading': 1.0,
        'earnings_season': 1.5,
        'fed_announcement': 2.0,
        'merger_activity': 1.3
    }
    
    vol_mult = volatility_multipliers.get(scenario, 1.0)
    
    market_data = {}
    for symbol in symbols:
        # Generate base price with trend
        base_price = 100 + step * random.uniform(-2, 2) * vol_mult
        volatility = random.uniform(0.01, 0.05) * vol_mult
        
        price = base_price * (1 + random.gauss(0, volatility))
        volume = random.randint(1000, 10000) * int(vol_mult)
        
        # Generate realistic bid/ask spread
        spread_bps = random.uniform(1, 10) * vol_mult  # basis points
        spread = price * spread_bps / 10000
        
        bid = price - spread / 2
        ask = price + spread / 2
        
        market_data[symbol] = {
            'last_trade': {
                'price': price,
                'volume': volume
            },
            'order_book': {
                'bids': [
                    {'price': bid - i * spread * 0.1, 'quantity': random.randint(100, 1000)}
                    for i in range(3)
                ],
                'asks': [
                    {'price': ask + i * spread * 0.1, 'quantity': random.randint(100, 1000)}
                    for i in range(3)
                ]
            },
            'price': price,
            'bid': bid,
            'ask': ask,
            'volume': volume
        }
    
    return market_data


def create_visualization_dashboard(analyzer: PerformanceAnalyzer):
    """Create visualization dashboard for simulation results"""
    
    if not analyzer.metrics_history:
        print("No metrics data available for visualization")
        return
    
    df = pd.DataFrame(analyzer.metrics_history)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LLM-ABIDES Integration Performance Dashboard', fontsize=16)
    
    # Plot 1: Orders and Signals over time
    axes[0, 0].plot(df['step'], df['total_orders'], label='Orders Generated', marker='o')
    axes[0, 0].plot(df['step'], df['active_signals'], label='Active Signals', marker='s')
    axes[0, 0].set_title('Orders and Signals Over Time')
    axes[0, 0].set_xlabel('Simulation Step')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Market Volatility
    axes[0, 1].plot(df['step'], df['market_volatility'], color='red', marker='d')
    axes[0, 1].set_title('Market Volatility')
    axes[0, 1].set_xlabel('Simulation Step')
    axes[0, 1].set_ylabel('Volatility Measure')
    axes[0, 1].grid(True)
    
    # Plot 3: Signal Diversity
    axes[1, 0].bar(df['step'], df['signal_diversity'], alpha=0.7, color='green')
    axes[1, 0].set_title('Signal Diversity per Step')
    axes[1, 0].set_xlabel('Simulation Step')
    axes[1, 0].set_ylabel('Diversity Score')
    axes[1, 0].grid(True)
    
    # Plot 4: Correlation scatter plot
    axes[1, 1].scatter(df['active_signals'], df['total_orders'], alpha=0.6)
    axes[1, 1].set_title('Signals vs Orders Correlation')
    axes[1, 1].set_xlabel('Active Signals')
    axes[1, 1].set_ylabel('Orders Generated')
    axes[1, 1].grid(True)
    
    # Add correlation coefficient
    corr = df['active_signals'].corr(df['total_orders'])
    axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=axes[1, 1].transAxes, fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('llm_abides_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Dashboard saved as 'llm_abides_dashboard.png'")


def main():
    """Main function to run the complete integration example"""
    
    print("Starting comprehensive LLM-ABIDES integration example...")
    
    # Run the full simulation
    bridge, analyzer = run_comprehensive_simulation()
    
    # Create visualizations
    create_visualization_dashboard(analyzer)
    
    # Export final data for further analysis
    bridge.llm_bridge.export_signals_for_abides("final_market_signals.json")
    
    print("\n=== Integration Example Complete ===")
    print("Files generated:")
    print("- performance_report.json: Detailed performance metrics")
    print("- final_llm_abides_config.json: ABIDES configuration")
    print("- final_market_signals.json: Market signals for ABIDES")
    print("- llm_abides_dashboard.png: Performance visualization")
    
    print("\nNext steps for ABIDES integration:")
    print("1. Load the configuration file into your ABIDES setup")
    print("2. Use the market signals to influence your HFT agents")
    print("3. Monitor the realistic market response to news events")
    print("4. Adjust LLM agent parameters based on performance metrics")


if __name__ == "__main__":
    main()