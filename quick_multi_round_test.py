#!/usr/bin/env python3
"""
Quick Multi-Round Market Simulation Test
========================================

Fast, lightweight test of market simulation concepts with multiple rounds
for demonstration and validation purposes.
"""

import time
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QuickTestConfig:
    """Configuration for a quick test round"""
    name: str
    symbols: List[str]
    agents_count: int
    duration_minutes: float
    news_frequency: float  # events per minute
    market_regime: str  # "bull", "bear", "volatile", "stable"
    volatility_level: float  # 0.1 to 1.0

@dataclass
class MockAgent:
    """Mock trading agent for testing"""
    agent_id: str
    strategy: str
    initial_capital: float
    current_cash: float
    positions: Dict[str, int]
    total_trades: int = 0
    total_pnl: float = 0.0
    
class MockMarketData:
    """Mock market data generator"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.prices = {symbol: 100.0 + random.uniform(-10, 10) for symbol in symbols}
        self.volatilities = {symbol: random.uniform(0.1, 0.4) for symbol in symbols}
    
    def update_prices(self, market_regime: str, volatility_level: float):
        """Update prices based on market regime"""
        
        for symbol in self.symbols:
            base_vol = self.volatilities[symbol] * volatility_level
            
            if market_regime == "bull":
                trend = random.uniform(0.001, 0.003)  # Positive trend
            elif market_regime == "bear":
                trend = random.uniform(-0.003, -0.001)  # Negative trend
            elif market_regime == "volatile":
                trend = random.uniform(-0.002, 0.002)  # No clear trend
                base_vol *= 2  # Higher volatility
            else:  # stable
                trend = random.uniform(-0.0005, 0.0005)  # Minimal trend
                base_vol *= 0.5  # Lower volatility
            
            # Update price with trend and random walk
            price_change = trend + random.gauss(0, base_vol)
            self.prices[symbol] *= (1 + price_change)
            
            # Keep prices reasonable
            self.prices[symbol] = max(10, min(1000, self.prices[symbol]))
    
    def get_current_data(self) -> Dict[str, Dict]:
        """Get current market data"""
        return {
            symbol: {
                'price': price,
                'bid': price * 0.999,
                'ask': price * 1.001,
                'volume': random.randint(1000, 10000)
            }
            for symbol, price in self.prices.items()
        }

class MockNewsGenerator:
    """Generate mock news events"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.news_templates = [
            "{company} reports {performance} quarterly earnings",
            "{company} announces {type} partnership with tech giant",
            "Analysts {action} {company} price target",
            "{company} {event} regulatory approval for new product",
            "Market volatility affects {company} trading",
            "{company} CEO announces {announcement} strategy"
        ]
    
    def generate_event(self) -> Dict[str, Any]:
        """Generate a news event"""
        
        symbol = random.choice(self.symbols)
        template = random.choice(self.news_templates)
        
        # Fill template
        sentiment = random.uniform(-0.8, 0.8)
        
        replacements = {
            'company': symbol,
            'performance': 'strong' if sentiment > 0.2 else 'weak' if sentiment < -0.2 else 'mixed',
            'type': random.choice(['strategic', 'technology', 'manufacturing']),
            'action': 'raise' if sentiment > 0 else 'lower' if sentiment < 0 else 'maintain',
            'event': 'receives' if sentiment > 0 else 'loses' if sentiment < -0.3 else 'awaits',
            'announcement': 'expansion' if sentiment > 0 else 'restructuring' if sentiment < 0 else 'operational'
        }
        
        headline = template.format(**replacements)
        
        return {
            'timestamp': datetime.now(),
            'headline': headline,
            'symbol': symbol,
            'sentiment': sentiment,
            'impact': random.uniform(0.1, 0.9)
        }

class QuickMarketSimulation:
    """Quick market simulation for testing"""
    
    def __init__(self, config: QuickTestConfig):
        self.config = config
        self.market_data = MockMarketData(config.symbols)
        self.news_generator = MockNewsGenerator(config.symbols)
        
        # Create mock agents
        self.agents = self._create_agents()
        
        # Simulation state
        self.current_time = datetime.now()
        self.news_events = []
        self.trade_history = []
        self.market_history = []
        
        logger.info(f"Initialized simulation: {config.name}")
        logger.info(f"Symbols: {config.symbols}")
        logger.info(f"Agents: {config.agents_count}")
        logger.info(f"Duration: {config.duration_minutes} minutes")
    
    def _create_agents(self) -> List[MockAgent]:
        """Create mock trading agents"""
        
        agents = []
        strategies = ["momentum", "mean_reversion", "trend_following", "contrarian", "arbitrage"]
        
        for i in range(self.config.agents_count):
            strategy = strategies[i % len(strategies)]
            agent = MockAgent(
                agent_id=f"Agent_{strategy}_{i+1}",
                strategy=strategy,
                initial_capital=1_000_000,
                current_cash=1_000_000,
                positions={symbol: 0 for symbol in self.config.symbols}
            )
            agents.append(agent)
        
        return agents
    
    def run_simulation(self) -> Dict[str, Any]:
        """Run the simulation"""
        
        start_time = datetime.now()
        steps = int(self.config.duration_minutes * 60 / 10)  # 10-second steps
        
        logger.info(f"Starting simulation with {steps} steps...")
        
        for step in range(steps):
            self._execute_simulation_step(step)
            
            # Brief pause for realism
            time.sleep(0.1)
            
            if step % 20 == 0:  # Log every 20 steps
                logger.info(f"Step {step}/{steps}: {len(self.news_events)} news events, {len(self.trade_history)} trades")
        
        end_time = datetime.now()
        
        # Compile results
        results = self._compile_results(start_time, end_time)
        
        logger.info(f"Simulation complete: {len(self.trade_history)} trades executed")
        
        return results
    
    def _execute_simulation_step(self, step: int):
        """Execute one simulation step"""
        
        # Update market data
        self.market_data.update_prices(self.config.market_regime, self.config.volatility_level)
        current_data = self.market_data.get_current_data()
        self.market_history.append({
            'step': step,
            'timestamp': self.current_time,
            'data': current_data.copy()
        })
        
        # Generate news events
        if random.random() < self.config.news_frequency / 6:  # Adjust for step frequency
            news_event = self.news_generator.generate_event()
            self.news_events.append(news_event)
            
            # Process news with agents
            self._process_news_with_agents(news_event, current_data)
        
        # Execute random trading for some agents
        self._execute_random_trading(current_data)
        
        # Update time
        self.current_time += timedelta(seconds=10)
    
    def _process_news_with_agents(self, news_event: Dict[str, Any], market_data: Dict[str, Dict]):
        """Process news event with agents"""
        
        affected_symbol = news_event['symbol']
        sentiment = news_event['sentiment']
        impact = news_event['impact']
        
        for agent in self.agents:
            # Agent decides whether to trade based on strategy and news
            if self._should_agent_trade(agent, news_event):
                trade = self._generate_trade(agent, affected_symbol, sentiment, impact, market_data)
                if trade:
                    self.trade_history.append(trade)
                    self._execute_trade(agent, trade, market_data)
    
    def _should_agent_trade(self, agent: MockAgent, news_event: Dict[str, Any]) -> bool:
        """Determine if agent should trade on news"""
        
        base_probability = 0.3  # 30% base chance
        
        # Strategy-based adjustments
        if agent.strategy == "momentum" and abs(news_event['sentiment']) > 0.5:
            base_probability += 0.3
        elif agent.strategy == "contrarian" and abs(news_event['sentiment']) > 0.4:
            base_probability += 0.2
        elif agent.strategy == "trend_following":
            base_probability += 0.1
        
        # Impact-based adjustment
        base_probability += news_event['impact'] * 0.2
        
        return random.random() < base_probability
    
    def _generate_trade(self, agent: MockAgent, symbol: str, sentiment: float, 
                       impact: float, market_data: Dict[str, Dict]) -> Optional[Dict[str, Any]]:
        """Generate a trade for an agent"""
        
        if symbol not in market_data:
            return None
        
        current_price = market_data[symbol]['price']
        
        # Determine trade direction based on strategy and sentiment
        if agent.strategy == "momentum":
            side = "BUY" if sentiment > 0 else "SELL"
        elif agent.strategy == "contrarian":
            side = "SELL" if sentiment > 0 else "BUY"
        else:
            side = random.choice(["BUY", "SELL"])
        
        # Calculate trade size
        risk_factor = min(impact, 0.5)  # Max 50% of capital at risk
        max_trade_value = agent.current_cash * risk_factor
        
        if side == "BUY":
            quantity = int(max_trade_value / current_price)
            if quantity < 1 or quantity * current_price > agent.current_cash:
                return None
        else:  # SELL
            quantity = min(100, abs(agent.positions.get(symbol, 0)))  # Can't sell more than owned
            if quantity < 1:
                return None
        
        return {
            'timestamp': self.current_time,
            'agent_id': agent.agent_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': current_price,
            'strategy': agent.strategy,
            'news_driven': True,
            'sentiment': sentiment
        }
    
    def _execute_random_trading(self, market_data: Dict[str, Dict]):
        """Execute some random trading activity"""
        
        for agent in self.agents:
            if random.random() < 0.1:  # 10% chance per step
                symbol = random.choice(self.config.symbols)
                
                # Random trade
                trade = {
                    'timestamp': self.current_time,
                    'agent_id': agent.agent_id,
                    'symbol': symbol,
                    'side': random.choice(["BUY", "SELL"]),
                    'quantity': random.randint(1, 50),
                    'price': market_data[symbol]['price'],
                    'strategy': agent.strategy,
                    'news_driven': False,
                    'sentiment': 0
                }
                
                self.trade_history.append(trade)
                self._execute_trade(agent, trade, market_data)
    
    def _execute_trade(self, agent: MockAgent, trade: Dict[str, Any], market_data: Dict[str, Dict]):
        """Execute trade and update agent portfolio"""
        
        symbol = trade['symbol']
        side = trade['side']
        quantity = trade['quantity']
        price = trade['price']
        
        if side == "BUY":
            cost = quantity * price
            if cost <= agent.current_cash:
                agent.current_cash -= cost
                agent.positions[symbol] += quantity
                agent.total_trades += 1
        else:  # SELL
            if agent.positions.get(symbol, 0) >= quantity:
                proceeds = quantity * price
                agent.current_cash += proceeds
                agent.positions[symbol] -= quantity
                agent.total_trades += 1
        
        # Update PnL (simplified)
        current_portfolio_value = self._calculate_portfolio_value(agent, market_data)
        agent.total_pnl = current_portfolio_value - agent.initial_capital
    
    def _calculate_portfolio_value(self, agent: MockAgent, market_data: Dict[str, Dict]) -> float:
        """Calculate current portfolio value"""
        
        total_value = agent.current_cash
        
        for symbol, quantity in agent.positions.items():
            if quantity > 0 and symbol in market_data:
                total_value += quantity * market_data[symbol]['price']
        
        return total_value
    
    def _compile_results(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Compile simulation results"""
        
        duration = (end_time - start_time).total_seconds()
        
        # Agent performance summary
        agent_performances = []
        total_pnl = 0
        
        final_market_data = self.market_data.get_current_data()
        
        for agent in self.agents:
            final_portfolio_value = self._calculate_portfolio_value(agent, final_market_data)
            pnl = final_portfolio_value - agent.initial_capital
            
            performance = {
                'agent_id': agent.agent_id,
                'strategy': agent.strategy,
                'initial_capital': agent.initial_capital,
                'final_value': final_portfolio_value,
                'pnl': pnl,
                'pnl_percent': (pnl / agent.initial_capital) * 100,
                'total_trades': agent.total_trades,
                'final_cash': agent.current_cash,
                'positions': dict(agent.positions)
            }
            
            agent_performances.append(performance)
            total_pnl += pnl
        
        # Trading statistics
        news_driven_trades = [t for t in self.trade_history if t.get('news_driven', False)]
        
        # Market statistics
        price_changes = {}
        if self.market_history:
            initial_prices = self.market_history[0]['data']
            final_prices = self.market_history[-1]['data']
            
            for symbol in self.config.symbols:
                initial = initial_prices[symbol]['price']
                final = final_prices[symbol]['price']
                price_changes[symbol] = (final - initial) / initial * 100
        
        return {
            'simulation_config': asdict(self.config),
            'execution_summary': {
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': duration,
                'total_trades': len(self.trade_history),
                'news_driven_trades': len(news_driven_trades),
                'news_events_count': len(self.news_events),
                'agents_count': len(self.agents)
            },
            'performance_summary': {
                'total_pnl': total_pnl,
                'avg_pnl_per_agent': total_pnl / len(self.agents) if self.agents else 0,
                'profitable_agents': len([p for p in agent_performances if p['pnl'] > 0]),
                'best_performer': max(agent_performances, key=lambda x: x['pnl']) if agent_performances else None,
                'worst_performer': min(agent_performances, key=lambda x: x['pnl']) if agent_performances else None
            },
            'market_summary': {
                'price_changes_percent': price_changes,
                'market_regime': self.config.market_regime,
                'volatility_level': self.config.volatility_level,
                'avg_price_change': np.mean(list(price_changes.values())) if price_changes else 0
            },
            'detailed_results': {
                'agent_performances': agent_performances,
                'trade_history': self.trade_history[-100:],  # Last 100 trades
                'news_events': self.news_events,
                'final_market_data': final_market_data
            }
        }

class QuickMultiRoundTester:
    """Run multiple quick simulation rounds"""
    
    def __init__(self, output_dir: str = "quick_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_configs = self._create_test_configs()
        self.results = []
        
        logger.info(f"Initialized quick tester with {len(self.test_configs)} configurations")
    
    def _create_test_configs(self) -> List[QuickTestConfig]:
        """Create test configurations"""
        
        return [
            QuickTestConfig(
                name="quick_stable_test",
                symbols=["AAPL", "MSFT"],
                agents_count=5,
                duration_minutes=1.0,
                news_frequency=2.0,  # 2 events per minute
                market_regime="stable",
                volatility_level=0.3
            ),
            
            QuickTestConfig(
                name="bull_market_test",
                symbols=["AAPL", "MSFT", "GOOGL"],
                agents_count=8,
                duration_minutes=1.5,
                news_frequency=1.5,
                market_regime="bull",
                volatility_level=0.4
            ),
            
            QuickTestConfig(
                name="bear_market_test",
                symbols=["AAPL", "MSFT", "GOOGL"],
                agents_count=8,
                duration_minutes=1.5,
                news_frequency=2.5,  # More news in bear markets
                market_regime="bear",
                volatility_level=0.5
            ),
            
            QuickTestConfig(
                name="volatile_market_test",
                symbols=["AAPL", "MSFT", "AMZN", "TSLA"],
                agents_count=10,
                duration_minutes=2.0,
                news_frequency=3.0,  # High news frequency
                market_regime="volatile",
                volatility_level=0.8
            ),
            
            QuickTestConfig(
                name="large_scale_test",
                symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"],
                agents_count=15,
                duration_minutes=2.5,
                news_frequency=1.0,
                market_regime="stable",
                volatility_level=0.4
            )
        ]
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test configurations"""
        
        logger.info("🚀 Starting Quick Multi-Round Market Simulation Tests")
        logger.info("=" * 70)
        
        total_start_time = datetime.now()
        
        for i, config in enumerate(self.test_configs, 1):
            logger.info(f"\n📊 Running Test {i}/{len(self.test_configs)}: {config.name}")
            logger.info(f"Market Regime: {config.market_regime.title()}")
            logger.info(f"Symbols: {len(config.symbols)}, Agents: {config.agents_count}")
            
            try:
                # Run simulation
                simulation = QuickMarketSimulation(config)
                result = simulation.run_simulation()
                result['test_number'] = i
                result['success'] = True
                
                self.results.append(result)
                
                # Log summary
                summary = result['performance_summary']
                logger.info(f"✅ Test completed successfully")
                logger.info(f"   Total PnL: ${summary['total_pnl']:,.2f}")
                logger.info(f"   Profitable Agents: {summary['profitable_agents']}/{config.agents_count}")
                logger.info(f"   Total Trades: {result['execution_summary']['total_trades']}")
                
            except Exception as e:
                logger.error(f"❌ Test {config.name} failed: {e}")
                self.results.append({
                    'simulation_config': asdict(config),
                    'test_number': i,
                    'success': False,
                    'error': str(e)
                })
        
        total_end_time = datetime.now()
        total_duration = (total_end_time - total_start_time).total_seconds()
        
        # Compile final analysis
        final_analysis = self._analyze_results(total_duration)
        
        # Save results
        self._save_results(final_analysis)
        
        # Print summary
        self._print_summary(final_analysis)
        
        return final_analysis
    
    def _analyze_results(self, total_duration: float) -> Dict[str, Any]:
        """Analyze all test results"""
        
        successful_results = [r for r in self.results if r.get('success', False)]
        
        if not successful_results:
            return {
                'success_rate': 0,
                'total_duration': total_duration,
                'error': 'No successful tests'
            }
        
        # Overall statistics
        total_trades = sum(r['execution_summary']['total_trades'] for r in successful_results)
        total_pnl = sum(r['performance_summary']['total_pnl'] for r in successful_results)
        total_agents = sum(r['execution_summary']['agents_count'] for r in successful_results)
        
        # Performance by market regime
        regime_performance = {}
        for result in successful_results:
            regime = result['simulation_config']['market_regime']
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(result['performance_summary']['total_pnl'])
        
        regime_summary = {}
        for regime, pnls in regime_performance.items():
            regime_summary[regime] = {
                'avg_pnl': np.mean(pnls),
                'total_pnl': sum(pnls),
                'test_count': len(pnls)
            }
        
        # Best and worst tests
        best_test = max(successful_results, key=lambda x: x['performance_summary']['total_pnl'])
        worst_test = min(successful_results, key=lambda x: x['performance_summary']['total_pnl'])
        
        return {
            'test_summary': {
                'total_tests': len(self.test_configs),
                'successful_tests': len(successful_results),
                'success_rate': len(successful_results) / len(self.test_configs),
                'total_duration_seconds': total_duration,
                'avg_test_duration': total_duration / len(self.test_configs)
            },
            'performance_summary': {
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'total_agents': total_agents,
                'avg_pnl_per_agent': total_pnl / total_agents if total_agents > 0 else 0,
                'trades_per_second': total_trades / total_duration if total_duration > 0 else 0
            },
            'regime_analysis': regime_summary,
            'best_test': {
                'name': best_test['simulation_config']['name'],
                'pnl': best_test['performance_summary']['total_pnl'],
                'regime': best_test['simulation_config']['market_regime']
            },
            'worst_test': {
                'name': worst_test['simulation_config']['name'],
                'pnl': worst_test['performance_summary']['total_pnl'],
                'regime': worst_test['simulation_config']['market_regime']
            },
            'detailed_results': self.results
        }
    
    def _save_results(self, final_analysis: Dict[str, Any]):
        """Save test results"""
        
        # Save main results
        results_file = self.output_dir / "quick_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_analysis, f, indent=2, default=str)
        
        logger.info(f"📊 Saved results to: {results_file}")
        
        # Save CSV summary
        if final_analysis.get('detailed_results'):
            csv_data = []
            for result in final_analysis['detailed_results']:
                if result.get('success'):
                    config = result['simulation_config']
                    performance = result['performance_summary']
                    
                    csv_data.append({
                        'test_name': config['name'],
                        'market_regime': config['market_regime'],
                        'symbols_count': len(config['symbols']),
                        'agents_count': config['agents_count'],
                        'duration_minutes': config['duration_minutes'],
                        'total_trades': result['execution_summary']['total_trades'],
                        'total_pnl': performance['total_pnl'],
                        'profitable_agents': performance['profitable_agents'],
                        'success': True
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_file = self.output_dir / "test_summary.csv"
                df.to_csv(csv_file, index=False)
                logger.info(f"📈 Saved CSV summary to: {csv_file}")
    
    def _print_summary(self, final_analysis: Dict[str, Any]):
        """Print comprehensive test summary"""
        
        print("\n" + "="*70)
        print("🎯 QUICK MULTI-ROUND TEST RESULTS")
        print("="*70)
        
        test_summary = final_analysis['test_summary']
        print(f"📊 Total Tests: {test_summary['total_tests']}")
        print(f"✅ Successful Tests: {test_summary['successful_tests']}")
        print(f"📈 Success Rate: {test_summary['success_rate']:.1%}")
        print(f"⏱️  Total Duration: {test_summary['total_duration_seconds']:.1f} seconds")
        print(f"⚡ Avg Test Duration: {test_summary['avg_test_duration']:.1f} seconds")
        
        if 'performance_summary' in final_analysis:
            perf = final_analysis['performance_summary']
            print(f"\n💹 OVERALL PERFORMANCE")
            print(f"📊 Total Trades: {perf['total_trades']:,}")
            print(f"💰 Total PnL: ${perf['total_pnl']:,.2f}")
            print(f"🤖 Total Agents: {perf['total_agents']}")
            print(f"📈 Avg PnL per Agent: ${perf['avg_pnl_per_agent']:,.2f}")
            print(f"⚡ Trades per Second: {perf['trades_per_second']:.2f}")
        
        if 'regime_analysis' in final_analysis:
            print(f"\n🌍 MARKET REGIME ANALYSIS")
            for regime, data in final_analysis['regime_analysis'].items():
                print(f"{regime.title()} Market:")
                print(f"   Avg PnL: ${data['avg_pnl']:,.2f}")
                print(f"   Total PnL: ${data['total_pnl']:,.2f}")
                print(f"   Tests: {data['test_count']}")
        
        if 'best_test' in final_analysis and 'worst_test' in final_analysis:
            print(f"\n🏆 BEST/WORST TESTS")
            best = final_analysis['best_test']
            worst = final_analysis['worst_test']
            print(f"🥇 Best: {best['name']} (PnL: ${best['pnl']:,.2f}, Regime: {best['regime']})")
            print(f"🥉 Worst: {worst['name']} (PnL: ${worst['pnl']:,.2f}, Regime: {worst['regime']})")
        
        print(f"\n📁 Results saved to: {self.output_dir}")
        print("="*70)
        print("🎉 Quick Multi-Round Test Complete!")
        print("="*70)


def main():
    """Main function to run quick tests"""
    
    print("🚀 Quick Multi-Round Market Simulation Tester")
    print("=" * 60)
    print("Running lightweight simulation tests to demonstrate functionality...")
    print()
    
    try:
        # Create and run tester
        tester = QuickMultiRoundTester()
        final_analysis = tester.run_all_tests()
        
        return final_analysis.get('test_summary', {}).get('success_rate', 0) > 0.5
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)