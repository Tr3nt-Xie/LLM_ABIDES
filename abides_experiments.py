"""
ABIDES-Style Experimental Framework
==================================

Comprehensive experimental framework for testing LLM trading agents
in realistic market simulations, implementing experiments similar to
those described in the ABIDES research papers.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import itertools

from enhanced_llm_abides_system import (
    LLMInterface, EnhancedLLMNewsAnalyzer, AdvancedLLMTradingAgent,
    RealisticNewsGenerator, NewsEvent, MarketSignal, NewsCategory
)
from realistic_order_book_system import Exchange, Order, OrderType, OrderSide

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for market experiments"""
    name: str
    description: str
    symbols: List[str]
    duration_minutes: int = 480  # 8 hours
    num_llm_agents: int = 5
    num_background_agents: int = 20
    initial_capital: float = 1000000
    news_frequency: float = 0.1  # events per minute
    save_results: bool = True
    record_order_book: bool = True
    random_seed: int = 42


@dataclass
class MarketImpactConfig:
    """Configuration for market impact experiments"""
    impact_agent_id: str = "IMPACT_AGENT"
    impact_symbol: str = "AAPL"
    impact_time_minutes: int = 60  # Impact occurs 60 minutes into simulation
    impact_order_sizes: List[int] = None  # Will be set to [1000, 5000, 10000, 25000]
    impact_direction: str = "BUY"  # BUY or SELL
    
    def __post_init__(self):
        if self.impact_order_sizes is None:
            self.impact_order_sizes = [1000, 5000, 10000, 25000]


class BackgroundAgent:
    """Simple background trading agent to provide market liquidity"""
    
    def __init__(self, agent_id: str, symbols: List[str], initial_capital: float):
        self.agent_id = agent_id
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {symbol: 0 for symbol in symbols}
        self.orders = []
        self.trade_frequency = np.random.uniform(30, 180)  # seconds between trades
        self.risk_tolerance = np.random.uniform(0.3, 0.8)
        
    def generate_order(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Generate a random order for market making"""
        if symbol not in market_data or 'bid_price' not in market_data[symbol]:
            return None
        
        bid = market_data[symbol].get('bid_price')
        ask = market_data[symbol].get('ask_price')
        last_price = market_data[symbol].get('last_trade_price')
        
        # Determine if we should trade
        if np.random.random() > 0.3:  # 30% chance to trade
            return None
        
        # Generate order parameters
        side = np.random.choice(['BUY', 'SELL'])
        quantity = np.random.randint(100, 1000)
        
        # Price around market with some randomness
        # Use available price data with smart fallbacks
        if bid and ask:
            if side == 'BUY':
                # For buy orders, use bid as base with small adjustment
                price = bid + np.random.uniform(-0.01, 0.02) * (bid if bid else 100.0)
            else:
                # For sell orders, use ask as base with small adjustment
                price = ask + np.random.uniform(-0.02, 0.01) * (ask if ask else 100.0)
        elif last_price:
            # Use last trade price if no bid/ask available
            price = last_price * (1 + np.random.uniform(-0.01, 0.01))
        else:
            # Fallback to a reasonable default price
            price = 100.0 * (1 + np.random.uniform(-0.01, 0.01))
        
        return {
            'agent_id': self.agent_id,
            'symbol': symbol,
            'side': side,
            'order_type': 'LIMIT',
            'quantity': quantity,
            'price': round(price, 2)
        }


class MarketSimulation:
    """Core market simulation engine"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.exchange = Exchange(config.symbols)
        self.llm_agents: List[AdvancedLLMTradingAgent] = []
        self.background_agents: List[BackgroundAgent] = []
        self.news_generator = RealisticNewsGenerator(config.symbols)
        self.news_analyzer = EnhancedLLMNewsAnalyzer(config.symbols)
        
        # Simulation state
        self.current_time = datetime.now()
        self.start_time = self.current_time
        self.end_time = self.start_time + timedelta(minutes=config.duration_minutes)
        self.is_running = False
        
        # Data collection
        self.market_data_history = []
        self.news_events = []
        self.agent_decisions = []
        self.performance_data = []
        
        # Initialize agents
        self._initialize_agents()
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        
        logger.info(f"Market simulation initialized: {config.name}")
    
    def _initialize_agents(self):
        """Initialize LLM and background agents"""
        # Create LLM agents with different strategies
        strategies = ['momentum', 'value', 'volatility', 'arbitrage']
        
        for i in range(self.config.num_llm_agents):
            strategy = strategies[i % len(strategies)]
            risk_tolerance = np.random.uniform(0.3, 0.8)
            
            agent = AdvancedLLMTradingAgent(
                agent_id=f"LLM_AGENT_{i+1}",
                strategy_type=strategy,
                initial_capital=self.config.initial_capital,
                symbols=self.config.symbols,
                risk_tolerance=risk_tolerance
            )
            self.llm_agents.append(agent)
        
        # Create background agents
        for i in range(self.config.num_background_agents):
            agent = BackgroundAgent(
                agent_id=f"BG_AGENT_{i+1}",
                symbols=self.config.symbols,
                initial_capital=self.config.initial_capital * 0.5  # Smaller capital
            )
            self.background_agents.append(agent)
        
        logger.info(f"Initialized {len(self.llm_agents)} LLM agents and {len(self.background_agents)} background agents")
    
    async def run_simulation(self) -> Dict:
        """Run the complete market simulation"""
        logger.info(f"Starting simulation: {self.config.name}")
        self.is_running = True
        
        # Initialize market with some orders
        await self._initialize_market()
        
        # Main simulation loop
        minute_count = 0
        while self.current_time < self.end_time and self.is_running:
            await self._simulation_step()
            minute_count += 1
            
            # Record market data
            if self.config.record_order_book:
                await self._record_market_data()
            
            # Progress update
            if minute_count % 60 == 0:
                hours = minute_count // 60
                logger.info(f"Simulation progress: {hours} hours completed")
            
            # Advance time
            self.current_time += timedelta(minutes=1)
        
        # Finalize simulation
        results = await self._finalize_simulation()
        
        logger.info(f"Simulation completed: {self.config.name}")
        return results
    
    async def _initialize_market(self):
        """Initialize the market with some initial orders"""
        # Create initial price levels for each symbol
        for symbol in self.config.symbols:
            base_price = np.random.uniform(90, 110)
            
            # Create some initial bid/ask orders
            for i in range(5):
                # Bid orders
                bid_price = base_price - (i + 1) * 0.25
                bid_quantity = np.random.randint(100, 500)
                
                self.exchange.submit_order(
                    agent_id="MARKET_MAKER",
                    symbol=symbol,
                    side="BUY",
                    order_type="LIMIT",
                    quantity=bid_quantity,
                    price=bid_price
                )
                
                # Ask orders
                ask_price = base_price + (i + 1) * 0.25
                ask_quantity = np.random.randint(100, 500)
                
                self.exchange.submit_order(
                    agent_id="MARKET_MAKER",
                    symbol=symbol,
                    side="SELL",
                    order_type="LIMIT",
                    quantity=ask_quantity,
                    price=ask_price
                )
    
    async def _simulation_step(self):
        """Execute one simulation step (1 minute)"""
        # Generate news events
        if np.random.random() < self.config.news_frequency:
            await self._generate_news_event()
        
        # Get current market data
        market_data = self._get_current_market_data()
        
        # LLM agents make decisions
        await self._process_llm_agents(market_data)
        
        # Background agents trade
        await self._process_background_agents(market_data)
        
        # Update agent performance metrics
        self._update_performance_metrics()
    
    async def _generate_news_event(self):
        """Generate and process a news event"""
        news_event = self.news_generator.generate_news_event()
        self.news_events.append(news_event)
        
        # Analyze news with LLM
        analysis = await self.news_analyzer.analyze_news(news_event)
        
        logger.info(f"News event: {news_event.headline}")
        logger.info(f"Sentiment: {analysis.get('sentiment_score', 0):.2f}")
        
        # Store news analysis for agents to use
        news_event.llm_analysis = analysis
    
    async def _process_llm_agents(self, market_data: Dict):
        """Process LLM agent decisions"""
        # Get recent news analysis
        recent_news = [event for event in self.news_events[-5:] 
                      if hasattr(event, 'llm_analysis')]
        
        for agent in self.llm_agents:
            try:
                # Generate trading signal
                news_analysis = recent_news[-1].llm_analysis if recent_news else None
                signal = await agent.generate_trading_signal(market_data, news_analysis)
                
                # Record decision
                self.agent_decisions.append({
                    'timestamp': self.current_time,
                    'agent_id': agent.agent_id,
                    'signal': signal.to_dict()
                })
                
                # Execute trade based on signal
                if signal.strength > 0.3:  # Minimum signal strength threshold
                    await self._execute_agent_trade(agent, signal, market_data)
                
            except Exception as e:
                logger.error(f"Error processing LLM agent {agent.agent_id}: {e}")
    
    async def _process_background_agents(self, market_data: Dict):
        """Process background agent trading"""
        for agent in self.background_agents:
            for symbol in self.config.symbols:
                order_data = agent.generate_order(symbol, market_data)
                if order_data:
                    result = self.exchange.submit_order(**order_data)
                    if result['status'] != 'ACCEPTED':
                        logger.debug(f"Background agent order rejected: {result.get('reason')}")
    
    async def _execute_agent_trade(self, agent: AdvancedLLMTradingAgent, 
                                  signal: MarketSignal, market_data: Dict):
        """Execute a trade for an LLM agent based on their signal"""
        symbol = signal.symbol
        
        # Determine order parameters
        if signal.strength > 0:
            side = "BUY"
        else:
            side = "SELL"
        
        # Calculate position size based on signal strength and risk tolerance
        max_position_value = agent.current_capital * 0.2  # Max 20% of capital per position
        
        if symbol in market_data and 'last_trade_price' in market_data[symbol]:
            last_price = market_data[symbol]['last_trade_price']
            if last_price:
                max_quantity = int(max_position_value / last_price)
                quantity = int(max_quantity * abs(signal.strength) * agent.risk_tolerance)
                quantity = max(100, min(quantity, max_quantity))  # Between 100 and max
                
                # Determine order type and price
                if signal.strength > 0.7:  # High conviction - market order
                    result = self.exchange.submit_order(
                        agent_id=agent.agent_id,
                        symbol=symbol,
                        side=side,
                        order_type="MARKET",
                        quantity=quantity
                    )
                else:  # Limit order
                    # Price slightly better than current market
                    if side == "BUY":
                        price = last_price * 0.999  # Slightly below market
                    else:
                        price = last_price * 1.001  # Slightly above market
                    
                    result = self.exchange.submit_order(
                        agent_id=agent.agent_id,
                        symbol=symbol,
                        side=side,
                        order_type="LIMIT",
                        quantity=quantity,
                        price=round(price, 2)
                    )
                
                # Update agent capital and positions based on trades
                if result['status'] == 'ACCEPTED' and result['trades']:
                    for trade in result['trades']:
                        trade_value = trade['quantity'] * trade['price']
                        if side == "BUY":
                            agent.current_capital -= trade_value
                            agent.positions[symbol] += trade['quantity']
                        else:
                            agent.current_capital += trade_value
                            agent.positions[symbol] -= trade['quantity']
                        
                        agent.trade_history.append({
                            'timestamp': self.current_time,
                            'symbol': symbol,
                            'side': side,
                            'quantity': trade['quantity'],
                            'price': trade['price'],
                            'signal_strength': signal.strength,
                            'signal_confidence': signal.confidence
                        })
    
    def _get_current_market_data(self) -> Dict:
        """Get current market data for all symbols"""
        market_data = {}
        
        for symbol in self.config.symbols:
            data = self.exchange.get_market_data(symbol)
            market_data[symbol] = data
        
        return market_data
    
    async def _record_market_data(self):
        """Record current market data for analysis"""
        market_data = self._get_current_market_data()
        
        record = {
            'timestamp': self.current_time,
            'market_data': market_data
        }
        
        self.market_data_history.append(record)
        
        # Save order book snapshots to database
        for symbol in self.config.symbols:
            self.exchange.save_order_book_snapshot(symbol)
    
    def _update_performance_metrics(self):
        """Update performance metrics for all agents"""
        current_market_data = self._get_current_market_data()
        
        for agent in self.llm_agents:
            # Calculate mark-to-market value
            portfolio_value = agent.current_capital
            for symbol, position in agent.positions.items():
                if symbol in current_market_data:
                    last_price = current_market_data[symbol].get('last_trade_price')
                    if last_price:
                        portfolio_value += position * last_price
            
            # Calculate performance metrics
            total_return = (portfolio_value / agent.initial_capital) - 1
            
            performance_record = {
                'timestamp': self.current_time,
                'agent_id': agent.agent_id,
                'portfolio_value': portfolio_value,
                'cash': agent.current_capital,
                'positions': agent.positions.copy(),
                'total_return': total_return,
                'num_trades': len(agent.trade_history)
            }
            
            self.performance_data.append(performance_record)
    
    async def _finalize_simulation(self) -> Dict:
        """Finalize simulation and generate results"""
        # Generate final market report
        market_report = self.exchange.generate_market_report()
        
        # Calculate final performance metrics
        final_performance = self._calculate_final_performance()
        
        # Generate analysis results
        results = {
            'config': asdict(self.config),
            'simulation_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.current_time.isoformat(),
                'duration_minutes': (self.current_time - self.start_time).total_seconds() / 60,
                'news_events_count': len(self.news_events),
                'total_decisions': len(self.agent_decisions)
            },
            'market_report': market_report,
            'agent_performance': final_performance,
            'news_events': [event.to_dict() for event in self.news_events],
            'agent_decisions': self.agent_decisions,
            'performance_history': self.performance_data
        }
        
        # Save results if requested
        if self.config.save_results:
            await self._save_results(results)
        
        return results
    
    def _calculate_final_performance(self) -> Dict:
        """Calculate final performance metrics for all agents"""
        current_market_data = self._get_current_market_data()
        performance = {}
        
        for agent in self.llm_agents:
            # Calculate final portfolio value
            portfolio_value = agent.current_capital
            for symbol, position in agent.positions.items():
                if symbol in current_market_data:
                    last_price = current_market_data[symbol].get('last_trade_price')
                    if last_price:
                        portfolio_value += position * last_price
            
            # Calculate metrics
            total_return = (portfolio_value / agent.initial_capital) - 1
            
            # Calculate Sharpe ratio if enough trades
            if len(agent.trade_history) > 1:
                trade_returns = []
                for trade in agent.trade_history:
                    # Simple return calculation
                    if trade['side'] == 'BUY':
                        trade_returns.append(-0.001)  # Assume small negative for buying
                    else:
                        trade_returns.append(0.001)   # Assume small positive for selling
                
                if trade_returns:
                    mean_return = np.mean(trade_returns)
                    std_return = np.std(trade_returns)
                    sharpe_ratio = mean_return / std_return if std_return > 0 else 0
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            performance[agent.agent_id] = {
                'strategy': agent.strategy_type,
                'initial_capital': agent.initial_capital,
                'final_portfolio_value': portfolio_value,
                'total_return': total_return,
                'cash_remaining': agent.current_capital,
                'final_positions': agent.positions.copy(),
                'num_trades': len(agent.trade_history),
                'sharpe_ratio': sharpe_ratio,
                'risk_tolerance': agent.risk_tolerance
            }
        
        return performance
    
    async def _save_results(self, results: Dict):
        """Save simulation results to files"""
        # Create results directory
        results_dir = Path("simulation_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save main results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.name}_{timestamp}.json"
        
        with open(results_dir / filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save trading history as CSV
        trading_data = []
        for agent in self.llm_agents:
            for trade in agent.trade_history:
                trade_record = trade.copy()
                trade_record['agent_id'] = agent.agent_id
                trade_record['strategy'] = agent.strategy_type
                trading_data.append(trade_record)
        
        if trading_data:
            df = pd.DataFrame(trading_data)
            df.to_csv(results_dir / f"trades_{self.config.name}_{timestamp}.csv", index=False)
        
        logger.info(f"Results saved to {results_dir / filename}")


class MarketImpactExperiment:
    """Experiment to study market impact of large orders"""
    
    def __init__(self, base_config: ExperimentConfig, impact_config: MarketImpactConfig):
        self.base_config = base_config
        self.impact_config = impact_config
        self.results = {}
    
    async def run_experiment(self) -> Dict:
        """Run market impact experiment with different order sizes"""
        logger.info("Starting Market Impact Experiment")
        
        # Run baseline simulation (no impact agent)
        logger.info("Running baseline simulation...")
        baseline_config = ExperimentConfig(
            name=f"{self.base_config.name}_baseline",
            description="Baseline simulation without impact agent",
            symbols=self.base_config.symbols,
            duration_minutes=self.base_config.duration_minutes,
            num_llm_agents=self.base_config.num_llm_agents,
            num_background_agents=self.base_config.num_background_agents,
            initial_capital=self.base_config.initial_capital,
            random_seed=self.base_config.random_seed
        )
        
        baseline_sim = MarketSimulation(baseline_config)
        baseline_results = await baseline_sim.run_simulation()
        self.results['baseline'] = baseline_results
        
        # Run impact simulations for different order sizes
        impact_results = {}
        
        for order_size in self.impact_config.impact_order_sizes:
            logger.info(f"Running impact simulation with order size: {order_size}")
            
            impact_sim_config = ExperimentConfig(
                name=f"{self.base_config.name}_impact_{order_size}",
                description=f"Impact simulation with {order_size} share order",
                symbols=self.base_config.symbols,
                duration_minutes=self.base_config.duration_minutes,
                num_llm_agents=self.base_config.num_llm_agents,
                num_background_agents=self.base_config.num_background_agents,
                initial_capital=self.base_config.initial_capital,
                random_seed=self.base_config.random_seed
            )
            
            impact_sim = MarketImpactSimulation(impact_sim_config, self.impact_config, order_size)
            impact_result = await impact_sim.run_simulation()
            impact_results[order_size] = impact_result
        
        self.results['impact_experiments'] = impact_results
        
        # Analyze impact results
        analysis = self._analyze_market_impact()
        self.results['impact_analysis'] = analysis
        
        logger.info("Market Impact Experiment completed")
        return self.results
    
    def _analyze_market_impact(self) -> Dict:
        """Analyze market impact results"""
        analysis = {
            'price_impact': {},
            'volume_impact': {},
            'spread_impact': {},
            'recovery_time': {}
        }
        
        baseline_data = self.results['baseline']
        impact_data = self.results['impact_experiments']
        
        for order_size, experiment_result in impact_data.items():
            # Calculate price impact
            baseline_prices = self._extract_price_series(baseline_data, self.impact_config.impact_symbol)
            impact_prices = self._extract_price_series(experiment_result, self.impact_config.impact_symbol)
            
            if baseline_prices and impact_prices:
                # Find the impact time point
                impact_time_idx = self.impact_config.impact_time_minutes
                
                if len(baseline_prices) > impact_time_idx and len(impact_prices) > impact_time_idx:
                    baseline_price = baseline_prices[impact_time_idx]
                    impact_price = impact_prices[impact_time_idx]
                    
                    price_impact = (impact_price - baseline_price) / baseline_price
                    analysis['price_impact'][order_size] = price_impact
                    
                    # Calculate recovery time (time to return to within 0.1% of baseline)
                    recovery_time = self._calculate_recovery_time(
                        baseline_prices[impact_time_idx:], 
                        impact_prices[impact_time_idx:],
                        threshold=0.001
                    )
                    analysis['recovery_time'][order_size] = recovery_time
        
        return analysis
    
    def _extract_price_series(self, simulation_data: Dict, symbol: str) -> List[float]:
        """Extract price series for a symbol from simulation data"""
        prices = []
        
        for record in simulation_data.get('performance_history', []):
            market_data = record.get('market_data', {})
            if symbol in market_data:
                last_price = market_data[symbol].get('last_trade_price')
                if last_price:
                    prices.append(last_price)
        
        return prices
    
    def _calculate_recovery_time(self, baseline_prices: List[float], 
                                impact_prices: List[float], threshold: float = 0.001) -> int:
        """Calculate time to recover from market impact"""
        for i, (baseline, impact) in enumerate(zip(baseline_prices, impact_prices)):
            if abs(impact - baseline) / baseline < threshold:
                return i
        
        return len(baseline_prices)  # Never recovered within observation period


class MarketImpactSimulation(MarketSimulation):
    """Specialized simulation for market impact experiments"""
    
    def __init__(self, config: ExperimentConfig, impact_config: MarketImpactConfig, order_size: int):
        super().__init__(config)
        self.impact_config = impact_config
        self.order_size = order_size
        self.impact_executed = False
    
    async def _simulation_step(self):
        """Extended simulation step with impact order execution"""
        await super()._simulation_step()
        
        # Check if it's time for the impact order
        minutes_elapsed = (self.current_time - self.start_time).total_seconds() / 60
        
        if (not self.impact_executed and 
            minutes_elapsed >= self.impact_config.impact_time_minutes):
            await self._execute_impact_order()
            self.impact_executed = True
    
    async def _execute_impact_order(self):
        """Execute the large impact order"""
        logger.info(f"Executing impact order: {self.order_size} shares of {self.impact_config.impact_symbol}")
        
        result = self.exchange.submit_order(
            agent_id=self.impact_config.impact_agent_id,
            symbol=self.impact_config.impact_symbol,
            side=self.impact_config.impact_direction,
            order_type="MARKET",
            quantity=self.order_size
        )
        
        logger.info(f"Impact order result: {result['status']}")
        if result['trades']:
            total_volume = sum(trade['quantity'] for trade in result['trades'])
            avg_price = sum(trade['quantity'] * trade['price'] for trade in result['trades']) / total_volume
            logger.info(f"Impact order executed: {total_volume} shares at average price ${avg_price:.2f}")


def generate_experiment_report(results: Dict, output_path: str = "experiment_report.html"):
    """Generate a comprehensive HTML report of experiment results"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ABIDES-LLM Experiment Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                      background-color: #e8f4fd; border-radius: 3px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>ABIDES-LLM Experiment Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Experiment Summary</h2>
            <div class="metric">
                <strong>Experiment:</strong> {results.get('config', {}).get('name', 'Unknown')}
            </div>
            <div class="metric">
                <strong>Duration:</strong> {results.get('simulation_summary', {}).get('duration_minutes', 0):.0f} minutes
            </div>
            <div class="metric">
                <strong>News Events:</strong> {results.get('simulation_summary', {}).get('news_events_count', 0)}
            </div>
            <div class="metric">
                <strong>Agent Decisions:</strong> {results.get('simulation_summary', {}).get('total_decisions', 0)}
            </div>
        </div>
        
        <div class="section">
            <h2>Agent Performance</h2>
            <table>
                <tr>
                    <th>Agent ID</th>
                    <th>Strategy</th>
                    <th>Total Return (%)</th>
                    <th>Final Value ($)</th>
                    <th>Number of Trades</th>
                    <th>Sharpe Ratio</th>
                </tr>
    """
    
    # Add agent performance data
    for agent_id, performance in results.get('agent_performance', {}).items():
        html_content += f"""
                <tr>
                    <td>{agent_id}</td>
                    <td>{performance.get('strategy', 'Unknown')}</td>
                    <td>{performance.get('total_return', 0) * 100:.2f}%</td>
                    <td>${performance.get('final_portfolio_value', 0):,.2f}</td>
                    <td>{performance.get('num_trades', 0)}</td>
                    <td>{performance.get('sharpe_ratio', 0):.3f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Market Statistics</h2>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Total Volume</th>
                    <th>Number of Trades</th>
                    <th>Value Traded ($)</th>
                    <th>Average Trade Size</th>
                </tr>
    """
    
    # Add market statistics
    for symbol, data in results.get('market_report', {}).get('symbols', {}).items():
        stats = data.get('stats', {})
        html_content += f"""
                <tr>
                    <td>{symbol}</td>
                    <td>{stats.get('total_volume', 0):,}</td>
                    <td>{stats.get('trade_count', 0):,}</td>
                    <td>${stats.get('value_traded', 0):,.2f}</td>
                    <td>{stats.get('avg_trade_size', 0):.0f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Experiment report saved to {output_path}")


# Example usage and experiment configurations
def create_market_impact_experiment() -> Tuple[ExperimentConfig, MarketImpactConfig]:
    """Create a market impact experiment configuration"""
    
    base_config = ExperimentConfig(
        name="market_impact_study",
        description="Study the impact of large orders on market prices",
        symbols=["AAPL", "MSFT", "GOOGL"],
        duration_minutes=240,  # 4 hours
        num_llm_agents=8,
        num_background_agents=30,
        initial_capital=1000000,
        news_frequency=0.05,  # Less frequent news for cleaner impact measurement
        random_seed=42
    )
    
    impact_config = MarketImpactConfig(
        impact_symbol="AAPL",
        impact_time_minutes=120,  # 2 hours into simulation
        impact_order_sizes=[5000, 15000, 30000, 50000],
        impact_direction="BUY"
    )
    
    return base_config, impact_config


def create_strategy_comparison_experiment() -> ExperimentConfig:
    """Create an experiment to compare different LLM trading strategies"""
    
    return ExperimentConfig(
        name="strategy_comparison",
        description="Compare performance of different LLM trading strategies",
        symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
        duration_minutes=480,  # 8 hours
        num_llm_agents=12,  # 3 agents per strategy type
        num_background_agents=40,
        initial_capital=1000000,
        news_frequency=0.1,
        random_seed=42
    )


async def run_example_experiments():
    """Run example experiments to demonstrate the framework"""
    
    # 1. Basic strategy comparison experiment
    logger.info("Running Strategy Comparison Experiment...")
    strategy_config = create_strategy_comparison_experiment()
    strategy_sim = MarketSimulation(strategy_config)
    strategy_results = await strategy_sim.run_simulation()
    
    # Generate report
    generate_experiment_report(strategy_results, "strategy_comparison_report.html")
    
    # 2. Market impact experiment
    logger.info("Running Market Impact Experiment...")
    base_config, impact_config = create_market_impact_experiment()
    impact_experiment = MarketImpactExperiment(base_config, impact_config)
    impact_results = await impact_experiment.run_experiment()
    
    # Save impact results
    with open("market_impact_results.json", 'w') as f:
        json.dump(impact_results, f, indent=2, default=str)
    
    logger.info("All experiments completed!")
    logger.info("Check the generated HTML reports and JSON files for detailed results.")


if __name__ == "__main__":
    # Run example experiments
    asyncio.run(run_example_experiments())