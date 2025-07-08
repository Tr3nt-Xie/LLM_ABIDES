"""
Realistic Market Simulation Engine
=================================

Complete simulation engine that orchestrates LLM agents, enhanced ABIDES agents,
realistic market data, and news events to create a comprehensive trading market simulation.
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from enhanced_llm_abides_system import (
    NewsEvent, MarketSignal, AdvancedLLMTradingAgent, 
    EnhancedLLMNewsAnalyzer, RealisticNewsGenerator, NewsCategory
)
from enhanced_abides_bridge import (
    EnhancedABIDESOrder, RealisticMarketDataGenerator, 
    EnhancedLLMInfluencedAgent, MarketState
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for market simulation"""
    symbols: List[str]
    simulation_duration_hours: int = 8  # Trading day length
    news_frequency: float = 0.1  # News events per minute
    llm_agents_count: int = 5
    abides_agents_count: int = 20
    initial_capital: float = 1000000  # $1M per agent
    real_time_factor: float = 1.0  # 1.0 = real time, 10.0 = 10x faster
    enable_learning: bool = True
    save_data: bool = True
    output_directory: str = "simulation_output"


class RealisticMarketSimulation:
    """Main simulation engine orchestrating all components"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.simulation_start_time = datetime.now()
        self.simulation_current_time = self.simulation_start_time
        self.is_running = False
        
        # Core components
        self.news_generator = RealisticNewsGenerator(config.symbols)
        self.market_data_generator = RealisticMarketDataGenerator(config.symbols)
        self.market_state = MarketState()
        
        # LLM components
        self.llm_news_analyzer = self._initialize_llm_analyzer()
        self.llm_agents = self._initialize_llm_agents()
        
        # Enhanced ABIDES agents
        self.abides_agents = self._initialize_abides_agents()
        
        # Data storage and analysis
        self.data_recorder = SimulationDataRecorder(config.output_directory)
        self.performance_analyzer = PerformanceAnalyzer()
        self.event_scheduler = EventScheduler()
        
        # Market data and state
        self.current_market_data = {}
        self.active_news_events = []
        self.all_market_signals = []
        self.execution_reports = []
        
        # Threading for real-time simulation
        self.simulation_thread = None
        self.stop_event = threading.Event()
        
        logger.info(f"Initialized simulation with {len(config.symbols)} symbols")
        logger.info(f"Created {config.llm_agents_count} LLM agents and {config.abides_agents_count} ABIDES agents")
    
    def _initialize_llm_analyzer(self) -> EnhancedLLMNewsAnalyzer:
        """Initialize LLM news analyzer"""
        try:
            llm_config = {
                "config_list": [
                    {
                        "model": "gpt-4o",
                        "api_key": "your-api-key-here",  # Replace with actual API key
                        "base_url": "",
                        "api_type": "openai"
                    }
                ],
                "timeout": 60,
                "temperature": 0.3
            }
            
            return EnhancedLLMNewsAnalyzer(
                name="MarketNewsAnalyzer",
                llm_config=llm_config
            )
        except Exception as e:
            logger.warning(f"Failed to initialize LLM analyzer: {e}")
            return None
    
    def _initialize_llm_agents(self) -> List[AdvancedLLMTradingAgent]:
        """Initialize sophisticated LLM trading agents"""
        agents = []
        
        # Different agent specializations and strategies
        agent_configs = [
            {"strategy": "momentum", "specialization": "high_frequency", "risk_tolerance": 0.8},
            {"strategy": "value", "specialization": "fundamental", "risk_tolerance": 0.3},
            {"strategy": "volatility", "specialization": "sector_Technology", "risk_tolerance": 0.6},
            {"strategy": "arbitrage", "specialization": "news_earnings", "risk_tolerance": 0.9},
            {"strategy": "momentum", "specialization": "generalist", "risk_tolerance": 0.5}
        ]
        
        try:
            llm_config = {
                "config_list": [
                    {
                        "model": "gpt-4o",
                        "api_key": "your-api-key-here",  # Replace with actual API key
                        "temperature": 0.3
                    }
                ],
                "timeout": 60
            }
            
            for i in range(self.config.llm_agents_count):
                config = agent_configs[i % len(agent_configs)]
                
                agent = AdvancedLLMTradingAgent(
                    name=f"LLM_{config['strategy'].title()}Agent_{i+1}",
                    strategy=config['strategy'],
                    risk_tolerance=config['risk_tolerance'],
                    specialization=config['specialization'],
                    llm_config=llm_config
                )
                
                agents.append(agent)
                
        except Exception as e:
            logger.warning(f"Failed to initialize LLM agents: {e}")
            # Create mock agents for testing
            for i in range(self.config.llm_agents_count):
                config = agent_configs[i % len(agent_configs)]
                agent = AdvancedLLMTradingAgent(
                    name=f"MockLLM_{config['strategy'].title()}Agent_{i+1}",
                    strategy=config['strategy'],
                    risk_tolerance=config['risk_tolerance'],
                    specialization=config['specialization']
                )
                agents.append(agent)
        
        return agents
    
    def _initialize_abides_agents(self) -> List[EnhancedLLMInfluencedAgent]:
        """Initialize enhanced ABIDES agents"""
        agents = []
        
        strategies = ["adaptive", "momentum", "mean_reversion", "volatility", "arbitrage"]
        
        for i in range(self.config.abides_agents_count):
            strategy = strategies[i % len(strategies)]
            
            agent = EnhancedLLMInfluencedAgent(
                agent_id=f"ABIDES_{strategy.title()}Agent_{i+1}",
                strategy=strategy,
                base_capital=self.config.initial_capital
            )
            
            agents.append(agent)
        
        return agents
    
    def start_simulation(self, blocking: bool = True):
        """Start the market simulation"""
        if self.is_running:
            logger.warning("Simulation is already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        logger.info("Starting realistic market simulation...")
        
        # Initialize market data
        self.current_market_data = self.market_data_generator.update_market_data()
        
        # Schedule initial events
        self._schedule_initial_events()
        
        if blocking:
            self._run_simulation_loop()
        else:
            self.simulation_thread = threading.Thread(target=self._run_simulation_loop)
            self.simulation_thread.start()
    
    def stop_simulation(self):
        """Stop the market simulation"""
        if not self.is_running:
            logger.warning("Simulation is not running")
            return
        
        logger.info("Stopping simulation...")
        self.stop_event.set()
        self.is_running = False
        
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=5)
        
        # Final data processing
        self._process_final_results()
        
        logger.info("Simulation stopped successfully")
    
    def _run_simulation_loop(self):
        """Main simulation loop"""
        
        simulation_end_time = (
            self.simulation_start_time + 
            timedelta(hours=self.config.simulation_duration_hours)
        )
        
        step_interval = 1.0 / self.config.real_time_factor  # Seconds between steps
        step_count = 0
        
        while self.is_running and self.simulation_current_time < simulation_end_time:
            if self.stop_event.is_set():
                break
            
            step_start_time = time.time()
            
            try:
                # Run simulation step
                self._execute_simulation_step(step_count)
                
                # Update simulation time
                self.simulation_current_time += timedelta(minutes=1)
                step_count += 1
                
                # Log progress periodically
                if step_count % 60 == 0:  # Every hour
                    hours_elapsed = step_count // 60
                    logger.info(f"Simulation hour {hours_elapsed}: {len(self.active_news_events)} active news events, "
                              f"{len(self.all_market_signals)} total signals")
                
                # Sleep to maintain real-time factor
                elapsed = time.time() - step_start_time
                sleep_time = max(0, step_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in simulation step {step_count}: {e}")
                if step_count % 10 == 0:  # Only continue after repeated errors
                    break
        
        self.is_running = False
        logger.info(f"Simulation completed after {step_count} steps")
    
    def _execute_simulation_step(self, step_count: int):
        """Execute one simulation step"""
        
        # 1. Process scheduled events
        scheduled_events = self.event_scheduler.get_due_events(self.simulation_current_time)
        
        # 2. Generate news events
        new_news_events = self._generate_news_events()
        
        # 3. Process news with LLM analyzer
        analyzed_news = []
        for news_event in new_news_events:
            if self.llm_news_analyzer:
                try:
                    analysis = self.llm_news_analyzer.analyze_news_comprehensive(
                        news_event, self._get_market_context()
                    )
                    analyzed_news.append((news_event, analysis))
                except Exception as e:
                    logger.warning(f"LLM analysis failed for news: {e}")
                    analyzed_news.append((news_event, {}))
            else:
                analyzed_news.append((news_event, {}))
        
        # 4. Generate signals from LLM agents
        new_signals = []
        for news_event, analysis in analyzed_news:
            for agent in self.llm_agents:
                try:
                    agent_signals = agent.process_news_with_specialization(
                        news_event, analysis, self.current_market_data
                    )
                    new_signals.extend(agent_signals)
                except Exception as e:
                    logger.warning(f"LLM agent {agent.name} signal generation failed: {e}")
        
        # 5. Process signals with ABIDES agents and generate orders
        all_orders = []
        for agent in self.abides_agents:
            try:
                # Get relevant signals for this agent
                relevant_signals = [s for s in new_signals + self.all_market_signals 
                                  if self._is_signal_active(s)]
                
                # Generate orders
                agent_orders = agent.process_signals(relevant_signals, self.current_market_data)
                all_orders.extend(agent_orders)
                
                # Check conditional orders (stop loss, take profit)
                conditional_orders = agent.order_manager.check_conditional_orders(self.current_market_data)
                all_orders.extend(conditional_orders)
                
            except Exception as e:
                logger.warning(f"ABIDES agent {agent.agent_id} order generation failed: {e}")
        
        # 6. Execute orders and update market data
        execution_reports = self._execute_orders(all_orders)
        
        # 7. Update market data with news impact and order flow
        self.current_market_data = self.market_data_generator.update_market_data(
            news_events=new_news_events,
            external_orders=all_orders
        )
        
        # 8. Update market state
        self.market_state.update_state(self.current_market_data, new_news_events)
        
        # 9. Update agent performance
        self._update_agent_performance(execution_reports)
        
        # 10. Record data
        if self.config.save_data:
            self._record_step_data(step_count, new_news_events, new_signals, all_orders, execution_reports)
        
        # 11. Clean up expired events and signals
        self._cleanup_expired_data()
    
    def _generate_news_events(self) -> List[NewsEvent]:
        """Generate news events for this step"""
        news_events = []
        
        # Check if we should generate news this step
        if np.random.random() < self.config.news_frequency:
            # Generate 1-3 news events
            num_events = np.random.poisson(1) + 1
            
            for _ in range(min(num_events, 3)):
                try:
                    # Set market cycle for realistic news generation
                    self.news_generator.market_cycle = self.market_state.regime
                    
                    news_event = self.news_generator.generate_realistic_event()
                    news_events.append(news_event)
                    
                except Exception as e:
                    logger.warning(f"News generation failed: {e}")
        
        # Add to active news events
        self.active_news_events.extend(news_events)
        
        return news_events
    
    def _get_market_context(self) -> Dict:
        """Get current market context for LLM analysis"""
        if not self.current_market_data:
            return {}
        
        # Calculate market-wide metrics
        prices = [data.get('price', 100) for data in self.current_market_data.values()]
        spreads = [data.get('spread', 1) for data in self.current_market_data.values()]
        volumes = [data.get('volume', 1000) for data in self.current_market_data.values()]
        
        return {
            'volatility': self.market_state.volatility,
            'trend': self.market_state.trend,
            'sentiment': self.market_state.sentiment,
            'regime': self.market_state.regime,
            'avg_spread': np.mean(spreads) if spreads else 0,
            'avg_volume': np.mean(volumes) if volumes else 1000,
            'price_range': (min(prices), max(prices)) if prices else (100, 100),
            'symbols_count': len(self.current_market_data)
        }
    
    def _is_signal_active(self, signal: MarketSignal) -> bool:
        """Check if signal is still active"""
        time_elapsed = (self.simulation_current_time - signal.timestamp).total_seconds() / 60
        return time_elapsed < signal.duration
    
    def _execute_orders(self, orders: List[EnhancedABIDESOrder]) -> List[Dict]:
        """Execute orders and generate execution reports"""
        execution_reports = []
        
        for order in orders:
            try:
                # Simulate order execution
                execution_report = self._simulate_order_execution(order)
                execution_reports.append(execution_report)
                
            except Exception as e:
                logger.warning(f"Order execution failed for {order.order_id}: {e}")
        
        self.execution_reports.extend(execution_reports)
        return execution_reports
    
    def _simulate_order_execution(self, order: EnhancedABIDESOrder) -> Dict:
        """Simulate realistic order execution"""
        symbol_data = self.current_market_data.get(order.symbol, {})
        current_price = symbol_data.get('price', 100.0)
        spread = symbol_data.get('spread', current_price * 0.001)
        
        # Determine execution price based on order type
        if order.order_type == 'MARKET':
            # Market orders execute immediately at current price + slippage
            slippage = min(spread * 0.5, current_price * order.max_slippage)
            if order.side == 'BUY':
                execution_price = current_price + slippage
            else:
                execution_price = current_price - slippage
            
            fill_quantity = order.quantity
            execution_status = 'FILLED'
            
        elif order.order_type == 'LIMIT':
            # Limit orders may or may not execute based on price
            if order.side == 'BUY' and order.limit_price >= current_price - spread * 0.3:
                execution_price = min(order.limit_price, current_price)
                fill_quantity = order.quantity
                execution_status = 'FILLED'
            elif order.side == 'SELL' and order.limit_price <= current_price + spread * 0.3:
                execution_price = max(order.limit_price, current_price)
                fill_quantity = order.quantity
                execution_status = 'FILLED'
            else:
                # Order doesn't execute
                execution_price = order.limit_price
                fill_quantity = 0
                execution_status = 'PENDING'
        
        else:  # STOP_LOSS, TAKE_PROFIT
            execution_price = current_price
            fill_quantity = order.quantity
            execution_status = 'FILLED'
        
        # Calculate fees and slippage costs
        notional_value = fill_quantity * execution_price
        commission = max(1.0, notional_value * 0.0001)  # 1 bps commission, min $1
        
        # Calculate PnL (simplified)
        if order.side == 'BUY':
            market_impact = execution_price - current_price
        else:
            market_impact = current_price - execution_price
        
        total_cost = commission + abs(market_impact * fill_quantity)
        
        return {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity_ordered': order.quantity,
            'quantity_filled': fill_quantity,
            'execution_price': execution_price,
            'execution_time': self.simulation_current_time,
            'execution_status': execution_status,
            'commission': commission,
            'market_impact': market_impact,
            'total_cost': total_cost,
            'agent_id': order.agent_id,
            'notional_value': notional_value
        }
    
    def _update_agent_performance(self, execution_reports: List[Dict]):
        """Update agent performance based on executions"""
        for report in execution_reports:
            if report['execution_status'] == 'FILLED':
                # Find the agent and update their portfolio
                agent_id = report['agent_id']
                
                for agent in self.abides_agents:
                    if agent.agent_id == agent_id:
                        # Update portfolio
                        quantity = report['quantity_filled']
                        if report['side'] == 'SELL':
                            quantity = -quantity
                        
                        agent.portfolio.add_position(
                            report['symbol'],
                            quantity,
                            report['execution_price']
                        )
                        
                        # Update performance tracking
                        agent.update_performance(report)
                        break
    
    def _record_step_data(self, step_count: int, news_events: List[NewsEvent], 
                         signals: List[MarketSignal], orders: List[EnhancedABIDESOrder],
                         executions: List[Dict]):
        """Record step data for analysis"""
        
        step_data = {
            'step': step_count,
            'simulation_time': self.simulation_current_time.isoformat(),
            'market_data': self.current_market_data,
            'market_state': {
                'regime': self.market_state.regime,
                'volatility': self.market_state.volatility,
                'trend': self.market_state.trend,
                'sentiment': self.market_state.sentiment
            },
            'news_events': [event.to_dict() for event in news_events],
            'signals_generated': len(signals),
            'orders_generated': len(orders),
            'orders_executed': len([e for e in executions if e['execution_status'] == 'FILLED']),
            'total_volume': sum(e['notional_value'] for e in executions if e['execution_status'] == 'FILLED'),
            'active_news_count': len(self.active_news_events),
            'active_signals_count': len([s for s in self.all_market_signals if self._is_signal_active(s)])
        }
        
        self.data_recorder.record_step(step_data)
    
    def _cleanup_expired_data(self):
        """Remove expired news events and signals"""
        current_time = self.simulation_current_time
        
        # Remove expired news events
        self.active_news_events = [
            event for event in self.active_news_events
            if (current_time - event.timestamp).total_seconds() / 60 < event.impact_duration
        ]
        
        # Remove expired signals
        self.all_market_signals = [
            signal for signal in self.all_market_signals
            if self._is_signal_active(signal)
        ]
    
    def _schedule_initial_events(self):
        """Schedule initial market events"""
        # Schedule market open/close events
        market_open = self.simulation_start_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = self.simulation_start_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if market_open > self.simulation_start_time:
            self.event_scheduler.schedule_event(market_open, "market_open", {})
        
        if market_close > self.simulation_start_time:
            self.event_scheduler.schedule_event(market_close, "market_close", {})
        
        # Schedule periodic events
        for hour in range(24):
            for minute in [0, 30]:
                event_time = self.simulation_start_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if event_time > self.simulation_start_time:
                    self.event_scheduler.schedule_event(event_time, "market_update", {})
    
    def _process_final_results(self):
        """Process and save final simulation results"""
        logger.info("Processing final simulation results...")
        
        # Generate final performance report
        final_report = self.performance_analyzer.generate_final_report(
            self.abides_agents,
            self.execution_reports,
            self.data_recorder.get_all_data()
        )
        
        # Save results
        if self.config.save_data:
            self.data_recorder.save_final_results(final_report)
            
            # Generate visualizations
            self._generate_visualizations()
        
        # Print summary
        self._print_simulation_summary(final_report)
    
    def _generate_visualizations(self):
        """Generate simulation visualizations"""
        try:
            viz_generator = VisualizationGenerator(self.data_recorder.get_all_data())
            
            # Price charts
            viz_generator.create_price_charts(self.config.symbols)
            
            # Volume analysis
            viz_generator.create_volume_analysis()
            
            # Signal effectiveness
            viz_generator.create_signal_analysis()
            
            # Agent performance
            viz_generator.create_agent_performance_charts()
            
            # Market regime analysis
            viz_generator.create_market_regime_analysis()
            
            logger.info("Visualizations generated successfully")
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
    
    def _print_simulation_summary(self, report: Dict):
        """Print simulation summary"""
        print("\n" + "="*80)
        print("REALISTIC MARKET SIMULATION SUMMARY")
        print("="*80)
        
        print(f"Simulation Duration: {self.config.simulation_duration_hours} hours")
        print(f"Symbols Traded: {', '.join(self.config.symbols)}")
        print(f"LLM Agents: {len(self.llm_agents)}")
        print(f"ABIDES Agents: {len(self.abides_agents)}")
        
        print(f"\nMarket Activity:")
        print(f"  Total News Events: {report.get('total_news_events', 0)}")
        print(f"  Total Signals Generated: {report.get('total_signals', 0)}")
        print(f"  Total Orders: {report.get('total_orders', 0)}")
        print(f"  Orders Executed: {report.get('executed_orders', 0)}")
        print(f"  Total Volume: ${report.get('total_volume', 0):,.2f}")
        
        print(f"\nTop Performing Agents:")
        for agent_perf in report.get('top_agents', [])[:5]:
            print(f"  {agent_perf['agent_id']}: {agent_perf['total_pnl']:+.2f} "
                  f"({agent_perf['win_rate']:.1%} win rate)")
        
        print(f"\nMarket Statistics:")
        print(f"  Average Volatility: {report.get('avg_volatility', 0):.2%}")
        print(f"  Signal-to-Order Conversion: {report.get('signal_conversion_rate', 0):.1%}")
        print(f"  Average Order Fill Rate: {report.get('fill_rate', 0):.1%}")
        
        print("="*80)


class EventScheduler:
    """Schedule and manage simulation events"""
    
    def __init__(self):
        self.scheduled_events = []
    
    def schedule_event(self, event_time: datetime, event_type: str, event_data: Dict):
        """Schedule an event"""
        self.scheduled_events.append({
            'time': event_time,
            'type': event_type,
            'data': event_data
        })
        
        # Keep events sorted by time
        self.scheduled_events.sort(key=lambda x: x['time'])
    
    def get_due_events(self, current_time: datetime) -> List[Dict]:
        """Get events that are due"""
        due_events = []
        remaining_events = []
        
        for event in self.scheduled_events:
            if event['time'] <= current_time:
                due_events.append(event)
            else:
                remaining_events.append(event)
        
        self.scheduled_events = remaining_events
        return due_events


class SimulationDataRecorder:
    """Record and manage simulation data"""
    
    def __init__(self, output_directory: str):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.output_directory / "simulation_data.db"
        self._initialize_database()
        
        # In-memory data for fast access
        self.step_data = []
        self.performance_data = []
    
    def _initialize_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulation_steps (
                step INTEGER,
                simulation_time TEXT,
                market_data TEXT,
                market_state TEXT,
                news_events TEXT,
                signals_generated INTEGER,
                orders_generated INTEGER,
                orders_executed INTEGER,
                total_volume REAL,
                active_news_count INTEGER,
                active_signals_count INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS executions (
                order_id TEXT,
                symbol TEXT,
                side TEXT,
                quantity_ordered INTEGER,
                quantity_filled INTEGER,
                execution_price REAL,
                execution_time TEXT,
                execution_status TEXT,
                commission REAL,
                market_impact REAL,
                total_cost REAL,
                agent_id TEXT,
                notional_value REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_step(self, step_data: Dict):
        """Record step data"""
        self.step_data.append(step_data)
        
        # Also save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO simulation_steps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            step_data['step'],
            step_data['simulation_time'],
            json.dumps(step_data['market_data']),
            json.dumps(step_data['market_state']),
            json.dumps(step_data['news_events']),
            step_data['signals_generated'],
            step_data['orders_generated'],
            step_data['orders_executed'],
            step_data['total_volume'],
            step_data['active_news_count'],
            step_data['active_signals_count']
        ))
        
        conn.commit()
        conn.close()
    
    def record_execution(self, execution: Dict):
        """Record execution data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO executions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            execution['order_id'],
            execution['symbol'],
            execution['side'],
            execution['quantity_ordered'],
            execution['quantity_filled'],
            execution['execution_price'],
            execution['execution_time'],
            execution['execution_status'],
            execution['commission'],
            execution['market_impact'],
            execution['total_cost'],
            execution['agent_id'],
            execution['notional_value']
        ))
        
        conn.commit()
        conn.close()
    
    def get_all_data(self) -> Dict:
        """Get all recorded data"""
        return {
            'step_data': self.step_data,
            'db_path': str(self.db_path)
        }
    
    def save_final_results(self, final_report: Dict):
        """Save final results to file"""
        results_file = self.output_directory / "final_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"Final results saved to {results_file}")


class PerformanceAnalyzer:
    """Analyze simulation performance and generate reports"""
    
    def generate_final_report(self, agents: List[EnhancedLLMInfluencedAgent],
                             executions: List[Dict], all_data: Dict) -> Dict:
        """Generate comprehensive final report"""
        
        # Agent performance analysis
        agent_performance = []
        for agent in agents:
            portfolio_value = agent.portfolio.total_value
            total_pnl = agent.portfolio.realized_pnl + agent.portfolio.unrealized_pnl
            
            trades = agent.trades_executed
            win_rate = len([t for t in trades if t.get('pnl', 0) > 0]) / len(trades) if trades else 0
            
            agent_performance.append({
                'agent_id': agent.agent_id,
                'strategy': agent.strategy,
                'portfolio_value': portfolio_value,
                'total_pnl': total_pnl,
                'realized_pnl': agent.portfolio.realized_pnl,
                'num_trades': len(trades),
                'win_rate': win_rate
            })
        
        # Sort by performance
        agent_performance.sort(key=lambda x: x['total_pnl'], reverse=True)
        
        # Market statistics
        filled_executions = [e for e in executions if e['execution_status'] == 'FILLED']
        
        total_volume = sum(e['notional_value'] for e in filled_executions)
        total_orders = len(executions)
        fill_rate = len(filled_executions) / total_orders if total_orders > 0 else 0
        
        # Signal analysis
        step_data = all_data.get('step_data', [])
        total_signals = sum(s.get('signals_generated', 0) for s in step_data)
        total_news = sum(len(s.get('news_events', [])) for s in step_data)
        
        signal_conversion = total_orders / total_signals if total_signals > 0 else 0
        
        # Volatility analysis
        volatilities = [s.get('market_state', {}).get('volatility', 0) for s in step_data]
        avg_volatility = np.mean(volatilities) if volatilities else 0
        
        return {
            'simulation_end_time': datetime.now().isoformat(),
            'agent_performance': agent_performance,
            'top_agents': agent_performance[:10],
            'total_news_events': total_news,
            'total_signals': total_signals,
            'total_orders': total_orders,
            'executed_orders': len(filled_executions),
            'total_volume': total_volume,
            'fill_rate': fill_rate,
            'signal_conversion_rate': signal_conversion,
            'avg_volatility': avg_volatility,
            'market_statistics': {
                'avg_commission': np.mean([e['commission'] for e in filled_executions]) if filled_executions else 0,
                'avg_market_impact': np.mean([abs(e['market_impact']) for e in filled_executions]) if filled_executions else 0,
                'total_commission': sum(e['commission'] for e in filled_executions)
            }
        }


class VisualizationGenerator:
    """Generate visualizations for simulation results"""
    
    def __init__(self, data: Dict):
        self.data = data
        self.step_data = data.get('step_data', [])
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_price_charts(self, symbols: List[str]):
        """Create price evolution charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Market Price Evolution', fontsize=16)
        
        # Extract price data
        price_data = {}
        timestamps = []
        
        for step in self.step_data:
            timestamps.append(step['simulation_time'])
            market_data = step.get('market_data', {})
            
            for symbol in symbols:
                if symbol not in price_data:
                    price_data[symbol] = []
                
                symbol_data = market_data.get(symbol, {})
                price_data[symbol].append(symbol_data.get('price', 100))
        
        # Plot price evolution
        ax = axes[0, 0]
        for symbol in symbols[:4]:  # Plot first 4 symbols
            if symbol in price_data:
                ax.plot(range(len(price_data[symbol])), price_data[symbol], label=symbol)
        ax.set_title('Price Evolution')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True)
        
        # Plot volatility
        ax = axes[0, 1]
        volatilities = [s.get('market_state', {}).get('volatility', 0) for s in self.step_data]
        ax.plot(volatilities)
        ax.set_title('Market Volatility')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Volatility')
        ax.grid(True)
        
        # Plot trading volume
        ax = axes[1, 0]
        volumes = [s.get('total_volume', 0) for s in self.step_data]
        ax.bar(range(len(volumes)), volumes, alpha=0.7)
        ax.set_title('Trading Volume')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Volume ($)')
        ax.grid(True)
        
        # Plot news events impact
        ax = axes[1, 1]
        news_counts = [s.get('active_news_count', 0) for s in self.step_data]
        signal_counts = [s.get('active_signals_count', 0) for s in self.step_data]
        
        ax.plot(news_counts, label='Active News', marker='o')
        ax.plot(signal_counts, label='Active Signals', marker='s')
        ax.set_title('News Events and Signals')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('market_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_volume_analysis(self):
        """Create volume analysis charts"""
        # Implementation for volume analysis visualization
        pass
    
    def create_signal_analysis(self):
        """Create signal effectiveness analysis"""
        # Implementation for signal analysis visualization
        pass
    
    def create_agent_performance_charts(self):
        """Create agent performance visualization"""
        # Implementation for agent performance charts
        pass
    
    def create_market_regime_analysis(self):
        """Create market regime analysis"""
        # Implementation for market regime visualization
        pass


# Example usage and testing
def run_example_simulation():
    """Run example realistic market simulation"""
    
    # Configuration
    config = SimulationConfig(
        symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        simulation_duration_hours=2,  # 2 hour simulation for testing
        news_frequency=0.2,  # Higher frequency for testing
        llm_agents_count=3,
        abides_agents_count=10,
        real_time_factor=60.0,  # 60x faster for testing
        save_data=True
    )
    
    # Create and run simulation
    simulation = RealisticMarketSimulation(config)
    
    try:
        simulation.start_simulation(blocking=True)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        simulation.stop_simulation()
    except Exception as e:
        print(f"Simulation error: {e}")
        simulation.stop_simulation()


if __name__ == "__main__":
    run_example_simulation()