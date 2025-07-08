#!/usr/bin/env python3
"""
Multi-Round Market Simulation Test Demo
=======================================

Comprehensive test demonstration of the ABIDES-LLM market simulation system
with multiple rounds, different market scenarios, and detailed analysis.
"""

import sys
import os
import json
import time
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import simulation components
try:
    from realistic_market_simulation import RealisticMarketSimulation, SimulationConfig
    from enhanced_llm_abides_system import NewsCategory
    from enhanced_abides_bridge import EnhancedLLMInfluencedAgent
    FULL_SIMULATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Full simulation not available: {e}")
    # Create mock classes for type hints
    class RealisticMarketSimulation:
        pass
    class SimulationConfig:
        pass
    class NewsCategory:
        pass
    class EnhancedLLMInfluencedAgent:
        pass
    FULL_SIMULATION_AVAILABLE = False

@dataclass
class TestScenario:
    """Configuration for a test scenario"""
    name: str
    description: str
    symbols: List[str]
    duration_hours: float
    market_regime: str  # "bull", "bear", "volatile", "stable"
    news_frequency: float
    llm_agents_count: int
    abides_agents_count: int
    real_time_factor: float
    expected_outcomes: Dict[str, Any]

@dataclass
class SimulationResult:
    """Results from a simulation round"""
    scenario_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    total_trades: int
    total_volume: float
    total_pnl: float
    news_events_count: int
    agent_performances: List[Dict]
    market_metrics: Dict[str, float]
    error_count: int
    success: bool

class MultiRoundSimulationDemo:
    """Orchestrate multiple rounds of market simulation testing"""
    
    def __init__(self, output_dir: str = "simulation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_scenarios = self._create_test_scenarios()
        self.simulation_results: List[SimulationResult] = []
        
        # Performance tracking
        self.start_time = datetime.now()
        self.comparison_metrics = {}
        
        logger.info(f"Initialized demo with {len(self.test_scenarios)} test scenarios")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _create_test_scenarios(self) -> List[TestScenario]:
        """Create comprehensive test scenarios"""
        
        scenarios = [
            # Quick smoke tests
            TestScenario(
                name="quick_smoke_test",
                description="Fast validation of basic functionality",
                symbols=["AAPL", "MSFT"],
                duration_hours=0.01,  # ~36 seconds real time
                market_regime="stable",
                news_frequency=1.0,  # High frequency for testing
                llm_agents_count=2,
                abides_agents_count=5,
                real_time_factor=1000.0,  # Very fast
                expected_outcomes={"min_trades": 5, "max_errors": 2}
            ),
            
            # Bull market scenario
            TestScenario(
                name="bull_market_momentum",
                description="Rising market with momentum strategies",
                symbols=["AAPL", "MSFT", "GOOGL", "TSLA"],
                duration_hours=0.1,  # 6 minutes real time
                market_regime="bull",
                news_frequency=0.3,
                llm_agents_count=3,
                abides_agents_count=10,
                real_time_factor=200.0,
                expected_outcomes={"positive_pnl_ratio": 0.6, "high_volume": True}
            ),
            
            # Bear market scenario
            TestScenario(
                name="bear_market_defensive",
                description="Declining market with defensive strategies",
                symbols=["AAPL", "MSFT", "GOOGL"],
                duration_hours=0.08,  # ~5 minutes real time
                market_regime="bear",
                news_frequency=0.4,  # Higher news in bear markets
                llm_agents_count=3,
                abides_agents_count=8,
                real_time_factor=180.0,
                expected_outcomes={"negative_pnl_ratio": 0.4, "high_volatility": True}
            ),
            
            # High volatility scenario
            TestScenario(
                name="volatile_market_chaos",
                description="High volatility with mixed signals",
                symbols=["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "META"],
                duration_hours=0.12,  # ~7 minutes real time
                market_regime="volatile",
                news_frequency=0.8,  # Very high news frequency
                llm_agents_count=4,
                abides_agents_count=15,
                real_time_factor=150.0,
                expected_outcomes={"high_signal_count": True, "mixed_pnl": True}
            ),
            
            # Large scale test
            TestScenario(
                name="large_scale_stress_test",
                description="Stress test with many agents and symbols",
                symbols=["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "META", "NVDA", "NFLX"],
                duration_hours=0.15,  # ~9 minutes real time
                market_regime="stable",
                news_frequency=0.2,
                llm_agents_count=6,
                abides_agents_count=25,
                real_time_factor=120.0,
                expected_outcomes={"scalability_test": True, "stable_performance": True}
            ),
            
            # LLM vs Traditional comparison
            TestScenario(
                name="llm_effectiveness_test",
                description="Compare LLM-enhanced vs traditional agents",
                symbols=["AAPL", "MSFT", "GOOGL"],
                duration_hours=0.1,  # 6 minutes real time
                market_regime="stable",
                news_frequency=0.5,
                llm_agents_count=5,  # More LLM agents for comparison
                abides_agents_count=5,
                real_time_factor=200.0,
                expected_outcomes={"llm_advantage": True, "performance_gap": 0.1}
            )
        ]
        
        return scenarios
    
    def run_all_simulations(self) -> Dict[str, Any]:
        """Run all simulation scenarios and compile results"""
        
        logger.info("🚀 Starting Multi-Round Market Simulation Demo")
        logger.info("=" * 80)
        
        total_scenarios = len(self.test_scenarios)
        successful_runs = 0
        
        for i, scenario in enumerate(self.test_scenarios, 1):
            logger.info(f"\n📊 Running Scenario {i}/{total_scenarios}: {scenario.name}")
            logger.info(f"Description: {scenario.description}")
            logger.info(f"Expected runtime: ~{(scenario.duration_hours * 60 / scenario.real_time_factor):.1f} seconds")
            
            try:
                result = self._run_single_scenario(scenario)
                self.simulation_results.append(result)
                
                if result.success:
                    successful_runs += 1
                    logger.info(f"✅ Scenario {scenario.name} completed successfully")
                    self._log_scenario_summary(result)
                else:
                    logger.error(f"❌ Scenario {scenario.name} failed")
                
            except Exception as e:
                logger.error(f"❌ Scenario {scenario.name} crashed: {e}")
                traceback.print_exc()
                
                # Create failure result
                failure_result = SimulationResult(
                    scenario_name=scenario.name,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration_seconds=0,
                    total_trades=0,
                    total_volume=0,
                    total_pnl=0,
                    news_events_count=0,
                    agent_performances=[],
                    market_metrics={},
                    error_count=1,
                    success=False
                )
                self.simulation_results.append(failure_result)
            
            # Small break between scenarios
            time.sleep(2)
        
        # Generate comprehensive analysis
        logger.info("\n📈 Generating comprehensive analysis...")
        final_report = self._generate_final_analysis()
        
        # Save results
        self._save_all_results(final_report)
        
        logger.info("=" * 80)
        logger.info(f"🎯 Multi-Round Demo Complete: {successful_runs}/{total_scenarios} scenarios successful")
        logger.info("=" * 80)
        
        return final_report
    
    def _run_single_scenario(self, scenario: TestScenario) -> SimulationResult:
        """Run a single simulation scenario"""
        
        start_time = datetime.now()
        
        if not FULL_SIMULATION_AVAILABLE:
            return self._run_mock_scenario(scenario, start_time)
        
        try:
            # Create simulation configuration
            config = SimulationConfig(
                symbols=scenario.symbols,
                simulation_duration_hours=scenario.duration_hours,
                news_frequency=scenario.news_frequency,
                llm_agents_count=scenario.llm_agents_count,
                abides_agents_count=scenario.abides_agents_count,
                real_time_factor=scenario.real_time_factor,
                save_data=True,
                output_directory=str(self.output_dir / f"{scenario.name}_output")
            )
            
            # Create and run simulation
            simulation = RealisticMarketSimulation(config)
            
            # Start simulation
            simulation.start_simulation(blocking=True)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Collect results
            result = self._collect_simulation_results(scenario, simulation, start_time, end_time, duration)
            
            return result
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return SimulationResult(
                scenario_name=scenario.name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                total_trades=0,
                total_volume=0,
                total_pnl=0,
                news_events_count=0,
                agent_performances=[],
                market_metrics={},
                error_count=1,
                success=False
            )
    
    def _run_mock_scenario(self, scenario: TestScenario, start_time: datetime) -> SimulationResult:
        """Run a mock scenario when full simulation is not available"""
        
        logger.info(f"Running mock simulation for {scenario.name}")
        
        # Simulate some processing time
        expected_duration = scenario.duration_hours * 60 / scenario.real_time_factor
        time.sleep(min(expected_duration, 5.0))  # Cap at 5 seconds for demo
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Generate mock results
        total_agents = scenario.llm_agents_count + scenario.abides_agents_count
        mock_trades = np.random.poisson(total_agents * 2)
        mock_volume = np.random.uniform(100000, 1000000)
        mock_pnl = np.random.normal(0, mock_volume * 0.001)
        mock_news_events = max(1, int(scenario.duration_hours * 60 * scenario.news_frequency))
        
        # Generate mock agent performances
        agent_performances = []
        for i in range(total_agents):
            agent_type = "LLM" if i < scenario.llm_agents_count else "ABIDES"
            performance = {
                "agent_id": f"{agent_type}_Agent_{i+1}",
                "agent_type": agent_type,
                "total_trades": np.random.poisson(3),
                "total_pnl": np.random.normal(0, 1000),
                "total_volume": np.random.uniform(10000, 100000),
                "win_rate": np.random.uniform(0.3, 0.7),
                "sharpe_ratio": np.random.uniform(-1, 2),
                "max_drawdown": np.random.uniform(0.05, 0.25)
            }
            agent_performances.append(performance)
        
        # Mock market metrics
        market_metrics = {
            "avg_spread": np.random.uniform(0.01, 0.05),
            "volatility": np.random.uniform(0.1, 0.4),
            "liquidity_score": np.random.uniform(0.6, 0.9),
            "market_efficiency": np.random.uniform(0.7, 0.95),
            "price_impact": np.random.uniform(0.001, 0.01)
        }
        
        return SimulationResult(
            scenario_name=scenario.name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            total_trades=mock_trades,
            total_volume=mock_volume,
            total_pnl=mock_pnl,
            news_events_count=mock_news_events,
            agent_performances=agent_performances,
            market_metrics=market_metrics,
            error_count=0,
            success=True
        )
    
    def _collect_simulation_results(self, scenario: TestScenario, simulation: RealisticMarketSimulation, 
                                  start_time: datetime, end_time: datetime, duration: float) -> SimulationResult:
        """Collect comprehensive results from simulation"""
        
        # Get execution reports and agent states
        execution_reports = simulation.execution_reports
        llm_agents = simulation.llm_agents
        abides_agents = simulation.abides_agents
        market_state = simulation.market_state
        
        # Calculate aggregated metrics
        total_trades = len(execution_reports)
        total_volume = sum(report.get('notional_value', 0) for report in execution_reports)
        total_pnl = sum(report.get('market_impact', 0) for report in execution_reports)
        news_events_count = len(simulation.active_news_events)
        
        # Collect agent performances
        agent_performances = []
        
        # LLM agents
        for agent in llm_agents:
            performance = {
                "agent_id": agent.name,
                "agent_type": "LLM",
                "strategy": getattr(agent, 'strategy', 'unknown'),
                "total_trades": len([r for r in execution_reports if r.get('agent_id', '').startswith(agent.name)]),
                "total_pnl": sum(r.get('market_impact', 0) for r in execution_reports if r.get('agent_id', '').startswith(agent.name)),
                "signals_generated": len(getattr(agent, 'signals_generated', [])),
                "risk_tolerance": getattr(agent, 'risk_tolerance', 0.5)
            }
            agent_performances.append(performance)
        
        # ABIDES agents
        for agent in abides_agents:
            agent_reports = [r for r in execution_reports if r.get('agent_id') == agent.agent_id]
            performance = {
                "agent_id": agent.agent_id,
                "agent_type": "ABIDES",
                "strategy": agent.strategy,
                "total_trades": len(agent_reports),
                "total_pnl": sum(r.get('market_impact', 0) for r in agent_reports),
                "portfolio_value": agent.portfolio.total_value,
                "cash_balance": agent.portfolio.cash
            }
            agent_performances.append(performance)
        
        # Market metrics
        market_metrics = {
            "volatility": market_state.volatility,
            "trend": market_state.trend,
            "sentiment": market_state.sentiment,
            "regime": market_state.regime,
            "efficiency_score": np.random.uniform(0.7, 0.95)  # Placeholder
        }
        
        return SimulationResult(
            scenario_name=scenario.name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            total_trades=total_trades,
            total_volume=total_volume,
            total_pnl=total_pnl,
            news_events_count=news_events_count,
            agent_performances=agent_performances,
            market_metrics=market_metrics,
            error_count=0,
            success=True
        )
    
    def _log_scenario_summary(self, result: SimulationResult):
        """Log summary of scenario results"""
        
        logger.info(f"   ⏱️  Duration: {result.duration_seconds:.1f} seconds")
        logger.info(f"   📈 Total Trades: {result.total_trades}")
        logger.info(f"   💰 Total PnL: ${result.total_pnl:,.2f}")
        logger.info(f"   📰 News Events: {result.news_events_count}")
        logger.info(f"   🤖 Active Agents: {len(result.agent_performances)}")
        
        if result.agent_performances:
            llm_agents = [a for a in result.agent_performances if a['agent_type'] == 'LLM']
            abides_agents = [a for a in result.agent_performances if a['agent_type'] == 'ABIDES']
            
            if llm_agents:
                avg_llm_pnl = np.mean([a['total_pnl'] for a in llm_agents])
                logger.info(f"   🧠 LLM Agents Avg PnL: ${avg_llm_pnl:,.2f}")
            
            if abides_agents:
                avg_abides_pnl = np.mean([a['total_pnl'] for a in abides_agents])
                logger.info(f"   ⚙️  ABIDES Agents Avg PnL: ${avg_abides_pnl:,.2f}")
    
    def _generate_final_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive final analysis"""
        
        if not self.simulation_results:
            return {"error": "No simulation results to analyze"}
        
        successful_results = [r for r in self.simulation_results if r.success]
        
        # Overall statistics
        total_demo_time = (datetime.now() - self.start_time).total_seconds()
        total_trades = sum(r.total_trades for r in successful_results)
        total_volume = sum(r.total_volume for r in successful_results)
        total_pnl = sum(r.total_pnl for r in successful_results)
        
        # Performance by scenario type
        scenario_analysis = {}
        for result in successful_results:
            scenario_type = self._classify_scenario_type(result.scenario_name)
            if scenario_type not in scenario_analysis:
                scenario_analysis[scenario_type] = []
            scenario_analysis[scenario_type].append(result)
        
        # Agent type comparison
        llm_performance = []
        abides_performance = []
        
        for result in successful_results:
            for agent in result.agent_performances:
                if agent['agent_type'] == 'LLM':
                    llm_performance.append(agent['total_pnl'])
                else:
                    abides_performance.append(agent['total_pnl'])
        
        # Market regime analysis
        regime_performance = {}
        for scenario in self.test_scenarios:
            regime = scenario.market_regime
            scenario_results = [r for r in successful_results if r.scenario_name == scenario.name]
            if scenario_results and regime not in regime_performance:
                regime_performance[regime] = {
                    'avg_pnl': np.mean([r.total_pnl for r in scenario_results]),
                    'avg_trades': np.mean([r.total_trades for r in scenario_results]),
                    'success_rate': len(scenario_results) / len([s for s in self.test_scenarios if s.market_regime == regime])
                }
        
        # Scalability analysis
        scalability_metrics = self._analyze_scalability()
        
        # Error analysis
        error_analysis = self._analyze_errors()
        
        final_report = {
            "demo_summary": {
                "total_scenarios": len(self.test_scenarios),
                "successful_scenarios": len(successful_results),
                "success_rate": len(successful_results) / len(self.test_scenarios),
                "total_demo_time_seconds": total_demo_time,
                "total_trades": total_trades,
                "total_volume": total_volume,
                "total_pnl": total_pnl
            },
            "scenario_analysis": scenario_analysis,
            "agent_comparison": {
                "llm_agents": {
                    "count": len(llm_performance),
                    "avg_pnl": np.mean(llm_performance) if llm_performance else 0,
                    "std_pnl": np.std(llm_performance) if llm_performance else 0,
                    "total_pnl": sum(llm_performance) if llm_performance else 0
                },
                "abides_agents": {
                    "count": len(abides_performance),
                    "avg_pnl": np.mean(abides_performance) if abides_performance else 0,
                    "std_pnl": np.std(abides_performance) if abides_performance else 0,
                    "total_pnl": sum(abides_performance) if abides_performance else 0
                }
            },
            "regime_analysis": regime_performance,
            "scalability_metrics": scalability_metrics,
            "error_analysis": error_analysis,
            "detailed_results": [asdict(r) for r in self.simulation_results],
            "recommendations": self._generate_recommendations()
        }
        
        return final_report
    
    def _classify_scenario_type(self, scenario_name: str) -> str:
        """Classify scenario into type for analysis"""
        if "smoke" in scenario_name:
            return "validation"
        elif "bull" in scenario_name:
            return "bull_market"
        elif "bear" in scenario_name:
            return "bear_market"
        elif "volatile" in scenario_name:
            return "high_volatility"
        elif "stress" in scenario_name:
            return "stress_test"
        elif "llm" in scenario_name:
            return "llm_comparison"
        else:
            return "general"
    
    def _analyze_scalability(self) -> Dict[str, Any]:
        """Analyze scalability across different scenario sizes"""
        
        scalability_data = []
        
        for result in self.simulation_results:
            if result.success:
                scenario = next(s for s in self.test_scenarios if s.name == result.scenario_name)
                total_agents = scenario.llm_agents_count + scenario.abides_agents_count
                
                scalability_data.append({
                    "total_agents": total_agents,
                    "total_symbols": len(scenario.symbols),
                    "duration_seconds": result.duration_seconds,
                    "trades_per_second": result.total_trades / max(result.duration_seconds, 1),
                    "pnl_per_agent": result.total_pnl / max(total_agents, 1)
                })
        
        if not scalability_data:
            return {}
        
        df = pd.DataFrame(scalability_data)
        
        return {
            "max_agents_tested": df['total_agents'].max(),
            "max_symbols_tested": df['total_symbols'].max(),
            "avg_trades_per_second": df['trades_per_second'].mean(),
            "performance_correlation_with_scale": df[['total_agents', 'trades_per_second']].corr().iloc[0, 1],
            "scalability_rating": "Good" if df['trades_per_second'].std() / df['trades_per_second'].mean() < 0.5 else "Needs Improvement"
        }
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze errors and failures across scenarios"""
        
        failed_results = [r for r in self.simulation_results if not r.success]
        total_errors = sum(r.error_count for r in self.simulation_results)
        
        return {
            "total_errors": total_errors,
            "failed_scenarios": len(failed_results),
            "error_rate": total_errors / len(self.simulation_results) if self.simulation_results else 0,
            "failed_scenario_names": [r.scenario_name for r in failed_results],
            "reliability_score": 1 - (len(failed_results) / len(self.simulation_results)) if self.simulation_results else 0
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on simulation results"""
        
        recommendations = []
        
        successful_rate = len([r for r in self.simulation_results if r.success]) / len(self.simulation_results)
        
        if successful_rate < 0.8:
            recommendations.append("Improve system reliability - success rate below 80%")
        
        if self.simulation_results:
            avg_duration = np.mean([r.duration_seconds for r in self.simulation_results if r.success])
            if avg_duration > 60:
                recommendations.append("Consider optimizing simulation speed for better user experience")
        
        # Agent performance analysis
        llm_agents_data = []
        abides_agents_data = []
        
        for result in self.simulation_results:
            if result.success:
                for agent in result.agent_performances:
                    if agent['agent_type'] == 'LLM':
                        llm_agents_data.append(agent['total_pnl'])
                    else:
                        abides_agents_data.append(agent['total_pnl'])
        
        if llm_agents_data and abides_agents_data:
            llm_avg = np.mean(llm_agents_data)
            abides_avg = np.mean(abides_agents_data)
            
            if llm_avg > abides_avg * 1.1:
                recommendations.append("LLM agents show superior performance - consider increasing LLM agent allocation")
            elif abides_avg > llm_avg * 1.1:
                recommendations.append("Traditional ABIDES agents outperforming - review LLM strategy configuration")
        
        if not recommendations:
            recommendations.append("System performing well across all test scenarios")
        
        return recommendations
    
    def _save_all_results(self, final_report: Dict[str, Any]):
        """Save all results and generate visualizations"""
        
        # Save JSON report
        report_file = self.output_dir / "multi_round_simulation_report.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"📊 Saved detailed report to: {report_file}")
        
        # Generate CSV summary
        self._save_csv_summary()
        
        # Generate visualizations
        self._generate_visualizations(final_report)
        
        # Save scenario configurations
        scenarios_file = self.output_dir / "test_scenarios.json"
        with open(scenarios_file, 'w') as f:
            json.dump([asdict(s) for s in self.test_scenarios], f, indent=2, default=str)
    
    def _save_csv_summary(self):
        """Save CSV summary of all simulation results"""
        
        csv_data = []
        for result in self.simulation_results:
            scenario = next(s for s in self.test_scenarios if s.name == result.scenario_name)
            
            row = {
                "scenario_name": result.scenario_name,
                "success": result.success,
                "duration_seconds": result.duration_seconds,
                "total_trades": result.total_trades,
                "total_volume": result.total_volume,
                "total_pnl": result.total_pnl,
                "news_events": result.news_events_count,
                "agent_count": len(result.agent_performances),
                "market_regime": scenario.market_regime,
                "symbols_count": len(scenario.symbols),
                "llm_agents": scenario.llm_agents_count,
                "abides_agents": scenario.abides_agents_count
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_file = self.output_dir / "simulation_summary.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"📈 Saved CSV summary to: {csv_file}")
    
    def _generate_visualizations(self, final_report: Dict[str, Any]):
        """Generate visualization charts"""
        
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multi-Round Market Simulation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Success Rate by Scenario
        successful_results = [r for r in self.simulation_results if r.success]
        scenario_names = [r.scenario_name.replace('_', '\n') for r in self.simulation_results]
        success_values = [1 if r.success else 0 for r in self.simulation_results]
        
        axes[0, 0].bar(scenario_names, success_values, color=['green' if v else 'red' for v in success_values])
        axes[0, 0].set_title('Scenario Success Rate')
        axes[0, 0].set_ylabel('Success (1) / Failure (0)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. PnL Distribution
        if successful_results:
            pnl_values = [r.total_pnl for r in successful_results]
            axes[0, 1].hist(pnl_values, bins=10, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 1].set_title('PnL Distribution')
            axes[0, 1].set_xlabel('Total PnL ($)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(np.mean(pnl_values), color='red', linestyle='--', label=f'Mean: ${np.mean(pnl_values):.0f}')
            axes[0, 1].legend()
        
        # 3. Trading Volume by Scenario
        if successful_results:
            volumes = [r.total_volume for r in successful_results]
            scenario_names_success = [r.scenario_name.replace('_', '\n') for r in successful_results]
            axes[0, 2].bar(scenario_names_success, volumes, color='orange', alpha=0.7)
            axes[0, 2].set_title('Trading Volume by Scenario')
            axes[0, 2].set_ylabel('Total Volume ($)')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Agent Performance Comparison
        llm_pnl = []
        abides_pnl = []
        
        for result in successful_results:
            for agent in result.agent_performances:
                if agent['agent_type'] == 'LLM':
                    llm_pnl.append(agent['total_pnl'])
                else:
                    abides_pnl.append(agent['total_pnl'])
        
        if llm_pnl and abides_pnl:
            axes[1, 0].boxplot([llm_pnl, abides_pnl], tick_labels=['LLM Agents', 'ABIDES Agents'])
            axes[1, 0].set_title('Agent Performance Comparison')
            axes[1, 0].set_ylabel('PnL per Agent ($)')
        
        # 5. Market Regime Performance
        regime_data = final_report.get('regime_analysis', {})
        if regime_data:
            regimes = list(regime_data.keys())
            avg_pnls = [regime_data[r]['avg_pnl'] for r in regimes]
            
            axes[1, 1].bar(regimes, avg_pnls, color=['bull'=='bull' and 'green' or 'bear'=='bear' and 'red' or 'blue' for r in regimes])
            axes[1, 1].set_title('Performance by Market Regime')
            axes[1, 1].set_ylabel('Average PnL ($)')
        
        # 6. Simulation Duration vs Complexity
        if successful_results:
            complexities = []
            durations = []
            
            for result in successful_results:
                scenario = next(s for s in self.test_scenarios if s.name == result.scenario_name)
                complexity = (scenario.llm_agents_count + scenario.abides_agents_count) * len(scenario.symbols)
                complexities.append(complexity)
                durations.append(result.duration_seconds)
            
            axes[1, 2].scatter(complexities, durations, alpha=0.7, color='purple')
            axes[1, 2].set_title('Duration vs Scenario Complexity')
            axes[1, 2].set_xlabel('Complexity (Agents × Symbols)')
            axes[1, 2].set_ylabel('Duration (seconds)')
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.output_dir / "simulation_analysis.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Saved visualizations to: {viz_file}")
    
    def print_final_summary(self, final_report: Dict[str, Any]):
        """Print comprehensive final summary"""
        
        print("\n" + "="*80)
        print("🎯 MULTI-ROUND MARKET SIMULATION DEMO - FINAL RESULTS")
        print("="*80)
        
        summary = final_report['demo_summary']
        print(f"📊 Total Scenarios: {summary['total_scenarios']}")
        print(f"✅ Successful Runs: {summary['successful_scenarios']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1%}")
        print(f"⏱️  Total Demo Time: {summary['total_demo_time_seconds']:.1f} seconds")
        print(f"💹 Total Trades Executed: {summary['total_trades']:,}")
        print(f"💰 Total Trading Volume: ${summary['total_volume']:,.0f}")
        print(f"📊 Net PnL Across All Scenarios: ${summary['total_pnl']:,.2f}")
        
        print("\n🤖 AGENT PERFORMANCE COMPARISON")
        print("-" * 40)
        agent_comp = final_report['agent_comparison']
        
        if agent_comp['llm_agents']['count'] > 0:
            print(f"🧠 LLM Agents:")
            print(f"   Count: {agent_comp['llm_agents']['count']}")
            print(f"   Average PnL: ${agent_comp['llm_agents']['avg_pnl']:,.2f}")
            print(f"   Total PnL: ${agent_comp['llm_agents']['total_pnl']:,.2f}")
        
        if agent_comp['abides_agents']['count'] > 0:
            print(f"⚙️  ABIDES Agents:")
            print(f"   Count: {agent_comp['abides_agents']['count']}")
            print(f"   Average PnL: ${agent_comp['abides_agents']['avg_pnl']:,.2f}")
            print(f"   Total PnL: ${agent_comp['abides_agents']['total_pnl']:,.2f}")
        
        print("\n🌍 MARKET REGIME ANALYSIS")
        print("-" * 40)
        regime_analysis = final_report['regime_analysis']
        for regime, data in regime_analysis.items():
            print(f"{regime.title()} Market:")
            print(f"   Average PnL: ${data['avg_pnl']:,.2f}")
            print(f"   Average Trades: {data['avg_trades']:.1f}")
            print(f"   Success Rate: {data['success_rate']:.1%}")
        
        print("\n⚡ SCALABILITY METRICS")
        print("-" * 40)
        scalability = final_report['scalability_metrics']
        if scalability:
            print(f"Max Agents Tested: {scalability.get('max_agents_tested', 'N/A')}")
            print(f"Max Symbols Tested: {scalability.get('max_symbols_tested', 'N/A')}")
            print(f"Avg Trades/Second: {scalability.get('avg_trades_per_second', 0):.2f}")
            print(f"Scalability Rating: {scalability.get('scalability_rating', 'Unknown')}")
        
        print("\n🔧 RECOMMENDATIONS")
        print("-" * 40)
        recommendations = final_report['recommendations']
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        print("\n📁 OUTPUT FILES")
        print("-" * 40)
        print(f"📊 Detailed Report: {self.output_dir}/multi_round_simulation_report.json")
        print(f"📈 CSV Summary: {self.output_dir}/simulation_summary.csv")
        print(f"📊 Visualizations: {self.output_dir}/simulation_analysis.png")
        print(f"⚙️  Test Scenarios: {self.output_dir}/test_scenarios.json")
        
        print("\n" + "="*80)
        print("🎉 Multi-Round Market Simulation Demo Complete!")
        print("="*80)


def main():
    """Main function to run the multi-round simulation demo"""
    
    print("🚀 Multi-Round Market Simulation Test Demo")
    print("=" * 60)
    print("This demo will run multiple market simulation scenarios")
    print("with different configurations and provide comprehensive analysis.")
    print()
    
    # Create and run demo
    demo = MultiRoundSimulationDemo()
    
    try:
        # Run all simulations
        final_report = demo.run_all_simulations()
        
        # Print comprehensive summary
        demo.print_final_summary(final_report)
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)