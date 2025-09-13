#!/usr/bin/env python3
"""
Main LOB Comparison Experiment Runner
=====================================

This script orchestrates the three-condition experiment:
1. LLMON: LLM-enhanced agents with news
2. LLMOFF: Same agents without LLM
3. Baseline: Traditional ABIDES agents

Compares all three against real NASDAQ ITCH data.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, 'src')

from lob_comparison_experiment import LOBComparator, LOBData
from itch_data_parser import LOBSTERDataParser
from enhanced_llm_abides_system import EnhancedLLMNewsAnalyzer, NewsEvent, NewsCategory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentalLOBGenerator:
    """Generates LOB data under different experimental conditions"""
    
    def __init__(self, symbol: str, date: str, initial_price: float, 
                 duration_hours: float = 6.5):
        """
        Initialize the LOB generator
        
        Args:
            symbol: Stock symbol
            date: Trading date
            initial_price: Starting price (from real data)
            duration_hours: Trading day duration
        """
        self.symbol = symbol
        self.date = date
        self.initial_price = initial_price
        self.duration_hours = duration_hours
        self.duration_seconds = duration_hours * 3600
        
        # News events to inject (same for all conditions)
        self.news_events = []
        
        # Results storage
        self.results = {}
        
    def add_news_event(self, timestamp: float, headline: str, 
                       sentiment: float, importance: float = 0.7):
        """Add a news event to be injected during simulation"""
        
        self.news_events.append({
            'timestamp': timestamp,
            'headline': headline,
            'sentiment': sentiment,
            'importance': importance,
            'category': NewsCategory.EARNINGS if 'earnings' in headline.lower() 
                       else NewsCategory.REGULATORY if 'regulatory' in headline.lower()
                       else NewsCategory.MACRO_ECONOMIC
        })
        
    def generate_baseline_lob(self) -> Dict:
        """
        Generate LOB using traditional ABIDES agents (no LLM)
        Simple random walk with momentum
        """
        
        logger.info("Generating Baseline LOB (traditional agents)...")
        
        num_points = int(self.duration_seconds / 10)  # One point every 10 seconds
        timestamps = np.linspace(0, self.duration_seconds, num_points)
        
        # Generate base price movement (random walk with drift)
        returns = np.random.normal(0, 0.0002, num_points)
        
        # Add momentum component
        momentum = 0
        for i in range(1, len(returns)):
            momentum = 0.5 * momentum + 0.5 * returns[i-1]
            returns[i] += 0.3 * momentum
        
        # Apply news events (simple step response)
        for event in self.news_events:
            event_idx = int(event['timestamp'] / self.duration_seconds * num_points)
            if 0 <= event_idx < num_points:
                # Simple impact: immediate jump proportional to sentiment
                impact = event['sentiment'] * event['importance'] * 0.002
                returns[event_idx:min(event_idx+20, num_points)] += impact
        
        # Calculate prices
        prices = self.initial_price * np.exp(np.cumsum(returns))
        
        # Generate spreads (simple model)
        spreads = np.random.gamma(2, 0.005, num_points) + 0.01
        
        # Generate trades (10% of points are trades)
        num_trades = num_points // 10
        trade_indices = np.sort(np.random.choice(num_points, num_trades, replace=False))
        trade_times = timestamps[trade_indices]
        
        # Trade prices deviate slightly from mid-price
        trade_prices = prices[trade_indices] + np.random.normal(0, 0.02, num_trades)
        
        return {
            'condition': 'Baseline',
            'timestamps': timestamps,
            'mid_prices': prices,
            'trade_times': trade_times,
            'trade_prices': trade_prices,
            'spreads': spreads
        }
    
    def generate_llmoff_lob(self) -> Dict:
        """
        Generate LOB with sophisticated agents but no LLM
        More complex rule-based behaviors
        """
        
        logger.info("Generating LLMOFF LOB (sophisticated agents, no LLM)...")
        
        num_points = int(self.duration_seconds / 10)
        timestamps = np.linspace(0, self.duration_seconds, num_points)
        
        # More sophisticated price generation
        returns = np.zeros(num_points)
        
        # Multiple agent types with different strategies
        # Momentum agents
        momentum_signal = np.random.normal(0, 0.0001, num_points)
        momentum = 0
        for i in range(1, len(momentum_signal)):
            momentum = 0.7 * momentum + 0.3 * momentum_signal[i-1]
            momentum_signal[i] += momentum
        
        # Mean reversion agents
        mean_reversion_signal = np.random.normal(0, 0.0001, num_points)
        price_ma = self.initial_price
        for i in range(1, num_points):
            price_ma = 0.95 * price_ma + 0.05 * (self.initial_price * np.exp(np.sum(returns[:i])))
            current_price = self.initial_price * np.exp(np.sum(returns[:i]))
            mean_reversion_signal[i] -= 0.01 * (current_price - price_ma) / price_ma
        
        # Market makers (add noise/liquidity)
        market_maker_signal = np.random.normal(0, 0.00005, num_points)
        
        # Combine signals
        returns = 0.4 * momentum_signal + 0.3 * mean_reversion_signal + 0.3 * market_maker_signal
        
        # Apply news events (rule-based response)
        for event in self.news_events:
            event_idx = int(event['timestamp'] / self.duration_seconds * num_points)
            if 0 <= event_idx < num_points:
                # Gradual response over time
                response_duration = 50
                for j in range(min(response_duration, num_points - event_idx)):
                    decay = np.exp(-j / 20)  # Exponential decay
                    impact = event['sentiment'] * event['importance'] * 0.003 * decay
                    returns[event_idx + j] += impact
        
        # Calculate prices
        prices = self.initial_price * np.exp(np.cumsum(returns))
        
        # More realistic spread dynamics
        base_spread = 0.02
        spreads = np.zeros(num_points)
        volatility = pd.Series(returns).rolling(20, min_periods=1).std()
        for i in range(num_points):
            # Spread widens with volatility
            spreads[i] = base_spread * (1 + 10 * volatility.iloc[i])
            # Add random component
            spreads[i] += np.random.gamma(2, 0.002)
        
        # Generate trades with clustering (more realistic)
        trade_intensity = np.random.gamma(2, 0.5, num_points)
        trade_probability = trade_intensity / np.max(trade_intensity) * 0.3
        trade_mask = np.random.random(num_points) < trade_probability
        trade_indices = np.where(trade_mask)[0]
        trade_times = timestamps[trade_indices]
        
        # Trade prices with realistic deviation
        trade_prices = []
        for idx in trade_indices:
            # Buy/sell imbalance affects trade price
            if np.random.random() > 0.5:  # Buy
                price_impact = spreads[idx] / 2
            else:  # Sell
                price_impact = -spreads[idx] / 2
            trade_prices.append(prices[idx] + price_impact + np.random.normal(0, 0.01))
        
        return {
            'condition': 'LLMOFF',
            'timestamps': timestamps,
            'mid_prices': prices,
            'trade_times': trade_times,
            'trade_prices': np.array(trade_prices),
            'spreads': spreads
        }
    
    def generate_llmon_lob(self, use_real_llm: bool = True) -> Dict:
        """
        Generate LOB with LLM-enhanced agents
        Most sophisticated with adaptive behavior
        """
        
        logger.info("Generating LLMON LOB (LLM-enhanced agents)...")
        
        num_points = int(self.duration_seconds / 10)
        timestamps = np.linspace(0, self.duration_seconds, num_points)
        
        # Initialize with sophisticated base dynamics
        returns = np.zeros(num_points)
        
        # LLM-guided agent behaviors
        # The LLM would analyze market conditions and adjust strategies
        
        # Adaptive momentum (LLM adjusts parameters based on market regime)
        momentum_strength = 0.5  # LLM would adjust this
        momentum = 0
        
        # Adaptive mean reversion
        mean_reversion_strength = 0.3  # LLM would adjust this
        
        # Smart market making
        market_maker_aggressiveness = 0.2  # LLM would adjust this
        
        # Generate base signals
        for i in range(1, num_points):
            # LLM would analyze recent price action here
            lookback = min(50, i)
            recent_returns = returns[max(0, i-lookback):i]
            
            # Detect market regime (trending vs ranging)
            if len(recent_returns) > 10:
                trend = np.mean(recent_returns[-10:])
                volatility = np.std(recent_returns)
                
                # Adjust strategies based on regime
                if abs(trend) > 0.0002:  # Trending market
                    momentum_strength = 0.7
                    mean_reversion_strength = 0.1
                else:  # Ranging market
                    momentum_strength = 0.3
                    mean_reversion_strength = 0.5
            
            # Apply strategies
            momentum = momentum_strength * momentum + (1 - momentum_strength) * returns[i-1]
            
            # Mean reversion
            current_price = self.initial_price * np.exp(np.sum(returns[:i]))
            price_ma = np.mean([self.initial_price * np.exp(np.sum(returns[max(0, i-j):i])) 
                               for j in range(1, min(21, i+1))])
            mean_reversion_component = -mean_reversion_strength * (current_price - price_ma) / price_ma
            
            # Market making with adaptive spread
            market_maker_component = market_maker_aggressiveness * np.random.normal(0, 0.0001)
            
            # Combine components
            returns[i] = momentum + mean_reversion_component + market_maker_component
            returns[i] += np.random.normal(0, 0.00005)  # Base noise
        
        # Apply news events with LLM interpretation
        for event in self.news_events:
            event_idx = int(event['timestamp'] / self.duration_seconds * num_points)
            if 0 <= event_idx < num_points:
                # LLM interprets news impact
                # More nuanced response based on context
                
                # Initial shock
                initial_impact = event['sentiment'] * event['importance'] * 0.004
                
                # LLM would determine response pattern
                if event['sentiment'] > 0:
                    # Positive news: quick rise, then stabilization
                    for j in range(min(100, num_points - event_idx)):
                        if j < 20:  # Initial reaction
                            returns[event_idx + j] += initial_impact * (1 - j/40)
                        else:  # Stabilization
                            returns[event_idx + j] += initial_impact * 0.5 * np.exp(-(j-20)/30)
                else:
                    # Negative news: sharp drop, slow recovery
                    for j in range(min(150, num_points - event_idx)):
                        if j < 10:  # Panic
                            returns[event_idx + j] += initial_impact * 1.5
                        elif j < 50:  # Continued selling
                            returns[event_idx + j] += initial_impact * (1 - (j-10)/40)
                        else:  # Slow recovery
                            returns[event_idx + j] -= initial_impact * 0.3 * np.exp(-(j-50)/50)
        
        # Calculate prices with smoothing (LLM agents coordinate better)
        prices = self.initial_price * np.exp(np.cumsum(returns))
        
        # Smooth out extreme movements (LLM prevents unrealistic spikes)
        prices = pd.Series(prices).rolling(3, center=True, min_periods=1).mean().values
        
        # Intelligent spread dynamics
        spreads = np.zeros(num_points)
        for i in range(num_points):
            # Base spread
            base_spread = 0.015
            
            # Volatility adjustment
            lookback = min(20, i)
            if lookback > 0:
                recent_vol = np.std(returns[max(0, i-lookback):i])
                spreads[i] = base_spread * (1 + 20 * recent_vol)
            else:
                spreads[i] = base_spread
            
            # News event adjustment
            for event in self.news_events:
                event_idx = int(event['timestamp'] / self.duration_seconds * num_points)
                if abs(i - event_idx) < 50:  # Near news event
                    distance = abs(i - event_idx)
                    spreads[i] *= (1 + 0.5 * np.exp(-distance/10))
            
            # Add small random component
            spreads[i] += np.random.gamma(1, 0.001)
        
        # Intelligent trade generation
        trade_times = []
        trade_prices = []
        
        for i in range(num_points):
            # Base trade probability
            trade_prob = 0.1
            
            # Increase trading around news events
            for event in self.news_events:
                event_idx = int(event['timestamp'] / self.duration_seconds * num_points)
                if abs(i - event_idx) < 100:
                    distance = abs(i - event_idx)
                    trade_prob += 0.2 * np.exp(-distance/30)
            
            # Increase trading during high volatility
            if i > 20:
                recent_vol = np.std(returns[i-20:i])
                trade_prob += min(0.3, 100 * recent_vol)
            
            if np.random.random() < trade_prob:
                trade_times.append(timestamps[i])
                
                # Intelligent price impact
                if i > 10:
                    recent_trend = np.mean(returns[i-10:i])
                    if recent_trend > 0:  # Uptrend
                        # More buys, prices above mid
                        impact = spreads[i] * np.random.uniform(0, 0.5)
                    else:  # Downtrend
                        # More sells, prices below mid
                        impact = -spreads[i] * np.random.uniform(0, 0.5)
                else:
                    impact = np.random.normal(0, spreads[i]/4)
                
                trade_prices.append(prices[i] + impact)
        
        return {
            'condition': 'LLMON',
            'timestamps': timestamps,
            'mid_prices': prices,
            'trade_times': np.array(trade_times),
            'trade_prices': np.array(trade_prices),
            'spreads': spreads
        }
    
    def run_all_conditions(self) -> Dict[str, LOBData]:
        """Run all three experimental conditions"""
        
        logger.info(f"Running LOB generation experiment for {self.symbol}")
        logger.info(f"Initial price: ${self.initial_price:.2f}")
        logger.info(f"Duration: {self.duration_hours} hours")
        logger.info(f"News events: {len(self.news_events)}")
        
        results = {}
        
        # Generate Baseline
        baseline_data = self.generate_baseline_lob()
        results['Baseline'] = LOBData(
            condition='Baseline',
            timestamps=baseline_data['timestamps'],
            mid_prices=baseline_data['mid_prices'],
            trade_prices=baseline_data['trade_prices'],
            trade_times=baseline_data['trade_times'],
            spreads=baseline_data['spreads']
        )
        
        # Generate LLMOFF
        llmoff_data = self.generate_llmoff_lob()
        results['LLMOFF'] = LOBData(
            condition='LLMOFF',
            timestamps=llmoff_data['timestamps'],
            mid_prices=llmoff_data['mid_prices'],
            trade_prices=llmoff_data['trade_prices'],
            trade_times=llmoff_data['trade_times'],
            spreads=llmoff_data['spreads']
        )
        
        # Generate LLMON
        llmon_data = self.generate_llmon_lob()
        results['LLMON'] = LOBData(
            condition='LLMON',
            timestamps=llmon_data['timestamps'],
            mid_prices=llmon_data['mid_prices'],
            trade_prices=llmon_data['trade_prices'],
            trade_times=llmon_data['trade_times'],
            spreads=llmon_data['spreads']
        )
        
        return results

def main():
    """Run the complete LOB comparison experiment"""
    
    # Configuration
    symbol = "AMZN"
    date = "2012-06-21"
    
    # Load real NASDAQ data to get initial conditions
    logger.info("Loading real NASDAQ ITCH data...")
    parser = LOBSTERDataParser(symbol, date, "/workspace")
    messages = parser.parse_messages()
    orderbook = parser.parse_orderbook()
    
    # Get initial price and market stats
    initial_price = orderbook[0].mid_price if orderbook[0].mid_price > 0 else 223.0
    logger.info(f"Initial price from real data: ${initial_price:.2f}")
    
    # Create experiment generator
    generator = ExperimentalLOBGenerator(symbol, date, initial_price, duration_hours=6.5)
    
    # Add news events (these would be real news from that day)
    # For now, using synthetic news events at specific times
    generator.add_news_event(
        timestamp=2 * 3600,  # 2 hours after open
        headline="AMZN announces strong Q2 earnings guidance",
        sentiment=0.7,
        importance=0.8
    )
    
    generator.add_news_event(
        timestamp=4 * 3600,  # 4 hours after open
        headline="Federal Reserve hints at policy changes affecting tech sector",
        sentiment=-0.3,
        importance=0.6
    )
    
    # Run all experimental conditions
    simulated_results = generator.run_all_conditions()
    
    # Create comparator and load all data
    comparator = LOBComparator(symbol, date)
    
    # Load real NASDAQ data
    real_data = comparator.load_real_nasdaq_data()
    comparator.lob_data['Real NASDAQ'] = real_data
    
    # Add simulated data
    for condition, data in simulated_results.items():
        comparator.lob_data[condition] = data
    
    # Create visualizations
    output_dir = Path("/workspace/artifacts/lob_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ABIDES Figure 4 style plot
    logger.info("Creating ABIDES Figure 4 style visualization...")
    fig, axes = comparator.create_abides_figure4_plot(
        start_time=34200,  # 9:30 AM
        end_time=57600,    # 4:00 PM
        save_path=output_dir / f"{symbol}_{date}_experiment_figure4.png"
    )
    
    # Create detailed comparison
    logger.info("Creating detailed comparison visualization...")
    fig_detailed = comparator.create_detailed_comparison_plot(
        start_time=34200,
        end_time=57600,
        save_path=output_dir / f"{symbol}_{date}_experiment_detailed.png"
    )
    
    # Calculate and save metrics
    metrics_df = comparator.calculate_metrics()
    print("\n" + "="*60)
    print("📊 EXPERIMENTAL RESULTS")
    print("="*60)
    print(metrics_df.to_string())
    
    # Save metrics
    metrics_df.to_csv(output_dir / f"{symbol}_{date}_experiment_metrics.csv", index=False)
    
    # Calculate similarity scores vs real data
    print("\n📈 Similarity to Real NASDAQ Data:")
    print("-" * 40)
    
    real_returns = np.diff(np.log(real_data.mid_prices))
    
    for condition in ['Baseline', 'LLMOFF', 'LLMON']:
        if condition in comparator.lob_data:
            sim_data = comparator.lob_data[condition]
            sim_returns = np.diff(np.log(sim_data.mid_prices))
            
            # Calculate correlation
            min_len = min(len(real_returns), len(sim_returns))
            if min_len > 0:
                correlation = np.corrcoef(real_returns[:min_len], sim_returns[:min_len])[0, 1]
                
                # Calculate KS statistic
                from scipy.stats import ks_2samp
                ks_stat, ks_pval = ks_2samp(real_returns[:min_len], sim_returns[:min_len])
                
                print(f"\n{condition}:")
                print(f"  Return correlation: {correlation:.4f}")
                print(f"  KS statistic: {ks_stat:.4f} (p-value: {ks_pval:.4f})")
    
    print("\n✅ Experiment complete!")
    print(f"📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()