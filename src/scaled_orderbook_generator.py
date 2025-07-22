#!/usr/bin/env python3
"""
Scaled Order Book Data Generator
===============================

High-performance order book data generation system for large-scale market simulation.
Generates realistic order flow patterns, trading volumes, and market microstructure data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import random
import logging
import json
from pathlib import Path
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScaledOrderBookConfig:
    """Configuration for scaled order book generation"""
    
    # Simulation parameters
    num_agents: int = 10000           # Number of trading agents
    simulation_days: int = 30         # Number of trading days
    orders_per_minute: int = 1000     # Base orders per minute
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])
    
    # Market parameters
    initial_prices: Dict[str, float] = field(default_factory=lambda: {
        "AAPL": 150.0, "GOOGL": 2500.0, "MSFT": 300.0, "TSLA": 800.0, "AMZN": 3000.0
    })
    volatility: float = 0.02          # Daily volatility
    spread_bps: int = 5               # Bid-ask spread in basis points
    
    # Agent distribution
    institutional_agents: float = 0.1 # 10% institutional
    retail_agents: float = 0.7        # 70% retail
    hft_agents: float = 0.15          # 15% high-frequency
    market_makers: float = 0.05       # 5% market makers
    
    # Order characteristics
    avg_order_size: int = 100
    large_order_threshold: int = 10000
    market_order_ratio: float = 0.3   # 30% market orders
    
    # Performance settings
    batch_size: int = 10000           # Orders per batch
    parallel_workers: int = 4         # Parallel processing workers
    
    # Output settings
    output_format: str = "csv"        # csv, parquet, json
    compression: bool = True
    output_directory: str = "scaled_data"

class AgentProfile:
    """Profile for different types of trading agents"""
    
    def __init__(self, agent_type: str, agent_id: str):
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.setup_profile()
    
    def setup_profile(self):
        """Setup agent-specific characteristics"""
        if self.agent_type == "institutional":
            self.order_size_range = (1000, 50000)
            self.order_frequency = 0.1  # Orders per minute
            self.market_order_prob = 0.2
            self.cancel_prob = 0.15
            self.price_improvement = 0.3  # Probability of price improvement
            
        elif self.agent_type == "retail":
            self.order_size_range = (10, 1000)
            self.order_frequency = 0.05
            self.market_order_prob = 0.4
            self.cancel_prob = 0.1
            self.price_improvement = 0.1
            
        elif self.agent_type == "hft":
            self.order_size_range = (50, 500)
            self.order_frequency = 5.0  # Very high frequency
            self.market_order_prob = 0.1
            self.cancel_prob = 0.8  # High cancellation rate
            self.price_improvement = 0.9
            
        elif self.agent_type == "market_maker":
            self.order_size_range = (100, 5000)
            self.order_frequency = 2.0
            self.market_order_prob = 0.05
            self.cancel_prob = 0.5
            self.price_improvement = 0.95
        
        else:  # default
            self.order_size_range = (100, 1000)
            self.order_frequency = 0.1
            self.market_order_prob = 0.3
            self.cancel_prob = 0.2
            self.price_improvement = 0.3

class MarketDataGenerator:
    """Generate realistic market data patterns"""
    
    def __init__(self, config: ScaledOrderBookConfig):
        self.config = config
        self.current_prices = config.initial_prices.copy()
        self.price_history = {symbol: [price] for symbol, price in config.initial_prices.items()}
        
    def generate_price_movement(self, symbol: str, time_elapsed_minutes: int) -> float:
        """Generate realistic price movement using GBM with mean reversion"""
        current_price = self.current_prices[symbol]
        initial_price = self.config.initial_prices[symbol]
        
        # Mean reversion component
        mean_reversion_strength = 0.1
        mean_reversion = mean_reversion_strength * (initial_price - current_price) / initial_price
        
        # Random walk component
        dt = 1.0 / (252 * 24 * 60)  # 1 minute in years
        random_shock = np.random.normal(0, self.config.volatility * np.sqrt(dt))
        
        # Price update
        price_change = current_price * (mean_reversion * dt + random_shock)
        new_price = max(0.01, current_price + price_change)
        
        self.current_prices[symbol] = new_price
        self.price_history[symbol].append(new_price)
        
        return new_price
    
    def get_realistic_spread(self, symbol: str, price: float) -> Tuple[float, float]:
        """Calculate realistic bid-ask spread"""
        spread_dollars = price * (self.config.spread_bps / 10000.0)
        
        # Add some randomness to spread
        spread_multiplier = np.random.uniform(0.5, 2.0)
        actual_spread = spread_dollars * spread_multiplier
        
        bid = price - actual_spread / 2
        ask = price + actual_spread / 2
        
        return max(0.01, bid), ask
