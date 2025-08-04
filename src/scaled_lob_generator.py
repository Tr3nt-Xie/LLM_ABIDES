#!/usr/bin/env python3
"""
Scaled Limit Order Book (LOB) Data Generator
============================================

Advanced system for generating large-scale, detailed limit order book data
with comprehensive market microstructure patterns and database storage.

Features:
- Massive scale data generation (millions of orders)
- Detailed LOB depth tracking (Level 2+ data)
- Tick-by-tick order book snapshots
- Advanced agent behaviors and market patterns
- Optimized database storage with indexing
- Real-time LOB reconstruction capabilities
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import json
import logging
import random
from pathlib import Path
import threading
from contextlib import contextmanager
import time
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import concurrent.futures

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class DetailedOrderDB(Base):
    """Enhanced order table with more granular data"""
    __tablename__ = 'detailed_orders'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    microsecond = Column(Integer, default=0)  # For high-frequency precision
    agent_id = Column(String(50), nullable=False, index=True)
    agent_type = Column(String(20), nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    side = Column(String(4), nullable=False, index=True)
    order_type = Column(String(10), nullable=False)
    price = Column(Float, nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    display_quantity = Column(Integer, nullable=False)  # Visible quantity
    hidden_quantity = Column(Integer, default=0)  # Iceberg orders
    filled_quantity = Column(Integer, default=0)
    remaining_quantity = Column(Integer, nullable=False)
    status = Column(String(15), default='PENDING')
    time_in_force = Column(String(5), default='DAY')
    avg_fill_price = Column(Float, default=0.0)
    market_price_at_time = Column(Float, nullable=False)
    
    # Advanced order attributes
    order_priority = Column(Integer, default=0)  # Price-time priority
    is_aggressive = Column(Boolean, default=False)  # Crosses spread
    parent_order_id = Column(String(50))  # For iceberg/hidden orders
    execution_algo = Column(String(20))  # TWAP, VWAP, etc.
    
    # Market impact and timing
    submission_delay_ms = Column(Integer, default=0)  # Network latency
    cancel_time = Column(DateTime)
    modification_count = Column(Integer, default=0)

class DetailedTradeDB(Base):
    """Enhanced trade table with market impact details"""
    __tablename__ = 'detailed_trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    microsecond = Column(Integer, default=0)
    symbol = Column(String(10), nullable=False, index=True)
    price = Column(Float, nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    buy_order_id = Column(String(50), nullable=False, index=True)
    sell_order_id = Column(String(50), nullable=False, index=True)
    buy_agent_id = Column(String(50), nullable=False)
    sell_agent_id = Column(String(50), nullable=False)
    aggressor_side = Column(String(4), nullable=False)
    
    # Market impact metrics
    market_impact_bps = Column(Float, default=0.0)
    permanent_impact_bps = Column(Float, default=0.0)
    temporary_impact_bps = Column(Float, default=0.0)
    
    # Pre and post trade book state
    pre_trade_mid = Column(Float)
    post_trade_mid = Column(Float)
    pre_trade_spread = Column(Float)
    post_trade_spread = Column(Float)
    
    # Volume and liquidity metrics
    total_volume_at_price = Column(Integer, default=0)
    remaining_volume_at_price = Column(Integer, default=0)
    depth_consumed = Column(Integer, default=0)
    
    # Timing information
    matching_latency_microsec = Column(Integer, default=0)
    trade_sequence_number = Column(Integer, default=0)

class LOBSnapshotDB(Base):
    """Detailed order book snapshots with full depth"""
    __tablename__ = 'lob_snapshots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    microsecond = Column(Integer, default=0)
    symbol = Column(String(10), nullable=False, index=True)
    
    # Level 1 data (BBO)
    best_bid = Column(Float, index=True)
    best_ask = Column(Float, index=True)
    best_bid_size = Column(Integer)
    best_ask_size = Column(Integer)
    
    # Spread metrics
    absolute_spread = Column(Float)
    relative_spread_bps = Column(Float)
    effective_spread_bps = Column(Float)
    quoted_spread_bps = Column(Float)
    
    # Mid price and references
    mid_price = Column(Float, index=True)
    weighted_mid_price = Column(Float)  # Size-weighted
    microprice = Column(Float)  # High-frequency reference
    
    # Depth metrics (Level 2+ aggregated)
    total_bid_volume = Column(Integer)
    total_ask_volume = Column(Integer)
    bid_volume_5 = Column(Integer)  # Within 5 ticks
    ask_volume_5 = Column(Integer)
    bid_volume_10 = Column(Integer)  # Within 10 ticks
    ask_volume_10 = Column(Integer)
    
    # Order book imbalance
    volume_imbalance = Column(Float)  # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    depth_imbalance = Column(Float)   # Imbalance in top 5 levels
    order_count_imbalance = Column(Float)
    
    # Volatility and activity measures
    price_volatility_1min = Column(Float)
    volume_rate_1min = Column(Float)
    trade_count_1min = Column(Integer)
    order_arrival_rate_1min = Column(Float)
    
    # Full depth data (JSON for flexibility)
    bid_depth_json = Column(Text)  # {price: {size: X, count: Y, hidden: Z}}
    ask_depth_json = Column(Text)
    recent_trades_json = Column(Text)  # Last 10 trades for impact calculation

class MarketMicrostructureDB(Base):
    """Market microstructure statistics and patterns"""
    __tablename__ = 'market_microstructure'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    
    # Statistical measures (rolling windows)
    realized_volatility_5min = Column(Float)
    realized_volatility_15min = Column(Float)
    realized_volatility_1hour = Column(Float)
    
    # Autocorrelation measures
    return_autocorr_1tick = Column(Float)
    return_autocorr_5tick = Column(Float)
    return_autocorr_1min = Column(Float)
    
    # Volume patterns
    volume_autocorr_1min = Column(Float)
    volume_autocorr_5min = Column(Float)
    intraday_volume_pattern = Column(Float)  # Relative to daily average
    
    # Liquidity measures
    effective_tick_size = Column(Float)
    price_clustering = Column(Float)  # Measure of clustering at round prices
    depth_resilience_time_sec = Column(Float)  # Time to replenish after trade
    
    # Stylized facts compliance
    has_volatility_clustering = Column(Boolean)
    has_fat_tails = Column(Boolean)
    has_mean_reversion = Column(Boolean)
    has_leverage_effect = Column(Boolean)
    
    # Market efficiency measures
    variance_ratio_5 = Column(Float)  # 5-period variance ratio test
    variance_ratio_10 = Column(Float)
    hurst_exponent = Column(Float)
    
    # News and event impact
    unusual_activity_score = Column(Float)
    price_discovery_contribution = Column(Float)

@dataclass 
class ScaledLOBConfig:
    """Configuration for scaled LOB data generation"""
    
    # Scale parameters
    scale_factor: int = 1000  # Multiplier for order generation
    simulation_days: int = 10
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])
    
    # Agent distribution (total agents = num_base_agents * scale_factor)
    num_base_agents: int = 100
    agent_distribution: Dict[str, float] = field(default_factory=lambda: {
        "retail": 0.60,
        "institutional": 0.20,
        "hft": 0.15,
        "market_maker": 0.05
    })
    
    # Order generation parameters
    base_orders_per_second: int = 10  # Base rate per symbol
    peak_order_multiplier: float = 5.0  # Peak times multiplier
    min_order_multiplier: float = 0.2   # Quiet times multiplier
    
    # Market microstructure parameters
    tick_size: float = 0.01
    lot_size: int = 100
    max_depth_levels: int = 20  # Order book depth to track
    snapshot_frequency_ms: int = 100  # Snapshot every 100ms
    
    # Advanced features
    enable_iceberg_orders: bool = True
    enable_hidden_orders: bool = True
    enable_smart_order_routing: bool = True
    enable_latency_simulation: bool = True
    
    # Database configuration
    db_path: str = "scaled_lob_data.db"
    use_in_memory: bool = False
    batch_size: int = 10000
    enable_compression: bool = True
    parallel_workers: int = 4
    
    # Market patterns
    enable_momentum_bursts: bool = True
    enable_mean_reversion: bool = True
    enable_volatility_clustering: bool = True
    enable_intraday_patterns: bool = True
    
    # Performance optimization
    use_bulk_inserts: bool = True
    create_indexes: bool = True
    vacuum_frequency: int = 100000  # Records between VACUUM

class AdvancedAgent:
    """Advanced trading agent with sophisticated behavior patterns"""
    
    def __init__(self, agent_id: str, agent_type: str, symbols: List[str], config: ScaledLOBConfig):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.symbols = symbols
        self.config = config
        
        # Agent state
        self.cash = self._get_initial_cash()
        self.positions = {symbol: 0 for symbol in symbols}
        self.active_orders = {}
        self.order_history = []
        
        # Behavioral parameters
        self.setup_agent_profile()
        
        # Strategy state
        self.momentum_signals = {symbol: 0.0 for symbol in symbols}
        self.mean_reversion_signals = {symbol: 0.0 for symbol in symbols}
        self.inventory_target = {symbol: 0 for symbol in symbols}
        
        # Risk management
        self.max_position_size = self._calculate_max_position()
        self.daily_loss_limit = self.cash * 0.05  # 5% daily loss limit
        self.current_pnl = 0.0
        
        # Timing and latency
        self.last_order_time = {}
        self.processing_latency_ms = self._get_processing_latency()
        
    def _get_initial_cash(self) -> float:
        """Get initial cash based on agent type"""
        cash_by_type = {
            "retail": random.uniform(10_000, 100_000),
            "institutional": random.uniform(10_000_000, 100_000_000),
            "hft": random.uniform(1_000_000, 10_000_000),
            "market_maker": random.uniform(5_000_000, 50_000_000)
        }
        return cash_by_type.get(self.agent_type, 1_000_000)
    
    def setup_agent_profile(self):
        """Setup detailed agent-specific characteristics"""
        profiles = {
            "retail": {
                "order_size_range": (10, 2000),
                "order_frequency_per_hour": random.uniform(0.5, 5.0),
                "market_order_prob": 0.35,
                "cancel_prob": 0.20,
                "modify_prob": 0.15,
                "iceberg_prob": 0.05,
                "hidden_prob": 0.02,
                "price_improvement_ticks": 2,
                "latency_ms_range": (50, 500),
                "patience_seconds": random.uniform(60, 3600),
                "momentum_factor": random.uniform(0.1, 0.4),
                "mean_reversion_factor": random.uniform(0.2, 0.6),
                "news_sensitivity": random.uniform(0.3, 0.8)
            },
            
            "institutional": {
                "order_size_range": (5000, 100000),
                "order_frequency_per_hour": random.uniform(2.0, 20.0),
                "market_order_prob": 0.20,
                "cancel_prob": 0.30,
                "modify_prob": 0.25,
                "iceberg_prob": 0.40,
                "hidden_prob": 0.15,
                "price_improvement_ticks": 5,
                "latency_ms_range": (10, 100),
                "patience_seconds": random.uniform(300, 7200),
                "momentum_factor": random.uniform(0.3, 0.7),
                "mean_reversion_factor": random.uniform(0.4, 0.8),
                "news_sensitivity": random.uniform(0.7, 0.95)
            },
            
            "hft": {
                "order_size_range": (100, 5000),
                "order_frequency_per_hour": random.uniform(100.0, 10000.0),
                "market_order_prob": 0.15,
                "cancel_prob": 0.85,
                "modify_prob": 0.70,
                "iceberg_prob": 0.10,
                "hidden_prob": 0.05,
                "price_improvement_ticks": 1,
                "latency_ms_range": (0.1, 5.0),
                "patience_seconds": random.uniform(0.1, 30),
                "momentum_factor": random.uniform(0.8, 0.95),
                "mean_reversion_factor": random.uniform(0.9, 0.99),
                "news_sensitivity": random.uniform(0.2, 0.5)
            },
            
            "market_maker": {
                "order_size_range": (500, 20000),
                "order_frequency_per_hour": random.uniform(50.0, 500.0),
                "market_order_prob": 0.05,
                "cancel_prob": 0.60,
                "modify_prob": 0.80,
                "iceberg_prob": 0.30,
                "hidden_prob": 0.20,
                "price_improvement_ticks": 3,
                "latency_ms_range": (1.0, 20.0),
                "patience_seconds": random.uniform(5, 300),
                "momentum_factor": random.uniform(0.1, 0.3),
                "mean_reversion_factor": random.uniform(0.7, 0.9),
                "news_sensitivity": random.uniform(0.4, 0.7)
            }
        }
        
        profile = profiles.get(self.agent_type, profiles["retail"])
        for key, value in profile.items():
            setattr(self, key, value)
    
    def _calculate_max_position(self) -> Dict[str, int]:
        """Calculate maximum position size per symbol"""
        base_position = {
            "retail": 10000,
            "institutional": 1000000,
            "hft": 50000,
            "market_maker": 200000
        }
        
        max_pos = base_position.get(self.agent_type, 10000)
        return {symbol: max_pos for symbol in self.symbols}
    
    def _get_processing_latency(self) -> float:
        """Get agent's processing latency in milliseconds"""
        return random.uniform(*self.latency_ms_range)
    
    def should_trade(self, current_time: datetime, symbol: str, market_data: Dict) -> bool:
        """Determine if agent should place an order"""
        
        # Check if enough time has passed since last order
        if symbol in self.last_order_time:
            time_since_last = (current_time - self.last_order_time[symbol]).total_seconds()
            min_interval = 3600.0 / self.order_frequency_per_hour
            if time_since_last < min_interval:
                return False
        
        # Risk checks
        if abs(self.current_pnl) > self.daily_loss_limit:
            return False
        
        if abs(self.positions[symbol]) >= self.max_position_size[symbol]:
            return False
        
        # Market condition checks
        spread_bps = market_data.get('spread_bps', 10)
        if spread_bps > 50:  # Too wide spread
            return False
        
        volatility = market_data.get('volatility', 0.02)
        
        # Base probability adjusted by market conditions
        base_prob = self.order_frequency_per_hour / 3600.0
        
        # Volatility adjustment
        if self.agent_type == "hft":
            base_prob *= (1 + volatility * 10)  # HFT likes volatility
        elif self.agent_type == "retail":
            base_prob *= (1 - volatility * 2)   # Retail avoids volatility
        
        # Time of day adjustment
        hour = current_time.hour
        if hour in [9, 10, 15, 16]:  # Market open/close
            base_prob *= 2.0
        elif hour in [11, 12, 13, 14]:  # Midday
            base_prob *= 0.5
        
        return random.random() < base_prob
    
    def generate_order(self, symbol: str, current_time: datetime, market_data: Dict) -> Optional[Dict]:
        """Generate a sophisticated trading order"""
        
        # Update strategy signals
        self._update_strategy_signals(symbol, market_data)
        
        # Determine order side based on signals and inventory
        side = self._determine_order_side(symbol, market_data)
        if not side:
            return None
        
        # Determine order characteristics
        order_type, price = self._determine_order_type_and_price(symbol, side, market_data)
        quantity = self._determine_order_size(symbol, side, market_data)
        
        # Advanced order features
        display_qty, hidden_qty = self._determine_display_strategy(quantity)
        execution_algo = self._choose_execution_algorithm(quantity, market_data)
        
        # Generate unique order ID with timestamp precision
        microsecond = current_time.microsecond + random.randint(0, 999)
        order_id = f"ORD_{self.agent_type}_{self.agent_id}_{current_time.strftime('%Y%m%d_%H%M%S')}_{microsecond:06d}"
        
        # Update last order time
        self.last_order_time[symbol] = current_time
        
        return {
            "order_id": order_id,
            "timestamp": current_time,
            "microsecond": microsecond,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "price": round(price, 2),
            "quantity": quantity,
            "display_quantity": display_qty,
            "hidden_quantity": hidden_qty,
            "remaining_quantity": quantity,
            "status": "PENDING",
            "time_in_force": self._determine_time_in_force(),
            "market_price_at_time": market_data.get('mid_price', price),
            "is_aggressive": self._is_aggressive_order(side, price, market_data),
            "execution_algo": execution_algo,
            "submission_delay_ms": int(self.processing_latency_ms)
        }
    
    def _update_strategy_signals(self, symbol: str, market_data: Dict):
        """Update trading strategy signals"""
        
        # Momentum signal
        price_change = market_data.get('price_change_1min', 0.0)
        self.momentum_signals[symbol] = price_change * self.momentum_factor
        
        # Mean reversion signal
        current_price = market_data.get('mid_price', 100.0)
        vwap = market_data.get('vwap_5min', current_price)
        deviation = (current_price - vwap) / vwap
        self.mean_reversion_signals[symbol] = -deviation * self.mean_reversion_factor
    
    def _determine_order_side(self, symbol: str, market_data: Dict) -> Optional[str]:
        """Determine order side based on strategy and inventory"""
        
        # Combine signals
        total_signal = (self.momentum_signals[symbol] + 
                       self.mean_reversion_signals[symbol])
        
        # Inventory adjustment
        current_position = self.positions[symbol]
        inventory_signal = -current_position / self.max_position_size[symbol] * 0.5
        
        total_signal += inventory_signal
        
        # Add noise
        noise = random.gauss(0, 0.1)
        total_signal += noise
        
        # Decision threshold
        if total_signal > 0.1:
            return "BUY"
        elif total_signal < -0.1:
            return "SELL"
        else:
            return None
    
    def _determine_order_type_and_price(self, symbol: str, side: str, market_data: Dict) -> Tuple[str, float]:
        """Determine order type and price with advanced logic"""
        
        mid_price = market_data.get('mid_price', 100.0)
        spread = market_data.get('spread', 0.1)
        
        if random.random() < self.market_order_prob:
            return "MARKET", mid_price
        
        # Limit order pricing strategy
        if side == "BUY":
            # Place below mid, closer for more aggressive
            aggression = random.uniform(0.1, 0.9)
            price_offset = spread * 0.5 * (1 - aggression)
            price = mid_price - price_offset
        else:
            # Place above mid
            aggression = random.uniform(0.1, 0.9)
            price_offset = spread * 0.5 * (1 - aggression)
            price = mid_price + price_offset
        
        # Round to tick size
        tick_size = self.config.tick_size
        price = round(price / tick_size) * tick_size
        
        return "LIMIT", max(0.01, price)
    
    def _determine_order_size(self, symbol: str, side: str, market_data: Dict) -> int:
        """Determine order size with risk management"""
        
        # Base size from agent profile
        min_size, max_size = self.order_size_range
        base_size = random.randint(min_size, max_size)
        
        # Adjust for volatility
        volatility = market_data.get('volatility', 0.02)
        vol_adjustment = 1.0 - min(volatility * 10, 0.5)  # Reduce size in high vol
        
        # Adjust for position limits
        current_position = self.positions[symbol]
        remaining_capacity = self.max_position_size[symbol] - abs(current_position)
        
        # Final size
        adjusted_size = int(base_size * vol_adjustment)
        final_size = min(adjusted_size, remaining_capacity)
        
        return max(self.config.lot_size, final_size)
    
    def _determine_display_strategy(self, quantity: int) -> Tuple[int, int]:
        """Determine how much quantity to display vs hide"""
        
        if random.random() < self.iceberg_prob and quantity > 1000:
            # Iceberg order
            display_ratio = random.uniform(0.1, 0.3)
            display_qty = max(100, int(quantity * display_ratio))
            hidden_qty = quantity - display_qty
            return display_qty, hidden_qty
        
        elif random.random() < self.hidden_prob:
            # Fully hidden order
            return 0, quantity
        
        else:
            # Fully displayed
            return quantity, 0
    
    def _choose_execution_algorithm(self, quantity: int, market_data: Dict) -> str:
        """Choose execution algorithm based on order size and market conditions"""
        
        if self.agent_type == "institutional" and quantity > 10000:
            algos = ["TWAP", "VWAP", "Implementation_Shortfall", "POV"]
            return random.choice(algos)
        elif self.agent_type == "hft":
            return "SOR"  # Smart Order Routing
        else:
            return "Standard"
    
    def _determine_time_in_force(self) -> str:
        """Determine time in force based on agent type"""
        
        if self.agent_type == "hft":
            return random.choice(["IOC", "FOK", "DAY"])
        else:
            return "DAY"
    
    def _is_aggressive_order(self, side: str, price: float, market_data: Dict) -> bool:
        """Check if order crosses the spread (aggressive)"""
        
        best_bid = market_data.get('best_bid', 0)
        best_ask = market_data.get('best_ask', 999999)
        
        if side == "BUY" and price >= best_ask:
            return True
        elif side == "SELL" and price <= best_bid:
            return True
        
        return False

class ScaledLOBGenerator:
    """Main class for generating scaled LOB data with detailed tracking"""
    
    def __init__(self, config: ScaledLOBConfig):
        self.config = config
        
        # Initialize state variables first
        self.order_books = {}
        self.trade_sequence = 0
        
        # Performance tracking
        self.stats = {
            "orders_generated": 0,
            "trades_executed": 0,
            "snapshots_taken": 0,
            "start_time": None,
            "end_time": None,
            "peak_orders_per_second": 0
        }
        
        # Setup components
        self.setup_database()
        self.setup_agents()
        self.setup_market_state()
        
        logger.info(f"🚀 Scaled LOB Generator initialized")
        logger.info(f"Scale factor: {config.scale_factor}x")
        logger.info(f"Total agents: {len(self.agents):,}")
        logger.info(f"Estimated orders per day: {self._estimate_daily_orders():,}")
    
    def setup_database(self):
        """Initialize enhanced database with indexes"""
        
        if self.config.use_in_memory:
            db_url = "sqlite:///:memory:"
        else:
            db_url = f"sqlite:///{self.config.db_path}"
        
        self.engine = create_engine(
            db_url,
            poolclass=StaticPool,
            connect_args={'check_same_thread': False} if 'sqlite' in db_url else {},
            echo=False
        )
        
        # Create all tables
        Base.metadata.create_all(self.engine)
        
        # Create additional indexes for performance
        if self.config.create_indexes:
            self._create_performance_indexes()
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        logger.info(f"Database initialized: {db_url}")
    
    def _create_performance_indexes(self):
        """Create additional database indexes for query performance"""
        
        from sqlalchemy import text
        
        with self.engine.connect() as conn:
            # Composite indexes for common queries
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_orders_symbol_timestamp ON detailed_orders(symbol, timestamp)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_orders_agent_timestamp ON detailed_orders(agent_id, timestamp)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON detailed_trades(symbol, timestamp)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_timestamp ON lob_snapshots(symbol, timestamp)"))
            
            # Price-based indexes
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_orders_symbol_price ON detailed_orders(symbol, price)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_symbol_price ON detailed_trades(symbol, price)"))
            
            conn.commit()
    
    def setup_agents(self):
        """Initialize trading agents at scale"""
        
        self.agents = []
        total_agents = self.config.num_base_agents * self.config.scale_factor
        
        agent_counts = {}
        for agent_type, proportion in self.config.agent_distribution.items():
            count = int(total_agents * proportion)
            agent_counts[agent_type] = count
        
        agent_id = 0
        for agent_type, count in agent_counts.items():
            for i in range(count):
                agent = AdvancedAgent(
                    agent_id=f"{agent_type}_{i:06d}",
                    agent_type=agent_type,
                    symbols=self.config.symbols,
                    config=self.config
                )
                self.agents.append(agent)
                agent_id += 1
        
        logger.info(f"Initialized {len(self.agents):,} agents: {agent_counts}")
    
    def setup_market_state(self):
        """Initialize market state for all symbols"""
        
        self.market_state = {}
        
        for symbol in self.config.symbols:
            # Initialize with realistic prices
            base_prices = {
                "AAPL": 150.0, "GOOGL": 2500.0, "MSFT": 300.0, 
                "TSLA": 800.0, "AMZN": 3000.0
            }
            
            initial_price = base_prices.get(symbol, 100.0)
            
            self.market_state[symbol] = {
                "mid_price": initial_price,
                "best_bid": initial_price - 0.05,
                "best_ask": initial_price + 0.05,
                "spread": 0.10,
                "spread_bps": 10.0,
                "volatility": 0.02,
                "volume_1min": 0,
                "trade_count_1min": 0,
                "vwap_5min": initial_price,
                "price_change_1min": 0.0,
                "last_trade_price": initial_price,
                "last_update": datetime.now()
            }
            
            # Initialize order book
            self.order_books[symbol] = {
                "bids": defaultdict(list),  # price -> list of orders
                "asks": defaultdict(list),
                "bid_prices": [],
                "ask_prices": []
            }
    
    def _estimate_daily_orders(self) -> int:
        """Estimate orders per day for planning"""
        
        orders_per_second = 0
        for agent in self.agents:
            orders_per_second += agent.order_frequency_per_hour / 3600.0
        
        # Multiply by number of symbols and trading hours
        daily_orders = orders_per_second * len(self.config.symbols) * 8 * 3600
        return int(daily_orders)
    
    @contextmanager
    def get_db_session(self):
        """Get database session with proper cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def generate_scaled_data(self) -> Dict[str, Any]:
        """Main method to generate massive LOB data"""
        
        logger.info("🎯 Starting Scaled LOB Data Generation")
        logger.info(f"Target: {self.config.simulation_days} days, {len(self.config.symbols)} symbols")
        
        self.stats["start_time"] = datetime.now()
        
        # Simulate each day
        for day in range(self.config.simulation_days):
            logger.info(f"📅 Simulating day {day + 1}/{self.config.simulation_days}")
            self._simulate_trading_day(day)
            
            # Periodic maintenance
            if day % 5 == 0:
                self._perform_database_maintenance()
        
        self.stats["end_time"] = datetime.now()
        
        # Generate final summary
        summary = self._generate_comprehensive_summary()
        logger.info("✅ Scaled LOB generation completed!")
        
        return summary
    
    def _simulate_trading_day(self, day_number: int):
        """Simulate one complete trading day with high granularity"""
        
        base_time = datetime(2024, 1, 2) + timedelta(days=day_number)
        trading_start = base_time.replace(hour=9, minute=30, second=0, microsecond=0)
        trading_end = base_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        current_time = trading_start
        
        # Track day performance
        orders_today = 0
        trades_today = 0
        snapshots_today = 0
        
        # Simulate with high frequency (every 100ms)
        time_increment = timedelta(milliseconds=self.config.snapshot_frequency_ms)
        
        while current_time < trading_end:
            
            # Update market conditions
            self._update_market_conditions(current_time)
            
            # Generate orders from agents
            minute_orders = self._generate_orders_for_timestep(current_time)
            
            if minute_orders:
                # Process orders and execute trades
                new_trades = self._process_orders_and_match(minute_orders, current_time)
                trades_today += len(new_trades)
                orders_today += len(minute_orders)
                
                # Update order book state
                self._update_order_books(minute_orders, new_trades)
            
            # Take detailed snapshots
            if current_time.microsecond % (self.config.snapshot_frequency_ms * 1000) == 0:
                self._take_detailed_snapshots(current_time)
                snapshots_today += len(self.config.symbols)
            
            # Move to next timestep
            current_time += time_increment
            
            # Progress reporting
            if current_time.minute % 30 == 0 and current_time.second == 0:
                elapsed = current_time - trading_start
                logger.info(f"  Progress: {elapsed} elapsed, {orders_today:,} orders, {trades_today:,} trades")
        
        # End of day statistics
        logger.info(f"  Day {day_number + 1} completed: {orders_today:,} orders, {trades_today:,} trades, {snapshots_today:,} snapshots")
        
        self.stats["orders_generated"] += orders_today
        self.stats["trades_executed"] += trades_today
        self.stats["snapshots_taken"] += snapshots_today
    
    def _update_market_conditions(self, current_time: datetime):
        """Update market conditions with realistic patterns"""
        
        for symbol in self.config.symbols:
            state = self.market_state[symbol]
            
            # Time of day effect
            hour = current_time.hour
            minute = current_time.minute
            
            # Intraday volatility pattern (U-shaped)
            if hour == 9 or (hour == 15 and minute >= 30):
                vol_multiplier = 2.0  # High volatility at open/close
            elif 11 <= hour <= 14:
                vol_multiplier = 0.6  # Low volatility midday
            else:
                vol_multiplier = 1.0
            
            # Random price movement
            base_volatility = 0.02  # 2% annual volatility
            dt = self.config.snapshot_frequency_ms / (1000 * 3600 * 24 * 252)  # Time fraction
            
            price_change = random.gauss(0, base_volatility * vol_multiplier * np.sqrt(dt))
            
            # Apply price change
            old_price = state["mid_price"]
            new_price = max(0.01, old_price * (1 + price_change))
            
            state["mid_price"] = new_price
            state["price_change_1min"] = (new_price - old_price) / old_price
            state["volatility"] = base_volatility * vol_multiplier
            
            # Update spread (proportional to volatility and inverse to liquidity)
            base_spread = 0.05  # 5 cents base spread
            spread = base_spread * (1 + state["volatility"] * 10)
            
            state["spread"] = spread
            state["spread_bps"] = (spread / new_price) * 10000
            state["best_bid"] = new_price - spread / 2
            state["best_ask"] = new_price + spread / 2
            
            state["last_update"] = current_time
    
    def _generate_orders_for_timestep(self, current_time: datetime) -> List[Dict]:
        """Generate orders for current timestep"""
        
        orders = []
        
        # Calculate order intensity based on time of day
        base_intensity = self.config.base_orders_per_second * self.config.scale_factor
        
        # Time-based multiplier
        hour = current_time.hour
        if hour in [9, 15, 16]:
            intensity = base_intensity * self.config.peak_order_multiplier
        elif hour in [11, 12, 13, 14]:
            intensity = base_intensity * self.config.min_order_multiplier
        else:
            intensity = base_intensity
        
        # Generate orders from subset of agents (not all agents trade every timestep)
        timestep_probability = intensity / len(self.agents) / 10  # Probability per agent per timestep
        
        for agent in self.agents:
            for symbol in self.config.symbols:
                if random.random() < timestep_probability:
                    if agent.should_trade(current_time, symbol, self.market_state[symbol]):
                        order = agent.generate_order(symbol, current_time, self.market_state[symbol])
                        if order:
                            orders.append(order)
        
        return orders
    
    def _process_orders_and_match(self, orders: List[Dict], current_time: datetime) -> List[Dict]:
        """Process orders and execute trades with detailed matching"""
        
        trades = []
        
        # Store orders in database
        with self.get_db_session() as session:
            for order in orders:
                db_order = DetailedOrderDB(**order)
                session.add(db_order)
            
            # Simple matching logic (enhanced version)
            trades = self._match_orders_advanced(orders, current_time, session)
            
            # Store trades
            for trade in trades:
                db_trade = DetailedTradeDB(**trade)
                session.add(db_trade)
        
        return trades
    
    def _match_orders_advanced(self, orders: List[Dict], current_time: datetime, session: Session) -> List[Dict]:
        """Advanced order matching with detailed market impact tracking"""
        
        trades = []
        
        # Group orders by symbol
        symbol_orders = defaultdict(lambda: {"buys": [], "sells": []})
        for order in orders:
            symbol = order["symbol"]
            if order["side"] == "BUY":
                symbol_orders[symbol]["buys"].append(order)
            else:
                symbol_orders[symbol]["sells"].append(order)
        
        # Process each symbol
        for symbol, order_dict in symbol_orders.items():
            buys = sorted(order_dict["buys"], key=lambda x: (-x["price"], x["timestamp"]))  # Price priority
            sells = sorted(order_dict["sells"], key=lambda x: (x["price"], x["timestamp"]))
            
            # Match orders
            symbol_trades = self._execute_matches(symbol, buys, sells, current_time)
            trades.extend(symbol_trades)
        
        return trades
    
    def _execute_matches(self, symbol: str, buy_orders: List[Dict], 
                        sell_orders: List[Dict], current_time: datetime) -> List[Dict]:
        """Execute order matches for a symbol"""
        
        trades = []
        market_state = self.market_state[symbol]
        
        # Simple crossing logic
        for buy_order in buy_orders:
            for sell_order in sell_orders:
                if self._can_match(buy_order, sell_order):
                    trade = self._create_trade(buy_order, sell_order, current_time, market_state)
                    if trade:
                        trades.append(trade)
                        
                        # Update market state
                        market_state["last_trade_price"] = trade["price"]
                        market_state["volume_1min"] += trade["quantity"]
                        market_state["trade_count_1min"] += 1
                        
                        # For simplicity, only match one trade per buy order
                        break
        
        return trades
    
    def _can_match(self, buy_order: Dict, sell_order: Dict) -> bool:
        """Check if orders can be matched"""
        
        if buy_order["symbol"] != sell_order["symbol"]:
            return False
        
        # Market orders always match
        if buy_order["order_type"] == "MARKET" or sell_order["order_type"] == "MARKET":
            return True
        
        # Limit orders match if buy price >= sell price
        return buy_order["price"] >= sell_order["price"]
    
    def _create_trade(self, buy_order: Dict, sell_order: Dict, 
                     current_time: datetime, market_state: Dict) -> Dict:
        """Create detailed trade record"""
        
        # Determine trade price
        if buy_order["order_type"] == "MARKET":
            price = sell_order["price"]
        elif sell_order["order_type"] == "MARKET":
            price = buy_order["price"]
        else:
            # For limit orders, use the resting order's price (price-time priority)
            if buy_order["timestamp"] < sell_order["timestamp"]:
                price = buy_order["price"]
            else:
                price = sell_order["price"]
        
        # Determine quantity
        quantity = min(buy_order["remaining_quantity"], sell_order["remaining_quantity"])
        
        # Calculate market impact
        pre_mid = market_state["mid_price"]
        post_mid = price  # Simplified
        impact_bps = abs(price - pre_mid) / pre_mid * 10000
        
        # Generate trade ID
        self.trade_sequence += 1
        trade_id = f"TRD_{current_time.strftime('%Y%m%d_%H%M%S')}_{self.trade_sequence:08d}"
        
        return {
            "trade_id": trade_id,
            "timestamp": current_time,
            "microsecond": current_time.microsecond,
            "symbol": buy_order["symbol"],
            "price": round(price, 2),
            "quantity": quantity,
            "buy_order_id": buy_order["order_id"],
            "sell_order_id": sell_order["order_id"],
            "buy_agent_id": buy_order["agent_id"],
            "sell_agent_id": sell_order["agent_id"],
            "aggressor_side": "BUY" if buy_order["order_type"] == "MARKET" else "SELL",
            "market_impact_bps": impact_bps,
            "permanent_impact_bps": impact_bps * 0.3,  # Simplified
            "temporary_impact_bps": impact_bps * 0.7,
            "pre_trade_mid": pre_mid,
            "post_trade_mid": post_mid,
            "pre_trade_spread": market_state["spread"],
            "post_trade_spread": market_state["spread"],  # Simplified
            "matching_latency_microsec": random.randint(10, 1000),
            "trade_sequence_number": self.trade_sequence
        }
    
    def _update_order_books(self, orders: List[Dict], trades: List[Dict]):
        """Update order book state tracking"""
        # This would maintain the live order book state
        # For now, simplified implementation
        pass
    
    def _take_detailed_snapshots(self, current_time: datetime):
        """Take detailed order book snapshots"""
        
        snapshots = []
        
        for symbol in self.config.symbols:
            snapshot = self._create_detailed_snapshot(symbol, current_time)
            snapshots.append(snapshot)
        
        # Store snapshots in database
        if snapshots:
            with self.get_db_session() as session:
                for snapshot in snapshots:
                    db_snapshot = LOBSnapshotDB(**snapshot)
                    session.add(db_snapshot)
    
    def _create_detailed_snapshot(self, symbol: str, current_time: datetime) -> Dict:
        """Create detailed order book snapshot"""
        
        market_state = self.market_state[symbol]
        
        # Generate realistic order book depth
        mid_price = market_state["mid_price"]
        spread = market_state["spread"]
        
        # Create bid/ask depth (simplified realistic generation)
        bid_depth = {}
        ask_depth = {}
        
        total_bid_vol = 0
        total_ask_vol = 0
        
        for level in range(self.config.max_depth_levels):
            # Bid side (decreasing prices)
            bid_price = mid_price - spread/2 - level * self.config.tick_size
            bid_size = max(100, int(np.random.exponential(1000)))  # Exponential decay
            bid_depth[str(round(bid_price, 2))] = {
                "size": bid_size,
                "count": random.randint(1, 10),
                "hidden": random.randint(0, bid_size // 10)
            }
            total_bid_vol += bid_size
            
            # Ask side (increasing prices)
            ask_price = mid_price + spread/2 + level * self.config.tick_size
            ask_size = max(100, int(np.random.exponential(1000)))
            ask_depth[str(round(ask_price, 2))] = {
                "size": ask_size,
                "count": random.randint(1, 10),
                "hidden": random.randint(0, ask_size // 10)
            }
            total_ask_vol += ask_size
        
        # Calculate metrics
        best_bid = mid_price - spread/2
        best_ask = mid_price + spread/2
        volume_imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
        
        return {
            "timestamp": current_time,
            "microsecond": current_time.microsecond,
            "symbol": symbol,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "best_bid_size": bid_depth[str(round(best_bid, 2))]["size"],
            "best_ask_size": ask_depth[str(round(best_ask, 2))]["size"],
            "absolute_spread": spread,
            "relative_spread_bps": (spread / mid_price) * 10000,
            "effective_spread_bps": (spread / mid_price) * 10000,  # Simplified
            "quoted_spread_bps": (spread / mid_price) * 10000,
            "mid_price": mid_price,
            "weighted_mid_price": mid_price,  # Simplified
            "microprice": mid_price,  # Simplified
            "total_bid_volume": total_bid_vol,
            "total_ask_volume": total_ask_vol,
            "bid_volume_5": sum(d["size"] for d in list(bid_depth.values())[:5]),
            "ask_volume_5": sum(d["size"] for d in list(ask_depth.values())[:5]),
            "bid_volume_10": sum(d["size"] for d in list(bid_depth.values())[:10]),
            "ask_volume_10": sum(d["size"] for d in list(ask_depth.values())[:10]),
            "volume_imbalance": volume_imbalance,
            "depth_imbalance": volume_imbalance,  # Simplified
            "order_count_imbalance": 0.0,  # Simplified
            "price_volatility_1min": market_state["volatility"],
            "volume_rate_1min": market_state["volume_1min"],
            "trade_count_1min": market_state["trade_count_1min"],
            "order_arrival_rate_1min": 10.0,  # Simplified
            "bid_depth_json": json.dumps(bid_depth),
            "ask_depth_json": json.dumps(ask_depth),
            "recent_trades_json": json.dumps([])  # Simplified
        }
    
    def _perform_database_maintenance(self):
        """Perform database optimization"""
        
        from sqlalchemy import text
        
        if self.stats["orders_generated"] % self.config.vacuum_frequency == 0:
            logger.info("🔧 Performing database maintenance...")
            with self.engine.connect() as conn:
                conn.execute(text("VACUUM"))
                conn.execute(text("ANALYZE"))
    
    def _generate_comprehensive_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of the simulation"""
        
        duration = self.stats["end_time"] - self.stats["start_time"]
        
        # Get database statistics
        with self.get_db_session() as session:
            total_orders = session.query(DetailedOrderDB).count()
            total_trades = session.query(DetailedTradeDB).count()
            total_snapshots = session.query(LOBSnapshotDB).count()
        
        # Calculate performance metrics
        orders_per_second = total_orders / duration.total_seconds() if duration.total_seconds() > 0 else 0
        fill_rate = total_trades / total_orders if total_orders > 0 else 0
        
        return {
            "simulation_duration": str(duration),
            "total_orders": total_orders,
            "total_trades": total_trades,
            "total_snapshots": total_snapshots,
            "orders_per_second": orders_per_second,
            "trades_per_second": total_trades / duration.total_seconds() if duration.total_seconds() > 0 else 0,
            "fill_rate": fill_rate,
            "symbols": self.config.symbols,
            "agents_count": len(self.agents),
            "scale_factor": self.config.scale_factor,
            "database_path": self.config.db_path,
            "database_size_mb": self._get_database_size_mb(),
            "peak_orders_per_second": self.stats["peak_orders_per_second"]
        }
    
    def _get_database_size_mb(self) -> float:
        """Get database file size in MB"""
        try:
            db_path = Path(self.config.db_path)
            if db_path.exists():
                return db_path.stat().st_size / (1024 * 1024)
        except:
            pass
        return 0.0

def main():
    """Main function for testing scaled LOB generation"""
    print("🚀 Scaled LOB Data Generator")
    print("=" * 50)
    
    # Large scale configuration
    config = ScaledLOBConfig(
        scale_factor=100,  # 100x scale
        simulation_days=2,
        symbols=["AAPL", "GOOGL", "MSFT"],
        num_base_agents=50,
        base_orders_per_second=20,
        db_path="scaled_lob_data.db",
        batch_size=5000,
        parallel_workers=4
    )
    
    # Generate data
    generator = ScaledLOBGenerator(config)
    summary = generator.generate_scaled_data()
    
    # Print results
    print("\n📊 SCALED LOB GENERATION SUMMARY")
    print("=" * 50)
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()