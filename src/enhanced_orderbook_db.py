#!/usr/bin/env python3
"""
Enhanced Order Book Database System
===================================

A comprehensive order book implementation with database storage for realistic
market simulation and analysis, replacing CSV-based storage with robust DB backend.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import json
import logging
import random
from pathlib import Path
import threading
from contextlib import contextmanager
import pickle
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

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

class OrderDB(Base):
    """Database model for orders"""
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    agent_id = Column(String(50), nullable=False)
    agent_type = Column(String(20), nullable=False)
    symbol = Column(String(10), nullable=False)
    side = Column(String(4), nullable=False)  # BUY/SELL
    order_type = Column(String(10), nullable=False)  # LIMIT/MARKET
    price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    filled_quantity = Column(Integer, default=0)
    remaining_quantity = Column(Integer, nullable=False)
    status = Column(String(15), default='PENDING')  # PENDING/PARTIAL/FILLED/CANCELLED
    time_in_force = Column(String(5), default='DAY')
    avg_fill_price = Column(Float, default=0.0)
    market_price_at_time = Column(Float, nullable=False)

class TradeDB(Base):
    """Database model for trades"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(10), nullable=False)
    price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    buy_order_id = Column(String(50), nullable=False)
    sell_order_id = Column(String(50), nullable=False)
    buy_agent_id = Column(String(50), nullable=False)
    sell_agent_id = Column(String(50), nullable=False)
    aggressor_side = Column(String(4), nullable=False)
    market_impact = Column(Float, default=0.0)

class OrderBookSnapshotDB(Base):
    """Database model for order book snapshots"""
    __tablename__ = 'orderbook_snapshots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(10), nullable=False)
    best_bid = Column(Float)
    best_ask = Column(Float)
    bid_depth_json = Column(Text)  # JSON string of bid depth
    ask_depth_json = Column(Text)  # JSON string of ask depth
    spread = Column(Float)
    mid_price = Column(Float)
    last_trade_price = Column(Float)
    total_volume = Column(Integer, default=0)
    volatility = Column(Float, default=0.0)
    
class MarketStatsDB(Base):
    """Database model for market statistics"""
    __tablename__ = 'market_stats'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    date = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    vwap = Column(Float, nullable=False)
    num_trades = Column(Integer, nullable=False)
    avg_spread = Column(Float, nullable=False)

@dataclass
class EnhancedOrderBookConfig:
    """Configuration for enhanced order book generation"""
    
    # Database settings
    db_path: str = "orderbook.db"
    db_type: str = "sqlite"  # sqlite, postgresql
    use_in_memory: bool = False
    
    # Simulation parameters
    num_agents: int = 5000
    simulation_days: int = 5
    trading_hours_start: int = 9
    trading_hours_end: int = 16
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])
    # New: start date override in UTC (e.g., 2025-08-21T13:30:00Z)
    simulation_start_utc: Optional[str] = None
    
    # Market microstructure parameters
    initial_prices: Dict[str, float] = field(default_factory=lambda: {
        "AAPL": 150.0, "GOOGL": 2500.0, "MSFT": 300.0, "TSLA": 800.0, "AMZN": 3000.0
    })
    volatility_per_symbol: Dict[str, float] = field(default_factory=lambda: {
        "AAPL": 0.02, "GOOGL": 0.025, "MSFT": 0.018, "TSLA": 0.04, "AMZN": 0.022
    })
    
    # Order flow parameters
    base_orders_per_minute: int = 200
    market_order_ratio: float = 0.25
    hidden_order_ratio: float = 0.05
    iceberg_order_ratio: float = 0.02
    
    # Agent distribution (must sum to 1.0)
    agent_distribution: Dict[str, float] = field(default_factory=lambda: {
        "retail": 0.65,
        "institutional": 0.15,
        "hft": 0.12,
        "market_maker": 0.08
    })
    
    # Microstructure patterns
    enable_momentum_trading: bool = True
    enable_mean_reversion: bool = True
    enable_news_impact: bool = True
    enable_intraday_patterns: bool = True
    
    # Performance settings
    batch_size: int = 5000
    snapshot_frequency_seconds: int = 60
    enable_real_time_processing: bool = False

class EnhancedAgent:
    """Enhanced trading agent with realistic behavior patterns"""
    
    def __init__(self, agent_id: str, agent_type: str, symbols: List[str], config: EnhancedOrderBookConfig):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.symbols = symbols
        self.config = config
        
        # Agent-specific parameters
        self.setup_agent_profile()
        
        # State tracking
        self.holdings = {symbol: 0 for symbol in symbols}
        self.cash = 1_000_000  # Start with $1M
        self.active_orders = {}
        self.trade_history = []
        self.risk_position = 0.0
        
        # Behavioral parameters
        self.momentum_factor = random.uniform(0.1, 0.9)
        self.mean_reversion_factor = random.uniform(0.1, 0.9)
        self.risk_tolerance = random.uniform(0.1, 1.0)
        
    def setup_agent_profile(self):
        """Setup agent-specific trading characteristics"""
        profiles = {
            "retail": {
                "order_size_range": (10, 1000),
                "order_frequency": 0.05,  # orders per minute
                "market_order_prob": 0.4,
                "cancel_prob": 0.15,
                "price_improvement_prob": 0.2,
                "patience_factor": 0.3,
                "news_sensitivity": 0.6
            },
            "institutional": {
                "order_size_range": (1000, 100000),
                "order_frequency": 0.08,
                "market_order_prob": 0.15,
                "cancel_prob": 0.25,
                "price_improvement_prob": 0.6,
                "patience_factor": 0.8,
                "news_sensitivity": 0.9
            },
            "hft": {
                "order_size_range": (50, 1000),
                "order_frequency": 8.0,  # Very high frequency
                "market_order_prob": 0.1,
                "cancel_prob": 0.85,
                "price_improvement_prob": 0.95,
                "patience_factor": 0.05,  # Very impatient
                "news_sensitivity": 0.3
            },
            "market_maker": {
                "order_size_range": (100, 10000),
                "order_frequency": 2.0,
                "market_order_prob": 0.05,
                "cancel_prob": 0.6,
                "price_improvement_prob": 0.9,
                "patience_factor": 0.7,
                "news_sensitivity": 0.4
            }
        }
        
        profile = profiles.get(self.agent_type, profiles["retail"])
        for key, value in profile.items():
            setattr(self, key, value)
    
    def should_trade(self, current_time: datetime, market_data: Dict) -> bool:
        """Determine if agent should trade based on various factors"""
        
        # Base probability from order frequency
        base_prob = self.order_frequency / 60.0  # Convert to per-second probability
        
        # Intraday pattern adjustments
        hour = current_time.hour
        if hour in [9, 10, 15, 16]:  # Market open/close periods
            base_prob *= 2.0
        elif hour in [11, 12, 13, 14]:  # Midday quiet period
            base_prob *= 0.5
        
        # Market volatility adjustments
        if 'volatility' in market_data:
            vol_multiplier = 1.0 + market_data['volatility'] * 2.0
            base_prob *= vol_multiplier
        
        return random.random() < base_prob
    
    def generate_order(self, symbol: str, current_price: float, market_data: Dict, timestamp: datetime) -> Optional[Dict]:
        """Generate a trading order based on agent behavior"""
        
        # Determine order side based on agent strategy
        side = self._determine_order_side(symbol, current_price, market_data)
        if not side:
            return None
        
        # Determine order size
        quantity = self._determine_order_size(symbol, current_price)
        
        # Determine order type and price
        order_type, price = self._determine_order_type_and_price(side, current_price, market_data)
        
        # Generate order ID with uuid suffix to avoid collisions
        u_sfx = uuid.uuid4().hex[:8]
        order_id = f"ORD_{self.agent_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{u_sfx}"
        
        return {
            "order_id": order_id,
            "timestamp": timestamp,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "price": round(price, 2),
            "quantity": quantity,
            "market_price_at_time": round(current_price, 2),
            "time_in_force": "DAY",
            "remaining_quantity": quantity,
            "status": "PENDING"
        }
    
    def _determine_order_side(self, symbol: str, current_price: float, market_data: Dict) -> Optional[str]:
        """Determine whether to buy or sell based on agent strategy"""
        
        # Get recent price movement
        price_change = market_data.get('price_change', 0.0)
        
        # Momentum trading logic
        momentum_signal = 0.0
        if self.config.enable_momentum_trading:
            momentum_signal = price_change * self.momentum_factor
        
        # Mean reversion logic  
        mean_reversion_signal = 0.0
        if self.config.enable_mean_reversion:
            fair_value = market_data.get('fair_value', current_price)
            mean_reversion_signal = -(current_price - fair_value) / fair_value * self.mean_reversion_factor
        
        # Combine signals
        total_signal = momentum_signal + mean_reversion_signal
        
        # Add random noise
        noise = random.gauss(0, 0.1) * (1.0 - self.patience_factor)
        total_signal += noise
        
        # Determine side based on signal strength
        if total_signal > 0.05:
            return "BUY"
        elif total_signal < -0.05:
            return "SELL"
        else:
            return None  # No trade
    
    def _determine_order_size(self, symbol: str, current_price: float) -> int:
        """Determine order size based on agent type and risk tolerance"""
        min_size, max_size = self.order_size_range
        
        # Base size from range
        base_size = random.randint(min_size, max_size)
        
        # Adjust for risk tolerance
        risk_adjustment = self.risk_tolerance
        adjusted_size = int(base_size * risk_adjustment)
        
        # Ensure minimum size
        return max(adjusted_size, min_size)
    
    def _determine_order_type_and_price(self, side: str, current_price: float, market_data: Dict) -> Tuple[str, float]:
        """Determine order type and price"""
        
        if random.random() < self.market_order_prob:
            return "MARKET", current_price
        
        # Limit order - calculate price based on spread and agent behavior
        spread = market_data.get('spread', current_price * 0.001)
        
        if side == "BUY":
            # Buy orders typically placed below market
            price_offset = random.uniform(0, spread * 2) * self.price_improvement_prob
            price = current_price - price_offset
        else:
            # Sell orders typically placed above market
            price_offset = random.uniform(0, spread * 2) * self.price_improvement_prob
            price = current_price + price_offset
        
        return "LIMIT", max(0.01, price)

class EnhancedOrderBookDB:
    """Enhanced order book with comprehensive database storage and realistic patterns"""
    
    def __init__(self, config: EnhancedOrderBookConfig):
        self.config = config
        
        # Initialize runtime state first
        # Use provided simulation start or default to today at 13:30 UTC (approx 9:30 ET during DST)
        if self.config.simulation_start_utc:
            try:
                self.current_time = pd.to_datetime(self.config.simulation_start_utc, utc=True).to_pydatetime()
            except Exception:
                self.current_time = datetime.utcnow().replace(hour=13, minute=30, second=0, microsecond=0)
        else:
            self.current_time = datetime.utcnow().replace(hour=13, minute=30, second=0, microsecond=0)
        # Unique run identifier to ensure globally unique IDs across runs
        self.run_id = uuid.uuid4().hex[:8]
        self.order_books = {symbol: {"bids": {}, "asks": {}} for symbol in config.symbols}
        self.last_trade_prices = config.initial_prices.copy()
        self.market_data = {symbol: {"volatility": 0.02, "spread": price * 0.001, "fair_value": price, "price_change": 0.0} 
                          for symbol, price in config.initial_prices.items()}
        
        # Setup components
        self.setup_database()
        self.setup_agents()
        self.setup_market_data()
        
        # Statistics
        self.stats = {
            "orders_generated": 0,
            "trades_executed": 0,
            "snapshots_taken": 0,
            "start_time": None,
            "end_time": None
        }
        
        # Trade ID counter for uniqueness
        self.trade_counter = 0
    
    def setup_database(self):
        """Initialize database connection and tables"""
        if self.config.use_in_memory:
            db_url = "sqlite:///:memory:"
        else:
            db_url = f"sqlite:///{self.config.db_path}"
        
        self.engine = create_engine(
            db_url, 
            poolclass=StaticPool,
            connect_args={'check_same_thread': False} if 'sqlite' in db_url else {}
        )
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        logger.info(f"Database initialized: {db_url}")
    
    def setup_agents(self):
        """Initialize trading agents"""
        self.agents = []
        agent_id = 0
        
        for agent_type, proportion in self.config.agent_distribution.items():
            num_agents = int(self.config.num_agents * proportion)
            
            for i in range(num_agents):
                agent = EnhancedAgent(
                    agent_id=f"{agent_type}_{i:04d}",
                    agent_type=agent_type,
                    symbols=self.config.symbols,
                    config=self.config
                )
                self.agents.append(agent)
                agent_id += 1
        
        logger.info(f"Initialized {len(self.agents)} agents: {dict(self.config.agent_distribution)}")
    
    def setup_market_data(self):
        """Initialize market data structures"""
        for symbol in self.config.symbols:
            # Initialize with some market depth
            price = self.config.initial_prices[symbol]
            spread = price * 0.001
            
            self.market_data[symbol].update({
                "last_update": self.current_time,
                "volume_today": 0,
                "trades_today": 0,
                "high_today": price,
                "low_today": price,
                "open_today": price
            })
    
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
    
    def generate_simulation_data(self) -> Dict[str, Any]:
        """Main method to generate comprehensive order book data"""
        logger.info("🚀 Starting Enhanced Order Book Data Generation")
        logger.info(f"Symbols: {len(self.config.symbols)}")
        logger.info(f"Agents: {len(self.agents)}")
        logger.info(f"Days: {self.config.simulation_days}")
        
        self.stats["start_time"] = datetime.now()
        
        # Simulate each day
        for day in range(self.config.simulation_days):
            logger.info(f"Simulating day {day + 1}/{self.config.simulation_days}")
            self._simulate_trading_day()
        
        self.stats["end_time"] = datetime.now()
        
        # Generate final summary
        summary = self._generate_summary()
        logger.info("✅ Simulation completed!")
        
        return summary
    
    def _simulate_trading_day(self):
        """Simulate one complete trading day"""
        # Reset daily stats
        for symbol in self.config.symbols:
            self.market_data[symbol]["volume_today"] = 0
            self.market_data[symbol]["trades_today"] = 0
            self.market_data[symbol]["open_today"] = self.last_trade_prices[symbol]
            self.market_data[symbol]["high_today"] = self.last_trade_prices[symbol]
            self.market_data[symbol]["low_today"] = self.last_trade_prices[symbol]
        
        # Simulate each minute of the trading day
        trading_minutes = (self.config.trading_hours_end - self.config.trading_hours_start) * 60
        
        for minute in range(trading_minutes):
            minute_time = self.current_time + timedelta(minutes=minute)
            self._simulate_minute(minute_time)
            
            # Take snapshots periodically
            if minute % (self.config.snapshot_frequency_seconds // 60) == 0:
                self._take_orderbook_snapshots(minute_time)
        
        # Save daily market stats
        self._save_daily_market_stats()
        
        # Move to next day
        self.current_time += timedelta(days=1)
    
    def _simulate_minute(self, minute_time: datetime):
        """Simulate one minute of trading activity"""
        minute_orders = []
        
        # Update market prices with realistic movement
        self._update_market_prices()
        
        # Generate orders from agents
        for agent in self.agents:
            for symbol in self.config.symbols:
                if agent.should_trade(minute_time, self.market_data[symbol]):
                    order = agent.generate_order(
                        symbol=symbol,
                        current_price=self.last_trade_prices[symbol],
                        market_data=self.market_data[symbol],
                        timestamp=minute_time
                    )
                    if order:
                        minute_orders.append(order)
                        self.stats["orders_generated"] += 1
        
        # Process orders and execute trades
        if minute_orders:
            self._process_orders_batch(minute_orders)
    
    def _update_market_prices(self):
        """Update market prices with realistic movement patterns"""
        for symbol in self.config.symbols:
            current_price = self.last_trade_prices[symbol]
            volatility = self.config.volatility_per_symbol.get(symbol, 0.02)
            
            # Generate price movement with multiple components
            dt = 1.0 / (252 * 24 * 60)  # 1 minute in years
            
            # Random walk component
            random_shock = np.random.normal(0, volatility * np.sqrt(dt))
            
            # Mean reversion component
            fair_value = self.config.initial_prices[symbol]
            mean_reversion = 0.1 * (fair_value - current_price) / fair_value * dt
            
            # Momentum component based on recent trades
            momentum = 0.0
            if symbol in self.market_data:
                momentum = self.market_data[symbol].get("price_change", 0.0) * 0.1
            
            # Combine components
            total_change = random_shock + mean_reversion + momentum
            new_price = max(0.01, current_price * (1 + total_change))
            
            # Update market data
            price_change = (new_price - current_price) / current_price
            self.last_trade_prices[symbol] = new_price
            self.market_data[symbol]["price_change"] = price_change
            self.market_data[symbol]["fair_value"] = fair_value
            
            # Update daily stats
            self.market_data[symbol]["high_today"] = max(self.market_data[symbol]["high_today"], new_price)
            self.market_data[symbol]["low_today"] = min(self.market_data[symbol]["low_today"], new_price)
    
    def _process_orders_batch(self, orders: List[Dict]):
        """Process a batch of orders and store in database"""
        with self.get_db_session() as session:
            # Save orders to database
            for order in orders:
                db_order = OrderDB(**order)
                session.add(db_order)
            
            # Simulate some trade executions (simplified matching)
            self._simulate_trade_matching(session, orders)
    
    def _simulate_trade_matching(self, session: Session, orders: List[Dict]):
        """Simulate trade matching (simplified version)"""
        # Group orders by symbol
        symbol_orders = defaultdict(list)
        for order in orders:
            symbol_orders[order["symbol"]].append(order)
        
        # Process each symbol
        for symbol, orders_list in symbol_orders.items():
            buys = [o for o in orders_list if o["side"] == "BUY"]
            sells = [o for o in orders_list if o["side"] == "SELL"]
            
            # Simple matching logic (market orders get filled, some limit orders)
            for buy_order in buys:
                for sell_order in sells:
                    if self._can_match_orders(buy_order, sell_order):
                        trade = self._execute_trade(buy_order, sell_order)
                        if trade:
                            # Save trade to database
                            db_trade = TradeDB(**trade)
                            session.add(db_trade)
                            self.stats["trades_executed"] += 1
                            break
    
    def _can_match_orders(self, buy_order: Dict, sell_order: Dict) -> bool:
        """Check if two orders can be matched"""
        if buy_order["symbol"] != sell_order["symbol"]:
            return False
        
        # Market orders always match
        if buy_order["order_type"] == "MARKET" or sell_order["order_type"] == "MARKET":
            return True
        
        # Limit orders match if buy price >= sell price
        return buy_order["price"] >= sell_order["price"]
    
    def _execute_trade(self, buy_order: Dict, sell_order: Dict) -> Optional[Dict]:
        """Execute a trade between two orders"""
        # Determine trade price (simplified)
        if buy_order["order_type"] == "MARKET":
            price = sell_order["price"]
        elif sell_order["order_type"] == "MARKET":
            price = buy_order["price"]
        else:
            price = (buy_order["price"] + sell_order["price"]) / 2
        
        # Determine quantity (minimum of both orders)
        quantity = min(buy_order["remaining_quantity"], sell_order["remaining_quantity"])
        
        # Generate unique trade ID (prefix with run_id to avoid collisions across runs)
        self.trade_counter += 1
        trade_id = (
            f"TRD_{self.run_id}_{buy_order['timestamp'].strftime('%Y%m%d_%H%M%S')}_"
            f"{self.trade_counter:06d}"
        )
        
        # Update market data
        symbol = buy_order["symbol"]
        old_price = self.last_trade_prices[symbol]
        self.last_trade_prices[symbol] = price
        self.market_data[symbol]["volume_today"] += quantity
        self.market_data[symbol]["trades_today"] += 1
        
        return {
            "trade_id": trade_id,
            "timestamp": buy_order["timestamp"],
            "symbol": symbol,
            "price": round(price, 2),
            "quantity": quantity,
            "buy_order_id": buy_order["order_id"],
            "sell_order_id": sell_order["order_id"],
            "buy_agent_id": buy_order["agent_id"],
            "sell_agent_id": sell_order["agent_id"],
            "aggressor_side": "BUY" if buy_order["order_type"] == "MARKET" else "SELL",
            "market_impact": abs(price - old_price) / old_price if old_price > 0 else 0.0
        }
    
    def _take_orderbook_snapshots(self, timestamp: datetime):
        """Take snapshots of order book state"""
        with self.get_db_session() as session:
            for symbol in self.config.symbols:
                # Calculate order book metrics (simplified)
                price = self.last_trade_prices[symbol]
                spread = price * 0.001  # 0.1% spread
                
                # Create bid/ask depth (simplified)
                bid_depth = {}
                ask_depth = {}
                
                # Generate some depth levels
                for i in range(5):
                    bid_price = price - (i + 1) * spread / 2
                    ask_price = price + (i + 1) * spread / 2
                    bid_depth[str(round(bid_price, 2))] = random.randint(100, 5000)
                    ask_depth[str(round(ask_price, 2))] = random.randint(100, 5000)
                
                best_bid = max(float(p) for p in bid_depth.keys()) if bid_depth else None
                best_ask = min(float(p) for p in ask_depth.keys()) if ask_depth else None
                
                snapshot = OrderBookSnapshotDB(
                    timestamp=timestamp,
                    symbol=symbol,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    bid_depth_json=json.dumps(bid_depth),
                    ask_depth_json=json.dumps(ask_depth),
                    spread=spread,
                    mid_price=(best_bid + best_ask) / 2 if best_bid and best_ask else price,
                    last_trade_price=price,
                    total_volume=self.market_data[symbol]["volume_today"],
                    volatility=self.config.volatility_per_symbol.get(symbol, 0.02)
                )
                
                session.add(snapshot)
                self.stats["snapshots_taken"] += 1
    
    def _save_daily_market_stats(self):
        """Save daily market statistics"""
        with self.get_db_session() as session:
            for symbol in self.config.symbols:
                data = self.market_data[symbol]
                
                # Calculate VWAP (simplified)
                vwap = self.last_trade_prices[symbol]  # Simplified
                
                stats = MarketStatsDB(
                    symbol=symbol,
                    date=self.current_time.date(),
                    open_price=data["open_today"],
                    high_price=data["high_today"],
                    low_price=data["low_today"],
                    close_price=self.last_trade_prices[symbol],
                    volume=data["volume_today"],
                    vwap=vwap,
                    num_trades=data["trades_today"],
                    avg_spread=self.last_trade_prices[symbol] * 0.001  # Simplified
                )
                
                session.add(stats)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive simulation summary"""
        duration = self.stats["end_time"] - self.stats["start_time"]
        
        # Get database statistics
        with self.get_db_session() as session:
            total_orders = session.query(OrderDB).count()
            total_trades = session.query(TradeDB).count()
            total_snapshots = session.query(OrderBookSnapshotDB).count()
        
        return {
            "simulation_duration": str(duration),
            "total_orders": total_orders,
            "total_trades": total_trades,
            "total_snapshots": total_snapshots,
            "orders_per_second": total_orders / duration.total_seconds() if duration.total_seconds() > 0 else 0,
            "symbols": self.config.symbols,
            "final_prices": self.last_trade_prices,
            "database_path": self.config.db_path,
            "agents_used": len(self.agents),
            "agent_distribution": dict(self.config.agent_distribution),
            "simulation_days": self.config.simulation_days
        }
    
    def export_data_analysis(self) -> Dict[str, pd.DataFrame]:
        """Export data for analysis"""
        logger.info("📊 Exporting data for analysis...")
        
        with self.get_db_session() as session:
            # Export orders
            orders_query = session.query(OrderDB)
            orders_df = pd.read_sql(orders_query.statement, session.bind)
            
            # Export trades
            trades_query = session.query(TradeDB)
            trades_df = pd.read_sql(trades_query.statement, session.bind)
            
            # Export snapshots
            snapshots_query = session.query(OrderBookSnapshotDB)
            snapshots_df = pd.read_sql(snapshots_query.statement, session.bind)
            
            # Export market stats
            stats_query = session.query(MarketStatsDB)
            stats_df = pd.read_sql(stats_query.statement, session.bind)
        
        return {
            "orders": orders_df,
            "trades": trades_df,
            "snapshots": snapshots_df,
            "market_stats": stats_df
        }
    
    def get_analysis_report(self) -> str:
        """Generate a comprehensive analysis report"""
        data = self.export_data_analysis()
        
        report = ["📊 ENHANCED ORDER BOOK ANALYSIS REPORT", "=" * 50, ""]
        
        # Summary statistics
        report.append("📈 SUMMARY STATISTICS")
        report.append("-" * 30)
        report.append(f"Total Orders: {len(data['orders']):,}")
        report.append(f"Total Trades: {len(data['trades']):,}")
        report.append(f"Total Snapshots: {len(data['snapshots']):,}")
        report.append(f"Fill Rate: {len(data['trades']) / len(data['orders']) * 100:.1f}%")
        report.append("")
        
        # Symbol analysis
        report.append("🏷️  SYMBOL ANALYSIS")
        report.append("-" * 30)
        for symbol in self.config.symbols:
            symbol_orders = data['orders'][data['orders']['symbol'] == symbol]
            symbol_trades = data['trades'][data['trades']['symbol'] == symbol]
            
            if not symbol_trades.empty:
                volume = symbol_trades['quantity'].sum()
                avg_price = symbol_trades['price'].mean()
                volatility = symbol_trades['price'].std() / avg_price * 100
                
                report.append(f"{symbol}:")
                report.append(f"  Orders: {len(symbol_orders):,}")
                report.append(f"  Trades: {len(symbol_trades):,}")
                report.append(f"  Volume: {volume:,}")
                report.append(f"  Avg Price: ${avg_price:.2f}")
                report.append(f"  Volatility: {volatility:.2f}%")
                report.append("")
        
        # Agent analysis
        report.append("🤖 AGENT ANALYSIS")
        report.append("-" * 30)
        agent_stats = data['orders'].groupby('agent_type').agg({
            'order_id': 'count',
            'quantity': 'sum'
        }).rename(columns={'order_id': 'orders'})
        
        for agent_type, stats in agent_stats.iterrows():
            report.append(f"{agent_type.title()}:")
            report.append(f"  Orders: {stats['orders']:,}")
            report.append(f"  Total Quantity: {stats['quantity']:,}")
            report.append("")
        
        # Market microstructure insights
        if not data['trades'].empty:
            report.append("🔬 MARKET MICROSTRUCTURE")
            report.append("-" * 30)
            
            trades_df = data['trades']
            avg_trade_size = trades_df['quantity'].mean()
            median_trade_size = trades_df['quantity'].median()
            
            # Market impact analysis
            avg_impact = trades_df['market_impact'].mean() * 10000  # in basis points
            
            report.append(f"Average Trade Size: {avg_trade_size:.1f}")
            report.append(f"Median Trade Size: {median_trade_size:.1f}")
            report.append(f"Average Market Impact: {avg_impact:.2f} bps")
            report.append("")
        
        return "\n".join(report)

def main():
    """Main function for testing the enhanced order book system"""
    print("🚀 Enhanced Order Book Database System")
    print("=" * 50)
    
    # Create configuration
    config = EnhancedOrderBookConfig(
        num_agents=1000,
        simulation_days=2,
        symbols=["AAPL", "GOOGL", "MSFT"],
        base_orders_per_minute=100
    )
    
    # Create and run simulation
    orderbook = EnhancedOrderBookDB(config)
    summary = orderbook.generate_simulation_data()
    
    # Print summary
    print("\n📊 SIMULATION SUMMARY")
    print("=" * 30)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Generate analysis report
    print("\n" + orderbook.get_analysis_report())

if __name__ == "__main__":
    main()