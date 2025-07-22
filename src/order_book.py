#!/usr/bin/env python3
"""
Order Book Recording System for ABIDES-LLM Integration
=====================================================

A comprehensive order book implementation with detailed recording capabilities
for market microstructure analysis and experimental studies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import pickle
import logging

@dataclass
class Order:
    """Individual order in the order book"""
    order_id: str
    timestamp: datetime
    agent_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    price: float
    quantity: int
    order_type: str = "LIMIT"  # "LIMIT" or "MARKET"
    time_in_force: str = "DAY"  # "DAY", "IOC", "FOK"
    
    # Execution tracking
    filled_quantity: int = 0
    remaining_quantity: int = field(init=False)
    avg_fill_price: float = 0.0
    status: str = "PENDING"  # "PENDING", "PARTIAL", "FILLED", "CANCELLED"
    
    def __post_init__(self):
        self.remaining_quantity = self.quantity

@dataclass
class Trade:
    """Executed trade record"""
    trade_id: str
    timestamp: datetime
    symbol: str
    price: float
    quantity: int
    buy_order_id: str
    sell_order_id: str
    buy_agent_id: str
    sell_agent_id: str
    aggressor_side: str  # "BUY" or "SELL"

@dataclass
class OrderBookSnapshot:
    """Snapshot of order book state at a point in time"""
    timestamp: datetime
    symbol: str
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bid_depth: Dict[float, int] = field(default_factory=dict)
    ask_depth: Dict[float, int] = field(default_factory=dict)
    spread: Optional[float] = None
    mid_price: Optional[float] = None
    last_trade_price: Optional[float] = None
    total_volume: int = 0

class OrderBook:
    """Advanced order book with comprehensive recording and ABIDES-style matching"""
    
    def __init__(self, symbol: str, tick_size: float = 0.01):
        self.symbol = symbol
        self.tick_size = tick_size
        
        # Order storage
        self.bids = defaultdict(deque)  # price -> deque of orders
        self.asks = defaultdict(deque)  # price -> deque of orders
        self.orders = {}  # order_id -> Order
        
        # Price levels (sorted)
        self.bid_prices = []  # sorted descending
        self.ask_prices = []  # sorted ascending
        
        # Trade and snapshot history
        self.trades = []
        self.snapshots = []
        self.order_history = []
        
        # Market data
        self.last_trade_price = None
        self.last_trade_time = None
        self.daily_open = None
        self.daily_high = None
        self.daily_low = None
        self.daily_volume = 0
        
        # Logging
        self.logger = logging.getLogger(f"OrderBook_{symbol}")
