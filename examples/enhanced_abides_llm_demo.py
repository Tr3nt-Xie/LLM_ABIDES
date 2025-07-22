#!/usr/bin/env python3
"""
Enhanced ABIDES-LLM Demo with Order Book Recording
=================================================

This enhanced demo includes:
- Complete order book recording and tracking
- Market microstructure analysis
- Basic ABIDES-style experiments
- Comprehensive data export for analysis
"""

import os
import sys
import random
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OrderBookOrder:
    """Order for the order book system"""
    order_id: str
    timestamp: datetime
    agent_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    price: float
    quantity: int
    order_type: str = "LIMIT"
    
    # Execution tracking
    filled_quantity: int = 0
    remaining_quantity: int = 0
    avg_fill_price: float = 0.0
    status: str = "PENDING"
    
    def __post_init__(self):
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity

@dataclass
class TradeRecord:
    """Record of executed trade"""
    trade_id: str
    timestamp: datetime
    symbol: str
    price: float
    quantity: int
    buy_order_id: str
    sell_order_id: str
    buy_agent_id: str
    sell_agent_id: str
    aggressor_side: str

class EnhancedOrderBook:
    """Enhanced order book with comprehensive recording"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # Order storage
        self.bids = {}  # price -> list of orders
        self.asks = {}  # price -> list of orders
        self.orders = {}  # order_id -> order
        
        # Price levels
        self.bid_prices = []  # sorted descending
        self.ask_prices = []  # sorted ascending
        
        # Trade history
        self.trades = []
        self.order_history = []
        
        # Market data
        self.last_trade_price = None
        self.last_trade_time = None
        self.daily_volume = 0
        
        # Snapshots for analysis
        self.snapshots = []
