#!/usr/bin/env python3
"""
Microstructure-aware LOB Generator with realistic trade price distribution.
Implements proper bid-ask mechanics and multiple price levels.
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from enum import Enum
import random
from dataclasses import dataclass
import bisect

class OrderType(Enum):
    """Order types that affect execution price"""
    MARKET = "market"           # Executes at best available
    LIMIT = "limit"             # Executes at specific price or better
    MARKETABLE_LIMIT = "marketable_limit"  # Limit order that crosses spread

@dataclass
class OrderBookLevel:
    """Single price level in the order book"""
    price: float
    size: int
    num_orders: int

class OrderBook:
    """Realistic order book with multiple price levels"""
    
    def __init__(self, initial_price: float, tick_size: float = 0.01):
        """
        Initialize order book
        
        Args:
            initial_price: Starting mid price
            tick_size: Minimum price increment ($0.01 for most stocks)
        """
        self.tick_size = tick_size
        self.bids = []  # List of (price, size) tuples, sorted descending
        self.asks = []  # List of (price, size) tuples, sorted ascending
        
        # Initialize with some depth
        self._initialize_book(initial_price)
    
    def _initialize_book(self, mid_price: float):
        """Create initial order book with multiple levels"""
        # Create 5 levels on each side
        for i in range(1, 6):
            # Bid side
            bid_price = mid_price - i * self.tick_size
            bid_size = np.random.randint(100, 1000) * 100
            self.bids.append((bid_price, bid_size))
            
            # Ask side
            ask_price = mid_price + i * self.tick_size
            ask_size = np.random.randint(100, 1000) * 100
            self.asks.append((ask_price, ask_size))
    
    def get_best_bid(self) -> Tuple[float, int]:
        """Get best bid price and size"""
        return self.bids[0] if self.bids else (0, 0)
    
    def get_best_ask(self) -> Tuple[float, int]:
        """Get best ask price and size"""
        return self.asks[0] if self.asks else (0, 0)
    
    def get_mid_price(self) -> float:
        """Calculate mid price"""
        bid_price, _ = self.get_best_bid()
        ask_price, _ = self.get_best_ask()
        if bid_price > 0 and ask_price > 0:
            return (bid_price + ask_price) / 2
        return 0
    
    def get_spread(self) -> float:
        """Get bid-ask spread"""
        bid_price, _ = self.get_best_bid()
        ask_price, _ = self.get_best_ask()
        if bid_price > 0 and ask_price > 0:
            return ask_price - bid_price
        return 0
    
    def update_levels(self, mid_price_change: float):
        """Update all price levels based on mid price change"""
        # Shift all levels
        new_bids = []
        for price, size in self.bids:
            new_price = price + mid_price_change
            # Add some randomness to size
            new_size = max(100, size + np.random.randint(-100, 100) * 100)
            new_bids.append((new_price, new_size))
        self.bids = new_bids
        
        new_asks = []
        for price, size in self.asks:
            new_price = price + mid_price_change
            new_size = max(100, size + np.random.randint(-100, 100) * 100)
            new_asks.append((new_price, new_size))
        self.asks = new_asks
        
        # Occasionally add/remove levels
        if random.random() < 0.1:
            self._rebalance_book()
    
    def _rebalance_book(self):
        """Rebalance order book depth"""
        mid = self.get_mid_price()
        
        # Ensure we have 3-7 levels on each side
        while len(self.bids) < 3:
            worst_bid = self.bids[-1][0] if self.bids else mid - self.tick_size
            new_price = worst_bid - self.tick_size
            new_size = np.random.randint(100, 500) * 100
            self.bids.append((new_price, new_size))
        
        while len(self.asks) < 3:
            worst_ask = self.asks[-1][0] if self.asks else mid + self.tick_size
            new_price = worst_ask + self.tick_size
            new_size = np.random.randint(100, 500) * 100
            self.asks.append((new_price, new_size))
        
        # Remove levels too far from mid
        self.bids = self.bids[:7]
        self.asks = self.asks[:7]

class MicrostructureLOBGenerator:
    """
    Generates realistic LOB with proper market microstructure
    """
    
    def __init__(self, symbol: str, date: str, initial_price: float, 
                 duration_seconds: int, news_events: List[Dict], 
                 condition: str = "Baseline"):
        """
        Initialize the LOB generator with microstructure awareness
        """
        self.symbol = symbol
        self.date = date
        self.initial_price = initial_price
        self.duration_seconds = duration_seconds
        self.news_events = news_events
        self.condition = condition
        
        # Market microstructure parameters
        self.tick_size = 0.01  # Penny increments
        self.min_spread = 0.01  # Minimum 1 penny spread
        self.typical_spread = 0.02  # Typical 2 penny spread
        
        # Order flow parameters
        self.market_order_prob = 0.4  # 40% market orders
        self.aggressive_limit_prob = 0.3  # 30% aggressive limit orders
        self.passive_limit_prob = 0.3  # 30% passive limit orders
        
        # Sub-second timing
        self.trades_per_second_base = 2
        self.trades_per_second_active = 10  # During active periods
        
        # Initialize order book
        self.order_book = OrderBook(initial_price, self.tick_size)
    
    def generate_trades_with_microstructure(self, timestamp: float, mid_price: float, 
                                           volatility: float, news_impact: float) -> List[Dict]:
        """
        Generate trades with realistic microstructure at a given timestamp
        
        Returns:
            List of trade dictionaries with sub-second timestamps and varied prices
        """
        trades = []
        
        # Determine trade intensity
        base_intensity = self.trades_per_second_base
        if abs(news_impact) > 0.001:
            base_intensity = self.trades_per_second_active
        
        # Number of trades in this second (Poisson distributed)
        num_trades = np.random.poisson(base_intensity * (1 + volatility * 10))
        
        if num_trades == 0:
            return trades
        
        # Generate sub-second timestamps
        sub_second_times = sorted(np.random.uniform(0, 1, num_trades))
        
        # Update order book based on mid price
        price_change = mid_price - self.order_book.get_mid_price()
        if abs(price_change) > self.tick_size / 2:
            self.order_book.update_levels(price_change)
        
        # Get current book state
        best_bid, bid_size = self.order_book.get_best_bid()
        best_ask, ask_size = self.order_book.get_best_ask()
        spread = best_ask - best_bid
        
        # Generate trades
        for i, sub_time in enumerate(sub_second_times):
            exact_timestamp = timestamp + sub_time
            
            # Determine order type
            rand = random.random()
            if rand < self.market_order_prob:
                # Market order - executes at best available
                if random.random() < 0.5:
                    # Buy market order - executes at ask
                    trade_price = best_ask
                    side = 'B'
                    # May walk the book for large orders
                    if random.random() < 0.1:  # 10% chance of walking book
                        trade_price += random.choice([0, self.tick_size, 2*self.tick_size])
                else:
                    # Sell market order - executes at bid
                    trade_price = best_bid
                    side = 'S'
                    if random.random() < 0.1:
                        trade_price -= random.choice([0, self.tick_size, 2*self.tick_size])
                        
            elif rand < self.market_order_prob + self.aggressive_limit_prob:
                # Aggressive limit order - crosses the spread
                if random.random() < 0.5:
                    # Buy limit at or above ask
                    trade_price = best_ask
                    side = 'B'
                else:
                    # Sell limit at or below bid
                    trade_price = best_bid
                    side = 'S'
                    
            else:
                # Passive limit order - may get price improvement
                if random.random() < 0.5:
                    # Buy limit - might execute between bid and ask
                    if spread > self.min_spread and random.random() < 0.3:
                        # Price improvement
                        trade_price = best_bid + self.tick_size
                    else:
                        trade_price = best_bid
                    side = 'B'
                else:
                    # Sell limit
                    if spread > self.min_spread and random.random() < 0.3:
                        trade_price = best_ask - self.tick_size
                    else:
                        trade_price = best_ask
                    side = 'S'
            
            # Add some noise for hidden orders, odd lots, etc.
            if random.random() < 0.05:  # 5% chance of off-spread execution
                trade_price += np.random.choice([-1, 1]) * self.tick_size * np.random.randint(0, 3)
            
            # Ensure price is positive and reasonable
            trade_price = max(trade_price, mid_price * 0.95)
            trade_price = min(trade_price, mid_price * 1.05)
            
            # Trade size - follows power law
            if random.random() < 0.7:  # 70% small trades
                size = np.random.choice([100, 200, 300, 400, 500])
            elif random.random() < 0.9:  # 20% medium trades
                size = np.random.choice([600, 700, 800, 900, 1000])
            else:  # 10% large trades
                size = np.random.randint(1100, 5000)
            
            trades.append({
                'timestamp': exact_timestamp,
                'price': round(trade_price, 2),  # Round to penny
                'size': size,
                'side': side
            })
            
            # Update book state after trade (simplified)
            if side == 'B' and random.random() < 0.3:
                # Buying pressure might increase ask
                best_ask += self.tick_size * random.choice([0, 0, 1])
            elif side == 'S' and random.random() < 0.3:
                # Selling pressure might decrease bid
                best_bid -= self.tick_size * random.choice([0, 0, 1])
        
        return trades
    
    def generate_price_path(self, num_steps: int) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate price path and all trades with microstructure
        """
        prices = np.zeros(num_steps)
        prices[0] = self.initial_price
        all_trades = []
        trade_id = 1
        
        # Parameters from heterogeneous model
        base_volatility = 0.0001
        mean_reversion_strength = 0.15
        max_price_change = 0.02
        news_decay_rate = 0.001
        max_news_impact = 0.005
        
        fundamental_price = self.initial_price
        momentum = 0
        
        for i in range(1, num_steps):
            timestamp = i  # Each step is 1 second
            
            # Calculate news impact
            total_news_impact = 0
            for news in self.news_events:
                time_since_news = timestamp - news['timestamp']
                if time_since_news >= 0:
                    decay = np.exp(-news_decay_rate * time_since_news)
                    raw_impact = news['sentiment'] * news.get('importance', 0.5)
                    capped_impact = np.sign(raw_impact) * min(abs(raw_impact), max_news_impact)
                    total_news_impact += capped_impact * decay
            
            # Price dynamics (from heterogeneous model)
            price_deviation = (prices[i-1] - fundamental_price) / fundamental_price
            
            if i > 10:
                recent_return = (prices[i-1] - prices[i-10]) / prices[i-10]
                momentum = 0.7 * momentum + 0.3 * recent_return * 10
            
            # Price update
            random_component = np.random.normal(0, base_volatility)
            pressure_component = momentum * 0.0001
            reversion_component = -price_deviation * mean_reversion_strength * 0.001
            news_component = total_news_impact * 0.001
            
            total_return = random_component + pressure_component + reversion_component + news_component
            total_return = np.clip(total_return, -max_price_change, max_price_change)
            
            prices[i] = prices[i-1] * (1 + total_return)
            
            # Generate trades with microstructure
            volatility = abs(total_return)
            trades = self.generate_trades_with_microstructure(
                timestamp, prices[i], volatility, total_news_impact
            )
            
            # Add trade IDs
            for trade in trades:
                trade['trade_id'] = trade_id
                trade_id += 1
                all_trades.append(trade)
        
        return prices, all_trades
    
    def generate_and_save(self, db_path: str):
        """
        Generate complete LOB data with microstructure and save to database
        """
        print(f"Generating {self.condition} LOB with market microstructure...")
        
        # Generate price path and trades
        prices, trades = self.generate_price_path(self.duration_seconds)
        
        # Create database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                timestamp REAL,
                message_type TEXT,
                order_id INTEGER,
                trade_id INTEGER,
                price REAL,
                size INTEGER,
                side TEXT,
                PRIMARY KEY (timestamp, message_type, order_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                timestamp REAL,
                trade_id INTEGER PRIMARY KEY,
                price REAL,
                size INTEGER,
                side TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orderbook (
                timestamp REAL PRIMARY KEY,
                bid_price_1 REAL,
                bid_size_1 INTEGER,
                ask_price_1 REAL,
                ask_size_1 INTEGER,
                mid_price REAL,
                spread REAL
            )
        ''')
        
        # Insert trades
        for trade in trades:
            cursor.execute('''
                INSERT INTO trades (timestamp, trade_id, price, size, side)
                VALUES (?, ?, ?, ?, ?)
            ''', (trade['timestamp'], trade['trade_id'], trade['price'], 
                  trade['size'], trade['side']))
            
            # Also insert as message
            cursor.execute('''
                INSERT OR IGNORE INTO messages (timestamp, message_type, order_id, trade_id, price, size, side)
                VALUES (?, 'E', ?, ?, ?, ?, ?)
            ''', (trade['timestamp'], trade['trade_id'], trade['trade_id'], 
                  trade['price'], trade['size'], trade['side']))
        
        # Generate orderbook snapshots (1 per second)
        for i in range(self.duration_seconds):
            timestamp = i
            mid_price = prices[i]
            
            # Calculate spread based on volatility
            if i > 0:
                volatility = abs(prices[i] - prices[i-1]) / prices[i-1]
            else:
                volatility = 0
            
            # Spread increases with volatility
            base_spread = 0.01
            volatility_adjustment = min(0.05, volatility * 100)
            spread = base_spread + volatility_adjustment
            
            # Round to tick size
            spread = round(spread / self.tick_size) * self.tick_size
            spread = max(self.min_spread, spread)
            
            bid_price = round(mid_price - spread/2, 2)
            ask_price = round(mid_price + spread/2, 2)
            
            # Size based on volatility (less size when volatile)
            base_size = 500
            size_multiplier = max(0.5, 1 - volatility * 50)
            bid_size = int(base_size * size_multiplier * (0.8 + random.random() * 0.4))
            ask_size = int(base_size * size_multiplier * (0.8 + random.random() * 0.4))
            
            cursor.execute('''
                INSERT INTO orderbook (timestamp, bid_price_1, bid_size_1, 
                                      ask_price_1, ask_size_1, mid_price, spread)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, bid_price, bid_size, ask_price, ask_size, mid_price, spread))
        
        conn.commit()
        conn.close()
        
        # Print summary
        print(f"  Generated {len(trades)} trades")
        print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"  Final price: ${prices[-1]:.2f} ({(prices[-1]/prices[0]-1)*100:.2f}%)")
        
        # Analyze trade distribution
        trade_df = pd.DataFrame(trades)
        if len(trade_df) > 0:
            grouped = trade_df.groupby(trade_df['timestamp'].astype(int))
            avg_trades_per_second = grouped.size().mean()
            avg_unique_prices = grouped['price'].nunique().mean()
            print(f"  Avg trades per second: {avg_trades_per_second:.1f}")
            print(f"  Avg unique prices per second: {avg_unique_prices:.1f}")
        
        print(f"  Saved to {db_path}")

def main():
    """
    Generate microstructure-aware LOB databases for all three conditions
    """
    # Real news events from 2012-06-21
    news_events = [
        {
            'timestamp': 3600,
            'sentiment': -0.3,
            'importance': 0.6,
            'description': 'Fed maintains cautious stance on economy'
        },
        {
            'timestamp': 7200,
            'sentiment': -0.4,
            'importance': 0.7,
            'description': 'Spain borrowing costs hit new highs'
        },
        {
            'timestamp': 10800,
            'sentiment': -0.2,
            'importance': 0.5,
            'description': 'Tech sector shows weakness'
        },
        {
            'timestamp': 14400,
            'sentiment': -0.1,
            'importance': 0.4,
            'description': 'Market fails to hold morning recovery'
        },
        {
            'timestamp': 18000,
            'sentiment': 0.2,
            'importance': 0.5,
            'description': 'Some buying interest emerges'
        }
    ]
    
    # Parameters from real NASDAQ data
    symbol = "AMZN"
    date = "2012-06-21"
    initial_price = 223.56
    duration_seconds = 23400  # 6.5 hours
    
    # Output directory
    import os
    output_dir = "/workspace/lob_databases_microstructure"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate for each condition
    for condition in ["LLMON", "LLMOFF", "Baseline"]:
        generator = MicrostructureLOBGenerator(
            symbol=symbol,
            date=date,
            initial_price=initial_price,
            duration_seconds=duration_seconds,
            news_events=news_events,
            condition=condition
        )
        
        db_path = f"{output_dir}/{symbol}_{date}_{condition}_microstructure.db"
        generator.generate_and_save(db_path)
    
    print("\n✅ All microstructure-aware LOB databases generated successfully!")

if __name__ == "__main__":
    main()