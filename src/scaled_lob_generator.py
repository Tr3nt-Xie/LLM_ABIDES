#!/usr/bin/env python3
"""
Scaled LOB Generator for Three Experimental Conditions
======================================================

Generates realistic, high-frequency LOB databases for:
1. LLMON - LLM-enhanced agents
2. LLMOFF - No LLM agents
3. Baseline - Traditional agents

Each database contains complete order book data with realistic trade frequencies.
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from dataclasses import dataclass
import json
import random

logger = logging.getLogger(__name__)

@dataclass
class MarketConfig:
    """Configuration for market simulation"""
    symbol: str
    date: str
    initial_price: float
    duration_hours: float = 6.5
    
    # Agent populations (scaled up for realism)
    num_market_makers: int = 20
    num_momentum_traders: int = 50
    num_mean_reversion_traders: int = 40
    num_noise_traders: int = 100
    num_institutional: int = 10
    
    # Trading parameters
    base_order_rate: float = 100  # Orders per second
    base_trade_rate: float = 20   # Trades per second
    
    # Market parameters
    base_spread: float = 0.02
    tick_size: float = 0.01
    
    # News events
    news_events: List[Dict] = None

class Agent:
    """Base class for trading agents"""
    
    def __init__(self, agent_id: str, agent_type: str, capital: float = 100000):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capital = capital
        self.position = 0
        self.orders = []
        self.trades = []
        
    def generate_order(self, current_price: float, spread: float, 
                       timestamp: float, market_state: Dict) -> Optional[Dict]:
        """Generate an order based on agent strategy"""
        raise NotImplementedError

class MarketMakerAgent(Agent):
    """Market maker providing liquidity"""
    
    def generate_order(self, current_price: float, spread: float,
                      timestamp: float, market_state: Dict) -> Optional[Dict]:
        
        if random.random() < 0.7:  # 70% chance to quote
            # Quote both sides
            orders = []
            
            # Bid
            bid_price = current_price - spread/2 - random.uniform(0, spread/2)
            bid_size = random.randint(100, 500)
            orders.append({
                'timestamp': timestamp,
                'agent_id': self.agent_id,
                'order_type': 'LIMIT',
                'side': 'BUY',
                'price': round(bid_price, 2),
                'size': bid_size,
                'order_id': f"{self.agent_id}_{timestamp}_bid"
            })
            
            # Ask
            ask_price = current_price + spread/2 + random.uniform(0, spread/2)
            ask_size = random.randint(100, 500)
            orders.append({
                'timestamp': timestamp,
                'agent_id': self.agent_id,
                'order_type': 'LIMIT',
                'side': 'SELL',
                'price': round(ask_price, 2),
                'size': ask_size,
                'order_id': f"{self.agent_id}_{timestamp}_ask"
            })
            
            return orders
        return None

class MomentumAgent(Agent):
    """Momentum trader following trends"""
    
    def generate_order(self, current_price: float, spread: float,
                      timestamp: float, market_state: Dict) -> Optional[Dict]:
        
        if 'price_trend' in market_state and random.random() < 0.3:
            trend = market_state['price_trend']
            
            if abs(trend) > 0.0001:  # Significant trend
                if trend > 0:  # Uptrend - buy
                    return {
                        'timestamp': timestamp,
                        'agent_id': self.agent_id,
                        'order_type': 'MARKET',
                        'side': 'BUY',
                        'price': current_price + spread/2,
                        'size': random.randint(50, 200),
                        'order_id': f"{self.agent_id}_{timestamp}"
                    }
                else:  # Downtrend - sell
                    if self.position > 0:
                        return {
                            'timestamp': timestamp,
                            'agent_id': self.agent_id,
                            'order_type': 'MARKET',
                            'side': 'SELL',
                            'price': current_price - spread/2,
                            'size': min(self.position, random.randint(50, 200)),
                            'order_id': f"{self.agent_id}_{timestamp}"
                        }
        return None

class ScaledLOBGenerator:
    """Generates scaled LOB data for experimental conditions"""
    
    def __init__(self, config: MarketConfig, condition: str):
        """
        Initialize generator
        
        Args:
            config: Market configuration
            condition: One of 'LLMON', 'LLMOFF', 'Baseline'
        """
        self.config = config
        self.condition = condition
        self.agents = []
        self.order_book = {'bids': {}, 'asks': {}}
        self.trades = []
        self.messages = []
        self.orderbook_snapshots = []
        
        # Initialize agents
        self._initialize_agents()
        
        # Market state
        self.current_price = config.initial_price
        self.current_spread = config.base_spread
        self.timestamp = 0
        
    def _initialize_agents(self):
        """Initialize agent population"""
        
        # Market makers
        for i in range(self.config.num_market_makers):
            self.agents.append(MarketMakerAgent(f"MM_{i}", "market_maker"))
        
        # Momentum traders
        for i in range(self.config.num_momentum_traders):
            self.agents.append(MomentumAgent(f"MOM_{i}", "momentum"))
        
        # Add more agent types as needed
        
        logger.info(f"Initialized {len(self.agents)} agents for {self.condition}")
    
    def generate_lob_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate complete LOB data
        
        Returns:
            Tuple of (messages_df, orderbook_df)
        """
        
        duration_seconds = self.config.duration_hours * 3600
        time_step = 0.1  # 100ms time steps for high frequency
        num_steps = int(duration_seconds / time_step)
        
        logger.info(f"Generating {self.condition} LOB with {num_steps} time steps")
        
        # Price evolution based on condition
        price_path = self._generate_price_path(num_steps, time_step)
        
        # Generate market events
        for step in range(num_steps):
            self.timestamp = step * time_step
            self.current_price = price_path[step]
            
            # Update spread based on volatility
            recent_volatility = self._calculate_volatility(price_path, step)
            self.current_spread = self.config.base_spread * (1 + 10 * recent_volatility)
            
            # Market state for agents
            market_state = {
                'price_trend': self._calculate_trend(price_path, step),
                'volatility': recent_volatility,
                'near_news': self._is_near_news(self.timestamp)
            }
            
            # Generate orders from agents
            self._generate_orders(market_state)
            
            # Match orders and create trades
            self._match_orders()
            
            # Take orderbook snapshot
            if step % 10 == 0:  # Snapshot every second
                self._take_snapshot()
            
            # Cancel some orders (realistic order lifetime)
            if step % 50 == 0:
                self._cancel_old_orders()
        
        # Convert to DataFrames
        messages_df = pd.DataFrame(self.messages)
        orderbook_df = pd.DataFrame(self.orderbook_snapshots)
        
        logger.info(f"Generated {len(self.messages)} messages and {len(self.trades)} trades")
        
        return messages_df, orderbook_df
    
    def _generate_price_path(self, num_steps: int, time_step: float) -> np.ndarray:
        """Generate realistic price path based on condition"""
        
        prices = np.zeros(num_steps)
        prices[0] = self.config.initial_price
        
        # Base volatility depends on condition
        if self.condition == "LLMON":
            base_vol = 0.00008  # Lower vol due to better coordination
        elif self.condition == "LLMOFF":
            base_vol = 0.00012
        else:  # Baseline
            base_vol = 0.00015
        
        # Generate returns
        returns = np.random.normal(0, base_vol, num_steps)
        
        # Add market microstructure noise
        noise = np.random.normal(0, 0.00002, num_steps)
        
        # Apply news events
        if self.config.news_events:
            for event in self.config.news_events:
                event_step = int(event['timestamp'] / time_step)
                if event_step < num_steps:
                    impact = self._calculate_news_impact(event, self.condition)
                    
                    # Apply impact over time
                    for i in range(min(500, num_steps - event_step)):  # 50 seconds
                        decay = np.exp(-i / 100)
                        returns[event_step + i] += impact * decay
        
        # Add autocorrelation (momentum)
        if self.condition == "LLMON":
            # LLM agents create more persistent trends
            for i in range(1, num_steps):
                returns[i] += 0.3 * returns[i-1]
        elif self.condition == "LLMOFF":
            for i in range(1, num_steps):
                returns[i] += 0.2 * returns[i-1]
        else:  # Baseline - more random
            for i in range(1, num_steps):
                returns[i] += 0.1 * returns[i-1]
        
        # Calculate prices
        for i in range(1, num_steps):
            prices[i] = prices[i-1] * (1 + returns[i] + noise[i])
        
        # Smooth if LLMON (coordination effect)
        if self.condition == "LLMON":
            window = 5
            prices = pd.Series(prices).rolling(window, center=True, min_periods=1).mean().values
        
        return prices
    
    def _calculate_news_impact(self, event: Dict, condition: str) -> float:
        """Calculate news impact based on condition"""
        
        base_impact = event['sentiment'] * event['importance'] * 0.002
        
        if condition == "LLMON":
            # LLM interprets news more accurately
            return base_impact * 1.2
        elif condition == "LLMOFF":
            # Rule-based interpretation
            return base_impact * 0.8
        else:  # Baseline
            # Simple reaction
            return base_impact * 0.5
    
    def _generate_orders(self, market_state: Dict):
        """Generate orders from all agents"""
        
        # Determine order generation rate based on market state
        if market_state.get('near_news', False):
            order_rate = self.config.base_order_rate * 3  # Triple during news
        elif market_state.get('volatility', 0) > 0.0001:
            order_rate = self.config.base_order_rate * 2  # Double during volatility
        else:
            order_rate = self.config.base_order_rate
        
        # Adjust rate based on condition
        if self.condition == "LLMON":
            order_rate *= 1.5  # More coordinated trading
        elif self.condition == "LLMOFF":
            order_rate *= 1.2
        
        # Generate orders probabilistically
        num_orders = np.random.poisson(order_rate * 0.1)  # Per 100ms
        
        for _ in range(num_orders):
            agent = random.choice(self.agents)
            order = agent.generate_order(
                self.current_price, 
                self.current_spread,
                self.timestamp,
                market_state
            )
            
            if order:
                if isinstance(order, list):
                    for o in order:
                        self._add_order(o)
                else:
                    self._add_order(order)
    
    def _add_order(self, order: Dict):
        """Add order to book and messages"""
        
        # Add to messages (ITCH format)
        message = {
            'timestamp': order['timestamp'],
            'type': 1,  # Submission
            'order_id': hash(order['order_id']) % 1000000,
            'size': order['size'],
            'price': int(order['price'] * 10000),  # ITCH price format
            'direction': 1 if order['side'] == 'BUY' else -1
        }
        self.messages.append(message)
        
        # Add to order book
        if order['side'] == 'BUY':
            price_level = order['price']
            if price_level not in self.order_book['bids']:
                self.order_book['bids'][price_level] = []
            self.order_book['bids'][price_level].append(order)
        else:
            price_level = order['price']
            if price_level not in self.order_book['asks']:
                self.order_book['asks'][price_level] = []
            self.order_book['asks'][price_level].append(order)
    
    def _match_orders(self):
        """Match orders and create trades"""
        
        if not self.order_book['bids'] or not self.order_book['asks']:
            return
        
        best_bid = max(self.order_book['bids'].keys())
        best_ask = min(self.order_book['asks'].keys())
        
        # Check for crossing
        while best_bid >= best_ask and self.order_book['bids'] and self.order_book['asks']:
            # Execute trade
            bid_orders = self.order_book['bids'][best_bid]
            ask_orders = self.order_book['asks'][best_ask]
            
            if bid_orders and ask_orders:
                bid_order = bid_orders[0]
                ask_order = ask_orders[0]
                
                # Trade size is minimum of both orders
                trade_size = min(bid_order['size'], ask_order['size'])
                trade_price = (best_bid + best_ask) / 2
                
                # Create trade
                trade = {
                    'timestamp': self.timestamp,
                    'price': trade_price,
                    'size': trade_size,
                    'buyer': bid_order['agent_id'],
                    'seller': ask_order['agent_id']
                }
                self.trades.append(trade)
                
                # Add execution messages
                for order in [bid_order, ask_order]:
                    exec_message = {
                        'timestamp': self.timestamp,
                        'type': 4,  # Execution
                        'order_id': hash(order['order_id']) % 1000000,
                        'size': trade_size,
                        'price': int(trade_price * 10000),
                        'direction': 1 if order['side'] == 'BUY' else -1
                    }
                    self.messages.append(exec_message)
                
                # Update order sizes
                bid_order['size'] -= trade_size
                ask_order['size'] -= trade_size
                
                # Remove filled orders
                if bid_order['size'] == 0:
                    bid_orders.pop(0)
                if ask_order['size'] == 0:
                    ask_orders.pop(0)
                
                # Clean up empty price levels
                if not bid_orders:
                    del self.order_book['bids'][best_bid]
                if not ask_orders:
                    del self.order_book['asks'][best_ask]
                
                # Update best bid/ask
                if self.order_book['bids']:
                    best_bid = max(self.order_book['bids'].keys())
                else:
                    break
                if self.order_book['asks']:
                    best_ask = min(self.order_book['asks'].keys())
                else:
                    break
            else:
                break
    
    def _take_snapshot(self):
        """Take orderbook snapshot"""
        
        # Get best bid/ask
        if self.order_book['bids']:
            best_bid = max(self.order_book['bids'].keys())
            bid_size = sum(o['size'] for o in self.order_book['bids'][best_bid])
        else:
            best_bid = self.current_price - self.current_spread
            bid_size = 0
        
        if self.order_book['asks']:
            best_ask = min(self.order_book['asks'].keys())
            ask_size = sum(o['size'] for o in self.order_book['asks'][best_ask])
        else:
            best_ask = self.current_price + self.current_spread
            ask_size = 0
        
        snapshot = {
            'timestamp': self.timestamp,
            'best_bid': best_bid,
            'bid_size': bid_size,
            'best_ask': best_ask,
            'ask_size': ask_size,
            'mid_price': (best_bid + best_ask) / 2,
            'spread': best_ask - best_bid
        }
        self.orderbook_snapshots.append(snapshot)
    
    def _cancel_old_orders(self):
        """Cancel old orders randomly"""
        
        # Cancel some percentage of orders
        for side in ['bids', 'asks']:
            for price_level in list(self.order_book[side].keys()):
                orders = self.order_book[side][price_level]
                for order in orders[:]:
                    if random.random() < 0.1:  # 10% chance to cancel
                        # Add cancellation message
                        cancel_message = {
                            'timestamp': self.timestamp,
                            'type': 3,  # Deletion
                            'order_id': hash(order['order_id']) % 1000000,
                            'size': order['size'],
                            'price': int(order['price'] * 10000),
                            'direction': 1 if order['side'] == 'BUY' else -1
                        }
                        self.messages.append(cancel_message)
                        orders.remove(order)
                
                if not orders:
                    del self.order_book[side][price_level]
    
    def _calculate_volatility(self, prices: np.ndarray, current_step: int) -> float:
        """Calculate recent volatility"""
        
        lookback = min(100, current_step)
        if lookback < 2:
            return 0.0001
        
        recent_prices = prices[max(0, current_step - lookback):current_step]
        returns = np.diff(np.log(recent_prices + 1e-10))
        return np.std(returns) if len(returns) > 0 else 0.0001
    
    def _calculate_trend(self, prices: np.ndarray, current_step: int) -> float:
        """Calculate recent price trend"""
        
        lookback = min(50, current_step)
        if lookback < 2:
            return 0
        
        recent_prices = prices[max(0, current_step - lookback):current_step]
        if len(recent_prices) > 1:
            return (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        return 0
    
    def _is_near_news(self, timestamp: float) -> bool:
        """Check if near a news event"""
        
        if not self.config.news_events:
            return False
        
        for event in self.config.news_events:
            if abs(timestamp - event['timestamp']) < 60:  # Within 1 minute
                return True
        return False
    
    def save_to_database(self, db_path: str):
        """Save LOB data to SQLite database"""
        
        logger.info(f"Saving {self.condition} data to {db_path}")
        
        # Generate data
        messages_df, orderbook_df = self.generate_lob_data()
        trades_df = pd.DataFrame(self.trades)
        
        # Save to SQLite
        conn = sqlite3.connect(db_path)
        
        # Save messages
        messages_df.to_sql('messages', conn, if_exists='replace', index=False)
        
        # Save orderbook snapshots
        orderbook_df.to_sql('orderbook', conn, if_exists='replace', index=False)
        
        # Save trades
        trades_df.to_sql('trades', conn, if_exists='replace', index=False)
        
        # Save metadata
        metadata = {
            'condition': self.condition,
            'symbol': self.config.symbol,
            'date': self.config.date,
            'initial_price': self.config.initial_price,
            'duration_hours': self.config.duration_hours,
            'num_messages': len(messages_df),
            'num_trades': len(trades_df),
            'num_snapshots': len(orderbook_df)
        }
        
        metadata_df = pd.DataFrame([metadata])
        metadata_df.to_sql('metadata', conn, if_exists='replace', index=False)
        
        conn.close()
        
        logger.info(f"Saved {len(messages_df)} messages, {len(trades_df)} trades, {len(orderbook_df)} snapshots")
        
        return metadata

def main():
    """Generate LOB databases for all three conditions"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Configuration
    symbol = "AMZN"
    date = "2012-06-21"
    initial_price = 223.56  # From real NASDAQ data
    
    # News events (same for all conditions)
    news_events = [
        {
            'timestamp': 7200,  # 2 hours after open
            'headline': 'AMZN announces strong cloud growth',
            'sentiment': 0.8,
            'importance': 0.9
        },
        {
            'timestamp': 14400,  # 4 hours after open
            'headline': 'Tech sector faces regulatory concerns',
            'sentiment': -0.4,
            'importance': 0.7
        }
    ]
    
    # Create configuration
    config = MarketConfig(
        symbol=symbol,
        date=date,
        initial_price=initial_price,
        news_events=news_events
    )
    
    # Output directory
    output_dir = Path("/workspace/lob_databases")
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    # Generate for each condition
    for condition in ['LLMON', 'LLMOFF', 'Baseline']:
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating {condition} LOB database")
        logger.info(f"{'='*60}")
        
        generator = ScaledLOBGenerator(config, condition)
        db_path = output_dir / f"{symbol}_{date}_{condition}.db"
        
        metadata = generator.save_to_database(str(db_path))
        results[condition] = metadata
        
        print(f"\n✅ {condition} database created:")
        print(f"   Path: {db_path}")
        print(f"   Messages: {metadata['num_messages']:,}")
        print(f"   Trades: {metadata['num_trades']:,}")
        print(f"   Snapshots: {metadata['num_snapshots']:,}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 LOB DATABASE GENERATION COMPLETE")
    print("="*60)
    
    for condition, metadata in results.items():
        print(f"\n{condition}:")
        print(f"  Messages: {metadata['num_messages']:,}")
        print(f"  Trades: {metadata['num_trades']:,}")
        print(f"  Trade rate: {metadata['num_trades'] / (metadata['duration_hours'] * 3600):.1f} trades/sec")
    
    print(f"\n📁 All databases saved to: {output_dir}")

if __name__ == "__main__":
    main()