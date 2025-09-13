#!/usr/bin/env python3
"""
Calibrated LOB Generator with realistic trade volumes matching real market
"""

import numpy as np
import pandas as pd
import sqlite3
from typing import List, Dict, Tuple
import random
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    """Different agent types"""
    MARKET_MAKER = "market_maker"
    MOMENTUM_SMART = "momentum_smart"
    MOMENTUM_SIMPLE = "momentum_simple"
    CONTRARIAN = "contrarian"
    MEAN_REVERSION = "mean_reversion"
    NOISE = "noise"
    INSTITUTIONAL = "institutional"

@dataclass
class Agent:
    """Agent with specific characteristics"""
    agent_type: AgentType
    intelligence: float
    reaction_speed: float
    risk_tolerance: float

class OrderBook:
    """Simple order book"""
    def __init__(self, initial_price: float, tick_size: float = 0.01):
        self.tick_size = tick_size
        self.mid_price = initial_price
        self.spread = 0.02
        
    def get_best_bid(self) -> float:
        return round(self.mid_price - self.spread/2, 2)
    
    def get_best_ask(self) -> float:
        return round(self.mid_price + self.spread/2, 2)
    
    def update(self, new_mid_price: float, volatility: float):
        self.mid_price = new_mid_price
        self.spread = min(0.10, 0.01 + volatility * 10)
        self.spread = round(self.spread / self.tick_size) * self.tick_size

class CalibratedLOBGenerator:
    """
    Generator calibrated to match real market statistics
    """
    
    def __init__(self, symbol: str, date: str, initial_price: float, 
                 duration_seconds: int, news_events: List[Dict], 
                 condition: str = "Baseline", target_trades: int = 11419):
        """
        Initialize with target trade count matching real market
        
        Args:
            target_trades: Target number of trades (default from real NASDAQ)
        """
        self.symbol = symbol
        self.date = date
        self.initial_price = initial_price
        self.duration_seconds = duration_seconds
        self.news_events = news_events
        self.condition = condition
        self.target_trades = target_trades
        
        # Calculate trade rate to match target
        self.avg_trades_per_second = target_trades / duration_seconds
        
        # Create heterogeneous agents
        self.agents = self._create_agent_population()
        
        # Market parameters
        self.base_volatility = 0.0002
        self.mean_reversion_strength = 0.1
        self.max_price_change_per_second = 0.0005
        
        # News impact calibrated for conditions
        self.news_decay_rate = 0.0008
        self.news_impact_multiplier = {
            "LLMON": 1.8,
            "LLMOFF": 0.15,
            "Baseline": 0.05
        }
        
        # Microstructure
        self.tick_size = 0.01
        self.order_book = OrderBook(initial_price, self.tick_size)
        
        # Trade generation parameters calibrated to target
        self.base_trade_rate = self.avg_trades_per_second * 0.8  # Normal periods
        self.active_trade_rate = self.avg_trades_per_second * 1.5  # Active periods
        
        print(f"  Target trades: {target_trades:,}")
        print(f"  Avg trades/second: {self.avg_trades_per_second:.2f}")
    
    def _create_agent_population(self) -> List[Agent]:
        """Create diverse agent population based on condition"""
        agents = []
        
        if self.condition == "LLMON":
            # Smart agents for LLMON
            for _ in range(6):
                agents.append(Agent(
                    agent_type=AgentType.MOMENTUM_SMART,
                    intelligence=np.random.uniform(0.7, 0.9),
                    reaction_speed=np.random.uniform(0.6, 0.8),
                    risk_tolerance=np.random.uniform(0.5, 0.7)
                ))
            for _ in range(3):
                agents.append(Agent(
                    agent_type=AgentType.CONTRARIAN,
                    intelligence=np.random.uniform(0.5, 0.7),
                    reaction_speed=np.random.uniform(0.4, 0.6),
                    risk_tolerance=np.random.uniform(0.4, 0.6)
                ))
            for _ in range(7):
                agents.append(Agent(
                    agent_type=AgentType.MOMENTUM_SIMPLE,
                    intelligence=np.random.uniform(0.3, 0.5),
                    reaction_speed=np.random.uniform(0.4, 0.6),
                    risk_tolerance=np.random.uniform(0.4, 0.6)
                ))
                
        elif self.condition == "LLMOFF":
            # Traditional agents
            for _ in range(10):
                agents.append(Agent(
                    agent_type=AgentType.MOMENTUM_SIMPLE,
                    intelligence=np.random.uniform(0.3, 0.5),
                    reaction_speed=np.random.uniform(0.3, 0.5),
                    risk_tolerance=np.random.uniform(0.4, 0.6)
                ))
            for _ in range(3):
                agents.append(Agent(
                    agent_type=AgentType.CONTRARIAN,
                    intelligence=np.random.uniform(0.4, 0.6),
                    reaction_speed=np.random.uniform(0.3, 0.5),
                    risk_tolerance=np.random.uniform(0.4, 0.6)
                ))
                
        else:  # Baseline
            for _ in range(8):
                agents.append(Agent(
                    agent_type=AgentType.NOISE,
                    intelligence=np.random.uniform(0.2, 0.4),
                    reaction_speed=np.random.uniform(0.2, 0.4),
                    risk_tolerance=np.random.uniform(0.3, 0.5)
                ))
            for _ in range(5):
                agents.append(Agent(
                    agent_type=AgentType.MEAN_REVERSION,
                    intelligence=np.random.uniform(0.3, 0.5),
                    reaction_speed=np.random.uniform(0.3, 0.5),
                    risk_tolerance=np.random.uniform(0.3, 0.5)
                ))
        
        # Add common agents
        for _ in range(2):
            agents.append(Agent(
                agent_type=AgentType.MARKET_MAKER,
                intelligence=0.7,
                reaction_speed=0.8,
                risk_tolerance=0.7
            ))
        for _ in range(2):
            agents.append(Agent(
                agent_type=AgentType.INSTITUTIONAL,
                intelligence=0.8,
                reaction_speed=0.3,
                risk_tolerance=0.6
            ))
        
        return agents
    
    def _calculate_news_impact(self, timestamp: float) -> float:
        """Calculate news impact at given time"""
        total_impact = 0
        for news in self.news_events:
            time_since_news = timestamp - news['timestamp']
            if time_since_news >= 0:
                decay = np.exp(-self.news_decay_rate * time_since_news)
                raw_impact = news['sentiment'] * news.get('importance', 0.5)
                multiplier = self.news_impact_multiplier.get(self.condition, 1.0)
                total_impact += raw_impact * decay * multiplier
        return total_impact
    
    def _calculate_agent_pressure(self, current_price: float, fundamental_price: float,
                                 news_impact: float, momentum: float) -> float:
        """Calculate aggregate trading pressure"""
        total_pressure = 0
        active_agents = 0
        
        price_deviation = (current_price - fundamental_price) / fundamental_price
        
        for agent in self.agents:
            if random.random() < agent.reaction_speed:
                pressure = 0
                
                if agent.agent_type == AgentType.MARKET_MAKER:
                    if abs(price_deviation) > 0.02:
                        pressure = -np.sign(price_deviation) * 0.3
                elif agent.agent_type == AgentType.MOMENTUM_SMART:
                    pressure = (momentum * 0.3 + news_impact * 0.7) * agent.intelligence
                elif agent.agent_type == AgentType.MOMENTUM_SIMPLE:
                    pressure = momentum * 0.8 + news_impact * 0.2
                elif agent.agent_type == AgentType.CONTRARIAN:
                    if abs(price_deviation) > 0.015:
                        pressure = -np.sign(price_deviation) * 0.5
                elif agent.agent_type == AgentType.MEAN_REVERSION:
                    pressure = -price_deviation * 3
                elif agent.agent_type == AgentType.INSTITUTIONAL:
                    if random.random() < 0.1:
                        pressure = -price_deviation * 2 + news_impact * 0.3
                else:  # NOISE
                    pressure = np.random.normal(0, 0.5)
                
                pressure *= agent.risk_tolerance
                total_pressure += np.clip(pressure, -1, 1)
                active_agents += 1
        
        if active_agents > 0:
            return total_pressure / active_agents
        return 0
    
    def generate_price_path(self) -> np.ndarray:
        """Generate price path"""
        num_steps = self.duration_seconds
        prices = np.zeros(num_steps)
        prices[0] = self.initial_price
        
        fundamental_price = self.initial_price
        momentum = 0
        
        for i in range(1, num_steps):
            timestamp = i
            
            news_impact = self._calculate_news_impact(timestamp)
            
            if i > 10:
                recent_return = (prices[i-1] - prices[i-10]) / prices[i-10]
                momentum = 0.8 * momentum + 0.2 * recent_return * 20
            
            agent_pressure = self._calculate_agent_pressure(
                prices[i-1], fundamental_price, news_impact, momentum
            )
            
            # Price components
            random_walk = np.random.normal(0, self.base_volatility)
            news_component = news_impact * 0.0006
            agent_component = agent_pressure * 0.0003
            mean_reversion = -(prices[i-1] - fundamental_price) / fundamental_price * self.mean_reversion_strength * 0.001
            
            total_return = random_walk + news_component + agent_component + mean_reversion
            total_return = np.clip(total_return, -self.max_price_change_per_second, 
                                  self.max_price_change_per_second)
            
            prices[i] = prices[i-1] * (1 + total_return)
            
            if abs(news_impact) > 0.1:
                fundamental_price = fundamental_price * (1 + news_impact * 0.00005)
        
        return prices
    
    def generate_trades_with_controlled_volume(self, timestamp: float, price: float,
                                              volatility: float, trades_so_far: int) -> List[Dict]:
        """Generate trades with volume control to match target"""
        trades = []
        
        # Calculate how many trades we should have by now
        expected_trades = (timestamp / self.duration_seconds) * self.target_trades
        trades_deficit = expected_trades - trades_so_far
        
        # Adjust trade rate based on deficit
        if trades_deficit > 0:
            # We're behind target, increase rate
            adjustment_factor = 1 + (trades_deficit / self.target_trades) * 2
        else:
            # We're ahead, decrease rate
            adjustment_factor = 0.8
        
        # Determine base rate for this second
        if volatility > 0.0002:
            base_rate = self.active_trade_rate
        else:
            base_rate = self.base_trade_rate
        
        # Apply adjustment
        adjusted_rate = base_rate * adjustment_factor
        
        # Generate trades (Poisson distributed)
        num_trades = np.random.poisson(adjusted_rate)
        
        if num_trades == 0:
            return trades
        
        # Update order book
        self.order_book.update(price, volatility)
        best_bid = self.order_book.get_best_bid()
        best_ask = self.order_book.get_best_ask()
        
        # Generate sub-second timestamps
        sub_times = sorted(np.random.uniform(0, 1, num_trades))
        
        for sub_time in sub_times:
            exact_time = timestamp + sub_time
            
            # Determine trade price based on order type
            rand = random.random()
            if rand < 0.4:  # Market order
                if random.random() < 0.5:
                    trade_price = best_ask
                    side = 'B'
                else:
                    trade_price = best_bid
                    side = 'S'
            elif rand < 0.7:  # Aggressive limit
                if random.random() < 0.5:
                    trade_price = best_ask + random.choice([0, self.tick_size])
                    side = 'B'
                else:
                    trade_price = best_bid - random.choice([0, self.tick_size])
                    side = 'S'
            else:  # Passive limit
                if random.random() < 0.5:
                    trade_price = best_bid + random.choice([0, self.tick_size])
                    side = 'B'
                else:
                    trade_price = best_ask - random.choice([0, self.tick_size])
                    side = 'S'
            
            trade_price = max(price * 0.98, min(price * 1.02, trade_price))
            trade_price = round(trade_price, 2)
            
            # Trade size
            if random.random() < 0.8:
                size = random.choice([100, 200, 300, 400, 500])
            else:
                size = random.randint(600, 2000)
            
            trades.append({
                'timestamp': exact_time,
                'price': trade_price,
                'size': size,
                'side': side
            })
        
        return trades
    
    def generate_and_save(self, db_path: str):
        """Generate and save LOB data with controlled trade volume"""
        print(f"Generating {self.condition} LOB (Calibrated to real market volume)...")
        
        # Generate price path
        prices = self.generate_price_path()
        
        # Generate trades with volume control
        all_trades = []
        trade_id = 1
        
        for i in range(self.duration_seconds):
            if i > 0:
                volatility = abs(prices[i] - prices[i-1]) / prices[i-1]
            else:
                volatility = 0
            
            trades = self.generate_trades_with_controlled_volume(
                i, prices[i], volatility, len(all_trades)
            )
            
            for trade in trades:
                trade['trade_id'] = trade_id
                trade_id += 1
                all_trades.append(trade)
        
        # Save to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
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
        
        for trade in all_trades:
            cursor.execute('''
                INSERT INTO trades (timestamp, trade_id, price, size, side)
                VALUES (?, ?, ?, ?, ?)
            ''', (trade['timestamp'], trade['trade_id'], trade['price'], 
                  trade['size'], trade['side']))
        
        for i in range(self.duration_seconds):
            self.order_book.update(prices[i], 0.0001)
            cursor.execute('''
                INSERT INTO orderbook (timestamp, bid_price_1, bid_size_1, 
                                      ask_price_1, ask_size_1, mid_price, spread)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (i, self.order_book.get_best_bid(), random.randint(100, 1000),
                  self.order_book.get_best_ask(), random.randint(100, 1000),
                  prices[i], self.order_book.spread))
        
        conn.commit()
        conn.close()
        
        # Print summary
        price_change = (prices[-1] / prices[0] - 1) * 100
        print(f"  Generated {len(all_trades):,} trades (target: {self.target_trades:,})")
        print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"  Final price: ${prices[-1]:.2f} ({price_change:+.2f}%)")
        print(f"  Saved to {db_path}")

def main():
    """Generate calibrated LOB databases"""
    # Real news events
    news_events = [
        {'timestamp': 3600, 'sentiment': -0.3, 'importance': 0.6},
        {'timestamp': 7200, 'sentiment': -0.4, 'importance': 0.7},
        {'timestamp': 10800, 'sentiment': -0.2, 'importance': 0.5},
        {'timestamp': 14400, 'sentiment': -0.1, 'importance': 0.4},
        {'timestamp': 18000, 'sentiment': 0.2, 'importance': 0.5}
    ]
    
    # Parameters
    symbol = "AMZN"
    date = "2012-06-21"
    initial_price = 223.56
    duration_seconds = 23400
    target_trades = 11419  # From real NASDAQ data
    
    # Output directory
    import os
    output_dir = "/workspace/lob_databases_calibrated_volume"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate for each condition
    for condition in ["LLMON", "LLMOFF", "Baseline"]:
        generator = CalibratedLOBGenerator(
            symbol=symbol,
            date=date,
            initial_price=initial_price,
            duration_seconds=duration_seconds,
            news_events=news_events,
            condition=condition,
            target_trades=target_trades
        )
        
        db_path = f"{output_dir}/{symbol}_{date}_{condition}_calibrated.db"
        generator.generate_and_save(db_path)
    
    print("\n✅ All calibrated LOB databases generated successfully!")

if __name__ == "__main__":
    main()