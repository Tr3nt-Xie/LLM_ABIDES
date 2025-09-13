#!/usr/bin/env python3
"""
Final LOB Generator combining:
1. Heterogeneous agents (diversity without over-dampening)
2. Realistic microstructure (multiple prices per timestamp)
3. Appropriate price responses to news (allowing ~3% moves)
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from enum import Enum
import random
from dataclasses import dataclass

class AgentType(Enum):
    """Different agent types with varying behaviors"""
    MARKET_MAKER = "market_maker"
    MOMENTUM_SMART = "momentum_smart"      # LLM-enhanced
    MOMENTUM_SIMPLE = "momentum_simple"    # Traditional
    CONTRARIAN = "contrarian"              # Buys dips, sells rallies
    MEAN_REVERSION = "mean_reversion"
    NOISE = "noise"
    INSTITUTIONAL = "institutional"

@dataclass
class Agent:
    """Agent with specific characteristics"""
    agent_type: AgentType
    intelligence: float  # 0-1, affects decision quality
    reaction_speed: float  # 0-1, how quickly responds to news
    risk_tolerance: float  # 0-1, position sizing

class OrderBook:
    """Realistic order book with multiple price levels"""
    
    def __init__(self, initial_price: float, tick_size: float = 0.01):
        self.tick_size = tick_size
        self.mid_price = initial_price
        self.spread = 0.02  # Start with 2 cent spread
        
    def get_best_bid(self) -> float:
        return round(self.mid_price - self.spread/2, 2)
    
    def get_best_ask(self) -> float:
        return round(self.mid_price + self.spread/2, 2)
    
    def update(self, new_mid_price: float, volatility: float):
        """Update order book state"""
        self.mid_price = new_mid_price
        # Spread widens with volatility
        self.spread = min(0.10, 0.01 + volatility * 10)
        self.spread = round(self.spread / self.tick_size) * self.tick_size

class FinalLOBGenerator:
    """
    Final generator with balanced parameters
    """
    
    def __init__(self, symbol: str, date: str, initial_price: float, 
                 duration_seconds: int, news_events: List[Dict], 
                 condition: str = "Baseline"):
        self.symbol = symbol
        self.date = date
        self.initial_price = initial_price
        self.duration_seconds = duration_seconds
        self.news_events = news_events
        self.condition = condition
        
        # Create heterogeneous agent population
        self.agents = self._create_agent_population()
        
        # Market parameters - balanced for realistic response
        self.base_volatility = 0.0002  # Slightly higher than before
        self.mean_reversion_strength = 0.1  # Moderate mean reversion
        self.max_price_change_per_second = 0.0005  # 0.05% per second max
        
        # News impact parameters - calibrated for ~3% total move for LLMON
        self.news_decay_rate = 0.0008  # Slower decay for sustained impact
        self.news_impact_multiplier = {
            "LLMON": 1.8,    # Smart agents react to news (~3% impact)
            "LLMOFF": 0.15,  # Traditional agents minimal reaction
            "Baseline": 0.05 # Baseline almost no news reaction
        }
        
        # Microstructure
        self.tick_size = 0.01
        self.order_book = OrderBook(initial_price, self.tick_size)
        self.trades_per_second_base = 2
        self.trades_per_second_active = 8
        
    def _create_agent_population(self) -> List[Agent]:
        """Create a diverse but reactive population"""
        agents = []
        
        if self.condition == "LLMON":
            # LLMON: Smart agents that react appropriately to news
            # 30% smart momentum traders (increased from 20%)
            for _ in range(6):
                agents.append(Agent(
                    agent_type=AgentType.MOMENTUM_SMART,
                    intelligence=np.random.uniform(0.7, 0.9),
                    reaction_speed=np.random.uniform(0.6, 0.8),
                    risk_tolerance=np.random.uniform(0.5, 0.7)
                ))
            
            # 15% contrarian traders (reduced from 20%)
            for _ in range(3):
                agents.append(Agent(
                    agent_type=AgentType.CONTRARIAN,
                    intelligence=np.random.uniform(0.5, 0.7),
                    reaction_speed=np.random.uniform(0.4, 0.6),
                    risk_tolerance=np.random.uniform(0.4, 0.6)
                ))
            
            # 35% simple momentum
            for _ in range(7):
                agents.append(Agent(
                    agent_type=AgentType.MOMENTUM_SIMPLE,
                    intelligence=np.random.uniform(0.3, 0.5),
                    reaction_speed=np.random.uniform(0.4, 0.6),
                    risk_tolerance=np.random.uniform(0.4, 0.6)
                ))
                
        elif self.condition == "LLMOFF":
            # LLMOFF: Traditional agents
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
            # Basic agents with limited sophistication
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
        
        # Add common stabilizing agents
        # Market makers (2)
        for _ in range(2):
            agents.append(Agent(
                agent_type=AgentType.MARKET_MAKER,
                intelligence=0.7,
                reaction_speed=0.8,
                risk_tolerance=0.7
            ))
        
        # Institutional (2)
        for _ in range(2):
            agents.append(Agent(
                agent_type=AgentType.INSTITUTIONAL,
                intelligence=0.8,
                reaction_speed=0.3,
                risk_tolerance=0.6
            ))
        
        return agents
    
    def _calculate_news_impact(self, timestamp: float) -> float:
        """Calculate cumulative news impact at given time"""
        total_impact = 0
        
        for news in self.news_events:
            time_since_news = timestamp - news['timestamp']
            if time_since_news >= 0:
                # Exponential decay but slower
                decay = np.exp(-self.news_decay_rate * time_since_news)
                
                # News impact based on sentiment and importance
                raw_impact = news['sentiment'] * news.get('importance', 0.5)
                
                # Apply condition-specific multiplier
                multiplier = self.news_impact_multiplier.get(self.condition, 1.0)
                
                total_impact += raw_impact * decay * multiplier
        
        return total_impact
    
    def _calculate_agent_pressure(self, current_price: float, fundamental_price: float,
                                 news_impact: float, momentum: float) -> float:
        """Calculate aggregate trading pressure from all agents"""
        total_pressure = 0
        active_agents = 0
        
        price_deviation = (current_price - fundamental_price) / fundamental_price
        
        for agent in self.agents:
            # Agents act probabilistically
            if random.random() < agent.reaction_speed:
                pressure = 0
                
                if agent.agent_type == AgentType.MARKET_MAKER:
                    # Provide liquidity, dampen extreme moves
                    if abs(price_deviation) > 0.02:
                        pressure = -np.sign(price_deviation) * 0.3
                        
                elif agent.agent_type == AgentType.MOMENTUM_SMART:
                    # Smart agents consider news heavily
                    pressure = (momentum * 0.3 + news_impact * 0.7) * agent.intelligence
                    
                elif agent.agent_type == AgentType.MOMENTUM_SIMPLE:
                    # Simple momentum followers
                    pressure = momentum * 0.8 + news_impact * 0.2
                    
                elif agent.agent_type == AgentType.CONTRARIAN:
                    # Trade against extreme moves
                    if abs(price_deviation) > 0.015:
                        pressure = -np.sign(price_deviation) * 0.5
                        
                elif agent.agent_type == AgentType.MEAN_REVERSION:
                    # Always push toward fundamental
                    pressure = -price_deviation * 3
                    
                elif agent.agent_type == AgentType.INSTITUTIONAL:
                    # Slow, fundamental-based
                    if random.random() < 0.1:  # Act rarely
                        pressure = -price_deviation * 2 + news_impact * 0.3
                        
                else:  # NOISE
                    pressure = np.random.normal(0, 0.5)
                
                # Apply risk tolerance
                pressure *= agent.risk_tolerance
                
                total_pressure += np.clip(pressure, -1, 1)
                active_agents += 1
        
        if active_agents > 0:
            return total_pressure / active_agents
        return 0
    
    def generate_price_path(self) -> np.ndarray:
        """Generate realistic price path with appropriate news response"""
        num_steps = self.duration_seconds
        prices = np.zeros(num_steps)
        prices[0] = self.initial_price
        
        fundamental_price = self.initial_price
        momentum = 0
        
        for i in range(1, num_steps):
            timestamp = i
            
            # Calculate market forces
            news_impact = self._calculate_news_impact(timestamp)
            
            # Update momentum (10-second window)
            if i > 10:
                recent_return = (prices[i-1] - prices[i-10]) / prices[i-10]
                momentum = 0.8 * momentum + 0.2 * recent_return * 20
            
            # Get agent consensus
            agent_pressure = self._calculate_agent_pressure(
                prices[i-1], fundamental_price, news_impact, momentum
            )
            
            # Price components
            random_walk = np.random.normal(0, self.base_volatility)
            news_component = news_impact * 0.0006  # Moderate news impact
            agent_component = agent_pressure * 0.0003  # Agent reactions
            mean_reversion = -(prices[i-1] - fundamental_price) / fundamental_price * self.mean_reversion_strength * 0.001
            
            # Total return
            total_return = random_walk + news_component + agent_component + mean_reversion
            
            # Limit extreme moves per second
            total_return = np.clip(total_return, -self.max_price_change_per_second, 
                                  self.max_price_change_per_second)
            
            # Update price
            prices[i] = prices[i-1] * (1 + total_return)
            
            # Slowly adjust fundamental based on persistent news
            if abs(news_impact) > 0.1:
                fundamental_price = fundamental_price * (1 + news_impact * 0.00005)
        
        return prices
    
    def generate_trades_with_microstructure(self, timestamp: float, price: float,
                                           volatility: float) -> List[Dict]:
        """Generate trades with realistic microstructure"""
        trades = []
        
        # Update order book
        self.order_book.update(price, volatility)
        best_bid = self.order_book.get_best_bid()
        best_ask = self.order_book.get_best_ask()
        
        # Determine trade intensity
        if volatility > 0.0002:
            trades_per_second = self.trades_per_second_active
        else:
            trades_per_second = self.trades_per_second_base
        
        # Number of trades (Poisson)
        num_trades = np.random.poisson(trades_per_second)
        
        if num_trades == 0:
            return trades
        
        # Generate sub-second timestamps
        sub_times = sorted(np.random.uniform(0, 1, num_trades))
        
        for sub_time in sub_times:
            exact_time = timestamp + sub_time
            
            # Determine trade price based on order type distribution
            rand = random.random()
            if rand < 0.4:  # Market order
                if random.random() < 0.5:
                    trade_price = best_ask  # Buy at ask
                    side = 'B'
                else:
                    trade_price = best_bid  # Sell at bid
                    side = 'S'
            elif rand < 0.7:  # Aggressive limit
                if random.random() < 0.5:
                    trade_price = best_ask + random.choice([0, self.tick_size])
                    side = 'B'
                else:
                    trade_price = best_bid - random.choice([0, self.tick_size])
                    side = 'S'
            else:  # Passive limit or price improvement
                if random.random() < 0.5:
                    trade_price = best_bid + random.choice([0, self.tick_size])
                    side = 'B'
                else:
                    trade_price = best_ask - random.choice([0, self.tick_size])
                    side = 'S'
            
            # Ensure reasonable price
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
        """Generate complete LOB data and save to database"""
        print(f"Generating {self.condition} LOB (Final Version)...")
        
        # Generate price path
        prices = self.generate_price_path()
        
        # Generate all trades
        all_trades = []
        trade_id = 1
        
        for i in range(self.duration_seconds):
            if i > 0:
                volatility = abs(prices[i] - prices[i-1]) / prices[i-1]
            else:
                volatility = 0
            
            trades = self.generate_trades_with_microstructure(i, prices[i], volatility)
            
            for trade in trades:
                trade['trade_id'] = trade_id
                trade_id += 1
                all_trades.append(trade)
        
        # Save to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
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
        for trade in all_trades:
            cursor.execute('''
                INSERT INTO trades (timestamp, trade_id, price, size, side)
                VALUES (?, ?, ?, ?, ?)
            ''', (trade['timestamp'], trade['trade_id'], trade['price'], 
                  trade['size'], trade['side']))
        
        # Insert orderbook snapshots
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
        print(f"  Generated {len(all_trades)} trades")
        print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"  Final price: ${prices[-1]:.2f} ({price_change:+.2f}%)")
        print(f"  Saved to {db_path}")

def main():
    """Generate final LOB databases"""
    # Real news events from 2012-06-21
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
    duration_seconds = 23400  # 6.5 hours
    
    # Output directory
    import os
    output_dir = "/workspace/lob_databases_final"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate for each condition
    for condition in ["LLMON", "LLMOFF", "Baseline"]:
        generator = FinalLOBGenerator(
            symbol=symbol,
            date=date,
            initial_price=initial_price,
            duration_seconds=duration_seconds,
            news_events=news_events,
            condition=condition
        )
        
        db_path = f"{output_dir}/{symbol}_{date}_{condition}_final.db"
        generator.generate_and_save(db_path)
    
    print("\n✅ All final LOB databases generated successfully!")

if __name__ == "__main__":
    main()