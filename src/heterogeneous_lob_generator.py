#!/usr/bin/env python3
"""
Heterogeneous LOB Generator with improved agent diversity and market stability.
Fixes the LLMON cascade effect by introducing:
1. Agent heterogeneity (different intelligence levels)
2. Contrarian traders
3. Market stabilizers
4. Better news impact decay
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
    LIQUIDITY_PROVIDER = "liquidity_provider"  # Stabilizes spreads

@dataclass
class Agent:
    """Agent with specific characteristics"""
    agent_type: AgentType
    intelligence: float  # 0-1, affects decision quality
    reaction_speed: float  # 0-1, how quickly responds to news
    risk_tolerance: float  # 0-1, position sizing
    contrarian_threshold: float  # price deviation to trigger contrarian

class HeterogeneousLOBGenerator:
    """
    Generates realistic LOB with heterogeneous agents to prevent cascade effects
    """
    
    def __init__(self, symbol: str, date: str, initial_price: float, 
                 duration_seconds: int, news_events: List[Dict], 
                 condition: str = "Baseline"):
        """
        Initialize the LOB generator with heterogeneous agents
        
        Args:
            symbol: Stock symbol
            date: Date string
            initial_price: Starting price
            duration_seconds: Simulation duration
            news_events: List of news events with timestamp and sentiment
            condition: "LLMON", "LLMOFF", or "Baseline"
        """
        self.symbol = symbol
        self.date = date
        self.initial_price = initial_price
        self.duration_seconds = duration_seconds
        self.news_events = news_events
        self.condition = condition
        
        # Create heterogeneous agent population
        self.agents = self._create_agent_population()
        
        # Market parameters (calibrated)
        self.base_volatility = 0.0001  # Reduced base volatility
        self.mean_reversion_strength = 0.15  # Stronger mean reversion
        self.max_price_change = 0.02  # 2% circuit breaker
        
        # News impact parameters
        self.news_decay_rate = 0.001  # Faster decay
        self.max_news_impact = 0.005  # Maximum 0.5% impact per news
        
        # Agent activity parameters
        self.base_order_rate = 5.0
        self.base_trade_rate = 2.0
        
    def _create_agent_population(self) -> List[Agent]:
        """Create a diverse population of agents"""
        agents = []
        
        if self.condition == "LLMON":
            # LLMON: Mix of smart and simple agents
            # 20% very smart momentum traders
            for _ in range(4):
                agents.append(Agent(
                    agent_type=AgentType.MOMENTUM_SMART,
                    intelligence=np.random.uniform(0.8, 1.0),
                    reaction_speed=np.random.uniform(0.7, 0.9),
                    risk_tolerance=np.random.uniform(0.4, 0.6),
                    contrarian_threshold=0.03
                ))
            
            # 20% contrarian traders (stabilizing force)
            for _ in range(4):
                agents.append(Agent(
                    agent_type=AgentType.CONTRARIAN,
                    intelligence=np.random.uniform(0.6, 0.8),
                    reaction_speed=np.random.uniform(0.5, 0.7),
                    risk_tolerance=np.random.uniform(0.5, 0.7),
                    contrarian_threshold=np.random.uniform(0.01, 0.02)
                ))
            
            # 30% simple momentum traders
            for _ in range(6):
                agents.append(Agent(
                    agent_type=AgentType.MOMENTUM_SIMPLE,
                    intelligence=np.random.uniform(0.3, 0.5),
                    reaction_speed=np.random.uniform(0.3, 0.5),
                    risk_tolerance=np.random.uniform(0.3, 0.5),
                    contrarian_threshold=0.04
                ))
                
        elif self.condition == "LLMOFF":
            # LLMOFF: Only traditional agents
            # 40% simple momentum
            for _ in range(8):
                agents.append(Agent(
                    agent_type=AgentType.MOMENTUM_SIMPLE,
                    intelligence=np.random.uniform(0.3, 0.5),
                    reaction_speed=np.random.uniform(0.3, 0.5),
                    risk_tolerance=np.random.uniform(0.3, 0.5),
                    contrarian_threshold=0.04
                ))
            
            # 20% contrarian
            for _ in range(4):
                agents.append(Agent(
                    agent_type=AgentType.CONTRARIAN,
                    intelligence=np.random.uniform(0.4, 0.6),
                    reaction_speed=np.random.uniform(0.4, 0.6),
                    risk_tolerance=np.random.uniform(0.4, 0.6),
                    contrarian_threshold=0.02
                ))
                
        else:  # Baseline
            # Mix of basic agents
            for _ in range(12):
                agent_type = np.random.choice([
                    AgentType.MOMENTUM_SIMPLE,
                    AgentType.MEAN_REVERSION,
                    AgentType.NOISE
                ])
                agents.append(Agent(
                    agent_type=agent_type,
                    intelligence=np.random.uniform(0.2, 0.4),
                    reaction_speed=np.random.uniform(0.2, 0.4),
                    risk_tolerance=np.random.uniform(0.3, 0.5),
                    contrarian_threshold=0.03
                ))
        
        # Add common agents to all conditions
        # Market makers (always present for liquidity)
        for _ in range(2):
            agents.append(Agent(
                agent_type=AgentType.MARKET_MAKER,
                intelligence=0.7,
                reaction_speed=0.9,
                risk_tolerance=0.8,
                contrarian_threshold=0.005
            ))
        
        # Liquidity providers
        for _ in range(2):
            agents.append(Agent(
                agent_type=AgentType.LIQUIDITY_PROVIDER,
                intelligence=0.6,
                reaction_speed=0.8,
                risk_tolerance=0.9,
                contrarian_threshold=0.01
            ))
        
        # Institutional (slow but large)
        agents.append(Agent(
            agent_type=AgentType.INSTITUTIONAL,
            intelligence=0.8,
            reaction_speed=0.3,
            risk_tolerance=0.7,
            contrarian_threshold=0.02
        ))
        
        # Mean reversion traders
        for _ in range(3):
            agents.append(Agent(
                agent_type=AgentType.MEAN_REVERSION,
                intelligence=np.random.uniform(0.5, 0.7),
                reaction_speed=np.random.uniform(0.4, 0.6),
                risk_tolerance=np.random.uniform(0.4, 0.6),
                contrarian_threshold=0.015
            ))
        
        return agents
    
    def _calculate_agent_action(self, agent: Agent, current_price: float, 
                               fundamental_price: float, news_impact: float,
                               market_momentum: float) -> float:
        """
        Calculate individual agent's trading decision
        
        Returns:
            Trading pressure (-1 to 1, negative = sell, positive = buy)
        """
        price_deviation = (current_price - fundamental_price) / fundamental_price
        
        if agent.agent_type == AgentType.MARKET_MAKER:
            # Market makers provide liquidity, trade against extreme moves
            if abs(price_deviation) > 0.01:
                return -np.sign(price_deviation) * 0.5
            return 0
            
        elif agent.agent_type == AgentType.MOMENTUM_SMART:
            # Smart momentum traders consider both momentum and news
            signal = market_momentum * 0.6 + news_impact * agent.intelligence * 0.4
            # But they're smart enough to not chase extreme moves
            if abs(price_deviation) > 0.02:
                signal *= 0.5
            return np.clip(signal * agent.risk_tolerance, -1, 1)
            
        elif agent.agent_type == AgentType.MOMENTUM_SIMPLE:
            # Simple momentum just follows the trend
            return np.clip(market_momentum * agent.risk_tolerance, -1, 1)
            
        elif agent.agent_type == AgentType.CONTRARIAN:
            # Contrarians trade against the trend when deviation is large
            if abs(price_deviation) > agent.contrarian_threshold:
                return -np.sign(price_deviation) * agent.risk_tolerance
            return 0
            
        elif agent.agent_type == AgentType.MEAN_REVERSION:
            # Mean reversion always trades toward fundamental value
            return -price_deviation * 10 * agent.risk_tolerance
            
        elif agent.agent_type == AgentType.LIQUIDITY_PROVIDER:
            # Liquidity providers dampen volatility
            if abs(market_momentum) > 0.5:
                return -market_momentum * 0.3
            return 0
            
        elif agent.agent_type == AgentType.INSTITUTIONAL:
            # Institutional traders are slow but trade on fundamentals
            if random.random() < agent.reaction_speed:
                return -price_deviation * 5 * agent.risk_tolerance
            return 0
            
        else:  # NOISE
            # Random trading
            return np.random.normal(0, 0.3) * agent.risk_tolerance
    
    def _calculate_news_impact(self, timestamp: float) -> float:
        """
        Calculate aggregate news impact with proper decay
        """
        total_impact = 0
        
        for news in self.news_events:
            time_since_news = timestamp - news['timestamp']
            if time_since_news >= 0:
                # Exponential decay with faster rate
                decay = np.exp(-self.news_decay_rate * time_since_news)
                
                # News impact is capped and decays
                raw_impact = news['sentiment'] * news.get('importance', 0.5)
                capped_impact = np.sign(raw_impact) * min(abs(raw_impact), self.max_news_impact)
                total_impact += capped_impact * decay
        
        return total_impact
    
    def generate_price_path(self, num_steps: int) -> np.ndarray:
        """
        Generate price path with heterogeneous agent interactions
        """
        prices = np.zeros(num_steps)
        prices[0] = self.initial_price
        
        fundamental_price = self.initial_price
        momentum = 0
        
        for i in range(1, num_steps):
            timestamp = i * (self.duration_seconds / num_steps)
            
            # Calculate market state
            news_impact = self._calculate_news_impact(timestamp)
            price_deviation = (prices[i-1] - fundamental_price) / fundamental_price
            
            # Update momentum with decay
            if i > 10:
                recent_return = (prices[i-1] - prices[i-10]) / prices[i-10]
                momentum = 0.7 * momentum + 0.3 * recent_return * 10
            
            # Aggregate agent actions
            total_pressure = 0
            active_agents = 0
            
            for agent in self.agents:
                # Agents act probabilistically based on reaction speed
                if random.random() < agent.reaction_speed:
                    action = self._calculate_agent_action(
                        agent, prices[i-1], fundamental_price, 
                        news_impact, momentum
                    )
                    
                    # Weight by intelligence for smart agents
                    if agent.agent_type in [AgentType.MOMENTUM_SMART, AgentType.INSTITUTIONAL]:
                        action *= (0.5 + 0.5 * agent.intelligence)
                    
                    total_pressure += action
                    active_agents += 1
            
            # Average pressure from active agents
            if active_agents > 0:
                avg_pressure = total_pressure / active_agents
            else:
                avg_pressure = 0
            
            # Price update with multiple components
            random_component = np.random.normal(0, self.base_volatility)
            pressure_component = avg_pressure * 0.001  # Scale down agent impact
            reversion_component = -price_deviation * self.mean_reversion_strength * 0.001
            
            # Total return
            total_return = random_component + pressure_component + reversion_component
            
            # Apply circuit breaker
            total_return = np.clip(total_return, -self.max_price_change, self.max_price_change)
            
            # Update price (multiplicative)
            prices[i] = prices[i-1] * (1 + total_return)
            
            # Update fundamental price slowly based on news
            fundamental_price = fundamental_price * (1 + news_impact * 0.0001)
        
        return prices
    
    def generate_trades(self, prices: np.ndarray, num_steps: int) -> Tuple[List, List]:
        """
        Generate trades based on price path and agent activity
        """
        trades = []
        trade_id = 1
        
        for i in range(num_steps):
            timestamp = i * (self.duration_seconds / num_steps)
            
            # Number of trades depends on price volatility and agent activity
            if i > 0:
                price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
                volatility_factor = 1 + price_change * 100
            else:
                volatility_factor = 1
            
            # More trades during news events
            news_factor = 1 + abs(self._calculate_news_impact(timestamp)) * 5
            
            # Poisson-distributed number of trades
            num_trades = np.random.poisson(self.base_trade_rate * volatility_factor * news_factor)
            
            for _ in range(num_trades):
                # Trade price with small deviation from mid price
                spread = 0.01 + np.random.exponential(0.005)
                if np.random.random() < 0.5:
                    trade_price = prices[i] * (1 + spread/2)  # Buy
                    side = 'B'
                else:
                    trade_price = prices[i] * (1 - spread/2)  # Sell
                    side = 'S'
                
                # Trade size based on agent type (some agents trade larger)
                if np.random.random() < 0.1:  # Institutional size
                    size = np.random.randint(500, 2000)
                else:  # Regular size
                    size = np.random.randint(100, 500)
                
                trades.append({
                    'timestamp': timestamp,
                    'trade_id': trade_id,
                    'price': trade_price,
                    'size': size,
                    'side': side
                })
                trade_id += 1
        
        return trades, []  # Return trades and empty messages for now
    
    def generate_and_save(self, db_path: str):
        """
        Generate complete LOB data and save to database
        """
        print(f"Generating {self.condition} LOB with heterogeneous agents...")
        
        # Generate price path
        num_steps = self.duration_seconds
        prices = self.generate_price_path(num_steps)
        
        # Generate trades
        trades, _ = self.generate_trades(prices, num_steps)
        
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
            
            # Also insert as message (use trade_id as order_id to ensure uniqueness)
            cursor.execute('''
                INSERT INTO messages (timestamp, message_type, order_id, trade_id, price, size, side)
                VALUES (?, 'E', ?, ?, ?, ?, ?)
            ''', (trade['timestamp'], trade['trade_id'], trade['trade_id'], trade['price'], 
                  trade['size'], trade['side']))
        
        # Generate orderbook snapshots
        for i in range(num_steps):
            timestamp = i
            mid_price = prices[i]
            
            # Generate realistic spread
            base_spread = 0.01
            volatility_spread = np.random.exponential(0.005)
            spread = base_spread + volatility_spread
            
            bid_price = mid_price * (1 - spread/2)
            ask_price = mid_price * (1 + spread/2)
            
            # Random sizes
            bid_size = np.random.randint(100, 1000)
            ask_size = np.random.randint(100, 1000)
            
            cursor.execute('''
                INSERT INTO orderbook (timestamp, bid_price_1, bid_size_1, 
                                      ask_price_1, ask_size_1, mid_price, spread)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, bid_price, bid_size, ask_price, ask_size, mid_price, spread))
        
        conn.commit()
        conn.close()
        
        # Print summary statistics
        print(f"  Generated {len(trades)} trades")
        print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"  Final price: ${prices[-1]:.2f} ({(prices[-1]/prices[0]-1)*100:.2f}%)")
        print(f"  Saved to {db_path}")

def main():
    """
    Generate heterogeneous LOB databases for all three conditions
    """
    # Real news events from 2012-06-21
    news_events = [
        {
            'timestamp': 3600,  # 10:30 AM
            'sentiment': -0.3,
            'importance': 0.6,
            'description': 'Fed maintains cautious stance on economy'
        },
        {
            'timestamp': 7200,  # 11:30 AM
            'sentiment': -0.4,
            'importance': 0.7,
            'description': 'Spain borrowing costs hit new highs'
        },
        {
            'timestamp': 10800,  # 12:30 PM
            'sentiment': -0.2,
            'importance': 0.5,
            'description': 'Tech sector shows weakness'
        },
        {
            'timestamp': 14400,  # 1:30 PM
            'sentiment': -0.1,
            'importance': 0.4,
            'description': 'Market fails to hold morning recovery'
        },
        {
            'timestamp': 18000,  # 2:30 PM
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
    output_dir = "/workspace/lob_databases_heterogeneous"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate for each condition
    for condition in ["LLMON", "LLMOFF", "Baseline"]:
        generator = HeterogeneousLOBGenerator(
            symbol=symbol,
            date=date,
            initial_price=initial_price,
            duration_seconds=duration_seconds,
            news_events=news_events,
            condition=condition
        )
        
        db_path = f"{output_dir}/{symbol}_{date}_{condition}_heterogeneous.db"
        generator.generate_and_save(db_path)
    
    print("\n✅ All heterogeneous LOB databases generated successfully!")

if __name__ == "__main__":
    main()