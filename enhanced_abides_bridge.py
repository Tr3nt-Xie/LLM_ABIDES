#!/usr/bin/env python3
"""
Enhanced ABIDES Bridge Module
============================

Bridge module providing enhanced ABIDES integration components.
This module provides the core classes and data structures needed for 
LLM-enhanced ABIDES simulations.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


@dataclass
class MarketState:
    """Current state of the market"""
    timestamp: datetime
    symbols: Dict[str, Dict[str, Any]]  # symbol -> {price, volume, bid, ask, etc.}
    news_events: List[Dict[str, Any]]
    market_conditions: Dict[str, Any]
    
    def __post_init__(self):
        if not self.symbols:
            self.symbols = {}
        if not self.news_events:
            self.news_events = []
        if not self.market_conditions:
            self.market_conditions = {
                'volatility': 0.02,
                'liquidity': 1.0,
                'sentiment': 0.0,
                'regime': 'normal'  # bull, bear, normal, volatile
            }
        # Add regime if missing
        if 'regime' not in self.market_conditions:
            self.market_conditions['regime'] = 'normal'
    
    @property
    def regime(self) -> str:
        """Get market regime"""
        return self.market_conditions.get('regime', 'normal')
    
    def get(self, key: str, default=None):
        """Get attribute by key for compatibility"""
        if hasattr(self, key):
            return getattr(self, key)
        return self.market_conditions.get(key, default)


@dataclass
class EnhancedABIDESOrder:
    """Enhanced order representation for ABIDES"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: str  # 'LIMIT', 'MARKET', etc.
    agent_id: str
    order_id: str = ""
    price: float = 0.0
    limit_price: float = None  # Compatibility field
    timestamp: datetime = None
    
    # Enhanced fields
    confidence: float = 0.5
    reasoning: str = ""
    llm_influenced: bool = False
    news_triggered: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if not self.order_id:
            self.order_id = f"{self.agent_id}_{self.symbol}_{int(time.time() * 1000)}"
        if self.limit_price is not None:
            self.price = self.limit_price  # Use limit_price if provided
    
    def to_abides_format(self) -> Dict[str, Any]:
        """Convert to ABIDES-compatible format"""
        return {
            'order_id': self.order_id,
            'agent_id': self.agent_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'price': self.price,
            'side': self.side,
            'order_type': self.order_type,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'llm_influenced': self.llm_influenced,
            'news_triggered': self.news_triggered
        }


class RealisticMarketDataGenerator:
    """Generates realistic market data for simulation"""
    
    def __init__(self, symbols: List[str], initial_prices: Optional[Dict[str, float]] = None):
        self.symbols = symbols
        self.initial_prices = initial_prices or {symbol: 100.0 for symbol in symbols}
        self.current_prices = self.initial_prices.copy()
        self.price_history = {symbol: [price] for symbol, price in self.current_prices.items()}
        self.volatility = {symbol: 0.02 for symbol in symbols}
        
    def generate_price_tick(self, symbol: str, news_impact: float = 0.0) -> Dict[str, Any]:
        """Generate a single price tick for a symbol"""
        if symbol not in self.symbols:
            raise ValueError(f"Unknown symbol: {symbol}")
            
        # Base random walk
        volatility = self.volatility[symbol]
        random_change = np.random.normal(0, volatility)
        
        # Apply news impact
        total_change = random_change + news_impact
        
        # Update price
        old_price = self.current_prices[symbol]
        new_price = old_price * (1 + total_change)
        new_price = max(new_price, 0.01)  # Prevent negative prices
        
        self.current_prices[symbol] = new_price
        self.price_history[symbol].append(new_price)
        
        # Generate bid/ask spread
        spread = new_price * 0.001  # 0.1% spread
        bid = new_price - spread/2
        ask = new_price + spread/2
        
        # Generate volume
        base_volume = 1000
        volume_mult = 1 + abs(total_change) * 10  # Higher volume with more movement
        volume = int(base_volume * volume_mult * random.uniform(0.5, 2.0))
        
        return {
            'symbol': symbol,
            'price': new_price,
            'bid': bid,
            'ask': ask,
            'volume': volume,
            'change': total_change,
            'timestamp': datetime.now()
        }
    
    def generate_market_state(self, news_events: List[Dict] = None) -> MarketState:
        """Generate current market state for all symbols"""
        news_events = news_events or []
        
        # Calculate news impact per symbol
        news_impact = {symbol: 0.0 for symbol in self.symbols}
        for news in news_events:
            for symbol in news.get('symbols', []):
                if symbol in news_impact:
                    sentiment = news.get('sentiment', 0.0)
                    impact = news.get('impact', 0.5)
                    news_impact[symbol] += sentiment * impact * 0.01
        
        # Generate data for all symbols
        symbols_data = {}
        for symbol in self.symbols:
            tick_data = self.generate_price_tick(symbol, news_impact[symbol])
            symbols_data[symbol] = tick_data
        
        return MarketState(
            timestamp=datetime.now(),
            symbols=symbols_data,
            news_events=news_events,
            market_conditions={
                'volatility': np.mean([self.volatility[s] for s in self.symbols]),
                'liquidity': 1.0,
                'sentiment': np.mean([news_impact[s] for s in self.symbols])
            }
        )
    
    def get_price_history(self, symbol: str, periods: int = 100) -> List[float]:
        """Get historical prices for a symbol"""
        if symbol not in self.price_history:
            return []
        return self.price_history[symbol][-periods:]
    
    def update_market_data(self, news_events: List[Dict] = None, external_orders: List = None) -> Dict[str, Any]:
        """Update market data and return current state - compatibility method"""
        # Process external orders if provided (for compatibility)
        if external_orders:
            # Simulate order impact on prices
            for order in external_orders:
                if hasattr(order, 'symbol') and order.symbol in self.symbols:
                    # Small impact from orders
                    impact = 0.001 * (order.quantity / 1000) * (1 if order.side == 'BUY' else -1)
                    old_price = self.current_prices.get(order.symbol, 100.0)
                    self.current_prices[order.symbol] = old_price * (1 + impact)
        
        market_state = self.generate_market_state(news_events)
        return {
            'market_state': market_state,
            'symbols': market_state.symbols,
            'timestamp': market_state.timestamp
        }


class EnhancedLLMInfluencedAgent:
    """Enhanced trading agent with LLM influence capabilities"""
    
    def __init__(self, agent_id: str, symbols: List[str] = None, strategy: str = "balanced", 
                 risk_tolerance: float = 0.5, initial_cash: float = 1000000, 
                 base_capital: float = None):
        self.agent_id = agent_id
        self.symbols = symbols or ['AAPL', 'MSFT', 'GOOGL']
        self.strategy = strategy
        self.risk_tolerance = risk_tolerance
        self.cash = base_capital or initial_cash
        self.base_capital = self.cash  # Compatibility
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.orders = []
        self.trade_history = []
        
        # Portfolio compatibility object
        self.portfolio = type('Portfolio', (), {
            'total_value': self.cash,
            'cash': self.cash,
            'positions': self.positions
        })()
        
        # LLM enhancement fields
        self.llm_enabled = False
        self.confidence_threshold = 0.6
        self.last_decision_reasoning = ""
        
    def analyze_market_state(self, market_state: MarketState) -> Dict[str, Any]:
        """Analyze current market state and generate trading signals"""
        analysis = {
            'timestamp': market_state.timestamp,
            'signals': {},
            'confidence': 0.5,
            'reasoning': "Basic technical analysis"
        }
        
        for symbol in self.symbols:
            if symbol not in market_state.symbols:
                continue
                
            symbol_data = market_state.symbols[symbol]
            price = symbol_data['price']
            change = symbol_data.get('change', 0.0)
            
            # Simple strategy logic
            signal_strength = 0.0
            action = "HOLD"
            
            if self.strategy == "momentum":
                if change > 0.01:  # 1% up
                    signal_strength = min(abs(change) * 10, 1.0)
                    action = "BUY"
                elif change < -0.01:  # 1% down
                    signal_strength = min(abs(change) * 10, 1.0)
                    action = "SELL"
                    
            elif self.strategy == "contrarian":
                if change > 0.01:  # 1% up - sell
                    signal_strength = min(abs(change) * 10, 1.0)
                    action = "SELL"
                elif change < -0.01:  # 1% down - buy
                    signal_strength = min(abs(change) * 10, 1.0)
                    action = "BUY"
                    
            elif self.strategy == "news_driven":
                # React to news sentiment
                news_sentiment = 0.0
                for news in market_state.news_events:
                    if symbol in news.get('symbols', []):
                        news_sentiment += news.get('sentiment', 0.0)
                
                if news_sentiment > 0.1:
                    signal_strength = min(abs(news_sentiment), 1.0)
                    action = "BUY"
                elif news_sentiment < -0.1:
                    signal_strength = min(abs(news_sentiment), 1.0)
                    action = "SELL"
            
            analysis['signals'][symbol] = {
                'action': action,
                'strength': signal_strength,
                'price': price,
                'reasoning': f"{self.strategy} strategy triggered by {change:.3f} price change"
            }
        
        return analysis
    
    def generate_order(self, symbol: str, action: str, strength: float, 
                      current_price: float, reasoning: str = "") -> Optional[EnhancedABIDESOrder]:
        """Generate an order based on signal"""
        if action == "HOLD" or strength < self.confidence_threshold:
            return None
            
        # Calculate order size based on strength and risk tolerance
        portfolio_value = self.cash + sum(pos * current_price for pos in self.positions.values())
        max_position_value = portfolio_value * self.risk_tolerance * strength
        quantity = int(max_position_value / current_price)
        
        if quantity <= 0:
            return None
            
        # Adjust for current position
        current_position = self.positions.get(symbol, 0)
        if action == "SELL":
            quantity = min(quantity, current_position) if current_position > 0 else 0
            if quantity <= 0:
                return None
                
        order = EnhancedABIDESOrder(
            order_id=f"{self.agent_id}_{symbol}_{int(time.time() * 1000)}",
            agent_id=self.agent_id,
            symbol=symbol,
            quantity=quantity,
            price=current_price,
            side=action,
            order_type="LIMIT",
            timestamp=datetime.now(),
            confidence=strength,
            reasoning=reasoning,
            llm_influenced=self.llm_enabled,
            news_triggered="news" in reasoning.lower()
        )
        
        self.orders.append(order)
        return order
    
    def process_market_update(self, market_state: MarketState) -> List[EnhancedABIDESOrder]:
        """Process market update and generate orders"""
        analysis = self.analyze_market_state(market_state)
        orders = []
        
        for symbol, signal in analysis['signals'].items():
            order = self.generate_order(
                symbol=symbol,
                action=signal['action'],
                strength=signal['strength'],
                current_price=signal['price'],
                reasoning=signal['reasoning']
            )
            if order:
                orders.append(order)
        
        return orders
    
    def execute_order(self, order: EnhancedABIDESOrder, fill_price: float = None) -> bool:
        """Execute an order (simulate fill)"""
        fill_price = fill_price or order.price
        
        if order.side == "BUY":
            cost = order.quantity * fill_price
            if self.cash >= cost:
                self.cash -= cost
                self.positions[order.symbol] += order.quantity
                success = True
            else:
                success = False
        else:  # SELL
            if self.positions.get(order.symbol, 0) >= order.quantity:
                proceeds = order.quantity * fill_price
                self.cash += proceeds
                self.positions[order.symbol] -= order.quantity
                success = True
            else:
                success = False
        
        if success:
            self.trade_history.append({
                'order': order,
                'fill_price': fill_price,
                'timestamp': datetime.now()
            })
        
        return success
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        value = self.cash
        for symbol, quantity in self.positions.items():
            if symbol in current_prices and quantity > 0:
                value += quantity * current_prices[symbol]
        return value
    
    def get_performance_metrics(self, current_prices: Dict[str, float], 
                              initial_value: float = None) -> Dict[str, Any]:
        """Get performance metrics for the agent"""
        current_value = self.get_portfolio_value(current_prices)
        initial_value = initial_value or 1000000  # Default initial cash
        
        return {
            'agent_id': self.agent_id,
            'strategy': self.strategy,
            'current_value': current_value,
            'initial_value': initial_value,
            'return_pct': (current_value - initial_value) / initial_value * 100,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'trades_count': len(self.trade_history),
            'orders_count': len(self.orders)
        }
    
    def process_signals(self, signals: List[Dict], market_state=None) -> List[EnhancedABIDESOrder]:
        """Process trading signals and generate orders - compatibility method"""
        orders = []
        
        for signal in signals:
            symbol = signal.get('symbol')
            action = signal.get('action', 'HOLD')
            strength = signal.get('strength', 0.0)
            price = signal.get('price', 100.0)
            reasoning = signal.get('reasoning', 'Signal-based trade')
            
            if symbol and symbol in self.symbols:
                order = self.generate_order(symbol, action, strength, price, reasoning)
                if order:
                    orders.append(order)
        
        return orders


# Export classes for import
__all__ = [
    'MarketState',
    'EnhancedABIDESOrder', 
    'RealisticMarketDataGenerator',
    'EnhancedLLMInfluencedAgent'
]