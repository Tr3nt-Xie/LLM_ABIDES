"""
Enhanced ABIDES Bridge Module
============================

Bridge module providing enhanced classes for ABIDES-LLM integration.
This module contains the classes that connect LLM agents with ABIDES market simulation.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Import from existing modules
try:
    from enhanced_llm_abides_system import NewsCategory, MarketSentiment, NewsEvent, MarketSignal
    from abides_llm_agents import ABIDESLLMTradingAgent, ABIDESLLMNewsAnalyzer
except ImportError:
    # Fallback definitions if imports fail
    class NewsCategory(Enum):
        EARNINGS = "earnings"
        MERGERS = "mergers"
        REGULATORY = "regulatory"
        MACRO_ECONOMIC = "macro_economic"
        COMPANY_SPECIFIC = "company_specific"
        GEOPOLITICAL = "geopolitical"
        TECHNICAL = "technical"
        FDA_APPROVAL = "fda_approval"
        ANALYST_UPGRADE = "analyst_upgrade"
        INSIDER_TRADING = "insider_trading"

    class MarketSentiment(Enum):
        VERY_BEARISH = -2
        BEARISH = -1
        NEUTRAL = 0
        BULLISH = 1
        VERY_BULLISH = 2

    @dataclass
    class NewsEvent:
        timestamp: datetime
        category: NewsCategory
        title: str
        description: str
        affected_symbols: List[str]
        sentiment_impact: MarketSentiment
        impact_magnitude: float
        duration_hours: int

    @dataclass
    class MarketSignal:
        timestamp: datetime
        symbol: str
        signal_type: str
        strength: float
        source: str
        confidence: float

logger = logging.getLogger(__name__)


@dataclass
class EnhancedABIDESOrder:
    """Enhanced order representation for ABIDES simulation"""
    order_id: str
    agent_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: str  # 'LIMIT', 'MARKET', 'STOP'
    limit_price: Optional[float] = None
    timestamp: datetime = None
    urgency: float = 0.5  # 0.0 to 1.0
    reasoning: str = ""
    confidence: float = 0.5
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert order to dictionary representation"""
        return asdict(self)


class MarketState:
    """Enhanced market state tracking for simulation"""
    
    def __init__(self):
        self.current_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.volume_history: Dict[str, List[Tuple[datetime, int]]] = {}
        self.volatility: Dict[str, float] = {}
        self.market_sentiment: MarketSentiment = MarketSentiment.NEUTRAL
        self.active_news: List[NewsEvent] = []
        self.market_session = "OPEN"  # OPEN, CLOSED, PRE_MARKET, AFTER_HOURS
        self.last_update: datetime = datetime.now()
        
    def update_price(self, symbol: str, price: float, volume: int = 0):
        """Update current price and history for a symbol"""
        timestamp = datetime.now()
        self.current_prices[symbol] = price
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append((timestamp, price))
        
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        if volume > 0:
            self.volume_history[symbol].append((timestamp, volume))
        
        # Keep only last 1000 entries for performance
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]
        if len(self.volume_history[symbol]) > 1000:
            self.volume_history[symbol] = self.volume_history[symbol][-1000:]
        
        self.last_update = timestamp
        self._calculate_volatility(symbol)
    
    def _calculate_volatility(self, symbol: str):
        """Calculate volatility for a symbol based on recent price history"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
            self.volatility[symbol] = 0.1  # Default volatility
            return
        
        recent_prices = [price for _, price in self.price_history[symbol][-50:]]
        if len(recent_prices) > 1:
            price_changes = np.diff(recent_prices) / recent_prices[:-1]
            self.volatility[symbol] = float(np.std(price_changes))
        else:
            self.volatility[symbol] = 0.1
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        return self.current_prices.get(symbol)
    
    def get_price_change(self, symbol: str, lookback_minutes: int = 60) -> float:
        """Get price change over specified time period"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return 0.0
        
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=lookback_minutes)
        
        current_price = self.current_prices.get(symbol, 0)
        
        # Find price at cutoff time
        past_price = None
        for timestamp, price in reversed(self.price_history[symbol]):
            if timestamp <= cutoff_time:
                past_price = price
                break
        
        if past_price is None and len(self.price_history[symbol]) > 0:
            past_price = self.price_history[symbol][0][1]
        
        if past_price and past_price > 0:
            return (current_price - past_price) / past_price
        return 0.0
    
    def add_news_event(self, news_event: NewsEvent):
        """Add news event to active news"""
        self.active_news.append(news_event)
        
        # Remove old news events (older than 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.active_news = [news for news in self.active_news if news.timestamp > cutoff_time]
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market state summary"""
        return {
            'current_prices': self.current_prices.copy(),
            'volatility': self.volatility.copy(),
            'market_sentiment': self.market_sentiment.value,
            'active_news_count': len(self.active_news),
            'market_session': self.market_session,
            'last_update': self.last_update.isoformat()
        }


class RealisticMarketDataGenerator:
    """Enhanced market data generator for realistic simulation"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.market_state = MarketState()
        self.base_prices = {}
        self.trends = {}
        self.news_generator = None
        
        # Initialize base prices and trends
        for symbol in symbols:
            self.base_prices[symbol] = random.uniform(50, 500)
            self.trends[symbol] = random.uniform(-0.02, 0.02)  # Daily trend
            self.market_state.update_price(symbol, self.base_prices[symbol])
    
    def generate_price_tick(self, symbol: str) -> float:
        """Generate next price tick for a symbol"""
        if symbol not in self.base_prices:
            return 100.0
        
        current_price = self.market_state.get_current_price(symbol) or self.base_prices[symbol]
        volatility = self.market_state.volatility.get(symbol, 0.1)
        trend = self.trends.get(symbol, 0.0)
        
        # Generate price movement
        random_factor = np.random.normal(0, volatility)
        trend_factor = trend / (24 * 60)  # Convert daily trend to per-minute
        
        # Apply news impact
        news_impact = self._calculate_news_impact(symbol)
        
        price_change = current_price * (trend_factor + random_factor + news_impact)
        new_price = max(current_price + price_change, 0.01)  # Ensure positive price
        
        # Generate volume
        base_volume = random.randint(100, 1000)
        volume_multiplier = 1 + abs(price_change / current_price) * 10  # Higher volume on bigger moves
        volume = int(base_volume * volume_multiplier)
        
        self.market_state.update_price(symbol, new_price, volume)
        return new_price
    
    def _calculate_news_impact(self, symbol: str) -> float:
        """Calculate price impact from active news events"""
        impact = 0.0
        
        for news in self.market_state.active_news:
            if symbol in news.affected_symbols:
                # Calculate time decay
                time_elapsed = (datetime.now() - news.timestamp).total_seconds() / 3600  # hours
                decay_factor = max(0, 1 - time_elapsed / news.duration_hours)
                
                # Calculate impact based on sentiment and magnitude
                sentiment_multiplier = news.sentiment_impact.value / 2.0  # -1 to 1
                impact += sentiment_multiplier * news.impact_magnitude * decay_factor * 0.001
        
        return impact
    
    def generate_market_data_batch(self, duration_minutes: int = 60) -> Dict[str, List[Dict]]:
        """Generate a batch of market data for specified duration"""
        data = {symbol: [] for symbol in self.symbols}
        
        for minute in range(duration_minutes):
            timestamp = datetime.now() + timedelta(minutes=minute)
            
            for symbol in self.symbols:
                price = self.generate_price_tick(symbol)
                volume = random.randint(100, 1000)
                
                data[symbol].append({
                    'timestamp': timestamp,
                    'price': price,
                    'volume': volume,
                    'volatility': self.market_state.volatility.get(symbol, 0.1)
                })
        
        return data
    
    def inject_news_event(self, news_event: NewsEvent):
        """Inject a news event that will affect market prices"""
        self.market_state.add_news_event(news_event)
        logger.info(f"Injected news event: {news_event.title} affecting {news_event.affected_symbols}")
    
    def get_historical_data(self, symbol: str, lookback_hours: int = 24) -> pd.DataFrame:
        """Get historical price data as DataFrame"""
        if symbol not in self.market_state.price_history:
            return pd.DataFrame()
        
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        filtered_data = [
            {'timestamp': ts, 'price': price}
            for ts, price in self.market_state.price_history[symbol]
            if ts >= cutoff_time
        ]
        
        return pd.DataFrame(filtered_data)


class EnhancedLLMInfluencedAgent:
    """Enhanced LLM-influenced trading agent for ABIDES simulation"""
    
    def __init__(self, agent_id: str, symbol: str, agent_type: str = "ENHANCED_LLM", 
                 starting_cash: float = 100000, max_position: int = 1000):
        self.agent_id = agent_id
        self.symbol = symbol
        self.agent_type = agent_type
        self.starting_cash = starting_cash
        self.current_cash = starting_cash
        self.max_position = max_position
        self.current_position = 0
        self.orders = []
        self.trades = []
        self.performance_metrics = {
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0
        }
        self.market_state = None
        self.news_analyzer = None
        self.risk_tolerance = random.uniform(0.3, 0.8)
        self.strategy_type = random.choice(['momentum', 'mean_reversion', 'news_driven', 'technical'])
        
        logger.info(f"Created enhanced LLM agent {agent_id} for {symbol} with strategy {self.strategy_type}")
    
    def set_market_state(self, market_state: MarketState):
        """Set reference to market state"""
        self.market_state = market_state
    
    def analyze_market_condition(self) -> Dict[str, Any]:
        """Analyze current market conditions"""
        if not self.market_state:
            return {'sentiment': 'neutral', 'volatility': 'medium', 'trend': 'sideways'}
        
        current_price = self.market_state.get_current_price(self.symbol)
        price_change_1h = self.market_state.get_price_change(self.symbol, 60)
        price_change_24h = self.market_state.get_price_change(self.symbol, 1440)
        volatility = self.market_state.volatility.get(self.symbol, 0.1)
        
        # Determine trend
        if price_change_24h > 0.02:
            trend = 'bullish'
        elif price_change_24h < -0.02:
            trend = 'bearish'
        else:
            trend = 'sideways'
        
        # Determine volatility level
        if volatility > 0.15:
            vol_level = 'high'
        elif volatility > 0.05:
            vol_level = 'medium'
        else:
            vol_level = 'low'
        
        return {
            'current_price': current_price,
            'price_change_1h': price_change_1h,
            'price_change_24h': price_change_24h,
            'volatility': volatility,
            'volatility_level': vol_level,
            'trend': trend,
            'active_news_count': len(self.market_state.active_news)
        }
    
    def generate_trading_signal(self) -> Optional[Dict[str, Any]]:
        """Generate trading signal based on strategy and market analysis"""
        market_analysis = self.analyze_market_condition()
        
        if not market_analysis.get('current_price'):
            return None
        
        signal = None
        confidence = 0.5
        reasoning = ""
        
        current_price = market_analysis['current_price']
        
        if self.strategy_type == 'momentum':
            if market_analysis['price_change_1h'] > 0.01 and market_analysis['trend'] == 'bullish':
                signal = 'BUY'
                confidence = 0.7
                reasoning = "Positive momentum and bullish trend detected"
            elif market_analysis['price_change_1h'] < -0.01 and market_analysis['trend'] == 'bearish':
                signal = 'SELL'
                confidence = 0.7
                reasoning = "Negative momentum and bearish trend detected"
        
        elif self.strategy_type == 'mean_reversion':
            if market_analysis['price_change_1h'] < -0.02:
                signal = 'BUY'
                confidence = 0.6
                reasoning = "Oversold condition, expecting mean reversion"
            elif market_analysis['price_change_1h'] > 0.02:
                signal = 'SELL'
                confidence = 0.6
                reasoning = "Overbought condition, expecting mean reversion"
        
        elif self.strategy_type == 'news_driven':
            if market_analysis['active_news_count'] > 0:
                # Analyze news sentiment (simplified)
                news_impact = sum(news.sentiment_impact.value for news in self.market_state.active_news 
                                if self.symbol in news.affected_symbols)
                if news_impact > 0:
                    signal = 'BUY'
                    confidence = 0.8
                    reasoning = "Positive news sentiment detected"
                elif news_impact < 0:
                    signal = 'SELL'
                    confidence = 0.8
                    reasoning = "Negative news sentiment detected"
        
        elif self.strategy_type == 'technical':
            # Simple technical analysis
            if market_analysis['volatility_level'] == 'low' and market_analysis['trend'] == 'bullish':
                signal = 'BUY'
                confidence = 0.6
                reasoning = "Low volatility uptrend - good entry opportunity"
        
        if signal and confidence >= self.risk_tolerance:
            return {
                'signal': signal,
                'confidence': confidence,
                'reasoning': reasoning,
                'target_price': current_price * (1.02 if signal == 'BUY' else 0.98),
                'urgency': confidence
            }
        
        return None
    
    def create_order(self, signal_data: Dict[str, Any]) -> Optional[EnhancedABIDESOrder]:
        """Create an order based on trading signal"""
        if not signal_data or not self.market_state:
            return None
        
        current_price = self.market_state.get_current_price(self.symbol)
        if not current_price:
            return None
        
        signal = signal_data['signal']
        confidence = signal_data['confidence']
        
        # Calculate position size based on confidence and risk tolerance
        max_risk_per_trade = self.current_cash * 0.02  # Risk 2% per trade
        position_size = int(max_risk_per_trade / (current_price * 0.05))  # Assume 5% stop loss
        position_size = min(position_size, self.max_position - abs(self.current_position))
        
        if position_size <= 0:
            return None
        
        # Adjust position size based on current position
        if signal == 'BUY' and self.current_position >= self.max_position:
            return None
        if signal == 'SELL' and self.current_position <= -self.max_position:
            return None
        
        order = EnhancedABIDESOrder(
            order_id=f"{self.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            agent_id=self.agent_id,
            symbol=self.symbol,
            side=signal,
            quantity=position_size,
            order_type='LIMIT',
            limit_price=signal_data.get('target_price', current_price),
            urgency=signal_data.get('urgency', 0.5),
            reasoning=signal_data.get('reasoning', ''),
            confidence=confidence
        )
        
        self.orders.append(order)
        return order
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """Update performance metrics based on trade result"""
        if trade_result.get('executed', False):
            pnl = trade_result.get('pnl', 0)
            self.performance_metrics['total_pnl'] += pnl
            self.performance_metrics['total_trades'] += 1
            self.trades.append(trade_result)
            
            # Update position and cash
            if trade_result['side'] == 'BUY':
                self.current_position += trade_result['quantity']
                self.current_cash -= trade_result['quantity'] * trade_result['price']
            else:
                self.current_position -= trade_result['quantity']
                self.current_cash += trade_result['quantity'] * trade_result['price']
            
            # Calculate win rate
            profitable_trades = sum(1 for trade in self.trades if trade.get('pnl', 0) > 0)
            self.performance_metrics['win_rate'] = profitable_trades / len(self.trades) if self.trades else 0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        total_value = self.current_cash
        if self.current_position and self.market_state:
            current_price = self.market_state.get_current_price(self.symbol)
            if current_price:
                total_value += self.current_position * current_price
        
        return {
            'agent_id': self.agent_id,
            'symbol': self.symbol,
            'strategy_type': self.strategy_type,
            'starting_cash': self.starting_cash,
            'current_cash': self.current_cash,
            'current_position': self.current_position,
            'total_value': total_value,
            'total_return': (total_value - self.starting_cash) / self.starting_cash,
            'performance_metrics': self.performance_metrics.copy(),
            'total_orders': len(self.orders),
            'total_trades': len(self.trades)
        }


# Export all classes
__all__ = [
    'EnhancedABIDESOrder',
    'MarketState', 
    'RealisticMarketDataGenerator',
    'EnhancedLLMInfluencedAgent',
    'NewsCategory',
    'MarketSentiment',
    'NewsEvent',
    'MarketSignal'
]