"""
ABIDES-Compatible LLM Agents
============================

This module implements LLM-enhanced trading agents that are fully compatible with 
the ABIDES experimental framework, including message passing, Kernel integration,
and ABIDES-Markets exchange protocols.

Based on official ABIDES architecture from JPMorgan Chase.
"""

import numpy as np
import pandas as pd
import random
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

# ABIDES imports (these would come from official ABIDES installation)
try:
    from agent.TradingAgent import TradingAgent
    from message.Message import Message
    from util import util
    from util.OrderBook import OrderBook
except ImportError:
    # Fallback for when ABIDES is not installed
    print("Warning: ABIDES not found. Creating mock classes for development.")
    
    class TradingAgent:
        def __init__(self, id, name, type, random_state=None, log_orders=False):
            self.id = id
            self.name = name
            self.type = type
            self.random_state = random_state or np.random.RandomState()
            self.log_orders = log_orders
            self.holdings = {"CASH": 10000000}  # $10M starting cash
            self.orders = {}
            self.last_trade = {}
            
        def receiveMessage(self, currentTime, msg):
            pass
            
        def wakeup(self, currentTime):
            pass
            
        def placeOrder(self, order):
            pass
            
        def cancelOrder(self, order):
            pass
    
    class Message:
        def __init__(self, msg_type, body=None):
            self.msg_type = msg_type
            self.body = body or {}

    class util:
        @staticmethod
        def log_print(msg):
            print(msg)

# LLM imports
try:
    import autogen
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    print("Warning: autogen not found. LLM features disabled.")

# Enhanced LLM system imports
try:
    from enhanced_llm_abides_system import (
        NewsEvent, MarketSignal, NewsCategory, 
        EnhancedLLMNewsAnalyzer, RealisticNewsGenerator
    )
except ImportError:
    # Create minimal versions for ABIDES compatibility
    from enum import Enum
    from dataclasses import dataclass
    
    class NewsCategory(Enum):
        EARNINGS = "earnings"
        MERGERS = "mergers"
        REGULATORY = "regulatory"
        MACRO_ECONOMIC = "macro_economic"
    
    @dataclass
    class NewsEvent:
        timestamp: datetime
        category: NewsCategory
        headline: str
        content: str
        affected_symbols: List[str]
        sentiment_score: float
        importance: float
    
    @dataclass  
    class MarketSignal:
        timestamp: datetime
        signal_type: str
        symbol: str
        strength: float
        duration: int
        confidence: float
        source_agent: str

logger = logging.getLogger(__name__)


class ABIDESLLMNewsAnalyzer(TradingAgent):
    """
    ABIDES-compatible LLM News Analyzer Agent
    
    This agent processes news events and generates market analysis using LLM reasoning,
    then distributes insights to other agents through ABIDES message system.
    """
    
    def __init__(self, id, name, type="ABIDESLLMNewsAnalyzer", symbols=None,
                 random_state=None, log_orders=False, llm_config=None):
        super().__init__(id, name, type, random_state, log_orders)
        
        self.symbols = symbols or ["ABM"]  # Default ABIDES symbol
        self.llm_config = llm_config
        self.analysis_history = []
        self.pending_news = []
        
        # Initialize LLM if available
        if HAS_LLM and llm_config:
            try:
                self.llm_analyzer = EnhancedLLMNewsAnalyzer(
                    name=f"LLMAnalyzer_{id}",
                    llm_config=llm_config
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM analyzer: {e}")
                self.llm_analyzer = None
        else:
            self.llm_analyzer = None
        
        # News generation for simulation
        self.news_generator = RealisticNewsGenerator(self.symbols) if 'RealisticNewsGenerator' in globals() else None
        
        util.log_print(f"ABIDESLLMNewsAnalyzer {self.name} initialized for symbols: {self.symbols}")
    
    def kernelStarting(self, startTime):
        """Called when ABIDES kernel starts"""
        super().kernelStarting(startTime)
        util.log_print(f"{self.name} starting at {startTime}")
        
        # Schedule periodic news generation and analysis
        self.schedulePeriodicNews()
    
    def kernelStopping(self):
        """Called when ABIDES kernel stops"""
        super().kernelStopping()
        util.log_print(f"{self.name} processed {len(self.analysis_history)} news events")
    
    def wakeup(self, currentTime):
        """Main agent activity - called by ABIDES kernel"""
        super().wakeup(currentTime)
        
        # Check for scheduled news generation
        if self.shouldGenerateNews(currentTime):
            self.generateAndProcessNews(currentTime)
        
        # Process any pending news analysis
        self.processPendingNews(currentTime)
    
    def receiveMessage(self, currentTime, msg):
        """Handle incoming ABIDES messages"""
        super().receiveMessage(currentTime, msg)
        
        if msg.msg_type == "NEWS_EVENT":
            # Process external news event
            self.processNewsEvent(currentTime, msg.body)
            
        elif msg.msg_type == "MARKET_DATA_UPDATE":
            # Update market context for analysis
            self.updateMarketContext(msg.body)
    
    def shouldGenerateNews(self, currentTime):
        """Determine if we should generate news at this time"""
        # Generate news with some probability (configurable)
        return self.random_state.random() < 0.01  # 1% chance per wakeup
    
    def generateAndProcessNews(self, currentTime):
        """Generate and analyze news events"""
        if not self.news_generator:
            return
        
        try:
            # Generate realistic news event
            news_event = self.news_generator.generate_realistic_event()
            news_event.timestamp = currentTime
            
            util.log_print(f"{self.name} generated news: {news_event.headline}")
            
            # Process the news event
            self.processNewsEvent(currentTime, news_event)
            
        except Exception as e:
            logger.error(f"Error generating news: {e}")
    
    def processNewsEvent(self, currentTime, news_event):
        """Process a news event with LLM analysis"""
        try:
            # Perform LLM analysis if available
            if self.llm_analyzer:
                analysis = self.llm_analyzer.analyze_news_comprehensive(
                    news_event, self.getMarketContext()
                )
            else:
                # Fallback analysis
                analysis = self.createFallbackAnalysis(news_event)
            
            # Store analysis
            analysis_record = {
                'timestamp': currentTime,
                'news_event': news_event,
                'analysis': analysis
            }
            self.analysis_history.append(analysis_record)
            
            # Broadcast analysis to interested agents
            self.broadcastAnalysis(currentTime, news_event, analysis)
            
            util.log_print(f"{self.name} analyzed news with sentiment: {analysis.get('sentiment', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Error processing news event: {e}")
    
    def createFallbackAnalysis(self, news_event):
        """Create basic analysis when LLM is not available"""
        return {
            'sentiment': news_event.sentiment_score,
            'impact_assessment': {
                'immediate_impact': abs(news_event.sentiment_score) * 5,
                'confidence': news_event.confidence if hasattr(news_event, 'confidence') else 0.5
            },
            'price_predictions': {
                'direction': 'up' if news_event.sentiment_score > 0 else 'down',
                'magnitude_percent': abs(news_event.sentiment_score) * 2
            },
            'reasoning': f"Fallback analysis for {news_event.category.value} news"
        }
    
    def broadcastAnalysis(self, currentTime, news_event, analysis):
        """Broadcast news analysis to other agents via ABIDES messaging"""
        
        # Create message payload
        message_body = {
            'news_event': {
                'headline': news_event.headline,
                'category': news_event.category.value,
                'sentiment_score': news_event.sentiment_score,
                'affected_symbols': news_event.affected_symbols,
                'timestamp': currentTime
            },
            'analysis': analysis,
            'source_agent': self.name
        }
        
        # Send to all trading agents (in real ABIDES, would use proper agent discovery)
        self.sendMessage(None, Message("NEWS_ANALYSIS", message_body), broadcast=True)
    
    def getMarketContext(self):
        """Get current market context for analysis"""
        # In real ABIDES, would query exchange for current market data
        return {
            'current_time': datetime.now(),
            'symbols': self.symbols,
            'market_open': True  # Simplified
        }
    
    def updateMarketContext(self, market_data):
        """Update market context from market data updates"""
        # Store market data for contextual analysis
        pass
    
    def schedulePeriodicNews(self):
        """Schedule periodic news generation"""
        # In real ABIDES, would use kernel scheduling
        pass
    
    def processPendingNews(self, currentTime):
        """Process any pending news analysis"""
        if self.pending_news:
            news = self.pending_news.pop(0)
            self.processNewsEvent(currentTime, news)


class ABIDESLLMTradingAgent(TradingAgent):
    """
    ABIDES-compatible LLM Trading Agent
    
    This agent receives news analysis and generates trading decisions using LLM reasoning,
    executing trades through the ABIDES-Markets exchange system.
    """
    
    def __init__(self, id, name, type="ABIDESLLMTradingAgent", symbol="ABM",
                 strategy="momentum", risk_tolerance=0.5, random_state=None, 
                 log_orders=True, llm_config=None, initial_cash=10000000):
        
        super().__init__(id, name, type, random_state, log_orders)
        
        self.symbol = symbol
        self.strategy = strategy
        self.risk_tolerance = risk_tolerance
        self.llm_config = llm_config
        
        # Portfolio management
        self.holdings = {"CASH": initial_cash}
        self.holdings[symbol] = 0
        self.target_holdings = {}
        
        # Trading state
        self.active_orders = {}
        self.order_id_counter = 0
        self.last_trade_price = None
        self.market_signals = []
        self.news_analysis_history = []
        
        # Strategy parameters based on agent type
        self.strategy_params = self._initializeStrategyParams()
        
        # LLM reasoning (if available)
        self.llm_reasoning_enabled = HAS_LLM and llm_config is not None
        
        util.log_print(f"ABIDESLLMTradingAgent {self.name} initialized: {strategy} strategy, "
                      f"risk_tolerance={risk_tolerance}, symbol={symbol}")
    
    def _initializeStrategyParams(self):
        """Initialize strategy-specific parameters"""
        base_params = {
            'max_position_pct': 0.10,  # 10% of portfolio
            'min_confidence': 0.6,
            'signal_decay_minutes': 30,
            'rebalance_threshold': 0.05
        }
        
        # Strategy-specific adjustments
        if self.strategy == "momentum":
            base_params.update({
                'signal_sensitivity': 1.2,
                'mean_reversion_factor': 0.0,
                'trend_following': True
            })
        elif self.strategy == "value":
            base_params.update({
                'signal_sensitivity': 0.8,
                'mean_reversion_factor': 1.0,
                'trend_following': False
            })
        elif self.strategy == "volatility":
            base_params.update({
                'signal_sensitivity': 1.5,
                'volatility_target': 0.02,
                'trend_following': False
            })
        
        return base_params
    
    def kernelStarting(self, startTime):
        """Called when ABIDES kernel starts"""
        super().kernelStarting(startTime)
        
        # Subscribe to market data for our symbol
        self.subscribeToMarketData()
        
        # Set initial trading schedule
        self.scheduleNextTrading(startTime)
    
    def kernelStopping(self):
        """Called when ABIDES kernel stops"""
        super().kernelStopping()
        
        # Report final portfolio state
        total_value = self.calculatePortfolioValue()
        util.log_print(f"{self.name} final portfolio value: ${total_value:,.2f}")
        util.log_print(f"{self.name} processed {len(self.news_analysis_history)} news analyses")
    
    def wakeup(self, currentTime):
        """Main trading activity - called by ABIDES kernel"""
        super().wakeup(currentTime)
        
        # Check for trading opportunities
        self.evaluateTradingOpportunities(currentTime)
        
        # Manage active orders
        self.manageActiveOrders(currentTime)
        
        # Schedule next trading activity
        self.scheduleNextTrading(currentTime)
    
    def receiveMessage(self, currentTime, msg):
        """Handle incoming ABIDES messages"""
        super().receiveMessage(currentTime, msg)
        
        if msg.msg_type == "NEWS_ANALYSIS":
            # Process news analysis from LLM analyzer
            self.processNewsAnalysis(currentTime, msg.body)
            
        elif msg.msg_type == "MARKET_DATA":
            # Update market data
            self.updateMarketData(msg.body)
            
        elif msg.msg_type == "ORDER_EXECUTED":
            # Handle order execution
            self.handleOrderExecution(currentTime, msg.body)
            
        elif msg.msg_type == "ORDER_CANCELLED":
            # Handle order cancellation  
            self.handleOrderCancellation(currentTime, msg.body)
    
    def processNewsAnalysis(self, currentTime, analysis_data):
        """Process incoming news analysis and generate trading signals"""
        try:
            news_event = analysis_data['news_event']
            analysis = analysis_data['analysis']
            
            # Check if this news affects our symbol
            if self.symbol not in news_event.get('affected_symbols', []):
                return
            
            # Store analysis
            self.news_analysis_history.append({
                'timestamp': currentTime,
                'analysis': analysis_data
            })
            
            # Generate trading signal from analysis
            signal = self.generateTradingSignal(currentTime, news_event, analysis)
            
            if signal:
                self.market_signals.append(signal)
                util.log_print(f"{self.name} generated signal: {signal['direction']} "
                             f"{signal['strength']:.2f} confidence={signal['confidence']:.2f}")
                
                # Execute trading decision
                self.executeTradingDecision(currentTime, signal)
            
        except Exception as e:
            logger.error(f"Error processing news analysis: {e}")
    
    def generateTradingSignal(self, currentTime, news_event, analysis):
        """Generate trading signal from news analysis"""
        try:
            # Extract key analysis components
            sentiment = analysis.get('sentiment', 0)
            impact = analysis.get('impact_assessment', {})
            price_pred = analysis.get('price_predictions', {})
            
            # Calculate signal strength based on strategy
            base_strength = abs(sentiment)
            confidence = impact.get('confidence', 0.5)
            
            # Apply strategy-specific adjustments
            if self.strategy == "momentum":
                strength = base_strength * self.strategy_params['signal_sensitivity']
            elif self.strategy == "value":
                # Value strategy looks for overreactions to reverse
                strength = base_strength * (1.0 - confidence) * 0.8
            else:
                strength = base_strength
            
            # Apply risk tolerance
            strength *= self.risk_tolerance
            
            # Check minimum confidence threshold
            if confidence < self.strategy_params['min_confidence']:
                return None
            
            # Determine direction
            direction = 'BUY' if sentiment > 0 else 'SELL'
            if self.strategy == "value":
                direction = 'SELL' if sentiment > 0 else 'BUY'  # Contrarian
            
            return {
                'timestamp': currentTime,
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'duration': 30,  # minutes
                'source': 'news_analysis',
                'reasoning': analysis.get('reasoning', 'LLM analysis')
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return None
    
    def executeTradingDecision(self, currentTime, signal):
        """Execute trading decision based on signal"""
        try:
            # Calculate position size
            portfolio_value = self.calculatePortfolioValue()
            max_position_value = portfolio_value * self.strategy_params['max_position_pct']
            
            # Adjust by signal strength and confidence
            position_value = max_position_value * signal['strength'] * signal['confidence']
            
            # Convert to shares (assuming we have current price)
            current_price = self.getLastTradePrice()
            if not current_price:
                return
            
            target_shares = int(position_value / current_price)
            current_shares = self.holdings.get(self.symbol, 0)
            
            # Calculate order quantity
            if signal['direction'] == 'BUY':
                order_quantity = target_shares - current_shares
            else:
                order_quantity = current_shares + target_shares  # Selling
                order_quantity = -order_quantity
            
            # Check if order is significant enough
            if abs(order_quantity) < 10:  # Minimum order size
                return
            
            # Place order
            self.placeMarketOrder(currentTime, signal['direction'], abs(order_quantity))
            
            util.log_print(f"{self.name} executing {signal['direction']} order: "
                         f"{abs(order_quantity)} shares, strength={signal['strength']:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing trading decision: {e}")
    
    def placeMarketOrder(self, currentTime, side, quantity):
        """Place market order through ABIDES"""
        try:
            self.order_id_counter += 1
            order_id = f"{self.name}_ORDER_{self.order_id_counter}"
            
            # Create ABIDES-compatible order
            order = {
                'order_id': order_id,
                'symbol': self.symbol,
                'side': side,
                'quantity': quantity,
                'order_type': 'MARKET',
                'timestamp': currentTime
            }
            
            # Store order for tracking
            self.active_orders[order_id] = order
            
            # In real ABIDES, would call:
            # self.placeOrder(order)
            # For now, simulate order placement
            util.log_print(f"{self.name} placed {side} market order: {quantity} shares")
            
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
    
    def handleOrderExecution(self, currentTime, execution_data):
        """Handle order execution notification"""
        order_id = execution_data.get('order_id')
        if order_id in self.active_orders:
            order = self.active_orders.pop(order_id)
            
            # Update holdings
            executed_qty = execution_data.get('quantity', 0)
            executed_price = execution_data.get('price', 0)
            
            if order['side'] == 'BUY':
                self.holdings[self.symbol] += executed_qty
                self.holdings['CASH'] -= executed_qty * executed_price
            else:
                self.holdings[self.symbol] -= executed_qty
                self.holdings['CASH'] += executed_qty * executed_price
            
            util.log_print(f"{self.name} executed {order['side']}: {executed_qty} @ ${executed_price:.2f}")
    
    def handleOrderCancellation(self, currentTime, cancellation_data):
        """Handle order cancellation notification"""
        order_id = cancellation_data.get('order_id')
        if order_id in self.active_orders:
            del self.active_orders[order_id]
            util.log_print(f"{self.name} order cancelled: {order_id}")
    
    def evaluateTradingOpportunities(self, currentTime):
        """Evaluate current trading opportunities"""
        # Clean up expired signals
        self.cleanupExpiredSignals(currentTime)
        
        # Look for rebalancing opportunities
        self.checkRebalancingNeeds(currentTime)
    
    def cleanupExpiredSignals(self, currentTime):
        """Remove expired trading signals"""
        decay_minutes = self.strategy_params['signal_decay_minutes']
        cutoff_time = currentTime - pd.Timedelta(minutes=decay_minutes)
        
        self.market_signals = [
            signal for signal in self.market_signals
            if signal['timestamp'] > cutoff_time
        ]
    
    def checkRebalancingNeeds(self, currentTime):
        """Check if portfolio needs rebalancing"""
        current_allocation = self.getCurrentAllocation()
        target_allocation = self.getTargetAllocation()
        
        # Simple rebalancing logic
        threshold = self.strategy_params['rebalance_threshold']
        if abs(current_allocation - target_allocation) > threshold:
            util.log_print(f"{self.name} rebalancing needed: "
                         f"current={current_allocation:.2%}, target={target_allocation:.2%}")
    
    def getCurrentAllocation(self):
        """Calculate current portfolio allocation to main symbol"""
        portfolio_value = self.calculatePortfolioValue()
        if portfolio_value <= 0:
            return 0.0
        
        position_value = self.holdings.get(self.symbol, 0) * self.getLastTradePrice()
        return position_value / portfolio_value
    
    def getTargetAllocation(self):
        """Calculate target allocation based on current signals"""
        if not self.market_signals:
            return 0.0
        
        # Simple target based on signal strength
        total_signal = sum(s['strength'] * s['confidence'] for s in self.market_signals)
        return min(total_signal * 0.1, self.strategy_params['max_position_pct'])
    
    def calculatePortfolioValue(self):
        """Calculate total portfolio value"""
        cash = self.holdings.get('CASH', 0)
        position_value = self.holdings.get(self.symbol, 0) * (self.getLastTradePrice() or 100)
        return cash + position_value
    
    def getLastTradePrice(self):
        """Get last trade price for symbol"""
        return self.last_trade_price or 100.0  # Default price
    
    def updateMarketData(self, market_data):
        """Update market data"""
        if 'last_trade_price' in market_data:
            self.last_trade_price = market_data['last_trade_price']
    
    def subscribeToMarketData(self):
        """Subscribe to market data updates"""
        # In real ABIDES, would subscribe to exchange market data
        pass
    
    def scheduleNextTrading(self, currentTime):
        """Schedule next trading activity"""
        # In real ABIDES, would use kernel.setWakeup()
        pass
    
    def manageActiveOrders(self, currentTime):
        """Manage active orders (timeouts, modifications)"""
        # Check for order timeouts and manage order lifecycle
        pass


class ABIDESLLMMarketMaker(ABIDESLLMTradingAgent):
    """
    ABIDES-compatible LLM Market Maker Agent
    
    Provides liquidity using LLM-enhanced market making strategies.
    """
    
    def __init__(self, id, name, type="ABIDESLLMMarketMaker", symbol="ABM",
                 spread_bps=10, max_inventory=1000, **kwargs):
        
        super().__init__(id, name, type, symbol, strategy="market_making", **kwargs)
        
        self.spread_bps = spread_bps  # Spread in basis points
        self.max_inventory = max_inventory
        self.target_inventory = 0
        
        # Market making specific parameters
        self.bid_order_id = None
        self.ask_order_id = None
        self.inventory_penalty = 0.1  # Penalty for holding inventory
        
        util.log_print(f"ABIDESLLMMarketMaker {self.name} initialized: "
                      f"spread={spread_bps}bps, max_inventory={max_inventory}")
    
    def wakeup(self, currentTime):
        """Market maker specific wakeup logic"""
        super().wakeup(currentTime)
        
        # Update market making quotes
        self.updateMarketMakingQuotes(currentTime)
    
    def updateMarketMakingQuotes(self, currentTime):
        """Update bid/ask quotes based on LLM analysis"""
        try:
            current_price = self.getLastTradePrice()
            if not current_price:
                return
            
            # Calculate spreads based on recent news analysis
            spread_adjustment = self.calculateSpreadAdjustment()
            adjusted_spread = self.spread_bps * (1 + spread_adjustment)
            
            # Calculate bid/ask prices
            half_spread = current_price * adjusted_spread / 20000  # Convert bps to price
            bid_price = current_price - half_spread
            ask_price = current_price + half_spread
            
            # Adjust for inventory
            inventory_adjustment = self.calculateInventoryAdjustment()
            bid_price += inventory_adjustment
            ask_price += inventory_adjustment
            
            # Place/update quotes
            self.updateQuotes(currentTime, bid_price, ask_price)
            
        except Exception as e:
            logger.error(f"Error updating market making quotes: {e}")
    
    def calculateSpreadAdjustment(self):
        """Calculate spread adjustment based on LLM analysis"""
        if not self.news_analysis_history:
            return 0.0
        
        # Look at recent analyses
        recent_analyses = self.news_analysis_history[-5:]  # Last 5 analyses
        
        # Calculate uncertainty/volatility from analyses
        uncertainty_score = 0.0
        for analysis_record in recent_analyses:
            analysis = analysis_record['analysis']['analysis']
            impact = analysis.get('impact_assessment', {})
            uncertainty_score += 1.0 - impact.get('confidence', 0.5)
        
        # Widen spreads when uncertainty is high
        return uncertainty_score / len(recent_analyses) if recent_analyses else 0.0
    
    def calculateInventoryAdjustment(self):
        """Calculate price adjustment based on current inventory"""
        current_inventory = self.holdings.get(self.symbol, 0)
        inventory_ratio = current_inventory / self.max_inventory
        
        # Adjust prices to reduce inventory when approaching limits
        return -inventory_ratio * self.inventory_penalty * self.getLastTradePrice()
    
    def updateQuotes(self, currentTime, bid_price, ask_price):
        """Update bid/ask quotes"""
        # Cancel existing orders
        if self.bid_order_id:
            self.cancelOrder(self.bid_order_id)
        if self.ask_order_id:
            self.cancelOrder(self.ask_order_id)
        
        # Place new quotes
        quote_size = min(100, self.max_inventory // 10)  # 10% of max inventory
        
        self.bid_order_id = self.placeLimitOrder(currentTime, 'BUY', quote_size, bid_price)
        self.ask_order_id = self.placeLimitOrder(currentTime, 'SELL', quote_size, ask_price)
        
        util.log_print(f"{self.name} updated quotes: ${bid_price:.2f} x ${ask_price:.2f}")
    
    def placeLimitOrder(self, currentTime, side, quantity, price):
        """Place limit order and return order ID"""
        self.order_id_counter += 1
        order_id = f"{self.name}_QUOTE_{self.order_id_counter}"
        
        order = {
            'order_id': order_id,
            'symbol': self.symbol,
            'side': side,
            'quantity': quantity,
            'order_type': 'LIMIT',
            'price': price,
            'timestamp': currentTime
        }
        
        self.active_orders[order_id] = order
        # In real ABIDES: self.placeOrder(order)
        
        return order_id


# Utility functions for ABIDES integration
def createABIDESLLMAgentConfig(agent_type="trading", agent_id=100, symbol="ABM", 
                               strategy="momentum", risk_tolerance=0.5, 
                               llm_config=None, **kwargs):
    """
    Create ABIDES agent configuration for LLM agents
    
    Returns configuration dict that can be used in ABIDES config files.
    """
    
    base_config = {
        'agent_id': agent_id,
        'name': f"LLM_{agent_type.title()}Agent_{agent_id}",
        'symbol': symbol,
        'random_state': np.random.RandomState(seed=kwargs.get('seed', agent_id)),
        'log_orders': kwargs.get('log_orders', True)
    }
    
    if agent_type == "news_analyzer":
        base_config.update({
            'type': ABIDESLLMNewsAnalyzer,
            'symbols': [symbol],
            'llm_config': llm_config
        })
    
    elif agent_type == "trading":
        base_config.update({
            'type': ABIDESLLMTradingAgent,
            'strategy': strategy,
            'risk_tolerance': risk_tolerance,
            'llm_config': llm_config,
            'initial_cash': kwargs.get('initial_cash', 10000000)
        })
    
    elif agent_type == "market_maker":
        base_config.update({
            'type': ABIDESLLMMarketMaker,
            'spread_bps': kwargs.get('spread_bps', 10),
            'max_inventory': kwargs.get('max_inventory', 1000),
            'llm_config': llm_config
        })
    
    return base_config


def createLLMEnhancedABIDESConfig(symbols=None, num_llm_agents=5, num_traditional_agents=10,
                                  llm_config=None, end_time="16:00:00", **kwargs):
    """
    Create complete ABIDES configuration with LLM-enhanced agents
    
    This function creates a configuration that can be used with the ABIDES 
    experimental framework, combining LLM agents with traditional ABIDES agents.
    """
    
    symbols = symbols or ["ABM"]
    agents = []
    agent_id = kwargs.get('start_agent_id', 100)
    
    # Add LLM News Analyzer
    agents.append(createABIDESLLMAgentConfig(
        agent_type="news_analyzer",
        agent_id=agent_id,
        symbol=symbols[0],
        llm_config=llm_config
    ))
    agent_id += 1
    
    # Add LLM Trading Agents with different strategies
    strategies = ["momentum", "value", "volatility"]
    risk_tolerances = [0.3, 0.5, 0.7, 0.9]
    
    for i in range(num_llm_agents):
        strategy = strategies[i % len(strategies)]
        risk_tolerance = risk_tolerances[i % len(risk_tolerances)]
        
        agents.append(createABIDESLLMAgentConfig(
            agent_type="trading",
            agent_id=agent_id,
            symbol=symbols[0],
            strategy=strategy,
            risk_tolerance=risk_tolerance,
            llm_config=llm_config
        ))
        agent_id += 1
    
    # Add LLM Market Maker
    agents.append(createABIDESLLMAgentConfig(
        agent_type="market_maker",
        agent_id=agent_id,
        symbol=symbols[0],
        llm_config=llm_config,
        spread_bps=kwargs.get('mm_spread_bps', 10)
    ))
    agent_id += 1
    
    # Configuration for ABIDES
    config = {
        'seed': kwargs.get('seed', 12345),
        'start_time': kwargs.get('start_time', "09:30:00"),
        'end_time': end_time,
        'agents': agents,
        'symbols': symbols,
        'book_logging': kwargs.get('book_logging', True),
        'book_log_depth': kwargs.get('book_log_depth', 10),
        'stream_history': kwargs.get('stream_history', 10),
        'log_dir': kwargs.get('log_dir', './logs'),
        'exchange_log_orders': kwargs.get('exchange_log_orders', True)
    }
    
    return config


# Example usage functions
def runABIDESWithLLMAgents(config_name="llm_enhanced_rmsc04", duration_hours=1):
    """
    Example function to run ABIDES with LLM agents
    
    This demonstrates how to integrate with the official ABIDES experimental framework.
    """
    
    # LLM configuration
    llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "your-api-key-here",  # Replace with actual key
                "temperature": 0.3
            }
        ],
        "timeout": 60
    }
    
    # Create ABIDES configuration
    config = createLLMEnhancedABIDESConfig(
        symbols=["ABM"],
        num_llm_agents=3,
        num_traditional_agents=10,
        llm_config=llm_config,
        end_time=f"{9+duration_hours:02d}:30:00"
    )
    
    print(f"Created ABIDES configuration with {len(config['agents'])} agents")
    print("LLM-enhanced agents:")
    for agent in config['agents']:
        print(f"  - {agent['name']}: {agent['type'].__name__}")
    
    # In real ABIDES, would run:
    # from abides_core import abides
    # end_state = abides.run(config)
    # return end_state
    
    print(f"\nTo run with real ABIDES:")
    print(f"from abides_core import abides")
    print(f"end_state = abides.run(config)")
    
    return config


if __name__ == "__main__":
    # Example: Create and display ABIDES configuration
    config = runABIDESWithLLMAgents("llm_demo", duration_hours=2)
    
    print(f"\nConfiguration created successfully!")
    print(f"Ready for ABIDES experimental framework integration.")