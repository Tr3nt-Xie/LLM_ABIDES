#!/usr/bin/env python3
"""
Simplified ABIDES-LLM Demo
==========================

This script demonstrates the core LLM-ABIDES integration concepts without requiring
external dependencies. It shows how the system would work conceptually.
"""

import os
import sys
import random
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MOCK CLASSES FOR MISSING DEPENDENCIES
# =============================================================================

class MockTradingAgent:
    """Mock ABIDES TradingAgent for demonstration"""
    def __init__(self, id, name, type, random_state=None, log_orders=False):
        self.id = id
        self.name = name
        self.type = type
        self.log_orders = log_orders
        self.holdings = {"CASH": 10_000_000}  # $10M starting cash
        self.orders = {}
        self.last_trade = {}
        
    def receiveMessage(self, currentTime, msg):
        pass
        
    def wakeup(self, currentTime):
        pass
        
    def placeOrder(self, order):
        print(f"[{self.name}] Placing order: {order}")
        
    def cancelOrder(self, order):
        print(f"[{self.name}] Canceling order: {order}")

class MockMessage:
    """Mock ABIDES Message for demonstration"""
    def __init__(self, msg_type, body=None):
        self.msg_type = msg_type
        self.body = body or {}

class MockUtil:
    """Mock ABIDES util for demonstration"""
    @staticmethod
    def log_print(msg):
        print(f"[ABIDES] {msg}")

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

class NewsCategory(Enum):
    EARNINGS = "earnings"
    MERGERS = "mergers"
    REGULATORY = "regulatory"
    MACRO_ECONOMIC = "macro_economic"
    PRODUCT_LAUNCH = "product_launch"

@dataclass
class NewsEvent:
    timestamp: datetime
    category: NewsCategory
    headline: str
    content: str
    affected_symbols: List[str]
    sentiment_score: float  # -1 to 1
    importance: float  # 0 to 1

@dataclass
class MarketSignal:
    timestamp: datetime
    signal_type: str
    symbol: str
    strength: float
    duration: int
    confidence: float
    source_agent: str

@dataclass
class TradeOrder:
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int
    price: Optional[float] = None
    order_type: str = "MARKET"

# =============================================================================
# SIMPLIFIED LLM NEWS ANALYZER
# =============================================================================

class SimpleLLMNewsAnalyzer(MockTradingAgent):
    """Simplified LLM News Analyzer for demonstration"""
    
    def __init__(self, id, name, symbols=None, llm_enabled=True):
        super().__init__(id, name, "SimpleLLMNewsAnalyzer")
        self.symbols = symbols or ["ABM"]
        self.llm_enabled = llm_enabled
        self.analysis_history = []
        
        print(f"✓ {self.name} initialized for symbols: {self.symbols}")
    
    def analyze_news(self, news_event: NewsEvent) -> Dict[str, Any]:
        """Analyze news event (simplified version)"""
        
        if self.llm_enabled:
            # Simulate LLM analysis
            analysis = self._simulate_llm_analysis(news_event)
        else:
            # Simple rule-based analysis
            analysis = self._simple_analysis(news_event)
        
        # Store analysis
        self.analysis_history.append({
            'timestamp': news_event.timestamp,
            'news': news_event,
            'analysis': analysis
        })
        
        print(f"[{self.name}] Analyzed news: {news_event.headline}")
        print(f"  Sentiment: {analysis['sentiment']:.2f}")
        print(f"  Impact: {analysis['impact_score']:.2f}")
        
        return analysis
    
    def _simulate_llm_analysis(self, news_event: NewsEvent) -> Dict[str, Any]:
        """Simulate what an LLM analysis would look like"""
        
        # Simulate more sophisticated analysis based on content
        sentiment_adjustment = 0
        impact_multiplier = 1.0
        
        # Keyword-based adjustments (simulating LLM reasoning)
        positive_keywords = ["growth", "success", "profit", "expansion", "partnership"]
        negative_keywords = ["loss", "decline", "lawsuit", "regulation", "bankruptcy"]
        
        content_lower = news_event.content.lower()
        for keyword in positive_keywords:
            if keyword in content_lower:
                sentiment_adjustment += 0.1
        
        for keyword in negative_keywords:
            if keyword in content_lower:
                sentiment_adjustment -= 0.1
        
        # Category-based impact
        if news_event.category == NewsCategory.EARNINGS:
            impact_multiplier = 1.5
        elif news_event.category == NewsCategory.REGULATORY:
            impact_multiplier = 1.2
        
        final_sentiment = max(-1, min(1, news_event.sentiment_score + sentiment_adjustment))
        impact_score = min(1, news_event.importance * impact_multiplier)
        
        return {
            'sentiment': final_sentiment,
            'impact_score': impact_score,
            'confidence': 0.7 + random.random() * 0.2,  # 0.7-0.9
            'reasoning': f"LLM analysis considering {news_event.category.value} context and content keywords",
            'price_direction': 'up' if final_sentiment > 0 else 'down',
            'magnitude_estimate': abs(final_sentiment) * 5  # % change estimate
        }
    
    def _simple_analysis(self, news_event: NewsEvent) -> Dict[str, Any]:
        """Simple rule-based analysis"""
        return {
            'sentiment': news_event.sentiment_score,
            'impact_score': news_event.importance,
            'confidence': 0.5,
            'reasoning': "Simple rule-based analysis",
            'price_direction': 'up' if news_event.sentiment_score > 0 else 'down',
            'magnitude_estimate': abs(news_event.sentiment_score) * 3
        }

# =============================================================================
# SIMPLIFIED LLM TRADING AGENT
# =============================================================================

class SimpleLLMTradingAgent(MockTradingAgent):
    """Simplified LLM Trading Agent for demonstration"""
    
    def __init__(self, id, name, symbol="ABM", strategy="momentum", 
                 risk_tolerance=0.5, llm_enabled=True):
        super().__init__(id, name, "SimpleLLMTradingAgent")
        self.symbol = symbol
        self.strategy = strategy
        self.risk_tolerance = risk_tolerance
        self.llm_enabled = llm_enabled
        
        # Portfolio state
        self.holdings[symbol] = 0
        self.portfolio_value = self.holdings["CASH"]
        self.max_position_size = int(self.portfolio_value * 0.1)  # 10% max position
        
        # Trading history
        self.signals_received = []
        self.trades_executed = []
        
        print(f"✓ {self.name} initialized: {strategy} strategy, risk={risk_tolerance}")
    
    def process_news_analysis(self, news_event: NewsEvent, analysis: Dict[str, Any]) -> Optional[MarketSignal]:
        """Process news analysis and generate trading signal"""
        
        # Check if news affects our symbol
        if self.symbol not in news_event.affected_symbols:
            return None
        
        # Generate trading signal based on strategy
        signal = self._generate_signal(news_event, analysis)
        
        if signal:
            self.signals_received.append(signal)
            print(f"[{self.name}] Generated signal: {signal.signal_type} {signal.strength:.2f}")
            
            # Execute trade based on signal
            trade = self._execute_trade(signal)
            if trade:
                self.trades_executed.append(trade)
        
        return signal
    
    def _generate_signal(self, news_event: NewsEvent, analysis: Dict[str, Any]) -> Optional[MarketSignal]:
        """Generate trading signal based on analysis"""
        
        sentiment = analysis['sentiment']
        confidence = analysis['confidence']
        impact = analysis['impact_score']
        
        # Check minimum confidence threshold
        min_confidence = 0.6
        if confidence < min_confidence:
            return None
        
        # Calculate signal strength based on strategy
        base_strength = abs(sentiment) * impact
        
        if self.strategy == "momentum":
            # Momentum strategy follows sentiment direction
            strength = base_strength * self.risk_tolerance
            signal_type = "BUY" if sentiment > 0 else "SELL"
            
        elif self.strategy == "contrarian":
            # Contrarian strategy goes against sentiment (value-style)
            strength = base_strength * self.risk_tolerance * 0.8
            signal_type = "SELL" if sentiment > 0 else "BUY"
            
        else:  # neutral strategy
            strength = base_strength * self.risk_tolerance * 0.5
            signal_type = "BUY" if sentiment > 0 else "SELL"
        
        # Apply LLM enhancement if enabled
        if self.llm_enabled:
            strength *= 1.2  # LLM provides better analysis
        
        return MarketSignal(
            timestamp=news_event.timestamp,
            signal_type=signal_type,
            symbol=self.symbol,
            strength=strength,
            duration=30,  # minutes
            confidence=confidence,
            source_agent=self.name
        )
    
    def _execute_trade(self, signal: MarketSignal) -> Optional[TradeOrder]:
        """Execute trade based on signal"""
        
        # Calculate position size based on signal strength and risk tolerance
        max_trade_value = self.max_position_size * signal.strength
        
        # Simulate current price (in real system would get from exchange)
        current_price = 100 + random.uniform(-5, 5)  # $95-$105
        
        quantity = int(max_trade_value / current_price)
        
        if quantity < 1:
            return None
        
        # Create and execute order
        order = TradeOrder(
            symbol=signal.symbol,
            side=signal.signal_type,
            quantity=quantity,
            price=current_price,
            order_type="MARKET"
        )
        
        # Update holdings (simplified)
        if order.side == "BUY":
            cost = quantity * current_price
            if cost <= self.holdings["CASH"]:
                self.holdings["CASH"] -= cost
                self.holdings[self.symbol] += quantity
                print(f"[{self.name}] EXECUTED: {order.side} {quantity} shares at ${current_price:.2f}")
            else:
                print(f"[{self.name}] INSUFFICIENT CASH for trade")
                return None
        else:  # SELL
            if quantity <= self.holdings[self.symbol]:
                proceeds = quantity * current_price
                self.holdings["CASH"] += proceeds
                self.holdings[self.symbol] -= quantity
                print(f"[{self.name}] EXECUTED: {order.side} {quantity} shares at ${current_price:.2f}")
            else:
                print(f"[{self.name}] INSUFFICIENT SHARES for trade")
                return None
        
        return order

# =============================================================================
# NEWS GENERATOR FOR SIMULATION
# =============================================================================

class SimpleNewsGenerator:
    """Generate realistic news events for simulation"""
    
    def __init__(self, symbols=None):
        self.symbols = symbols or ["ABM"]
        
        # News templates
        self.news_templates = {
            NewsCategory.EARNINGS: [
                ("Company reports {direction} quarterly earnings", 0.6),
                ("Q{quarter} results {direction} analyst expectations", 0.7),
                ("Revenue {direction} {percent}% year-over-year", 0.8)
            ],
            NewsCategory.REGULATORY: [
                ("New regulations may {impact} industry operations", 0.5),
                ("Government announces {policy} policy changes", 0.6),
                ("Regulatory approval {status} for new product", 0.7)
            ],
            NewsCategory.PRODUCT_LAUNCH: [
                ("Company launches {adjective} new product line", 0.4),
                ("Innovation in {area} drives {direction} outlook", 0.5),
                ("Partnership announced for {adjective} initiative", 0.6)
            ]
        }
    
    def generate_event(self) -> NewsEvent:
        """Generate a random news event"""
        
        category = random.choice(list(NewsCategory))
        symbol = random.choice(self.symbols)
        
        # Generate content based on category
        if category in self.news_templates:
            template, base_importance = random.choice(self.news_templates[category])
            headline, content = self._generate_content(template, category)
        else:
            headline = f"Market update affects {symbol}"
            content = f"Various market factors impact {symbol} trading."
            base_importance = 0.3
        
        # Generate sentiment
        sentiment = random.uniform(-0.8, 0.8)
        
        return NewsEvent(
            timestamp=datetime.now(),
            category=category,
            headline=headline,
            content=content,
            affected_symbols=[symbol],
            sentiment_score=sentiment,
            importance=base_importance + random.uniform(-0.1, 0.1)
        )
    
    def _generate_content(self, template: str, category: NewsCategory) -> tuple:
        """Generate headline and content from template"""
        
        # Fill in template variables
        replacements = {
            'direction': random.choice(['strong', 'weak', 'mixed']),
            'quarter': random.choice(['Q1', 'Q2', 'Q3', 'Q4']),
            'percent': random.randint(5, 25),
            'impact': random.choice(['benefit', 'challenge']),
            'policy': random.choice(['trade', 'environmental', 'financial']),
            'status': random.choice(['received', 'pending', 'denied']),
            'adjective': random.choice(['innovative', 'strategic', 'ambitious']),
            'area': random.choice(['technology', 'sustainability', 'efficiency'])
        }
        
        headline = template.format(**replacements)
        
        # Generate longer content
        content = f"{headline}. This development is expected to have significant " \
                 f"implications for the company's future performance and market position. " \
                 f"Analysts are closely monitoring the situation for further updates."
        
        return headline, content

# =============================================================================
# SIMULATION ORCHESTRATOR
# =============================================================================

class ABIDESLLMSimulation:
    """Orchestrate the simplified ABIDES-LLM simulation"""
    
    def __init__(self, symbols=None, llm_enabled=True):
        self.symbols = symbols or ["ABM"]
        self.llm_enabled = llm_enabled
        
        # Create agents
        self.news_analyzer = SimpleLLMNewsAnalyzer(
            id=1, name="NewsAnalyzer", symbols=symbols, llm_enabled=llm_enabled
        )
        
        self.traders = [
            SimpleLLMTradingAgent(
                id=10, name="MomentumTrader", strategy="momentum", 
                risk_tolerance=0.8, llm_enabled=llm_enabled
            ),
            SimpleLLMTradingAgent(
                id=11, name="ContrarianTrader", strategy="contrarian", 
                risk_tolerance=0.6, llm_enabled=llm_enabled
            ),
            SimpleLLMTradingAgent(
                id=12, name="NeutralTrader", strategy="neutral", 
                risk_tolerance=0.4, llm_enabled=llm_enabled
            )
        ]
        
        # News generator
        self.news_generator = SimpleNewsGenerator(symbols)
        
        # Simulation state
        self.events_processed = 0
        self.total_trades = 0
        
    def run_simulation(self, num_events=5):
        """Run the simulation for a specified number of news events"""
        
        print("="*60)
        print("🚀 Starting ABIDES-LLM Simulation")
        print("="*60)
        print(f"Symbols: {self.symbols}")
        print(f"LLM Enabled: {self.llm_enabled}")
        print(f"Traders: {len(self.traders)}")
        print()
        
        for i in range(num_events):
            print(f"\n--- Event {i+1}/{num_events} ---")
            
            # Generate news event
            news_event = self.news_generator.generate_event()
            print(f"📰 NEWS: {news_event.headline}")
            print(f"   Category: {news_event.category.value}")
            print(f"   Sentiment: {news_event.sentiment_score:.2f}")
            print(f"   Symbols: {news_event.affected_symbols}")
            
            # Analyze news
            analysis = self.news_analyzer.analyze_news(news_event)
            
            # Send analysis to traders
            for trader in self.traders:
                signal = trader.process_news_analysis(news_event, analysis)
                if signal:
                    self.total_trades += 1
            
            self.events_processed += 1
            
            # Small delay to show progression
            import time
            time.sleep(0.5)
        
        # Print simulation results
        self.print_results()
    
    def print_results(self):
        """Print simulation results"""
        
        print("\n" + "="*60)
        print("📊 SIMULATION RESULTS")
        print("="*60)
        
        print(f"Events Processed: {self.events_processed}")
        print(f"Total Signals Generated: {self.total_trades}")
        print(f"LLM Enhancement: {'Enabled' if self.llm_enabled else 'Disabled'}")
        
        print("\nTrader Performance:")
        for trader in self.traders:
            portfolio_value = trader.holdings["CASH"] + (trader.holdings[trader.symbol] * 100)  # Assume $100/share
            print(f"  {trader.name}:")
            print(f"    Cash: ${trader.holdings['CASH']:,.2f}")
            print(f"    Shares: {trader.holdings[trader.symbol]}")
            print(f"    Total Value: ${portfolio_value:,.2f}")
            print(f"    Signals: {len(trader.signals_received)}")
            print(f"    Trades: {len(trader.trades_executed)}")
        
        print("\nNews Analysis Summary:")
        print(f"  Total analyses: {len(self.news_analyzer.analysis_history)}")
        if self.news_analyzer.analysis_history:
            avg_sentiment = sum(a['analysis']['sentiment'] for a in self.news_analyzer.analysis_history) / len(self.news_analyzer.analysis_history)
            avg_confidence = sum(a['analysis']['confidence'] for a in self.news_analyzer.analysis_history) / len(self.news_analyzer.analysis_history)
            print(f"  Average sentiment: {avg_sentiment:.2f}")
            print(f"  Average confidence: {avg_confidence:.2f}")
        
        print("\n🎉 Simulation completed!")

# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    """Run the main demonstration"""
    
    print("ABIDES-LLM Integration Demo")
    print("===========================")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your-openai-api-key-here":
        print("✓ OpenAI API key found (LLM features would be enabled)")
        llm_enabled = True
    else:
        print("⚠️  No OpenAI API key found (using mock LLM)")
        llm_enabled = False
    
    print(f"LLM Enhancement: {'Enabled' if llm_enabled else 'Mock Mode'}")
    print()
    
    # Run simulation
    try:
        symbols = ["ABM", "XYZ", "TECH"]
        sim = ABIDESLLMSimulation(symbols=symbols[:1], llm_enabled=llm_enabled)  # Use just one symbol for demo
        sim.run_simulation(num_events=3)
        
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()