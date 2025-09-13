#!/usr/bin/env python3
"""
LLM-Enhanced Market Simulation Demo
===================================

This demo shows how LLM agents analyze real market news and make trading decisions
using OpenAI's API for sophisticated market analysis.
"""

import os
import sys
import json
import openai
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

# Import our modules
from enhanced_llm_abides_system import (
    NewsEvent, MarketSignal, NewsCategory,
    EnhancedLLMNewsAnalyzer, RealisticNewsGenerator
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class LLMMarketSimulator:
    """Enhanced market simulator with real LLM integration"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment")
        
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Initialize components
        self.symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        self.news_analyzer = EnhancedLLMNewsAnalyzer(self.symbols)
        self.news_generator = RealisticNewsGenerator(self.symbols)
        
        # Market state
        self.current_prices = {symbol: 100 + random.uniform(-20, 20) for symbol in self.symbols}
        self.portfolio = {"cash": 1000000}  # $1M starting capital
        for symbol in self.symbols:
            self.portfolio[symbol] = 0
        
        self.trades = []
        self.news_history = []
        
    def analyze_news_with_llm(self, news_event: NewsEvent) -> Dict:
        """Use real LLM to analyze news impact on market"""
        
        prompt = f"""
        You are a sophisticated financial analyst. Analyze this market news and provide trading recommendations.
        
        News: {news_event.headline}
        Content: {news_event.content}
        Category: {news_event.category.value}
        Affected Symbols: {', '.join(news_event.affected_symbols)}
        
        Provide your analysis in JSON format with the following structure:
        {{
            "sentiment": <float between -1 and 1>,
            "market_impact": <"high", "medium", or "low">,
            "price_direction": <"bullish", "bearish", or "neutral">,
            "confidence": <float between 0 and 1>,
            "trading_recommendation": {{
                "action": <"buy", "sell", or "hold">,
                "strength": <float between 0 and 1>,
                "reasoning": <brief explanation>
            }},
            "risk_assessment": <brief risk analysis>,
            "time_horizon": <"short", "medium", or "long">
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst providing market analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            logger.info(f"LLM Analysis: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            # Fallback to simple analysis
            return {
                "sentiment": news_event.sentiment_score,
                "market_impact": "medium",
                "price_direction": "bullish" if news_event.sentiment_score > 0 else "bearish",
                "confidence": 0.5,
                "trading_recommendation": {
                    "action": "hold",
                    "strength": 0.3,
                    "reasoning": "Analysis unavailable"
                },
                "risk_assessment": "Unable to assess",
                "time_horizon": "medium"
            }
    
    def execute_trade(self, symbol: str, action: str, quantity: int, price: float):
        """Execute a trade based on LLM recommendation"""
        
        if action == "buy":
            cost = quantity * price
            if self.portfolio["cash"] >= cost:
                self.portfolio["cash"] -= cost
                self.portfolio[symbol] = self.portfolio.get(symbol, 0) + quantity
                trade = {
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "action": "buy",
                    "quantity": quantity,
                    "price": price,
                    "total": cost
                }
                self.trades.append(trade)
                logger.info(f"EXECUTED BUY: {quantity} shares of {symbol} at ${price:.2f}")
                return True
        
        elif action == "sell":
            if self.portfolio.get(symbol, 0) >= quantity:
                self.portfolio[symbol] -= quantity
                self.portfolio["cash"] += quantity * price
                trade = {
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "action": "sell",
                    "quantity": quantity,
                    "price": price,
                    "total": quantity * price
                }
                self.trades.append(trade)
                logger.info(f"EXECUTED SELL: {quantity} shares of {symbol} at ${price:.2f}")
                return True
        
        return False
    
    def simulate_market_response(self, symbol: str, sentiment: float, impact: str):
        """Simulate how market prices respond to news"""
        
        impact_multipliers = {"high": 0.05, "medium": 0.02, "low": 0.01}
        multiplier = impact_multipliers.get(impact, 0.01)
        
        # Calculate price change
        price_change = sentiment * multiplier * self.current_prices[symbol]
        self.current_prices[symbol] += price_change
        
        # Add some random noise
        self.current_prices[symbol] *= (1 + random.uniform(-0.005, 0.005))
        
        logger.info(f"{symbol} price updated: ${self.current_prices[symbol]:.2f} (change: ${price_change:.2f})")
    
    def run_simulation(self, num_events: int = 5):
        """Run the market simulation with LLM analysis"""
        
        print("\n" + "="*60)
        print("🚀 LLM-ENHANCED MARKET SIMULATION")
        print("="*60)
        print(f"Starting Capital: ${self.portfolio['cash']:,.2f}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Using OpenAI API: ✓")
        print()
        
        for i in range(num_events):
            print(f"\n--- Event {i+1}/{num_events} ---")
            
            # Generate news event
            news_event = self.news_generator.generate_news_event()
            self.news_history.append(news_event)
            
            print(f"📰 NEWS: {news_event.headline}")
            print(f"   Category: {news_event.category.value}")
            print(f"   Affected: {', '.join(news_event.affected_symbols)}")
            
            # Analyze with LLM
            analysis = self.analyze_news_with_llm(news_event)
            
            print(f"\n🤖 LLM Analysis:")
            print(f"   Sentiment: {analysis['sentiment']:.2f}")
            print(f"   Impact: {analysis['market_impact']}")
            print(f"   Direction: {analysis['price_direction']}")
            print(f"   Confidence: {analysis['confidence']:.2f}")
            print(f"   Recommendation: {analysis['trading_recommendation']['action'].upper()}")
            print(f"   Reasoning: {analysis['trading_recommendation']['reasoning']}")
            
            # Make trading decisions
            for symbol in news_event.affected_symbols:
                if symbol in self.symbols:
                    action = analysis['trading_recommendation']['action']
                    strength = analysis['trading_recommendation']['strength']
                    
                    if action in ["buy", "sell"] and strength > 0.3:
                        # Calculate position size based on strength and confidence
                        max_position = 0.1 * self.portfolio["cash"] / self.current_prices[symbol]
                        quantity = int(max_position * strength * analysis['confidence'])
                        
                        if quantity > 0:
                            self.execute_trade(symbol, action, quantity, self.current_prices[symbol])
                    
                    # Update market prices
                    self.simulate_market_response(
                        symbol, 
                        analysis['sentiment'],
                        analysis['market_impact']
                    )
        
        # Calculate final portfolio value
        total_value = self.portfolio["cash"]
        for symbol, shares in self.portfolio.items():
            if symbol != "cash" and shares > 0:
                total_value += shares * self.current_prices[symbol]
        
        print("\n" + "="*60)
        print("📊 SIMULATION RESULTS")
        print("="*60)
        print(f"Final Cash: ${self.portfolio['cash']:,.2f}")
        print(f"Total Portfolio Value: ${total_value:,.2f}")
        print(f"P&L: ${total_value - 1000000:,.2f} ({(total_value/1000000 - 1)*100:.2f}%)")
        print(f"Total Trades: {len(self.trades)}")
        
        print("\nPortfolio Holdings:")
        for symbol, shares in self.portfolio.items():
            if symbol != "cash" and shares > 0:
                value = shares * self.current_prices[symbol]
                print(f"  {symbol}: {shares} shares (${value:,.2f})")
        
        # Save results
        self.save_results()
        
    def save_results(self):
        """Save simulation results to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"artifacts/llm_simulation_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trades
        with open(output_dir / "trades.json", "w") as f:
            json.dump([{
                "timestamp": t["timestamp"].isoformat(),
                "symbol": t["symbol"],
                "action": t["action"],
                "quantity": t["quantity"],
                "price": t["price"],
                "total": t["total"]
            } for t in self.trades], f, indent=2)
        
        # Save news history
        with open(output_dir / "news_history.json", "w") as f:
            json.dump([{
                "timestamp": n.timestamp.isoformat(),
                "headline": n.headline,
                "content": n.content,
                "category": n.category.value,
                "affected_symbols": n.affected_symbols,
                "sentiment_score": n.sentiment_score,
                "importance": n.importance
            } for n in self.news_history], f, indent=2)
        
        # Save final state
        with open(output_dir / "final_state.json", "w") as f:
            total_value = self.portfolio["cash"]
            for symbol, shares in self.portfolio.items():
                if symbol != "cash" and shares > 0:
                    total_value += shares * self.current_prices[symbol]
            
            json.dump({
                "portfolio": self.portfolio,
                "current_prices": self.current_prices,
                "total_value": total_value,
                "pnl": total_value - 1000000,
                "pnl_pct": (total_value/1000000 - 1) * 100,
                "num_trades": len(self.trades)
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_dir}")

def main():
    """Main entry point"""
    try:
        simulator = LLMMarketSimulator()
        simulator.run_simulation(num_events=5)
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        print(f"\n❌ Error: {e}")
        print("\nMake sure your OpenAI API key is set correctly in the .env file")

if __name__ == "__main__":
    main()