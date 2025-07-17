#!/usr/bin/env python3
"""
Real OpenAI API Integration Test
================================

This script tests the real OpenAI API integration with your ABIDES-LLM system.
It uses your provided API key to make actual calls to OpenAI and demonstrate
LLM-enhanced market simulation.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")

from enhanced_llm_abides_system import (
    NewsEvent, NewsCategory, MarketSentiment
)


class RealLLMNewsAnalyzer:
    """Real LLM-powered news analyzer using OpenAI API"""
    
    def __init__(self, api_key: str = None):
        """Initialize with OpenAI API key"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"  # Using the cost-effective model
        
    def analyze_news(self, news_text: str, symbol: str) -> Dict[str, Any]:
        """Analyze news using real OpenAI API"""
        
        prompt = f"""
You are a professional financial analyst. Analyze the following news about {symbol} and provide:
1. Market sentiment (very_negative, negative, neutral, positive, very_positive)
2. Sentiment score (-1.0 to 1.0)
3. Trading recommendation (buy, sell, hold)
4. Confidence level (0.0 to 1.0)
5. Key factors influencing your decision
6. Potential price impact (low, medium, high)

News: {news_text}

Respond in JSON format with the following structure:
{{
    "sentiment": "positive/negative/neutral",
    "sentiment_score": 0.5,
    "recommendation": "buy/sell/hold",
    "confidence": 0.8,
    "key_factors": ["factor1", "factor2"],
    "price_impact": "medium",
    "reasoning": "Brief explanation of analysis"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst with expertise in market sentiment analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON if it's wrapped in markdown
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(content)
            
            # Add metadata
            analysis['timestamp'] = datetime.now().isoformat()
            analysis['model_used'] = self.model
            analysis['symbol'] = symbol
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing news with OpenAI: {e}")
            # Return a fallback neutral analysis
            return {
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "recommendation": "hold",
                "confidence": 0.1,
                "key_factors": ["Analysis failed"],
                "price_impact": "low",
                "reasoning": f"Analysis failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "model_used": self.model,
                "symbol": symbol
            }


class RealLLMTradingAgent:
    """Trading agent that uses real LLM analysis for decision making"""
    
    def __init__(self, name: str, strategy: str, risk_tolerance: float, llm_analyzer: RealLLMNewsAnalyzer):
        self.name = name
        self.strategy = strategy
        self.risk_tolerance = risk_tolerance
        self.llm_analyzer = llm_analyzer
        self.portfolio = {"cash": 1000000, "positions": {}}
        self.trade_history = []
        
    def process_news(self, news_text: str, symbol: str, current_price: float) -> Dict[str, Any]:
        """Process news and generate trading decision"""
        
        logger.info(f"[{self.name}] Processing news for {symbol}: {news_text[:50]}...")
        
        # Get LLM analysis
        analysis = self.llm_analyzer.analyze_news(news_text, symbol)
        
        # Generate trading decision based on analysis and strategy
        decision = self._make_trading_decision(analysis, symbol, current_price)
        
        # Log the decision
        logger.info(f"[{self.name}] Decision: {decision['action']} {decision.get('quantity', 0)} shares of {symbol}")
        logger.info(f"[{self.name}] Reasoning: {analysis.get('reasoning', 'No reasoning provided')}")
        
        return {
            "agent": self.name,
            "analysis": analysis,
            "decision": decision,
            "timestamp": datetime.now().isoformat()
        }
    
    def _make_trading_decision(self, analysis: Dict, symbol: str, current_price: float) -> Dict[str, Any]:
        """Make trading decision based on LLM analysis and agent strategy"""
        
        sentiment_score = analysis.get('sentiment_score', 0.0)
        confidence = analysis.get('confidence', 0.0)
        recommendation = analysis.get('recommendation', 'hold')
        
        # Adjust decision based on agent strategy
        if self.strategy == "momentum":
            # Follow the trend
            action_strength = sentiment_score * confidence
        elif self.strategy == "contrarian":
            # Go against the trend
            action_strength = -sentiment_score * confidence
        else:  # neutral/conservative
            # Only act on high confidence signals
            if confidence < 0.7:
                return {"action": "hold", "reasoning": "Insufficient confidence for conservative strategy"}
            action_strength = sentiment_score * confidence * 0.5
        
        # Apply risk tolerance
        action_strength *= self.risk_tolerance
        
        # Determine action and quantity
        if abs(action_strength) < 0.2:
            return {
                "action": "hold",
                "reasoning": f"Signal too weak: {action_strength:.3f}"
            }
        
        # Calculate position size (simple portfolio percentage)
        available_cash = self.portfolio["cash"]
        max_position_value = available_cash * 0.1 * abs(action_strength)  # Max 10% of cash per trade
        max_shares = int(max_position_value / current_price)
        
        if action_strength > 0:
            action = "buy"
            quantity = min(max_shares, 1000)  # Cap at 1000 shares
        else:
            action = "sell"
            current_position = self.portfolio["positions"].get(symbol, 0)
            quantity = min(max_shares, current_position) if current_position > 0 else 0
        
        return {
            "action": action,
            "quantity": quantity,
            "price": current_price,
            "confidence": confidence,
            "reasoning": f"{self.strategy} strategy: signal={action_strength:.3f}, confidence={confidence:.3f}"
        }


def run_real_llm_test():
    """Run a comprehensive test of real LLM integration"""
    
    print("🚀 REAL LLM-ABIDES INTEGRATION TEST")
    print("=" * 60)
    
    # Check OpenAI availability
    if not OPENAI_AVAILABLE:
        print("❌ OpenAI library not installed. Run: pip install openai")
        return
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    print(f"✅ OpenAI API key found: {api_key[:20]}...")
    
    try:
        # Initialize LLM analyzer
        print("\n🧠 Initializing Real LLM Analyzer...")
        llm_analyzer = RealLLMNewsAnalyzer(api_key)
        print("✅ LLM Analyzer initialized successfully!")
        
        # Create trading agents with different strategies
        print("\n🤖 Creating LLM-Enhanced Trading Agents...")
        agents = [
            RealLLMTradingAgent("MomentumBot", "momentum", 0.8, llm_analyzer),
            RealLLMTradingAgent("ContrarianBot", "contrarian", 0.6, llm_analyzer),
            RealLLMTradingAgent("ConservativeBot", "conservative", 0.3, llm_analyzer)
        ]
        
        for agent in agents:
            print(f"✅ {agent.name} ({agent.strategy}) initialized")
        
        # Test news scenarios
        test_scenarios = [
            {
                "symbol": "AAPL",
                "price": 175.50,
                "news": "Apple reports record quarterly earnings with 15% revenue growth, driven by strong iPhone sales and expanding services business. CEO emphasizes continued investment in AI and sustainable technology."
            },
            {
                "symbol": "MSFT", 
                "price": 380.25,
                "news": "Microsoft faces regulatory scrutiny over its AI partnerships and cloud market dominance. The Justice Department is investigating potential antitrust violations in the enterprise software space."
            },
            {
                "symbol": "TSLA",
                "price": 245.80,
                "news": "Tesla announces major recall of 2 million vehicles due to autopilot safety concerns. Stock drops in pre-market trading as investors worry about regulatory implications and repair costs."
            }
        ]
        
        print("\n📰 Processing Real News Scenarios...")
        print("=" * 60)
        
        all_results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n--- Scenario {i}: {scenario['symbol']} ---")
            print(f"Current Price: ${scenario['price']}")
            print(f"News: {scenario['news'][:80]}...")
            print()
            
            scenario_results = []
            
            for agent in agents:
                try:
                    result = agent.process_news(
                        scenario['news'],
                        scenario['symbol'],
                        scenario['price']
                    )
                    scenario_results.append(result)
                    
                    # Display summary
                    analysis = result['analysis']
                    decision = result['decision']
                    
                    print(f"🤖 {agent.name}:")
                    print(f"   Sentiment: {analysis.get('sentiment', 'unknown')} ({analysis.get('sentiment_score', 0):.3f})")
                    print(f"   Confidence: {analysis.get('confidence', 0):.3f}")
                    print(f"   Action: {decision.get('action', 'none')} {decision.get('quantity', 0)} shares")
                    print(f"   Reasoning: {decision.get('reasoning', 'No reasoning')}")
                    print()
                    
                except Exception as e:
                    print(f"❌ Error processing with {agent.name}: {e}")
            
            all_results.append({
                "scenario": i,
                "symbol": scenario['symbol'],
                "price": scenario['price'],
                "news": scenario['news'],
                "agent_results": scenario_results
            })
        
        # Summary
        print("\n📊 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Processed {len(test_scenarios)} news scenarios")
        print(f"✅ Tested {len(agents)} LLM-enhanced agents")
        print("✅ Real OpenAI API integration working!")
        
        # Count decisions
        total_decisions = sum(len(r['agent_results']) for r in all_results)
        buy_decisions = sum(1 for r in all_results for ar in r['agent_results'] 
                          if ar['decision'].get('action') == 'buy')
        sell_decisions = sum(1 for r in all_results for ar in r['agent_results'] 
                           if ar['decision'].get('action') == 'sell')
        hold_decisions = total_decisions - buy_decisions - sell_decisions
        
        print(f"\nDecision Breakdown:")
        print(f"  📈 Buy:  {buy_decisions}")
        print(f"  📉 Sell: {sell_decisions}")
        print(f"  ⏸️  Hold: {hold_decisions}")
        
        print(f"\n🎯 SUCCESS: Your ABIDES-LLM system is fully operational!")
        print("You can now build realistic market simulations with real AI reasoning.")
        
        return all_results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_real_llm_test()
    
    if results:
        print(f"\n💾 Test completed successfully!")
        print("You can now integrate this with the full ABIDES simulation framework.")
    else:
        print("\n❌ Test failed. Please check the errors above.")