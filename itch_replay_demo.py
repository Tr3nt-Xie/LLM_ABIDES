#!/usr/bin/env python3
"""
NASDAQ ITCH Data Replay Demo with ABIDES-LLM
============================================

This demo shows how to replay real NASDAQ ITCH market data through the ABIDES-LLM
simulator, allowing LLM agents to trade based on actual historical market events.
"""

import json
import sys
import logging
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

# Import our modules
from itch_data_parser import LOBSTERDataParser, ITCHDataIntegrator
from enhanced_llm_abides_system import EnhancedLLMNewsAnalyzer
import openai
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ITCHReplaySimulator:
    """Replays NASDAQ ITCH data with LLM-enhanced trading agents"""
    
    def __init__(self, symbol: str, date: str):
        """
        Initialize replay simulator
        
        Args:
            symbol: Stock symbol to replay
            date: Date of data in YYYY-MM-DD format
        """
        self.symbol = symbol
        self.date = date
        
        # Initialize ITCH data parser
        self.parser = LOBSTERDataParser(symbol, date, "/workspace")
        
        # Parse data
        logger.info(f"Loading NASDAQ ITCH data for {symbol} on {date}")
        self.messages = self.parser.parse_messages()
        self.orderbook_snapshots = self.parser.parse_orderbook()
        self.trades_df = self.parser.get_trades()
        
        # Initialize LLM if available
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.use_llm = bool(self.api_key)
        if self.use_llm:
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info("✅ LLM integration enabled")
        else:
            logger.info("⚠️ No OpenAI API key - using rule-based analysis")
        
        # Trading agents
        self.agents = self.initialize_agents()
        
        # Market state
        self.current_orderbook = None
        self.portfolio = {
            "cash": 1000000,  # $1M starting capital
            symbol: 0
        }
        self.agent_trades = []
        
    def initialize_agents(self) -> Dict:
        """Initialize trading agents"""
        
        agents = {
            "momentum": {
                "name": "MomentumAgent",
                "strategy": "momentum",
                "position": 0,
                "cash": 333333,
                "trades": []
            },
            "mean_reversion": {
                "name": "MeanReversionAgent", 
                "strategy": "mean_reversion",
                "position": 0,
                "cash": 333333,
                "trades": []
            },
            "market_maker": {
                "name": "MarketMakerAgent",
                "strategy": "market_maker",
                "position": 0,
                "cash": 333334,
                "trades": []
            }
        }
        
        return agents
    
    def analyze_market_event(self, event_window: List) -> Dict:
        """
        Analyze a window of market events using LLM or rules
        
        Args:
            event_window: List of recent market events
            
        Returns:
            Analysis dict with trading signals
        """
        
        if self.use_llm and len(event_window) >= 10:
            # Use LLM for sophisticated analysis
            return self.llm_analysis(event_window)
        else:
            # Use rule-based analysis
            return self.rule_based_analysis(event_window)
    
    def llm_analysis(self, event_window: List) -> Dict:
        """Analyze market events using LLM"""
        
        # Prepare market summary
        trades = [e for e in event_window if e['type'] in ['VISIBLE_EXECUTION', 'HIDDEN_EXECUTION']]
        
        if not trades:
            return {"signal": "hold", "confidence": 0.5, "reasoning": "No recent trades"}
        
        # Calculate metrics
        prices = [t['price'] for t in trades]
        volumes = [t['size'] for t in trades]
        
        prompt = f"""
        Analyze this NASDAQ market data and provide trading signals:
        
        Symbol: {self.symbol}
        Recent trades: {len(trades)}
        Price range: ${min(prices):.2f} - ${max(prices):.2f}
        Average price: ${np.mean(prices):.2f}
        Total volume: {sum(volumes):,}
        Price trend: {"up" if prices[-1] > prices[0] else "down" if prices[-1] < prices[0] else "flat"}
        
        Current orderbook:
        - Best bid: ${self.current_orderbook.bid_price:.2f} ({self.current_orderbook.bid_size} shares)
        - Best ask: ${self.current_orderbook.ask_price:.2f} ({self.current_orderbook.ask_size} shares)
        - Spread: ${self.current_orderbook.spread:.4f}
        
        Provide analysis in JSON format:
        {{
            "signal": <"buy", "sell", or "hold">,
            "confidence": <float 0-1>,
            "target_size": <integer shares>,
            "reasoning": <brief explanation>,
            "market_condition": <"bullish", "bearish", or "neutral">
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a quantitative trader analyzing NASDAQ market data."},
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
            return self.rule_based_analysis(event_window)
    
    def rule_based_analysis(self, event_window: List) -> Dict:
        """Simple rule-based market analysis"""
        
        trades = [e for e in event_window if e['type'] in ['VISIBLE_EXECUTION', 'HIDDEN_EXECUTION']]
        
        if not trades or not self.current_orderbook:
            return {
                "signal": "hold",
                "confidence": 0.5,
                "target_size": 0,
                "reasoning": "Insufficient data",
                "market_condition": "neutral"
            }
        
        # Calculate simple momentum
        if len(trades) >= 2:
            recent_price = trades[-1]['price']
            prev_price = trades[0]['price']
            price_change = (recent_price - prev_price) / prev_price
            
            if price_change > 0.001:  # 0.1% up
                return {
                    "signal": "buy",
                    "confidence": min(0.7, abs(price_change) * 100),
                    "target_size": 100,
                    "reasoning": f"Positive momentum: {price_change:.2%}",
                    "market_condition": "bullish"
                }
            elif price_change < -0.001:  # 0.1% down
                return {
                    "signal": "sell",
                    "confidence": min(0.7, abs(price_change) * 100),
                    "target_size": 100,
                    "reasoning": f"Negative momentum: {price_change:.2%}",
                    "market_condition": "bearish"
                }
        
        return {
            "signal": "hold",
            "confidence": 0.5,
            "target_size": 0,
            "reasoning": "No clear signal",
            "market_condition": "neutral"
        }
    
    def execute_agent_trades(self, analysis: Dict, timestamp: float):
        """Execute trades based on analysis for each agent"""
        
        if not self.current_orderbook:
            return
        
        signal = analysis.get("signal", "hold")
        confidence = analysis.get("confidence", 0.5)
        target_size = analysis.get("target_size", 100)
        
        for agent_key, agent in self.agents.items():
            # Different agents react differently to signals
            if agent["strategy"] == "momentum":
                # Momentum agent follows the signal
                if signal == "buy" and confidence > 0.6:
                    self.execute_trade(agent, "buy", target_size, timestamp)
                elif signal == "sell" and confidence > 0.6:
                    self.execute_trade(agent, "sell", target_size, timestamp)
                    
            elif agent["strategy"] == "mean_reversion":
                # Mean reversion agent does opposite
                if signal == "buy" and confidence > 0.7:
                    self.execute_trade(agent, "sell", target_size // 2, timestamp)
                elif signal == "sell" and confidence > 0.7:
                    self.execute_trade(agent, "buy", target_size // 2, timestamp)
                    
            elif agent["strategy"] == "market_maker":
                # Market maker provides liquidity
                if self.current_orderbook.spread > 0.02:  # Wide spread
                    # Try to profit from spread
                    self.execute_trade(agent, "buy", 50, timestamp, 
                                     price=self.current_orderbook.bid_price + 0.01)
                    self.execute_trade(agent, "sell", 50, timestamp,
                                     price=self.current_orderbook.ask_price - 0.01)
    
    def execute_trade(self, agent: Dict, side: str, size: int, timestamp: float, price: float = None):
        """Execute a trade for an agent"""
        
        if price is None:
            # Use market price
            price = self.current_orderbook.ask_price if side == "buy" else self.current_orderbook.bid_price
        
        cost = size * price
        
        if side == "buy":
            if agent["cash"] >= cost:
                agent["cash"] -= cost
                agent["position"] += size
                trade = {
                    "timestamp": timestamp,
                    "agent": agent["name"],
                    "side": "buy",
                    "size": size,
                    "price": price,
                    "cost": cost
                }
                agent["trades"].append(trade)
                self.agent_trades.append(trade)
                logger.info(f"{agent['name']} BUY {size} @ ${price:.2f}")
                
        elif side == "sell" and agent["position"] >= size:
            agent["cash"] += size * price
            agent["position"] -= size
            trade = {
                "timestamp": timestamp,
                "agent": agent["name"],
                "side": "sell",
                "size": size,
                "price": price,
                "proceeds": size * price
            }
            agent["trades"].append(trade)
            self.agent_trades.append(trade)
            logger.info(f"{agent['name']} SELL {size} @ ${price:.2f}")
    
    def run_replay(self, start_idx: int = 0, num_events: int = 1000):
        """
        Run the replay simulation
        
        Args:
            start_idx: Starting index in the data
            num_events: Number of events to replay
        """
        
        print("\n" + "="*60)
        print("🔄 NASDAQ ITCH DATA REPLAY SIMULATION")
        print("="*60)
        print(f"Symbol: {self.symbol}")
        print(f"Date: {self.date}")
        print(f"Events to replay: {num_events}")
        print(f"LLM Integration: {'✅ Enabled' if self.use_llm else '❌ Disabled'}")
        print(f"Starting capital: ${sum(a['cash'] for a in self.agents.values()):,.2f}")
        print()
        
        # Process events
        event_window = []
        analysis_count = 0
        
        for i in range(start_idx, min(start_idx + num_events, len(self.messages))):
            msg = self.messages[i]
            
            # Update orderbook
            if i < len(self.orderbook_snapshots):
                self.current_orderbook = self.orderbook_snapshots[i]
            
            # Add to event window
            event_window.append({
                'timestamp': msg.timestamp,
                'type': msg.msg_type.name,
                'price': msg.price,
                'size': msg.size,
                'side': msg.side
            })
            
            # Keep window size manageable
            if len(event_window) > 50:
                event_window.pop(0)
            
            # Analyze every 100 events
            if i > 0 and i % 100 == 0:
                analysis = self.analyze_market_event(event_window)
                self.execute_agent_trades(analysis, msg.timestamp)
                analysis_count += 1
                
                if analysis_count % 10 == 0:
                    print(f"Processed {i} events, {len(self.agent_trades)} trades executed")
        
        # Calculate final results
        self.print_results()
    
    def print_results(self):
        """Print simulation results"""
        
        print("\n" + "="*60)
        print("📊 REPLAY SIMULATION RESULTS")
        print("="*60)
        
        total_pnl = 0
        
        for agent_key, agent in self.agents.items():
            # Calculate P&L
            initial_cash = 333333
            current_value = agent["cash"]
            if agent["position"] > 0 and self.current_orderbook:
                current_value += agent["position"] * self.current_orderbook.mid_price
            
            pnl = current_value - initial_cash
            pnl_pct = (pnl / initial_cash) * 100
            total_pnl += pnl
            
            print(f"\n{agent['name']}:")
            print(f"  Strategy: {agent['strategy']}")
            print(f"  Cash: ${agent['cash']:,.2f}")
            print(f"  Position: {agent['position']} shares")
            print(f"  Trades: {len(agent['trades'])}")
            print(f"  P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        
        print(f"\n📈 Total P&L: ${total_pnl:,.2f}")
        print(f"Total trades executed: {len(self.agent_trades)}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save replay results to file"""
        
        output_dir = Path("artifacts/itch_replay")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save agent trades
        if self.agent_trades:
            trades_df = pd.DataFrame(self.agent_trades)
            trades_file = output_dir / f"{self.symbol}_{self.date}_replay_trades.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"\n💾 Trades saved to: {trades_file}")
        
        # Save summary
        summary = {
            "symbol": self.symbol,
            "date": self.date,
            "total_events": len(self.messages),
            "total_trades": len(self.agent_trades),
            "llm_enabled": self.use_llm,
            "agents": {}
        }
        
        for agent_key, agent in self.agents.items():
            summary["agents"][agent_key] = {
                "name": agent["name"],
                "strategy": agent["strategy"],
                "final_cash": agent["cash"],
                "final_position": agent["position"],
                "num_trades": len(agent["trades"])
            }
        
        summary_file = output_dir / f"{self.symbol}_{self.date}_replay_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📋 Summary saved to: {summary_file}")

def main():
    """Main entry point"""
    
    # Check available data
    available = []
    for symbol in ["AMZN", "AAPL", "MSFT"]:
        if Path(f"/workspace/{symbol}_2012-06-21_34200000_57600000_message_1.csv").exists():
            available.append(symbol)
    
    if not available:
        print("❌ No NASDAQ ITCH data found. Please download LOBSTER sample files first.")
        return
    
    print(f"Available symbols: {', '.join(available)}")
    
    # Run replay for first available symbol
    symbol = available[0]
    simulator = ITCHReplaySimulator(symbol, "2012-06-21")
    
    # Run replay on first 2000 events (for demo purposes)
    simulator.run_replay(start_idx=1000, num_events=2000)

if __name__ == "__main__":
    main()