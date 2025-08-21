#!/usr/bin/env python3
"""
Demo: LLM-Powered News Sentiment Analysis Impact on Trading
===========================================================

This demonstrates how the ABIDES-LLM integration uses Large Language Models
to analyze market news and drive trading decisions.
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Check for OpenAI API key
has_openai = bool(os.getenv("OPENAI_API_KEY"))

def run_enhanced_demo():
    """Run the enhanced ABIDES-LLM demo"""
    print("=" * 70)
    print("🤖 ABIDES-LLM MARKET SIMULATOR DEMO")
    print("=" * 70)
    print()
    
    print("📋 Project Overview:")
    print("This market simulator combines ABIDES (Agent-Based Interactive Discrete")
    print("Event Simulation) with Large Language Models to create realistic market")
    print("simulations where trading agents can:")
    print()
    print("✅ Analyze news sentiment using LLM")
    print("✅ Make intelligent trading decisions")
    print("✅ React to market events dynamically")
    print("✅ Generate realistic market microstructure")
    print()
    
    if has_openai:
        print("🟢 OpenAI API Key detected - Full LLM features enabled!")
        from openai import OpenAI
        client = OpenAI()
        
        # Test the API with a simple news analysis
        print("\n📰 Testing LLM News Analysis...")
        test_news = "Apple announces record-breaking quarterly earnings, beating analyst expectations by 15%"
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Analyze the sentiment and market impact of this news."},
                    {"role": "user", "content": f"News: {test_news}\n\nProvide: 1) Sentiment (-1 to +1), 2) Expected price impact (%), 3) Brief reasoning"}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            print(f"\nNews: {test_news}")
            print(f"LLM Analysis: {response.choices[0].message.content}")
            
        except Exception as e:
            print(f"⚠️  Error calling OpenAI API: {e}")
            print("Please check your API key is valid and has credits.")
    else:
        print("🟡 No OpenAI API Key - Using mock LLM responses")
        print("   To enable real LLM analysis, set: export OPENAI_API_KEY='your-key'")
    
    print("\n" + "=" * 70)
    print("📊 AVAILABLE SIMULATIONS:")
    print("=" * 70)
    
    simulations = [
        {
            "name": "Basic Market Simulation",
            "command": "python3 main.py --demo",
            "description": "Simple demonstration with 3 traders and news events"
        },
        {
            "name": "Enhanced Order Book Recording",
            "command": "python3 enhanced_orderbook_main.py --config quick_test",
            "description": "Full order book simulation with data recording and analysis"
        },
        {
            "name": "Real Market Validation",
            "command": "python3 enhanced_orderbook_main.py --config quick_test --validate-real --val-symbol AAPL",
            "description": "Compare simulation with real market data from yfinance"
        },
        {
            "name": "Large Scale Data Generation",
            "command": "python3 main.py --scale-data",
            "description": "Generate massive datasets for research (millions of orders)"
        },
        {
            "name": "Market Microstructure Analysis",
            "command": "python3 src/validation_viz.py --db quick_test_orderbook.db --symbol AAPL",
            "description": "Generate plots and metrics for market quality analysis"
        }
    ]
    
    for i, sim in enumerate(simulations, 1):
        print(f"\n{i}. {sim['name']}")
        print(f"   Command: {sim['command']}")
        print(f"   {sim['description']}")
    
    print("\n" + "=" * 70)
    print("🚀 KEY FEATURES:")
    print("=" * 70)
    
    features = {
        "LLM Integration": [
            "News sentiment analysis with GPT models",
            "Dynamic trading strategy adaptation",
            "Market impact prediction"
        ],
        "Market Microstructure": [
            "Realistic bid-ask spreads",
            "Order book dynamics",
            "Price discovery mechanisms"
        ],
        "Agent Types": [
            "Retail traders (65%)",
            "Institutional traders (15%)",
            "High-frequency traders (12%)",
            "Market makers (8%)"
        ],
        "Data Export": [
            "Complete order flow",
            "Trade executions",
            "Market snapshots",
            "Agent performance metrics"
        ]
    }
    
    for category, items in features.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")
    
    print("\n" + "=" * 70)
    print("📈 SAMPLE OUTPUT:")
    print("=" * 70)
    
    # Show sample output from existing database if available
    db_path = "quick_test_orderbook.db"
    if os.path.exists(db_path):
        print(f"\n✅ Found existing simulation database: {db_path}")
        
        # Try to read some stats
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get trade count
            cursor.execute("SELECT COUNT(*) FROM trades")
            trade_count = cursor.fetchone()[0]
            
            # Get order count
            cursor.execute("SELECT COUNT(*) FROM orders")
            order_count = cursor.fetchone()[0]
            
            # Get symbols
            cursor.execute("SELECT DISTINCT symbol FROM trades")
            symbols = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            print(f"\nDatabase Statistics:")
            print(f"  Total Orders: {order_count:,}")
            print(f"  Total Trades: {trade_count:,}")
            print(f"  Symbols: {', '.join(symbols)}")
            
        except Exception as e:
            print(f"  (Could not read database: {e})")
    
    print("\n" + "=" * 70)
    print("🎯 NEXT STEPS:")
    print("=" * 70)
    print("\n1. Run a basic simulation:")
    print("   python3 main.py --demo")
    print("\n2. Generate order book data with LLM analysis:")
    print("   python3 enhanced_orderbook_main.py --config quick_test")
    print("\n3. Validate against real market:")
    print("   python3 enhanced_orderbook_main.py --config quick_test --validate-real --val-symbol AAPL")
    print("\n4. Generate visualization plots:")
    print("   python3 src/validation_viz.py --db quick_test_orderbook.db --symbol AAPL")
    
    print("\n" + "=" * 70)
    print("✨ Happy trading with AI! 🤖📈")
    print("=" * 70)

if __name__ == "__main__":
    run_enhanced_demo()