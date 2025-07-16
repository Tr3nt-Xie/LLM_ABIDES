#!/usr/bin/env python3
"""
Simple Test Script for Realistic Market Simulation
==================================================

This script tests the basic functionality of the market simulation system
without requiring API keys or external dependencies.
"""

import sys
import traceback
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported successfully"""
    print("🧪 Testing module imports...")
    
    try:
        from enhanced_llm_abides_system import (
            NewsEvent, MarketSignal, RealisticNewsGenerator, NewsCategory
        )
        print("✅ Enhanced LLM ABIDES system imported successfully")
    except Exception as e:
        print(f"❌ Failed to import enhanced_llm_abides_system: {e}")
        return False
    
    try:
        from enhanced_abides_bridge import (
            EnhancedABIDESOrder, RealisticMarketDataGenerator, 
            EnhancedLLMInfluencedAgent, MarketState
        )
        print("✅ Enhanced ABIDES bridge imported successfully")
    except Exception as e:
        print(f"❌ Failed to import enhanced_abides_bridge: {e}")
        return False
    
    try:
        from realistic_market_simulation import (
            RealisticMarketSimulation, SimulationConfig
        )
        print("✅ Realistic market simulation imported successfully")
    except Exception as e:
        print(f"❌ Failed to import realistic_market_simulation: {e}")
        return False
    
    return True


def test_news_generation():
    """Test news generation functionality"""
    print("\n📰 Testing news generation...")
    
    try:
        from enhanced_llm_abides_system import RealisticNewsGenerator, NewsCategory
        
        # Create news generator
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        news_gen = RealisticNewsGenerator(symbols)
        
        # Generate some news events
        for i in range(3):
            news_event = news_gen.generate_realistic_event()
            print(f"  📢 Generated news: {news_event.headline}")
            print(f"     Category: {news_event.category.value}")
            print(f"     Sentiment: {news_event.sentiment_score:.2f}")
            print(f"     Affected symbols: {news_event.affected_symbols}")
        
        print("✅ News generation test passed")
        return True
        
    except Exception as e:
        print(f"❌ News generation test failed: {e}")
        traceback.print_exc()
        return False


def test_market_data_generation():
    """Test market data generation"""
    print("\n📊 Testing market data generation...")
    
    try:
        from enhanced_abides_bridge import RealisticMarketDataGenerator
        
        # Create market data generator
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        market_gen = RealisticMarketDataGenerator(symbols)
        
        # Generate market data
        market_update = market_gen.update_market_data()
        market_data = market_update['symbols']  # Extract symbols data
        
        print(f"  📈 Generated market data for {len(market_data)} symbols:")
        for symbol, data in market_data.items():
            print(f"     {symbol}: Price=${data['price']:.2f}, "
                  f"Bid=${data['bid']:.2f}, Ask=${data['ask']:.2f}, "
                  f"Volume={data['volume']:,}")
        
        print("✅ Market data generation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Market data generation test failed: {e}")
        traceback.print_exc()
        return False


def test_agent_creation():
    """Test agent creation and initialization"""
    print("\n🤖 Testing agent creation...")
    
    try:
        from enhanced_abides_bridge import EnhancedLLMInfluencedAgent
        
        # Create test agent
        agent = EnhancedLLMInfluencedAgent(
            agent_id="TestAgent_001",
            strategy="momentum",
            base_capital=1000000
        )
        
        print(f"  🤖 Created agent: {agent.agent_id}")
        print(f"     Strategy: {agent.strategy}")
        print(f"     Initial capital: ${agent.base_capital:,.2f}")
        print(f"     Portfolio value: ${agent.portfolio.total_value:,.2f}")
        
        print("✅ Agent creation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        traceback.print_exc()
        return False


def test_order_creation():
    """Test order creation and processing"""
    print("\n📋 Testing order creation...")
    
    try:
        from enhanced_abides_bridge import EnhancedABIDESOrder
        from datetime import datetime
        
        # Create test order
        order = EnhancedABIDESOrder(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            order_type="LIMIT",
            limit_price=150.00,
            agent_id="TestAgent_001"
        )
        
        print(f"  📋 Created order: {order.order_id}")
        print(f"     Symbol: {order.symbol}")
        print(f"     Side: {order.side}")
        print(f"     Quantity: {order.quantity}")
        print(f"     Limit Price: ${order.limit_price}")
        
        # Test order conversion
        abides_format = order.to_abides_format()
        print(f"     ABIDES format keys: {list(abides_format.keys())}")
        
        print("✅ Order creation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Order creation test failed: {e}")
        traceback.print_exc()
        return False


def test_simulation_config():
    """Test simulation configuration"""
    print("\n⚙️  Testing simulation configuration...")
    
    try:
        from realistic_market_simulation import SimulationConfig
        
        # Create test configuration
        config = SimulationConfig(
            symbols=['AAPL', 'MSFT'],
            simulation_duration_hours=1,
            news_frequency=0.1,
            llm_agents_count=2,
            abides_agents_count=5,
            real_time_factor=120.0,
            save_data=False
        )
        
        print(f"  ⚙️  Configuration created:")
        print(f"     Symbols: {config.symbols}")
        print(f"     Duration: {config.simulation_duration_hours} hours")
        print(f"     Speed factor: {config.real_time_factor}x")
        print(f"     Total agents: {config.llm_agents_count + config.abides_agents_count}")
        
        print("✅ Simulation configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Simulation configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_mini_simulation():
    """Test a very short simulation run"""
    print("\n🚀 Testing mini simulation (no LLM APIs)...")
    
    try:
        from realistic_market_simulation import RealisticMarketSimulation, SimulationConfig
        
        # Create minimal configuration
        config = SimulationConfig(
            symbols=['AAPL', 'MSFT'],
            simulation_duration_hours=0.01,  # 36 seconds
            news_frequency=0.5,  # Higher frequency for testing
            llm_agents_count=1,
            abides_agents_count=3,
            real_time_factor=300.0,  # Very fast simulation
            save_data=False
        )
        
        # Create simulation
        simulation = RealisticMarketSimulation(config)
        
        print(f"  🚀 Mini simulation created with {len(config.symbols)} symbols")
        print(f"     Expected runtime: ~{(config.simulation_duration_hours * 60) / config.real_time_factor:.1f} seconds")
        
        # Run simulation
        print("  ⏳ Running mini simulation...")
        simulation.start_simulation(blocking=True)
        
        print("✅ Mini simulation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Mini simulation test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("🧪 Running Realistic Market Simulation Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_news_generation,
        test_market_data_generation,
        test_agent_creation,
        test_order_creation,
        test_simulation_config,
        test_mini_simulation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("🧪 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! The simulation system is ready to use.")
        print("\nNext steps:")
        print("1. Set up your OpenAI API key in .env file")
        print("2. Run: python run_simulation.py --config fast_test")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the error messages above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)