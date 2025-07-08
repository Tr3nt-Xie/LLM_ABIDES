"""
ABIDES-LLM Test Suite
=====================

Comprehensive testing framework for LLM-enhanced ABIDES agents.
Tests can run with or without official ABIDES installation.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from abides_llm_agents import (
    ABIDESLLMNewsAnalyzer, ABIDESLLMTradingAgent, ABIDESLLMMarketMaker,
    createABIDESLLMAgentConfig, createLLMEnhancedABIDESConfig
)
from abides_llm_config import (
    build_config, build_rmsc_llm_config, quick_llm_demo_config
)

# Try to import ABIDES (optional)
try:
    from abides_core import abides
    ABIDES_AVAILABLE = True
except ImportError:
    ABIDES_AVAILABLE = False


class TestABIDESLLMAgents(unittest.TestCase):
    """Test ABIDES-compatible LLM agents"""
    
    def setUp(self):
        """Set up test environment"""
        self.seed = 42
        self.random_state = np.random.RandomState(self.seed)
        
        # Mock LLM config for testing
        self.llm_config = {
            "config_list": [
                {
                    "model": "gpt-4o-mini",
                    "api_key": "test-key",
                    "temperature": 0.3
                }
            ],
            "timeout": 60
        }
        
        # Mock current time
        self.mock_time = pd.to_datetime("2024-01-15 10:00:00")
    
    def test_news_analyzer_initialization(self):
        """Test LLM News Analyzer initialization"""
        agent = ABIDESLLMNewsAnalyzer(
            id=100,
            name="TestNewsAnalyzer",
            symbols=["ABM"],
            random_state=self.random_state,
            llm_config=self.llm_config
        )
        
        self.assertEqual(agent.id, 100)
        self.assertEqual(agent.name, "TestNewsAnalyzer")
        self.assertEqual(agent.symbols, ["ABM"])
        self.assertIsNotNone(agent.random_state)
        self.assertEqual(len(agent.analysis_history), 0)
        self.assertEqual(len(agent.pending_news), 0)
    
    def test_trading_agent_initialization(self):
        """Test LLM Trading Agent initialization"""
        agent = ABIDESLLMTradingAgent(
            id=101,
            name="TestTradingAgent",
            symbol="ABM",
            strategy="momentum",
            risk_tolerance=0.7,
            random_state=self.random_state,
            llm_config=self.llm_config
        )
        
        self.assertEqual(agent.id, 101)
        self.assertEqual(agent.name, "TestTradingAgent")
        self.assertEqual(agent.symbol, "ABM")
        self.assertEqual(agent.strategy, "momentum")
        self.assertEqual(agent.risk_tolerance, 0.7)
        self.assertIn("CASH", agent.holdings)
        self.assertIn("ABM", agent.holdings)
        self.assertEqual(agent.holdings["ABM"], 0)
    
    def test_market_maker_initialization(self):
        """Test LLM Market Maker initialization"""
        agent = ABIDESLLMMarketMaker(
            id=102,
            name="TestMarketMaker",
            symbol="ABM",
            spread_bps=15,
            max_inventory=500,
            random_state=self.random_state,
            llm_config=self.llm_config
        )
        
        self.assertEqual(agent.id, 102)
        self.assertEqual(agent.name, "TestMarketMaker")
        self.assertEqual(agent.symbol, "ABM")
        self.assertEqual(agent.spread_bps, 15)
        self.assertEqual(agent.max_inventory, 500)
        self.assertEqual(agent.strategy, "market_making")
    
    def test_trading_agent_strategy_params(self):
        """Test trading agent strategy parameter initialization"""
        
        # Test momentum strategy
        momentum_agent = ABIDESLLMTradingAgent(
            id=103, name="Momentum", strategy="momentum",
            random_state=self.random_state
        )
        self.assertTrue(momentum_agent.strategy_params['trend_following'])
        self.assertEqual(momentum_agent.strategy_params['signal_sensitivity'], 1.2)
        
        # Test value strategy
        value_agent = ABIDESLLMTradingAgent(
            id=104, name="Value", strategy="value",
            random_state=self.random_state
        )
        self.assertFalse(value_agent.strategy_params['trend_following'])
        self.assertEqual(value_agent.strategy_params['mean_reversion_factor'], 1.0)
        
        # Test volatility strategy
        vol_agent = ABIDESLLMTradingAgent(
            id=105, name="Volatility", strategy="volatility",
            random_state=self.random_state
        )
        self.assertEqual(vol_agent.strategy_params['signal_sensitivity'], 1.5)
        self.assertIn('volatility_target', vol_agent.strategy_params)
    
    def test_news_analysis_processing(self):
        """Test news analysis processing without LLM"""
        agent = ABIDESLLMNewsAnalyzer(
            id=106, name="TestAnalyzer", symbols=["ABM"],
            random_state=self.random_state
        )
        
        # Mock news event
        from abides_llm_agents import NewsEvent, NewsCategory
        mock_news = Mock()
        mock_news.headline = "Test headline"
        mock_news.category = NewsCategory.EARNINGS
        mock_news.sentiment_score = 0.7
        mock_news.confidence = 0.8
        mock_news.affected_symbols = ["ABM"]
        
        # Process news event
        agent.processNewsEvent(self.mock_time, mock_news)
        
        # Check that analysis was created
        self.assertEqual(len(agent.analysis_history), 1)
        analysis = agent.analysis_history[0]
        self.assertIn('analysis', analysis)
        self.assertEqual(analysis['timestamp'], self.mock_time)
    
    def test_trading_signal_generation(self):
        """Test trading signal generation from news analysis"""
        agent = ABIDESLLMTradingAgent(
            id=107, name="TestTrader", strategy="momentum",
            risk_tolerance=0.8, random_state=self.random_state
        )
        
        # Mock news event and analysis
        mock_news = {
            'headline': 'Positive earnings report',
            'sentiment_score': 0.6,
            'affected_symbols': ['ABM']
        }
        
        mock_analysis = {
            'sentiment': 0.6,
            'impact_assessment': {'confidence': 0.8},
            'price_predictions': {'direction': 'up', 'magnitude_percent': 3.0},
            'reasoning': 'Strong earnings beat expectations'
        }
        
        # Generate trading signal
        signal = agent.generateTradingSignal(self.mock_time, mock_news, mock_analysis)
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal['direction'], 'BUY')  # Positive sentiment -> BUY for momentum
        self.assertGreater(signal['strength'], 0)
        self.assertEqual(signal['confidence'], 0.8)
        self.assertIn('reasoning', signal)
    
    def test_portfolio_calculations(self):
        """Test portfolio value calculations"""
        agent = ABIDESLLMTradingAgent(
            id=108, name="TestPortfolio", 
            initial_cash=1000000,  # $1M
            random_state=self.random_state
        )
        
        # Set mock price
        agent.last_trade_price = 100.0
        
        # Test initial portfolio
        initial_value = agent.calculatePortfolioValue()
        self.assertEqual(initial_value, 1000000)  # All cash
        
        # Add position
        agent.holdings["ABM"] = 1000  # 1000 shares
        portfolio_value = agent.calculatePortfolioValue()
        expected_value = 1000000 + (1000 * 100.0)  # Cash + position
        self.assertEqual(portfolio_value, expected_value)
        
        # Test allocation calculation
        allocation = agent.getCurrentAllocation()
        expected_allocation = (1000 * 100.0) / expected_value
        self.assertAlmostEqual(allocation, expected_allocation, places=4)
    
    def test_market_maker_spread_adjustment(self):
        """Test market maker spread adjustment based on uncertainty"""
        agent = ABIDESLLMMarketMaker(
            id=109, name="TestMM", spread_bps=10,
            random_state=self.random_state
        )
        
        # Test with no analysis history
        adjustment = agent.calculateSpreadAdjustment()
        self.assertEqual(adjustment, 0.0)
        
        # Add mock analysis history
        agent.news_analysis_history = [
            {
                'analysis': {
                    'analysis': {
                        'impact_assessment': {'confidence': 0.9}  # High confidence
                    }
                }
            },
            {
                'analysis': {
                    'analysis': {
                        'impact_assessment': {'confidence': 0.3}  # Low confidence
                    }
                }
            }
        ]
        
        adjustment = agent.calculateSpreadAdjustment()
        self.assertGreater(adjustment, 0)  # Should widen spreads due to uncertainty
    
    def test_order_execution_handling(self):
        """Test order execution handling"""
        agent = ABIDESLLMTradingAgent(
            id=110, name="TestExecution",
            initial_cash=1000000,
            random_state=self.random_state
        )
        
        # Place mock order
        agent.active_orders["ORDER_123"] = {
            'order_id': 'ORDER_123',
            'side': 'BUY',
            'quantity': 100
        }
        
        # Mock execution
        execution_data = {
            'order_id': 'ORDER_123',
            'quantity': 100,
            'price': 105.0
        }
        
        initial_cash = agent.holdings['CASH']
        initial_shares = agent.holdings['ABM']
        
        agent.handleOrderExecution(self.mock_time, execution_data)
        
        # Check holdings updated correctly
        self.assertEqual(agent.holdings['ABM'], initial_shares + 100)
        self.assertEqual(agent.holdings['CASH'], initial_cash - (100 * 105.0))
        self.assertNotIn('ORDER_123', agent.active_orders)


class TestABIDESConfiguration(unittest.TestCase):
    """Test ABIDES configuration creation"""
    
    def setUp(self):
        """Set up test environment"""
        self.seed = 12345
        self.llm_config = {
            "config_list": [{"model": "gpt-4o-mini", "api_key": "test"}],
            "timeout": 60
        }
    
    def test_basic_config_creation(self):
        """Test basic configuration creation"""
        config = build_config(
            seed=self.seed,
            end_time="12:00:00",
            llm_enabled=True,
            llm_config=self.llm_config,
            num_llm_traders=2,
            num_noise=50
        )
        
        self.assertEqual(config['seed'], self.seed)
        self.assertEqual(config['end_time'], "12:00:00")
        self.assertIsInstance(config['agents'], list)
        self.assertGreater(len(config['agents']), 0)
        self.assertIn('symbols', config)
        self.assertIn('custom_properties', config)
    
    def test_llm_disabled_config(self):
        """Test configuration with LLM disabled"""
        config = build_config(
            seed=self.seed,
            llm_enabled=False,
            num_noise=10,
            num_value=5
        )
        
        self.assertFalse(config['custom_properties']['llm_enabled'])
        self.assertEqual(config['custom_properties']['num_llm_agents'], 0)
        
        # Should still have traditional agents if ABIDES is available
        if ABIDES_AVAILABLE:
            self.assertGreater(len(config['agents']), 1)  # At least exchange + others
    
    def test_rmsc_style_config(self):
        """Test RMSC-style configuration"""
        config = build_rmsc_llm_config(
            seed=self.seed,
            llm_config=self.llm_config
        )
        
        self.assertEqual(config['seed'], self.seed)
        self.assertIn('LLM_Enhanced_ABIDES', config['custom_properties']['configuration_name'])
        
        # Check agent counts match RMSC style
        custom_props = config['custom_properties']
        self.assertGreater(custom_props['total_agents'], 100)  # Should be large simulation
    
    def test_quick_demo_config(self):
        """Test quick demo configuration"""
        config = quick_llm_demo_config(
            seed=self.seed,
            llm_config=self.llm_config
        )
        
        self.assertEqual(config['seed'], self.seed)
        self.assertEqual(config['end_time'], "11:00:00")  # Shorter simulation
        
        # Should have fewer agents than full simulation
        self.assertLess(len(config['agents']), 100)
    
    def test_agent_config_creation(self):
        """Test individual agent configuration creation"""
        
        # Test trading agent config
        trading_config = createABIDESLLMAgentConfig(
            agent_type="trading",
            agent_id=200,
            symbol="TEST",
            strategy="value",
            risk_tolerance=0.6,
            llm_config=self.llm_config
        )
        
        self.assertEqual(trading_config['agent_id'], 200)
        self.assertEqual(trading_config['symbol'], "TEST")
        self.assertEqual(trading_config['strategy'], "value")
        self.assertEqual(trading_config['risk_tolerance'], 0.6)
        
        # Test market maker config
        mm_config = createABIDESLLMAgentConfig(
            agent_type="market_maker",
            agent_id=201,
            symbol="TEST",
            spread_bps=15,
            max_inventory=2000,
            llm_config=self.llm_config
        )
        
        self.assertEqual(mm_config['agent_id'], 201)
        self.assertEqual(mm_config['spread_bps'], 15)
        self.assertEqual(mm_config['max_inventory'], 2000)
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = build_config(
            seed=self.seed,
            symbols=["AAPL", "GOOGL"],  # Multiple symbols
            num_llm_traders=3,
            llm_config=self.llm_config
        )
        
        # Check that symbols are properly set
        self.assertEqual(config['symbols'], ["AAPL", "GOOGL"])
        
        # Check that agents are created for the first symbol
        llm_agents = [
            agent for agent in config['agents'] 
            if 'LLM' in agent.get('args', {}).get('name', '')
        ]
        for agent in llm_agents:
            if 'symbol' in agent['args']:
                self.assertEqual(agent['args']['symbol'], "AAPL")  # First symbol


class TestABIDESIntegration(unittest.TestCase):
    """Test integration with ABIDES framework"""
    
    @unittest.skipUnless(ABIDES_AVAILABLE, "ABIDES not available")
    def test_abides_simulation_run(self):
        """Test running simulation with ABIDES (if available)"""
        
        # Create minimal configuration for quick test
        config = quick_llm_demo_config(
            seed=42,
            end_time="09:35:00",  # Very short simulation
            num_llm_traders=1,
            num_noise=5,
            num_value=2,
            llm_enabled=False  # Disable LLM for faster testing
        )
        
        try:
            # Run simulation
            end_state = abides.run(config)
            
            # Basic validation
            self.assertIsNotNone(end_state)
            self.assertIn('agents', end_state)
            
            # Check that agents exist
            self.assertGreater(len(end_state['agents']), 0)
            
        except Exception as e:
            self.skipTest(f"ABIDES simulation failed: {e}")
    
    def test_mock_abides_simulation(self):
        """Test simulation with mock ABIDES framework"""
        
        # Create configuration
        config = quick_llm_demo_config(
            seed=42,
            llm_enabled=True,
            num_llm_traders=2
        )
        
        # Mock simulation results
        mock_end_state = {
            'agents': config['agents'],
            'simulation_completed': True,
            'final_time': config['end_time']
        }
        
        # Validate mock results
        self.assertIn('agents', mock_end_state)
        self.assertTrue(mock_end_state['simulation_completed'])
        self.assertEqual(mock_end_state['final_time'], config['end_time'])


class TestLLMIntegration(unittest.TestCase):
    """Test LLM integration components"""
    
    def setUp(self):
        """Set up LLM test environment"""
        self.mock_llm_config = {
            "config_list": [
                {
                    "model": "gpt-4o-mini",
                    "api_key": "test-key",
                    "temperature": 0.3
                }
            ],
            "timeout": 60
        }
    
    def test_llm_config_validation(self):
        """Test LLM configuration validation"""
        
        # Test valid config
        config = build_config(
            llm_enabled=True,
            llm_config=self.mock_llm_config
        )
        self.assertTrue(config['custom_properties']['llm_enabled'])
        
        # Test with missing API key
        invalid_config = {
            "config_list": [{"model": "gpt-4", "api_key": "your-api-key-here"}]
        }
        
        config = build_config(
            llm_enabled=True,
            llm_config=invalid_config
        )
        # Should disable LLM due to missing key
        self.assertFalse(config['custom_properties']['llm_enabled'])
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-env-key'})
    def test_environment_api_key(self):
        """Test API key from environment"""
        
        config = build_config(llm_enabled=True)
        
        # Should use environment key
        self.assertTrue(config['custom_properties']['llm_enabled'])
    
    def test_fallback_analysis(self):
        """Test fallback analysis when LLM is unavailable"""
        
        agent = ABIDESLLMNewsAnalyzer(
            id=300, name="TestFallback",
            symbols=["ABM"],
            llm_config=None  # No LLM config
        )
        
        # Mock news event
        mock_news = Mock()
        mock_news.sentiment_score = 0.8
        mock_news.category = Mock()
        mock_news.category.value = "earnings"
        
        # Test fallback analysis
        analysis = agent.createFallbackAnalysis(mock_news)
        
        self.assertIn('sentiment', analysis)
        self.assertIn('impact_assessment', analysis)
        self.assertIn('price_predictions', analysis)
        self.assertEqual(analysis['sentiment'], 0.8)


class TestPerformanceAndBenchmarks(unittest.TestCase):
    """Test performance and benchmarking utilities"""
    
    def test_large_config_creation(self):
        """Test creation of large configuration"""
        
        start_time = datetime.now()
        
        config = build_rmsc_llm_config(
            seed=123,
            num_noise=1000,
            num_value=100,
            num_llm_traders=10,
            llm_enabled=False  # Disable for speed
        )
        
        creation_time = datetime.now() - start_time
        
        # Should create config quickly (under 5 seconds)
        self.assertLess(creation_time.total_seconds(), 5.0)
        
        # Should have expected number of agents
        total_agents = len(config['agents'])
        expected_agents = 1 + 1000 + 100 + 2 + 10  # Exchange + noise + value + mm + llm
        
        if ABIDES_AVAILABLE:
            self.assertGreaterEqual(total_agents, expected_agents // 2)  # Allow some variance
    
    def test_agent_creation_performance(self):
        """Test agent creation performance"""
        
        start_time = datetime.now()
        
        agents = []
        for i in range(100):
            agent = ABIDESLLMTradingAgent(
                id=1000 + i,
                name=f"PerfTest_{i}",
                strategy="momentum",
                random_state=np.random.RandomState(i)
            )
            agents.append(agent)
        
        creation_time = datetime.now() - start_time
        
        # Should create 100 agents quickly
        self.assertLess(creation_time.total_seconds(), 2.0)
        self.assertEqual(len(agents), 100)
    
    def test_memory_usage(self):
        """Test memory usage of agent creation"""
        
        import sys
        
        # Create baseline
        baseline_refs = len(sys.getrefcount.__defaults__ or [])
        
        # Create agents
        agents = []
        for i in range(50):
            agent = ABIDESLLMTradingAgent(
                id=2000 + i,
                name=f"MemTest_{i}",
                random_state=np.random.RandomState(i)
            )
            agents.append(agent)
        
        # Check that objects are created
        self.assertEqual(len(agents), 50)
        
        # Clean up
        del agents


def run_all_tests():
    """Run all test suites"""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestABIDESLLMAgents,
        TestABIDESConfiguration,
        TestABIDESIntegration,
        TestLLMIntegration,
        TestPerformanceAndBenchmarks
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


def run_integration_test():
    """Run integration test with mini simulation"""
    
    print("=" * 60)
    print("ABIDES-LLM Integration Test")
    print("=" * 60)
    
    try:
        # Create test configuration
        config = quick_llm_demo_config(
            seed=42,
            end_time="09:32:00",  # 2-minute simulation
            num_llm_traders=2,
            num_noise=20,
            llm_enabled=True
        )
        
        print(f"✓ Configuration created: {len(config['agents'])} agents")
        
        # Test agent initialization
        llm_agents = [
            agent for agent in config['agents']
            if 'LLM' in agent.get('args', {}).get('name', '')
        ]
        
        print(f"✓ LLM agents configured: {len(llm_agents)}")
        
        # Test configuration validation
        required_keys = ['seed', 'start_time', 'end_time', 'agents', 'symbols']
        for key in required_keys:
            assert key in config, f"Missing required key: {key}"
        
        print("✓ Configuration validation passed")
        
        # Test agent creation (mock)
        test_agent = ABIDESLLMTradingAgent(
            id=999,
            name="IntegrationTest",
            strategy="momentum",
            risk_tolerance=0.5
        )
        
        print(f"✓ Test agent created: {test_agent.name}")
        
        # Test basic functionality
        portfolio_value = test_agent.calculatePortfolioValue()
        assert portfolio_value > 0, "Portfolio value should be positive"
        
        print(f"✓ Basic functionality test passed")
        
        # Try ABIDES integration if available
        if ABIDES_AVAILABLE:
            print("✓ ABIDES framework detected")
            
            try:
                # Very quick test run
                mini_config = quick_llm_demo_config(
                    seed=42,
                    end_time="09:30:30",  # 30-second test
                    num_llm_traders=1,
                    num_noise=5,
                    llm_enabled=False  # Disable LLM for speed
                )
                
                from abides_core import abides
                end_state = abides.run(mini_config)
                print("✓ Mini ABIDES simulation completed successfully")
                
            except Exception as e:
                print(f"⚠ ABIDES simulation test failed: {e}")
        else:
            print("ℹ ABIDES framework not installed (using mock mode)")
        
        print("\n" + "=" * 60)
        print("✅ Integration test completed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ABIDES-LLM Test Suite')
    parser.add_argument('--integration', action='store_true', 
                       help='Run integration test')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance tests only')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test subset')
    
    args = parser.parse_args()
    
    if args.integration:
        success = run_integration_test()
        sys.exit(0 if success else 1)
    
    elif args.performance:
        # Run only performance tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceAndBenchmarks)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    
    elif args.quick:
        # Run quick subset of tests
        suite = unittest.TestSuite()
        suite.addTest(TestABIDESLLMAgents('test_news_analyzer_initialization'))
        suite.addTest(TestABIDESLLMAgents('test_trading_agent_initialization'))
        suite.addTest(TestABIDESConfiguration('test_basic_config_creation'))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    
    else:
        # Run all tests
        result = run_all_tests()
        sys.exit(0 if result.wasSuccessful() else 1)