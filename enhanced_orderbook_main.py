#!/usr/bin/env python3
"""
Enhanced Order Book Main System
===============================

Comprehensive main script that integrates:
- Enhanced order book generation with realistic patterns
- Database storage (SQLite/PostgreSQL)
- LLM-powered analysis and comparison with real market data
- Detailed reporting and visualization

This replaces CSV-based storage with robust database backend and adds
sophisticated analysis capabilities.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from enhanced_orderbook_db import EnhancedOrderBookDB, EnhancedOrderBookConfig
    from llm_analysis_system import LLMOrderBookAnalyzer, MarketDataComparison
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedOrderBookSystem:
    """Main system orchestrator for enhanced order book generation and analysis"""
    
    def __init__(self):
        self.output_dir = Path("enhanced_orderbook_output")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Enhanced Order Book System Initialized")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
    
    def run_comprehensive_simulation(self, config: EnhancedOrderBookConfig, 
                                   analyze_with_llm: bool = True) -> dict:
        """Run complete simulation with generation, storage, and analysis"""
        
        logger.info("=" * 80)
        logger.info("🎯 STARTING COMPREHENSIVE ORDER BOOK SIMULATION")
        logger.info("=" * 80)
        
        results = {
            "start_time": datetime.now(),
            "config": config,
            "generation_summary": None,
            "analysis_results": None,
            "database_path": None,
            "reports": {},
            "success": False
        }
        
        try:
            # Phase 1: Generate Order Book Data
            logger.info("\n📊 PHASE 1: GENERATING ORDER BOOK DATA")
            logger.info("-" * 50)
            
            orderbook_system = EnhancedOrderBookDB(config)
            generation_summary = orderbook_system.generate_simulation_data()
            results["generation_summary"] = generation_summary
            results["database_path"] = config.db_path
            
            self._print_generation_summary(generation_summary)
            
            # Phase 2: Export Data for Analysis
            logger.info("\n📈 PHASE 2: EXPORTING DATA FOR ANALYSIS")
            logger.info("-" * 50)
            
            data_dict = orderbook_system.export_data_analysis()
            logger.info(f"✅ Exported {len(data_dict)} datasets for analysis")
            
            # Save data summaries
            self._save_data_summaries(data_dict)
            
            # Phase 3: LLM-Powered Analysis
            if analyze_with_llm:
                logger.info("\n🤖 PHASE 3: LLM-POWERED ANALYSIS")
                logger.info("-" * 50)
                
                analyzer = LLMOrderBookAnalyzer(use_real_llm=True)
                comparison = analyzer.analyze_order_book_quality(data_dict)
                results["analysis_results"] = comparison
                
                # Generate comprehensive reports
                comparison_report = analyzer.generate_comparison_report(comparison)
                results["reports"]["llm_analysis"] = comparison_report
                
                # Save reports
                self._save_reports(results["reports"])
                
                logger.info("✅ LLM analysis completed")
            else:
                logger.info("\n⚠️  Skipping LLM analysis (disabled)")
            
            # Phase 4: Generate Final Summary
            logger.info("\n📋 PHASE 4: GENERATING FINAL SUMMARY")
            logger.info("-" * 50)
            
            final_summary = self._generate_final_summary(results)
            results["reports"]["final_summary"] = final_summary
            
            results["end_time"] = datetime.now()
            results["success"] = True
            
            logger.info("🎉 COMPREHENSIVE SIMULATION COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            logger.error(f"❌ Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            results["error"] = str(e)
            results["end_time"] = datetime.now()
        
        return results
    
    def _print_generation_summary(self, summary: dict):
        """Print order book generation summary"""
        
        print("\n" + "=" * 60)
        print("📊 ORDER BOOK GENERATION SUMMARY")
        print("=" * 60)
        
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"{key.replace('_', ' ').title()}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("=" * 60)
    
    def _save_data_summaries(self, data_dict: dict):
        """Save data summaries to files"""
        
        summary_dir = self.output_dir / "data_summaries"
        summary_dir.mkdir(exist_ok=True)
        
        for dataset_name, df in data_dict.items():
            if not df.empty:
                # Basic statistics
                stats_file = summary_dir / f"{dataset_name}_stats.txt"
                with open(stats_file, 'w') as f:
                    f.write(f"Dataset: {dataset_name}\n")
                    f.write(f"Shape: {df.shape}\n")
                    f.write(f"Columns: {list(df.columns)}\n\n")
                    f.write("Basic Statistics:\n")
                    f.write(str(df.describe()))
                
                # Sample data
                sample_file = summary_dir / f"{dataset_name}_sample.csv"
                df.head(100).to_csv(sample_file, index=False)
                
                logger.info(f"✅ Saved {dataset_name} summary ({len(df):,} rows)")
    
    def _save_reports(self, reports: dict):
        """Save analysis reports to files"""
        
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        for report_name, content in reports.items():
            report_file = reports_dir / f"{report_name}.txt"
            with open(report_file, 'w') as f:
                f.write(content)
            
            logger.info(f"✅ Saved {report_name} report")
    
    def _generate_final_summary(self, results: dict) -> str:
        """Generate comprehensive final summary"""
        
        summary = []
        summary.append("🎯 ENHANCED ORDER BOOK SYSTEM - FINAL SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        
        # Execution overview
        start_time = results["start_time"]
        end_time = results.get("end_time", datetime.now())
        duration = end_time - start_time
        
        summary.append("⏱️  EXECUTION OVERVIEW")
        summary.append("-" * 40)
        summary.append(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Total Duration: {duration}")
        summary.append(f"Status: {'✅ SUCCESS' if results['success'] else '❌ FAILED'}")
        summary.append("")
        
        # Configuration summary
        config = results["config"]
        summary.append("⚙️  CONFIGURATION")
        summary.append("-" * 40)
        summary.append(f"Agents: {config.num_agents:,}")
        summary.append(f"Symbols: {', '.join(config.symbols)}")
        summary.append(f"Simulation Days: {config.simulation_days}")
        summary.append(f"Database Path: {config.db_path}")
        summary.append(f"Trading Hours: {config.trading_hours_start}:00 - {config.trading_hours_end}:00")
        summary.append("")
        
        # Generation results
        if results["generation_summary"]:
            gen_summary = results["generation_summary"]
            summary.append("📊 GENERATION RESULTS")
            summary.append("-" * 40)
            summary.append(f"Total Orders: {gen_summary.get('total_orders', 'N/A'):,}")
            summary.append(f"Total Trades: {gen_summary.get('total_trades', 'N/A'):,}")
            summary.append(f"Total Snapshots: {gen_summary.get('total_snapshots', 'N/A'):,}")
            if 'orders_per_second' in gen_summary:
                summary.append(f"Orders/Second: {gen_summary['orders_per_second']:,.1f}")
            summary.append("")
        
        # Analysis results
        if results["analysis_results"]:
            analysis = results["analysis_results"]
            summary.append("🤖 LLM ANALYSIS RESULTS")
            summary.append("-" * 40)
            
            realism_score = analysis.realism_assessment.get("score", 0)
            confidence = analysis.confidence_score
            
            summary.append(f"Realism Score: {realism_score:.1f}/10")
            summary.append(f"Confidence Level: {confidence:.1%}")
            
            if analysis.similarity_scores:
                summary.append("\nSimilarity Scores:")
                for metric, score in analysis.similarity_scores.items():
                    status = "✅" if score > 0.7 else "⚠️" if score > 0.4 else "❌"
                    summary.append(f"  {status} {metric.replace('_', ' ').title()}: {score:.1%}")
            
            if analysis.improvement_suggestions:
                summary.append("\nTop Recommendations:")
                for i, suggestion in enumerate(analysis.improvement_suggestions[:3], 1):
                    summary.append(f"  {i}. {suggestion}")
            summary.append("")
        
        # Output files
        summary.append("📁 OUTPUT FILES")
        summary.append("-" * 40)
        summary.append(f"Database: {results.get('database_path', 'N/A')}")
        summary.append(f"Output Directory: {self.output_dir.absolute()}")
        summary.append(f"Reports: {len(results['reports'])} generated")
        
        if self.output_dir.exists():
            summary.append("\nGenerated Files:")
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(self.output_dir)
                    summary.append(f"  📄 {rel_path}")
        summary.append("")
        
        # Next steps
        summary.append("🎯 NEXT STEPS")
        summary.append("-" * 40)
        summary.append("1. Review the generated database and analysis reports")
        summary.append("2. Examine similarity scores and implement recommendations")
        summary.append("3. Use the database for further research and analysis")
        summary.append("4. Compare with real market data using provided benchmarks")
        summary.append("")
        
        summary.append("=" * 80)
        summary.append(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(summary)

def create_sample_configs():
    """Create sample configurations for different use cases"""
    
    configs = {}
    
    # Quick test configuration
    configs["quick_test"] = EnhancedOrderBookConfig(
        db_path="quick_test_orderbook.db",
        num_agents=500,
        simulation_days=1,
        symbols=["AAPL", "GOOGL"],
        base_orders_per_minute=50,
        use_in_memory=False
    )
    
    # Research configuration
    configs["research"] = EnhancedOrderBookConfig(
        db_path="research_orderbook.db",
        num_agents=2000,
        simulation_days=5,
        symbols=["AAPL", "GOOGL", "MSFT", "TSLA"],
        base_orders_per_minute=150,
        enable_momentum_trading=True,
        enable_mean_reversion=True,
        enable_news_impact=True
    )
    
    # Production configuration
    configs["production"] = EnhancedOrderBookConfig(
        db_path="production_orderbook.db",
        num_agents=5000,
        simulation_days=30,
        symbols=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
        base_orders_per_minute=200,
        enable_momentum_trading=True,
        enable_mean_reversion=True,
        enable_news_impact=True,
        enable_intraday_patterns=True
    )
    
    return configs

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Order Book System with Database Storage and LLM Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_orderbook_main.py --config quick_test
  python enhanced_orderbook_main.py --config research --no-llm
  python enhanced_orderbook_main.py --custom --agents 1000 --days 3 --symbols AAPL GOOGL
        """
    )
    
    parser.add_argument("--config", choices=["quick_test", "research", "production"],
                       help="Use predefined configuration")
    parser.add_argument("--custom", action="store_true",
                       help="Use custom configuration with following parameters")
    parser.add_argument("--agents", type=int, default=1000,
                       help="Number of trading agents (default: 1000)")
    parser.add_argument("--days", type=int, default=3,
                       help="Number of simulation days (default: 3)")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "GOOGL"],
                       help="Trading symbols (default: AAPL GOOGL)")
    parser.add_argument("--db-path", default="orderbook.db",
                       help="Database file path (default: orderbook.db)")
    parser.add_argument("--orders-per-minute", type=int, default=100,
                       help="Base orders per minute (default: 100)")
    parser.add_argument("--no-llm", action="store_true",
                       help="Skip LLM analysis")
    parser.add_argument("--in-memory", action="store_true",
                       help="Use in-memory database")
    
    args = parser.parse_args()
    
    # Initialize system
    system = EnhancedOrderBookSystem()
    
    # Create configuration
    if args.config:
        configs = create_sample_configs()
        config = configs[args.config]
        logger.info(f"Using predefined configuration: {args.config}")
    elif args.custom:
        config = EnhancedOrderBookConfig(
            db_path=args.db_path,
            num_agents=args.agents,
            simulation_days=args.days,
            symbols=args.symbols,
            base_orders_per_minute=args.orders_per_minute,
            use_in_memory=args.in_memory
        )
        logger.info("Using custom configuration")
    else:
        # Interactive mode
        print("\n🚀 Enhanced Order Book System")
        print("=" * 50)
        print("Available configurations:")
        print("1. Quick Test (500 agents, 1 day, 2 symbols)")
        print("2. Research (2000 agents, 5 days, 4 symbols)")
        print("3. Production (5000 agents, 30 days, 5 symbols)")
        print("4. Custom configuration")
        
        choice = input("\nSelect configuration (1-4): ").strip()
        
        configs = create_sample_configs()
        config_map = {"1": "quick_test", "2": "research", "3": "production"}
        
        if choice in config_map:
            config = configs[config_map[choice]]
            logger.info(f"Selected: {config_map[choice]}")
        elif choice == "4":
            # Custom configuration
            agents = int(input("Number of agents (1000): ") or "1000")
            days = int(input("Simulation days (3): ") or "3")
            symbols_input = input("Symbols (AAPL GOOGL): ") or "AAPL GOOGL"
            symbols = symbols_input.split()
            
            config = EnhancedOrderBookConfig(
                num_agents=agents,
                simulation_days=days,
                symbols=symbols
            )
        else:
            config = configs["quick_test"]
            logger.info("Using default quick_test configuration")
    
    # Display configuration
    print(f"\n📋 Configuration Summary:")
    print(f"  Agents: {config.num_agents:,}")
    print(f"  Days: {config.simulation_days}")
    print(f"  Symbols: {', '.join(config.symbols)}")
    print(f"  Database: {config.db_path}")
    print(f"  LLM Analysis: {'Disabled' if args.no_llm else 'Enabled'}")
    
    # Run simulation
    try:
        results = system.run_comprehensive_simulation(
            config=config,
            analyze_with_llm=not args.no_llm
        )
        
        # Print final summary
        if "final_summary" in results["reports"]:
            print("\n" + results["reports"]["final_summary"])
        
        if results["success"]:
            print(f"\n🎉 Simulation completed successfully!")
            print(f"📁 Results saved to: {system.output_dir.absolute()}")
            print(f"🗃️  Database: {results['database_path']}")
        else:
            print(f"\n❌ Simulation failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 Simulation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())