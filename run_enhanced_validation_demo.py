#!/usr/bin/env python3
"""
Enhanced ABIDES Validation Demo Script
=====================================

Quick demonstration of the enhanced validation system.
Run this script to test the validation framework with your existing simulator.

Usage:
    python run_enhanced_validation_demo.py
"""

import sys
import traceback
from pathlib import Path

def run_demo():
    """Run the enhanced validation demonstration"""
    
    print("🚀 ENHANCED ABIDES VALIDATION SYSTEM")
    print("="*60)
    print("Testing enhanced validation capabilities...")
    print()
    
    try:
        # Test 1: Import validation system
        print("📦 Test 1: Importing validation system...")
        from enhanced_abides_validation_system import ABIDESValidationFramework
        from enhanced_validation_integration import EnhancedValidationSimulation
        print("✅ Validation system imported successfully!")
        
        # Test 2: Initialize framework
        print("\n🔧 Test 2: Initializing validation framework...")
        validator = ABIDESValidationFramework("demo_validation_results")
        enhanced_sim = EnhancedValidationSimulation(validator)
        print("✅ Framework initialized successfully!")
        
        # Test 3: Run ABIDES paper experiments
        print("\n📊 Test 3: Running ABIDES paper experiments...")
        experiment_results = enhanced_sim.run_abides_paper_experiments()
        print("✅ ABIDES experiments completed!")
        
        # Test 4: Run stylized facts validation
        print("\n🔍 Test 4: Running stylized facts validation...")
        symbols = ['TEST1', 'TEST2', 'TEST3']
        validation_results = validator.run_comprehensive_validation(symbols)
        print("✅ Stylized facts validation completed!")
        
        # Test 5: Generate reports
        print("\n📝 Test 5: Generating validation reports...")
        report = validator.generate_validation_report()
        print("✅ Reports generated successfully!")
        
        # Test 6: Create visualizations
        print("\n📈 Test 6: Creating visualizations...")
        try:
            from enhanced_validation_integration import ValidationVisualization
            viz = ValidationVisualization("demo_validation_plots")
            viz.generate_comprehensive_report(experiment_results, validation_results)
            print("✅ Visualizations created successfully!")
        except Exception as e:
            print(f"⚠️  Visualization creation skipped: {e}")
        
        # Display results summary
        print("\n" + "="*60)
        print("📋 VALIDATION RESULTS SUMMARY")
        print("="*60)
        
        if 'summary' in validation_results:
            summary = validation_results['summary']
            compliance_score = summary.get('overall_compliance_score', 0)
            total_symbols = summary.get('total_symbols', 0)
            
            print(f"Overall Compliance Score: {compliance_score:.3f}")
            print(f"Symbols Analyzed: {total_symbols}")
            
            if 'stylized_facts_compliance' in summary:
                print("\nStylized Facts Compliance:")
                for fact, rate in summary['stylized_facts_compliance'].items():
                    status = "✅" if rate >= 0.8 else "⚠️ " if rate >= 0.6 else "❌"
                    print(f"  {status} {fact.replace('_', ' ').title()}: {rate:.2%}")
            
            # Overall assessment
            print(f"\n{'='*60}")
            if compliance_score >= 0.8:
                print("🎉 EXCELLENT: Your simulation meets high-fidelity standards!")
                print("   Market behavior is highly realistic.")
            elif compliance_score >= 0.6:
                print("👍 GOOD: Your simulation shows realistic behavior!")
                print("   Minor improvements may be needed.")
            elif compliance_score >= 0.4:
                print("⚠️  FAIR: Some unrealistic behaviors detected.")
                print("   Consider refining your market model.")
            else:
                print("❌ NEEDS IMPROVEMENT: Significant deviations detected.")
                print("   Review price generation and agent behavior models.")
        
        # Show experiment results
        print(f"\n{'='*60}")
        print("🧪 ABIDES PAPER EXPERIMENTS SUMMARY")
        print("="*60)
        
        for exp in experiment_results.get('experiments', []):
            print(f"\n{exp['name']}:")
            print(f"  Description: {exp['description']}")
            
            results = exp['results']
            if exp['name'] == 'Background Agent Validation':
                conclusions = results.get('conclusions', {})
                print(f"  ✓ Efficiency improves with agents: {conclusions.get('efficiency_improves_with_agents', False)}")
                print(f"  ✓ Recommended agent count: {conclusions.get('recommended_agent_count', 'N/A')}")
                
            elif exp['name'] == 'Market Impact Study':
                conclusions = results.get('conclusions', {})
                print(f"  ✓ Follows square-root law: {conclusions.get('follows_square_root_law', False)}")
                print(f"  ✓ Impact correlation: {conclusions.get('impact_correlation_with_sqrt_size', 0):.3f}")
                
            elif exp['name'] == 'Agent Strategy Comparison':
                best_strategy = results.get('best_strategy', 'Unknown')
                print(f"  ✓ Best performing strategy: {best_strategy}")
        
        # Output locations
        print(f"\n{'='*60}")
        print("📁 OUTPUT LOCATIONS")
        print("="*60)
        print(f"Validation Results: {validator.output_dir}")
        print(f"Validation Database: {validator.recorder.db_path}")
        if Path("demo_validation_plots").exists():
            print(f"Visualizations: demo_validation_plots/")
        
        print(f"\n{'='*60}")
        print("🎯 NEXT STEPS")
        print("="*60)
        print("1. Review the validation report in validation_results.json")
        print("2. Check individual stylized facts that need improvement")
        print("3. Integrate this validation system with your real simulation")
        print("4. Run validation after any changes to your market model")
        print("5. Use the database to analyze detailed trading patterns")
        
        print(f"\n✅ DEMO COMPLETED SUCCESSFULLY!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nPossible solutions:")
        print("1. Install missing dependencies: pip install numpy pandas matplotlib seaborn scipy scikit-learn")
        print("2. Ensure all validation system files are in the current directory")
        print("3. Check Python path configuration")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        print("\nPlease check the error details above and try again.")
        return False


def check_dependencies():
    """Check if required dependencies are available"""
    
    print("🔍 Checking dependencies...")
    missing_deps = []
    
    try:
        import numpy
        print("✅ numpy available")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import pandas
        print("✅ pandas available")
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import matplotlib
        print("✅ matplotlib available")
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import seaborn
        print("✅ seaborn available")
    except ImportError:
        missing_deps.append("seaborn")
    
    try:
        import scipy
        print("✅ scipy available")
    except ImportError:
        missing_deps.append("scipy")
    
    try:
        import sklearn
        print("✅ scikit-learn available")
    except ImportError:
        missing_deps.append("scikit-learn")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print(f"Install with: pip install {' '.join(missing_deps)}")
        return False
    else:
        print("✅ All dependencies available!")
        return True


def main():
    """Main entry point"""
    
    print("Enhanced ABIDES Validation System Demo")
    print("="*50)
    print()
    
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    print()
    
    # Run the demonstration
    success = run_demo()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("You can now integrate this validation system with your simulator.")
    else:
        print("\n❌ Demo failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()