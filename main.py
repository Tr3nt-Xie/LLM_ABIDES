#!/usr/bin/env python3
"""
ABIDES-LLM Integration Main Application
======================================

A complete integration of Large Language Models (LLM) with ABIDES 
for market simulation and algorithmic trading research.
"""

import sys
import os
import argparse
from pathlib import Path

def run_demo():
    """Run the demonstration"""
    try:
        os.chdir(Path(__file__).parent)
        sys.path.insert(0, "examples")
        import simple_abides_llm_demo
        simple_abides_llm_demo.main()
    except Exception as e:
        print(f"Error: {e}")
        os.system("python examples/simple_abides_llm_demo.py")

def test_config():
    """Test configuration"""
    print("🔧 Testing Configuration")
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    
    try:
        import numpy, pandas, matplotlib
        print("✅ Core packages OK")
    except ImportError as e:
        print(f"❌ Missing: {e}")

def main():
    parser = argparse.ArgumentParser(description="ABIDES-LLM Integration")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--config", action="store_true", help="Test config")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.config:
        test_config()
    else:
        print("🚀 ABIDES-LLM Integration")
        print("Use --demo to run demonstration")
        print("Use --config to test configuration")
        run_demo()

if __name__ == "__main__":
    main()
