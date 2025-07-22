#!/usr/bin/env python3
"""
ABIDES-LLM Integration Setup Script
==================================

Quick setup for the ABIDES-LLM integration project.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 ABIDES-LLM Integration Setup")
    print("=" * 40)
    
    # Check Python version
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need 3.8+")
        return
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("⚠️  Some dependencies may have failed to install")
    
    # Create environment template
    if not Path(".env").exists():
        with open(".env.template", "w") as f:
            f.write("OPENAI_API_KEY=your-openai-api-key-here\n")
        print("📄 Created .env.template - configure your API keys")
    
    # Test installation
    print("\n🧪 Testing Installation")
    if run_command("python main.py --config", "Testing configuration"):
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("• Configure .env with your OpenAI API key (optional)")
        print("• Run: python main.py --demo")
        print("• Or: python main.py for interactive mode")
    else:
        print("\n❌ Setup encountered issues")

if __name__ == "__main__":
    main()
