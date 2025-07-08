#!/usr/bin/env python3
"""
ABIDES-LLM Quick Start Script
============================

Interactive menu for running demonstrations and tests of the ABIDES-LLM integration.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    """Print welcome header"""
    print("=" * 60)
    print("🚀 ABIDES-LLM Integration Quick Start")
    print("=" * 60)
    print("Welcome to your ABIDES-LLM integration project!")
    print("This script helps you run demonstrations and tests.\n")

def check_environment():
    """Check system environment and dependencies"""
    print("🔍 Checking Environment...")
    
    # Check Python version
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
    else:
        print(f"⚠️  Python {version.major}.{version.minor}.{version.micro} (May have issues)")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your-openai-api-key-here":
        print("✅ OpenAI API Key Found")
        llm_available = True
    else:
        print("⚠️  No OpenAI API Key (Mock LLM mode)")
        llm_available = False
    
    # Check for core files
    core_files = [
        "simple_abides_llm_demo.py",
        "abides_llm_config.py", 
        "abides_llm_agents.py"
    ]
    
    missing_files = []
    for file in core_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (Missing)")
            missing_files.append(file)
    
    print()
    return llm_available, missing_files

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🏃 {description}")
    print(f"Running: {command}")
    print("-" * 40)
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def demo_simple():
    """Run the simplified demonstration"""
    print("\n" + "=" * 50)
    print("📊 Running Simplified ABIDES-LLM Demo")
    print("=" * 50)
    print("This demo shows core concepts without requiring external dependencies.")
    print("It includes news analysis, trading strategies, and agent interactions.\n")
    
    return run_command("python3 simple_abides_llm_demo.py", "Simplified Demo")

def demo_configuration():
    """Test the configuration system"""
    print("\n" + "=" * 50)
    print("⚙️  Testing Configuration System")
    print("=" * 50)
    print("This tests the ABIDES configuration generation.\n")
    
    test_command = """python3 -c "
try:
    from abides_llm_config import quick_llm_demo_config
    config = quick_llm_demo_config(end_time='10:00:00', num_llm_traders=2)
    print('✅ Configuration system working!')
    print(f'   Agents: {len(config[\"agents\"])}')
    print(f'   Symbols: {config.get(\"symbols\", [])}')
    print(f'   End time: {config.get(\"end_time\", \"N/A\")}')
except Exception as e:
    print(f'❌ Configuration test failed: {e}')
    import traceback
    traceback.print_exc()
"
"""
    
    return run_command(test_command, "Configuration Test")

def run_tests():
    """Run the test suite"""
    print("\n" + "=" * 50)
    print("🧪 Running Test Suite")
    print("=" * 50)
    print("This runs the comprehensive test suite to validate functionality.\n")
    
    return run_command("python3 abides_test_suite.py --quick", "Test Suite")

def show_project_info():
    """Show project information and file structure"""
    print("\n" + "=" * 50)
    print("📁 Project Information")
    print("=" * 50)
    
    project_root = Path(".")
    
    print("Core Files:")
    core_files = [
        ("simple_abides_llm_demo.py", "Simplified working demonstration"),
        ("abides_llm_config.py", "ABIDES configuration system"),
        ("abides_llm_agents.py", "LLM-enhanced trading agents"),
        ("enhanced_llm_abides_system.py", "Advanced LLM reasoning"),
        ("abides_test_suite.py", "Comprehensive test suite"),
        (".env", "Environment configuration"),
        ("PROJECT_STATUS.md", "Detailed project status")
    ]
    
    for filename, description in core_files:
        status = "✅" if Path(filename).exists() else "❌"
        print(f"  {status} {filename:<30} - {description}")
    
    print(f"\nProject Structure:")
    print(f"  📂 Root Directory: {project_root.absolute()}")
    
    # Count Python files
    py_files = list(project_root.glob("*.py"))
    print(f"  📄 Python Files: {len(py_files)}")
    
    # Show subdirectories
    subdirs = [d for d in project_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    print(f"  📁 Subdirectories: {len(subdirs)}")
    for subdir in subdirs:
        print(f"    - {subdir.name}/")

def show_usage_examples():
    """Show usage examples"""
    print("\n" + "=" * 50)
    print("💡 Usage Examples")
    print("=" * 50)
    
    examples = [
        ("Basic Demo", "python3 simple_abides_llm_demo.py"),
        ("Configuration Test", "python3 -c \"from abides_llm_config import quick_llm_demo_config; print('Works!')\""),
        ("Run Tests", "python3 abides_test_suite.py --quick"),
        ("Interactive Python", "python3 -i simple_abides_llm_demo.py"),
    ]
    
    for title, command in examples:
        print(f"\n{title}:")
        print(f"  {command}")
    
    print(f"\nFor detailed documentation:")
    print(f"  📖 Read PROJECT_STATUS.md")
    print(f"  📖 Read README.md")

def main_menu():
    """Show main menu and handle user input"""
    while True:
        print("\n" + "=" * 40)
        print("📋 Main Menu")
        print("=" * 40)
        print("1. 🚀 Run Simplified Demo")
        print("2. ⚙️  Test Configuration System")
        print("3. 🧪 Run Test Suite")
        print("4. 📁 Show Project Information")
        print("5. 💡 Show Usage Examples")
        print("6. 🔍 Check Environment Again")
        print("7. 📖 View Project Status")
        print("0. 🚪 Exit")
        
        try:
            choice = input("\nEnter your choice (0-7): ").strip()
            
            if choice == "0":
                print("👋 Thanks for using ABIDES-LLM! Happy researching!")
                break
            elif choice == "1":
                demo_simple()
            elif choice == "2":
                demo_configuration()
            elif choice == "3":
                run_tests()
            elif choice == "4":
                show_project_info()
            elif choice == "5":
                show_usage_examples()
            elif choice == "6":
                check_environment()
            elif choice == "7":
                if Path("PROJECT_STATUS.md").exists():
                    print("\n📖 Opening PROJECT_STATUS.md...")
                    run_command("cat PROJECT_STATUS.md", "View Project Status")
                else:
                    print("❌ PROJECT_STATUS.md not found")
            else:
                print("❌ Invalid choice. Please enter a number from 0-7.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function"""
    print_header()
    
    # Check environment
    llm_available, missing_files = check_environment()
    
    if missing_files:
        print("⚠️  Warning: Some core files are missing!")
        print("   Consider running the setup script: python3 setup_abides_llm.py")
        print()
    
    # Show available options
    print("🎯 What would you like to do?")
    print("   • Run demonstrations to see the system in action")
    print("   • Test configurations and functionality")
    print("   • View project information and examples")
    
    # Start main menu
    main_menu()

if __name__ == "__main__":
    main()