#!/usr/bin/env python3
"""
ABIDES-LLM Integration Setup Script
===================================

This script sets up the complete ABIDES-LLM integration environment,
including ABIDES installation, dependency management, and configuration.
"""

import os
import sys
import subprocess
import shutil
import tempfile
import argparse
import urllib.request
import zipfile
from pathlib import Path
import json


class ABIDESLLMSetup:
    """Setup manager for ABIDES-LLM integration"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent.absolute()
        self.venv_path = self.project_root / "venv_abides"
        
    def log(self, message, level="INFO"):
        """Log message with level"""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{level}] {message}")
    
    def run_command(self, command, check=True, capture_output=False):
        """Run shell command with error handling"""
        self.log(f"Running: {command}")
        
        try:
            if capture_output:
                result = subprocess.run(
                    command, shell=True, check=check, 
                    capture_output=True, text=True
                )
                return result.stdout.strip()
            else:
                subprocess.run(command, shell=True, check=check)
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {e}", "ERROR")
            if check:
                raise
    
    def check_python_version(self):
        """Check Python version compatibility"""
        self.log("Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            raise RuntimeError(
                f"Python 3.8+ required, found {version.major}.{version.minor}"
            )
        
        self.log(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    
    def create_virtual_environment(self):
        """Create virtual environment for the project"""
        self.log("Creating virtual environment...")
        
        if self.venv_path.exists():
            self.log("Virtual environment already exists")
            return
        
        # Use python3 instead of python
        self.run_command(f"python3 -m venv {self.venv_path}")
        self.log(f"✓ Virtual environment created at {self.venv_path}")
    
    def activate_virtual_environment(self):
        """Activate virtual environment"""
        if os.name == 'nt':  # Windows
            activate_script = self.venv_path / "Scripts" / "activate.bat"
            python_executable = self.venv_path / "Scripts" / "python.exe"
        else:  # Unix/Linux/macOS
            activate_script = self.venv_path / "bin" / "activate"
            python_executable = self.venv_path / "bin" / "python"
        
        if not python_executable.exists():
            raise RuntimeError("Virtual environment not properly created")
        
        # Update sys.executable for subsequent subprocess calls
        sys.executable = str(python_executable)
        self.log(f"✓ Using Python: {python_executable}")
    
    def install_base_requirements(self):
        """Install base Python requirements"""
        self.log("Installing base requirements...")
        
        # Look for requirements file in Setup directory or create minimal one
        requirements_file = self.project_root / "Setup" / "requirements_abides.txt"
        if not requirements_file.exists():
            requirements_file = self.project_root / "requirements_abides.txt"
            
        if not requirements_file.exists():
            self.log("Creating minimal requirements file...", "WARNING")
            minimal_reqs = [
                "numpy>=1.21.0",
                "pandas>=1.5.0", 
                "matplotlib>=3.5.0",
                "pyautogen>=0.2.0",
                "openai>=1.0.0",
                "pyyaml>=6.0",
                "python-dateutil>=2.8.0",
                "scipy>=1.9.0",
                "seaborn>=0.11.0",
                "requests>=2.28.0"
            ]
            with open(requirements_file, 'w') as f:
                f.write('\n'.join(minimal_reqs))
        
        pip_executable = sys.executable.replace('python', 'pip')
        self.run_command(f"{pip_executable} install --upgrade pip")
        self.run_command(f"{pip_executable} install -r {requirements_file}")
        self.log("✓ Base requirements installed")
    
    def move_files_to_root(self):
        """Move source files from subdirectories to root for easier access"""
        self.log("Organizing project files...")
        
        # Move core files from "Core ABIDES Integration" to root
        core_dir = self.project_root / "Core ABIDES Integration"
        if core_dir.exists():
            for file_path in core_dir.glob("*.py"):
                target_path = self.project_root / file_path.name
                if not target_path.exists():
                    shutil.copy2(file_path, target_path)
                    self.log(f"✓ Copied {file_path.name} to root")
        
        # Move test files from Testing&Validation to root  
        test_dir = self.project_root / "Testing&Validation"
        if test_dir.exists():
            for file_path in test_dir.glob("*.py"):
                target_path = self.project_root / file_path.name
                if not target_path.exists():
                    shutil.copy2(file_path, target_path)
                    self.log(f"✓ Copied {file_path.name} to root")
        
        # Copy requirements from Setup directory
        setup_dir = self.project_root / "Setup"
        if setup_dir.exists():
            req_file = setup_dir / "requirements_abides.txt"
            if req_file.exists():
                target_req = self.project_root / "requirements_abides.txt"
                if not target_req.exists():
                    shutil.copy2(req_file, target_req)
                    self.log("✓ Copied requirements file to root")
    
    def setup_abides(self, install_method="auto"):
        """Setup ABIDES framework"""
        self.log("Setting up ABIDES framework...")
        
        abides_path = self.project_root.parent / "abides-jpmc-public"
        
        if install_method == "auto":
            if not abides_path.exists():
                self.log("ABIDES not found, attempting to clone...")
                self.clone_abides(abides_path)
            else:
                self.log(f"Found existing ABIDES at {abides_path}")
        
        elif install_method == "clone":
            if abides_path.exists():
                self.log(f"Removing existing ABIDES at {abides_path}")
                shutil.rmtree(abides_path)
            self.clone_abides(abides_path)
        
        elif install_method == "skip":
            self.log("Skipping ABIDES installation")
            return
        
        # Install ABIDES
        if abides_path.exists():
            self.install_abides(abides_path)
    
    def clone_abides(self, target_path):
        """Clone ABIDES repository"""
        self.log("Cloning ABIDES repository...")
        
        repo_url = "https://github.com/jpmorganchase/abides-jpmc-public.git"
        
        try:
            self.run_command(f"git clone {repo_url} {target_path}")
            self.log(f"✓ ABIDES cloned to {target_path}")
        except subprocess.CalledProcessError:
            self.log("Git clone failed, trying alternative method...", "WARNING")
            self.download_abides_zip(target_path)
    
    def download_abides_zip(self, target_path):
        """Download ABIDES as ZIP (fallback method)"""
        self.log("Downloading ABIDES as ZIP archive...")
        
        zip_url = "https://github.com/jpmorganchase/abides-jpmc-public/archive/refs/heads/main.zip"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "abides.zip"
            
            urllib.request.urlretrieve(zip_url, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Move extracted directory to target
            extracted_dir = Path(temp_dir) / "abides-jpmc-public-main"
            shutil.move(str(extracted_dir), str(target_path))
        
        self.log(f"✓ ABIDES downloaded to {target_path}")
    
    def install_abides(self, abides_path):
        """Install ABIDES in development mode"""
        self.log("Installing ABIDES...")
        
        original_dir = os.getcwd()
        try:
            os.chdir(abides_path)
            
            pip_executable = sys.executable.replace('python', 'pip')
            
            # Install ABIDES requirements if they exist
            abides_requirements = abides_path / "requirements.txt"
            if abides_requirements.exists():
                self.run_command(f"{pip_executable} install -r requirements.txt")
            
            # Install ABIDES in development mode
            self.run_command(f"{pip_executable} install -e .")
            
            self.log("✓ ABIDES installed successfully")
            
        finally:
            os.chdir(original_dir)
    
    def setup_environment_file(self):
        """Create environment configuration file"""
        self.log("Setting up environment configuration...")
        
        env_file = self.project_root / ".env"
        
        if env_file.exists():
            self.log("Environment file already exists")
            return
        
        env_content = """# ABIDES-LLM Environment Configuration
# =====================================

# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_ORG_ID=your-org-id-here  # Optional

# Alternative LLM APIs (optional)  
ANTHROPIC_API_KEY=your-anthropic-api-key
COHERE_API_KEY=your-cohere-api-key

# ABIDES Configuration
ABIDES_LOG_LEVEL=INFO
ABIDES_LOG_DIR=./abides_logs

# Simulation Configuration
DEFAULT_SIMULATION_SEED=12345
DEFAULT_END_TIME=16:00:00

# Performance Settings
MAX_AGENTS=1000
ENABLE_PARALLEL_PROCESSING=true

# Database Configuration
DATABASE_URL=sqlite:///./abides_simulation.db

# Development Settings
DEBUG_MODE=false
VERBOSE_LOGGING=false
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        self.log(f"✓ Environment file created: {env_file}")
        self.log("Please edit .env file and add your API keys!")
    
    def create_example_script(self):
        """Create example usage script"""
        self.log("Creating example script...")
        
        example_script = self.project_root / "example_run.py"
        
        example_content = '''#!/usr/bin/env python3
"""
ABIDES-LLM Example Script
========================

This script demonstrates how to run ABIDES with LLM-enhanced agents.
"""

import os
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    """Run example ABIDES-LLM simulation"""
    
    print("ABIDES-LLM Example Simulation")
    print("=" * 40)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here":
        print("⚠️  Warning: No OpenAI API key found!")
        print("   Set OPENAI_API_KEY environment variable to enable LLM features")
        print("   Example: export OPENAI_API_KEY='your-key-here'")
        llm_enabled = False
    else:
        print("✓ OpenAI API key found")
        llm_enabled = True
    
    # Try to import configuration
    try:
        from abides_llm_config import quick_llm_demo_config
        
        print("\\nCreating simulation configuration...")
        config = quick_llm_demo_config(
            seed=42,
            end_time="09:32:00",  # 2-minute simulation
            num_llm_traders=2 if llm_enabled else 0,
            num_noise=20,
            llm_enabled=llm_enabled
        )
        
        print(f"✓ Configuration created with {len(config['agents'])} agents")
        
        # Try to run with ABIDES
        try:
            from abides_core import abides
            
            print("\\nStarting ABIDES simulation...")
            end_state = abides.run(config)
            
            print("\\n🎉 Simulation completed successfully!")
            print(f"   Final agents: {len(end_state.get('agents', []))}")
            
        except ImportError:
            print("\\n⚠️  ABIDES framework not found")
            print("   Install ABIDES to run actual simulations")
            print("   For now, showing configuration details:")
            
            llm_agents = [
                agent for agent in config['agents']
                if 'LLM' in str(agent.get('agent_class', ''))
            ]
            
            print(f"   - Total agents: {len(config['agents'])}")
            print(f"   - LLM agents: {len(llm_agents)}")
            print(f"   - Symbols: {config.get('symbols', [])}")
            print(f"   - Duration: {config.get('start_time', '')} to {config.get('end_time', '')}")
    
    except ImportError as e:
        print(f"\\n⚠️  Configuration module not found: {e}")
        print("   Make sure all source files are in the correct location")
    
    print("\\nExample completed!")

if __name__ == "__main__":
    main()
'''
        
        with open(example_script, 'w') as f:
            f.write(example_content)
        
        # Make executable
        os.chmod(example_script, 0o755)
        
        self.log(f"✓ Example script created: {example_script}")
    
    def run_tests(self, test_type="quick"):
        """Run test suite to verify installation"""
        self.log(f"Running {test_type} tests...")
        
        test_script = self.project_root / "abides_test_suite.py"
        python_executable = sys.executable
        
        if not test_script.exists():
            self.log("Test script not found, skipping tests", "WARNING")
            return
        
        try:
            if test_type == "quick":
                self.run_command(f"{python_executable} {test_script} --quick")
            elif test_type == "integration":
                self.run_command(f"{python_executable} {test_script} --integration")
            else:
                self.run_command(f"{python_executable} {test_script}")
            
            self.log("✓ All tests passed!")
            
        except subprocess.CalledProcessError:
            self.log("Some tests failed, but continuing setup", "WARNING")
    
    def print_completion_message(self):
        """Print setup completion message"""
        print("\\n" + "=" * 60)
        print("🎉 ABIDES-LLM Setup Completed!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Edit .env file and add your OpenAI API key")
        print("2. Activate virtual environment:")
        
        if os.name == 'nt':
            print(f"   {self.venv_path}\\\\Scripts\\\\activate")
        else:
            print(f"   source {self.venv_path}/bin/activate")
        
        print("3. Run example simulation:")
        print("   python3 example_run.py")
        print()
        print("4. Test the installation:")
        print("   python3 abides_test_suite.py --quick")
        print()
        print("Happy simulating! 🚀")


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description="Setup ABIDES-LLM Integration Environment"
    )
    
    parser.add_argument(
        "--abides", 
        choices=["auto", "clone", "skip"],
        default="auto",
        help="ABIDES installation method (default: auto)"
    )
    
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="Skip virtual environment creation"
    )
    
    parser.add_argument(
        "--test",
        choices=["none", "quick", "integration", "all"],
        default="quick",
        help="Test level to run (default: quick)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Initialize setup manager
    setup = ABIDESLLMSetup(verbose=args.verbose)
    
    try:
        # Check Python version
        setup.check_python_version()
        
        # Organize files
        setup.move_files_to_root()
        
        # Create virtual environment
        if not args.no_venv:
            setup.create_virtual_environment()
            setup.activate_virtual_environment()
        
        # Install requirements
        setup.install_base_requirements()
        
        # Setup ABIDES
        setup.setup_abides(args.abides)
        
        # Setup environment
        setup.setup_environment_file()
        
        # Create example script
        setup.create_example_script()
        
        # Run tests
        if args.test != "none":
            setup.run_tests(args.test)
        
        # Print completion message
        setup.print_completion_message()
        
    except Exception as e:
        setup.log(f"Setup failed: {e}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()