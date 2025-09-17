#!/usr/bin/env python3
"""
Simple smoke test for ABIDES-LLM demo
Runs a single-event simulation to verify basic functionality without ABIDES.
"""

import sys
import os


def main() -> int:
    # Ensure examples are importable
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(project_root, "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    try:
        from simple_abides_llm_demo import ABIDESLLMSimulation
    except Exception as e:
        print(f"❌ Failed to import demo: {e}")
        return 2

    try:
        sim = ABIDESLLMSimulation(symbols=["ABM"], llm_enabled=False)
        sim.run_simulation(num_events=1)
        print("✅ Smoke test passed")
        return 0
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

