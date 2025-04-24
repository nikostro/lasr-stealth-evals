import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import and run the task
from lasr_stealth_evals.pump_and_dump.task import main

if __name__ == "__main__":
    main() 