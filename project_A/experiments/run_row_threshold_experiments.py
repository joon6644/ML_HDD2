import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from experiments.run_unified_threshold_experiments import run_unified_threshold_experiments

def run_row_threshold_experiments():
    print("[NOTICE] Running unified experiments (combining proposed & row-level threshold evaluation)...")
    run_unified_threshold_experiments()

if __name__ == "__main__":
    run_row_threshold_experiments()
