# Make submodules accessible
import sys
import os

# Add the verl directory to sys.path if not already there
verl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if verl_dir not in sys.path:
    sys.path.insert(0, verl_dir)
