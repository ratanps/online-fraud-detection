"""
Streamlit Cloud entry point (app/streamlit_app.py).
Runs the main dashboard from the project root.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "streamlit_app.py"), run_name="__main__")
