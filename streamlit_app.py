"""
Streamlit entrypoint for Streamlit Community Cloud deployment.
Redirects to the main UI application at src/interfaces/ui/app.py.
"""

import runpy
from pathlib import Path

app_path = Path(__file__).parent / "src" / "interfaces" / "ui" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
