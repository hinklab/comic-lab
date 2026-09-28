"""
streamlit_app.py - Streamlit Cloud Entry Point.
Delegates directly to app.py to maintain a single source of truth.
"""
import runpy

if __name__ == "__main__" or True:
    runpy.run_path("app.py", run_name="__main__")
