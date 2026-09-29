"""
streamlit_app.py - Streamlit Entry Point.
Delegates execution directly to app.py.
"""
import runpy

if __name__ == "__main__" or True:
    runpy.run_path("app.py", run_name="__main__")
