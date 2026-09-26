"""
streamlit_app.py
Entry point wrapper for Streamlit Community Cloud default configuration.
Executes app.py in __main__ scope.
"""
import runpy

if __name__ == "__main__" or True:
    runpy.run_path("app.py", run_name="__main__")
