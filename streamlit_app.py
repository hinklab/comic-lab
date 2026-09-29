"""
streamlit_app.py - Streamlit Cloud Entry Point.
Profiles dependency import memory costs (Step 1) and delegates to app.py.
"""
import os
import sys
import runpy

# Step 1: Run one-off import memory profiling before any other project code loads
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scripts.import_profiler as profiler
    profiler.profile_imports()
except Exception as _prof_err:
    print(f"[IMPORT_PROFILER_WARN] Could not profile imports: {_prof_err}", flush=True)

if __name__ == "__main__" or True:
    runpy.run_path("app.py", run_name="__main__")
