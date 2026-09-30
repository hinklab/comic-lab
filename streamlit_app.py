"""
streamlit_app.py - Streamlit Entry Point.
Delegates execution directly to app.py.
"""
import sys
import runpy

# Ensure fresh imports of our repo's modules across hot-reloads
for mod in list(sys.modules.keys()):
    if mod in ["engine", "bubble_lettering", "bubble_mask_editor", "character_profiles", "naturalization_rules", "quality_gates", "local_translator", "history_manager", "sfx_engine"]:
        del sys.modules[mod]

if __name__ == "__main__" or True:
    runpy.run_path("app.py", run_name="__main__")

