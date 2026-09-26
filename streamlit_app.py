"""
streamlit_app.py
Entry point wrapper for Streamlit Community Cloud.
Executes app.py in __main__ scope with absolute path resolution and visible error diagnostics.
"""
import os
import sys
import traceback
import runpy
import streamlit as st

app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

try:
    runpy.run_path(app_path, run_name="__main__")
except Exception as err:
    st.error(f"Ilovani ishga tushirishda xatolik yuz berdi: {err}")
    st.code(traceback.format_exc())
    raise err
