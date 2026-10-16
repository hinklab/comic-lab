"""
app.py - Comic-Lab Ultra Studio Entrypoint.
Standalone ASGI application powered by Starlette & Uvicorn.
Zero Streamlit dependency.
Zero AI API Keys (free/local translation model); online services for data storage are fine.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import uvicorn
from server import app

def main():
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n=======================================================")
    print(f"[OK] COMIC-LAB ULTRA STUDIO (FastAPI/Starlette Engine)")
    print(f"[URL] Studio URL: http://localhost:{port}")
    print(f"[INFO] Zero Streamlit | RAM: ~35 MB | 60 FPS Canvas")
    print(f"=======================================================\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
