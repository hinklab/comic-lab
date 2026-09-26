"""
scripts/download_nllb_model.py
One-time download script for NLLB-200-distilled-600M CTranslate2 int8 model.
Once downloaded, the model runs 100% locally on CPU with zero internet access.
"""

import os
import sys
from huggingface_hub import snapshot_download

REPO_ID = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "nllb-200-distilled-600M-ct2-int8")

def download_model():
    print(f"Downloading {REPO_ID} to {LOCAL_DIR}...")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"Successfully downloaded to {LOCAL_DIR}!")

if __name__ == "__main__":
    download_model()
