"""
scripts/ocr_worker.py - Isolated subprocess worker for EasyOCR scanning.
Runs in a separate OS process so that PyTorch/EasyOCR memory is 100% reclaimed by the OS on exit.
"""
import os
import sys
import json
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import engine

def main():
    if len(sys.argv) < 3:
        print("Usage: python ocr_worker.py <input_image_path> <output_json_path>", file=sys.stderr)
        sys.exit(1)
    
    img_path = sys.argv[1]
    out_path = sys.argv[2]
    
    im = Image.open(img_path)
    bubbles = engine.scan_bubbles_ocr(im, use_subprocess=False)
    data = [b.model_dump() for b in bubbles]
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    
    print(f"[OCR_WORKER] Successfully extracted {len(bubbles)} bubbles to {out_path}", flush=True)

if __name__ == "__main__":
    main()
