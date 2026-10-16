"""
scripts/ocr_tile_worker.py - Short-lived subprocess worker for a SINGLE EasyOCR tile.
Runs EasyOCR inference on one cropped tile, writes the raw bounding boxes and text to JSON,
and exits immediately so 100% of PyTorch/CRAFT memory is returned to the OS.
"""
import os
import sys

# Balanced multi-threading for PyTorch CPU operations (prevents 1-thread bottlenecks while keeping RAM lean)
cpu_cnt = os.cpu_count() or 4
thread_count = str(min(4, max(2, cpu_cnt // 2)))
os.environ["OMP_NUM_THREADS"] = thread_count
os.environ["MKL_NUM_THREADS"] = thread_count
os.environ["OPENBLAS_NUM_THREADS"] = thread_count
os.environ["VECLIB_MAXIMUM_THREADS"] = thread_count
os.environ["NUMEXPR_NUM_THREADS"] = thread_count

import json
from PIL import Image
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("Usage: python ocr_tile_worker.py <tile_img_path> <tile_out_json> [canvas_size]", file=sys.stderr)
        sys.exit(1)

    img_path = sys.argv[1]
    out_path = sys.argv[2]
    canvas_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1600

    import easyocr
    import torch
    try:
        torch.set_num_threads(int(thread_count))
    except Exception:
        pass

    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    im = Image.open(img_path).convert("RGB")
    arr = np.array(im)

    with torch.no_grad():
        raw_results = reader.readtext(arr, paragraph=False, canvas_size=canvas_size)

    serializable = []
    for item in raw_results:
        if len(item) == 2:
            bbox, text = item
            conf = 1.0
        else:
            bbox, text, conf = item
        bbox_list = [[float(pt[0]), float(pt[1])] for pt in bbox]
        serializable.append({
            "bbox": bbox_list,
            "text": str(text).strip(),
            "conf": float(conf)
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)

    print(f"[OCR_TILE_WORKER] Extracted {len(serializable)} raw items to {out_path}", flush=True)

if __name__ == "__main__":
    main()
