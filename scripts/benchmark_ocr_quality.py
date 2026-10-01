import os
import sys
import time
import json
import psutil
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine

# Ensure tesseract path on Windows
import pytesseract
if sys.platform == "win32":
    for cand in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe")
    ]:
        if os.path.exists(cand):
            pytesseract.pytesseract.tesseract_cmd = cand
            break

def run_tesseract_raw(image: Image.Image, psm: int = 3, min_conf: float = 25.0):
    config = f"--psm {psm}"
    d = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    
    n_boxes = len(d['text'])
    words = []
    for i in range(n_boxes):
        text = d['text'][i].strip()
        conf = float(d['conf'][i])
        if not text or conf < min_conf:
            continue
        words.append({
            'x': int(d['left'][i]),
            'y': int(d['top'][i]),
            'w': int(d['width'][i]),
            'h': int(d['height'][i]),
            'text': text,
            'conf': conf
        })

    words.sort(key=lambda w: (w['y'], w['x']))

    lines = []
    for w in words:
        matched = False
        for line in lines:
            last_w = line['words'][-1]
            v_overlap = min(w['y'] + w['h'], last_w['y'] + last_w['h']) - max(w['y'], last_w['y'])
            min_h = min(w['h'], last_w['h'])
            if v_overlap > 0.35 * min_h:
                gap = w['x'] - (last_w['x'] + last_w['w'])
                max_gap = max(40, int(1.8 * max(w['h'], last_w['h'])))
                if -15 <= gap <= max_gap:
                    line['words'].append(w)
                    line['x1'] = max(line['x1'], w['x'] + w['w'])
                    line['y0'] = min(line['y0'], w['y'])
                    line['y1'] = max(line['y1'], w['y'] + w['h'])
                    matched = True
                    break
        if not matched:
            lines.append({
                'words': [w],
                'x0': w['x'],
                'y0': w['y'],
                'x1': w['x'] + w['w'],
                'y1': w['y'] + w['h']
            })

    results = []
    for line in lines:
        line_str = " ".join(w['text'] for w in line['words']).strip()
        if not line_str:
            continue
        avg_conf = (sum(w['conf'] for w in line['words']) / len(line['words'])) / 100.0
        x0, y0, x1, y1 = line['x0'], line['y0'], line['x1'], line['y1']
        bbox = [[float(x0), float(y0)], [float(x1), float(y0)], [float(x1), float(y1)], [float(x0), float(y1)]]
        results.append((bbox, line_str, avg_conf))
    
    return results

def get_tree_rss_mb():
    current = psutil.Process()
    total = current.memory_info().rss
    try:
        for child in current.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:
                pass
    except Exception:
        pass
    return total / (1024 * 1024)

def benchmark_page(page_path: str):
    print(f"\n==================================================")
    print(f"BENCHMARKING: {page_path}")
    print(f"==================================================")
    im = Image.open(page_path)
    
    # 1. Tesseract Benchmark
    rss_start_tess = get_tree_rss_mb()
    t0_tess = time.time()
    tess_raw = run_tesseract_raw(im)
    tess_bubbles = engine._extract_bubbles_from_ocr_results(im, tess_raw)
    tess_time = time.time() - t0_tess
    rss_end_tess = get_tree_rss_mb()
    tess_rss_delta = max(0.0, rss_end_tess - rss_start_tess)
    
    # 2. EasyOCR Benchmark (using isolated subprocess per-tile)
    rss_start_easy = get_tree_rss_mb()
    t0_easy = time.time()
    try:
        easy_bubbles = engine.scan_bubbles_ocr_subprocess(im, timeout_seconds=180)
        easy_time = time.time() - t0_easy
    except Exception as e:
        print(f"EasyOCR error: {e}")
        easy_bubbles = []
        easy_time = 0.0
    rss_end_easy = get_tree_rss_mb()
    easy_rss_delta = max(0.0, rss_end_easy - rss_start_easy)

    # Calculate metrics
    tess_avg_conf = sum(b.confidence for b in tess_bubbles) / len(tess_bubbles) if tess_bubbles else 0.0
    easy_avg_conf = sum(b.confidence for b in easy_bubbles) / len(easy_bubbles) if easy_bubbles else 0.0

    print(f"\n--- METRICS SUMMARY ---")
    print(f"Tesseract: {tess_time:.2f}s | Delta RSS: {tess_rss_delta:.1f} MB | Bubbles: {len(tess_bubbles)} | Avg Conf: {tess_avg_conf*100:.1f}%")
    print(f"EasyOCR:   {easy_time:.2f}s | Delta RSS: {easy_rss_delta:.1f} MB | Bubbles: {len(easy_bubbles)} | Avg Conf: {easy_avg_conf*100:.1f}%")

    print(f"\n--- TESSERACT BUBBLES ({len(tess_bubbles)}) ---")
    for i, b in enumerate(tess_bubbles):
        print(f"  [{i+1}] (conf={b.confidence:.2f}) [{b.x0},{b.y0},{b.x1},{b.y1}]: {b.original_text}")

    print(f"\n--- EASYOCR BUBBLES ({len(easy_bubbles)}) ---")
    for i, b in enumerate(easy_bubbles):
        print(f"  [{i+1}] (conf={b.confidence:.2f}) [{b.x0},{b.y0},{b.x1},{b.y1}]: {b.original_text}")

    return {
        "page": page_path,
        "tess": {
            "time_s": tess_time,
            "rss_delta_mb": tess_rss_delta,
            "bubble_count": len(tess_bubbles),
            "avg_conf": tess_avg_conf,
            "bubbles": [{"text": b.original_text, "conf": b.confidence, "box": [b.x0, b.y0, b.x1, b.y1]} for b in tess_bubbles]
        },
        "easy": {
            "time_s": easy_time,
            "rss_delta_mb": easy_rss_delta,
            "bubble_count": len(easy_bubbles),
            "avg_conf": easy_avg_conf,
            "bubbles": [{"text": b.original_text, "conf": b.confidence, "box": [b.x0, b.y0, b.x1, b.y1]} for b in easy_bubbles]
        }
    }

if __name__ == "__main__":
    pages = ["samples/page_4.png", "samples/page_18.png", "samples/page_22.png"]
    results = []
    for p in pages:
        if os.path.exists(p):
            res = benchmark_page(p)
            results.append(res)
    
    with open("scratch/ocr_quality_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nBenchmark complete! Saved to scratch/ocr_quality_benchmark_results.json")
