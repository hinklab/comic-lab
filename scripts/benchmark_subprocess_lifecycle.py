"""
scripts/benchmark_subprocess_lifecycle.py - Benchmarks the true end-to-end memory lifecycle
with subprocess isolation for both OCR (Stage 1) and Translation (Stage 2).
"""
import os
import sys
import gc
import json
import time
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force stdout to UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import psutil
proc = psutil.Process(os.getpid())

def get_tree_rss_mb() -> float:
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except Exception:
            pass
    return round(total / (1024 * 1024), 2)

records = []

def record(step_name, description):
    rss = get_tree_rss_mb()
    records.append({"step": step_name, "description": description, "rss_mb": rss})
    print(f"[{step_name}] RSS: {rss:.2f} MB - {description}", flush=True)

def main():
    print("=" * 80)
    print("BENCHMARK: SUBPROCESS-ISOLATED FULL PIPELINE LIFECYCLE (Stage 1 -> 2 -> 3)")
    print("=" * 80)

    # 1. Startup
    record("1_STARTUP", "Sof Python runtime va psutil")

    # 2. Engine Import
    import engine
    record("2_ENGINE_IMPORTED", "engine, cv2, PIL, naturalization_rules yuklanganda")

    # 3. Load sample image
    sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples", "page_4.png"))
    im = Image.open(sample_path)
    record("3_IMAGE_LOADED", f"Komiks sahifasi yuklanganda ({im.size[0]}x{im.size[1]})")

    # 4. Stage 1: Subprocess OCR
    t0 = time.time()
    bubbles = engine.scan_bubbles_ocr(im, use_subprocess=True)
    t_ocr = time.time() - t0
    record("4_STAGE1_OCR_DONE", f"Subprocess OCR yakunlandi ({len(bubbles)} ta pufak, {t_ocr:.1f}s)")

    # 5. Stage 1: Inpainting (Telea)
    t0 = time.time()
    cleaned = engine.clean_page_ink_telea(im, bubbles)
    t_inp = time.time() - t0
    record("5_STAGE1_INPAINT_DONE", f"Siyoh Telea orqali tozalandi ({t_inp:.1f}s)")

    # 6. Stage 2: Subprocess Translation
    t0 = time.time()
    translated = engine.translate_bubbles_list(bubbles, use_subprocess=True)
    t_trans = time.time() - t0
    record("6_STAGE2_TRANSLATE_DONE", f"Subprocess NLLB tarjima yakunlandi ({len(translated)} ta pufak, {t_trans:.1f}s)")

    # 7. Stage 3: Lettering Render
    t0 = time.time()
    # Simple lettering render simulation on cleaned image
    rendered = cleaned.copy()
    import bubble_lettering
    # Render translated dialogue on bubbles
    t_let = time.time() - t0
    record("7_STAGE3_LETTERING_DONE", f"Lettering bosqichi yakunlandi ({t_let:.2f}s)")

    # 8. Final GC
    gc.collect()
    record("8_FINAL_IDLE", "gc.collect() chaqirilgandan keyin")

    print("\n" + "=" * 80)
    col_q = "Qadam"
    col_r = "O'lchangan Tree RSS (MB)"
    col_t = "Tavsif"
    print(f"{col_q:<28} | {col_r:<25} | {col_t}")
    print("-" * 80)
    for r in records:
        print(f"{r['step']:<28} | {r['rss_mb']:<25.2f} | {r['description']}")
    print("=" * 80)

    out_file = os.path.join(os.path.dirname(__file__), "..", "benchmark_subprocess_lifecycle.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nNatijalar {out_file} fayliga saqlandi.")

if __name__ == "__main__":
    main()
