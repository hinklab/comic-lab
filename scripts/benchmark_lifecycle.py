import os
import sys
import gc
import json
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force stdout to UTF-8 to prevent cp1251 encoding errors on Windows
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import psutil
proc = psutil.Process(os.getpid())

def get_rss():
    return proc.memory_info().rss / (1024 * 1024)

records = []

def record(step_name, description):
    rss_val = get_rss()
    records.append({"step": step_name, "description": description, "rss_mb": round(rss_val, 2)})
    print(f"[{step_name}] RSS: {rss_val:.2f} MB - {description}", flush=True)

# 1. Startup
record("1_STARTUP", "Sof Python va psutil yuklanganda")

# 2. Engine import
import engine
record("2_ENGINE_IMPORTED", "engine, cv2, PIL, naturalization_rules import qilinganda")

# 3. EasyOCR reader creation
reader = engine.get_ocr_reader()
record("3_EASYOCR_CREATED", "EasyOCR Reader(['en'], gpu=False) yaratilganda")

# 4. Scan page_4.png
sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples", "page_4.png"))
im = Image.open(sample_path)
bubbles = engine.scan_bubbles_ocr(im)
record("4_SCAN_FINISHED", f"Tiled OCR orqali {len(bubbles)} ta pufak topilganda")

# 5. Inpainting
cleaned = engine.clean_page_ink_telea(im, bubbles)
record("5_INPAINTING_FINISHED", "Telea algoritmi orqali matnlar tozalanganda")

# 6. Release EasyOCR
del reader
del cleaned
engine.release_ocr_reader()
gc.collect()
record("6_AFTER_RELEASE_OCR", "EasyOCR reader xotiradan bo'shatilib gc.collect() qilinganda")

# 7. NLLB Load
import local_translator
translator = local_translator.get_translator()
record("7_NLLB_LOADED", "CTranslate2 NLLB-200 int8 modeli yuklanganda")

# 8. Translation
translated = engine.translate_bubbles_list(bubbles)
record("8_TRANSLATE_FINISHED", f"{len(translated)} ta pufak o'zbek tiliga tarjima qilinganda")

# 9. Release Translator
local_translator.release_translator()
gc.collect()
record("9_AFTER_RELEASE_TRANSLATOR", "NLLB modeli bo'shatilib gc.collect() qilinganda")

print("\n--- YAKUNIY O'LCHOV JADVALI ---", flush=True)
print("| Qadam | Tavsif | O'lchangan RSS (MB) |", flush=True)
print("| :--- | :--- | :---: |", flush=True)
for r in records:
    print(f"| {r['step']} | {r['description']} | **{r['rss_mb']:.2f} MB** |", flush=True)

with open("benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

print("\nBajarildi!", flush=True)
