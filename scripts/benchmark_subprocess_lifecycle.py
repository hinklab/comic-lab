import os
import sys
import gc
import json
import subprocess
import tempfile
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    print(f"[{step_name}] Parent Process RSS: {rss_val:.2f} MB - {description}", flush=True)

# 1. Startup
record("1_STARTUP", "Sof Python va psutil")

# 2. Import engine (parent process)
import engine
record("2_ENGINE_IMPORTED", "engine, cv2, PIL import qilinganda")

# 3. Run EasyOCR in SUBPROCESS
sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples", "page_4.png"))
worker_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "ocr_worker.py"))

with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
    out_json_path = tf.name

try:
    print("[PARENT] Subprocess ishga tushirilmoqda...", flush=True)
    res = subprocess.run(
        [sys.executable, "-u", worker_script, sample_path, out_json_path],
        capture_output=True,
        text=True,
        check=True
    )
    print(res.stdout, flush=True)
finally:
    pass

with open(out_json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
bubbles = [engine.SpeechBubble(**d) for d in raw_data]
try:
    os.remove(out_json_path)
except Exception:
    pass

record("3_AFTER_SUBPROCESS_EXIT", f"Subprocess tugagach va {len(bubbles)} ta pufak o'qilganda")

# 4. Inpainting in parent process
im = Image.open(sample_path)
cleaned = engine.clean_page_ink_telea(im, bubbles)
record("4_INPAINTING_FINISHED", "Telea algoritmi orqali matnlar tozalanganda")

# 5. Load NLLB in parent process
import local_translator
translator = local_translator.get_translator()
record("5_NLLB_LOADED", "CTranslate2 NLLB-200 int8 modeli yuklanganda")

# 6. Translation in parent process
translated = engine.translate_bubbles_list(bubbles)
record("6_TRANSLATE_FINISHED", f"{len(translated)} ta pufak o'zbek tiliga tarjima qilinganda")

# 7. Release NLLB
local_translator.release_translator()
gc.collect()
record("7_AFTER_RELEASE_TRANSLATOR", "NLLB modeli bo'shatilib gc.collect() qilinganda")

print("\n--- SUBPROCESS VARIANTI YAKUNIY O'LCHOV JADVALI ---", flush=True)
print("| Qadam | Tavsif | O'lchangan RSS (MB) |", flush=True)
print("| :--- | :--- | :---: |", flush=True)
for r in records:
    print(f"| {r['step']} | {r['description']} | **{r['rss_mb']:.2f} MB** |", flush=True)

with open("benchmark_subprocess_results.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

print("\nSubprocess testi yakunlandi!", flush=True)
