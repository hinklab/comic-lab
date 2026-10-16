import time
from PIL import Image
import engine
import gemini_enhancements
import cv2
import numpy as np

img_path = r'C:\Users\Ozod\.gemini\antigravity\brain\b9e18c00-a13c-4536-8eb0-753bbc58e136\.user_uploaded\media_1791797877109_bf8766fc.jpg'
pil_img = Image.open(img_path)

print("Running engine.scan_bubbles_ocr (which server.py calls)...")
bubbles = engine.scan_bubbles_ocr(pil_img, ocr_engine="gemini", force_refresh=True)
print(f"Scanned {len(bubbles)} bubbles.")
for b in bubbles:
    print(f"#{b.bubble_id} [{b.x0}, {b.y0}, {b.x1}, {b.y1}] orig: '{b.original_text}' | uz: '{b.uzbek_translation}'")

print("Running engine.clean_page_ink_telea...")
cleaned = engine.clean_page_ink_telea(pil_img, bubbles)
cleaned.save("scratch/debug_server_clean.png")
print("Saved scratch/debug_server_clean.png")
