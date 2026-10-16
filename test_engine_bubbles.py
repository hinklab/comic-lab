import engine
from PIL import Image

img_path = r'C:\Users\Ozod\.gemini\antigravity\brain\b9e18c00-a13c-4536-8eb0-753bbc58e136\.user_uploaded\media_1791797877109_bf8766fc.jpg'
img = Image.open(img_path)

bubbles = engine.scan_bubbles_ocr(img, ocr_engine="gemini", force_refresh=True)
print(f"Total bubbles scanned by engine: {len(bubbles)}")
for b in bubbles:
    bw = b.x1 - b.x0
    bh = b.y1 - b.y0
    print(f"ID #{b.bubble_id} | box: [{b.x0}, {b.y0}, {b.x1}, {b.y1}] (w={bw}, h={bh}) | shape={b.shape_type} | orig: '{b.original_text}' | uz: '{b.uzbek_translation}'")
