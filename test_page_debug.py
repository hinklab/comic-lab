import gemini_enhancements
from PIL import Image

img_path = r'C:\Users\Ozod\.gemini\antigravity\brain\b9e18c00-a13c-4536-8eb0-753bbc58e136\.user_uploaded\media_1791797877109_bf8766fc.jpg'
img = Image.open(img_path)
print('Image loaded:', img.size)

results = gemini_enhancements.gemini_extract_comic_bubbles(img, use_cache=False)
print('Total bubbles extracted by Gemini:', len(results))
for i, b in enumerate(results):
    print(f"[{i+1}] box: {b.get('box')} | orig: {b.get('original_text')} | uz: {b.get('uzbek_translation')}")
