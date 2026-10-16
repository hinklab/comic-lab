import gemini_enhancements
from PIL import Image

img_path = r'C:\Users\Ozod\.gemini\antigravity\brain\b9e18c00-a13c-4536-8eb0-753bbc58e136\.user_uploaded\media_1791797877109_bf8766fc.jpg'
img = Image.open(img_path)

print("Calling gemini_extract_comic_bubbles with use_cache=False...")
res = gemini_enhancements.gemini_extract_comic_bubbles(img, use_cache=False)
print(f"Direct call: {len(res)} bubbles")
for b in res:
    print(b)
