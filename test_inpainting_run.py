import json
from PIL import Image
import cv2
import numpy as np
import engine
import gemini_enhancements

img_path = r'C:\Users\Ozod\.gemini\antigravity\brain\b9e18c00-a13c-4536-8eb0-753bbc58e136\.user_uploaded\media_1791797877109_bf8766fc.jpg'
img = Image.open(img_path)

# Let's extract with clean prompt
with open('test_clean_prompt.py', 'r') as f:
    pass

import test_clean_prompt
data = json.loads(test_clean_prompt.raw)
bubbles_raw = data['bubbles']

W_img, H_img = img.size
speech_bubbles = []
for i, b in enumerate(bubbles_raw):
    ymin, xmin, ymax, xmax = b['box_2d']
    x0 = int(xmin * W_img / 1000.0)
    y0 = int(ymin * H_img / 1000.0)
    x1 = int(xmax * W_img / 1000.0)
    y1 = int(ymax * H_img / 1000.0)
    sb = engine.SpeechBubble(
        bubble_id=i+1,
        x0=x0, y0=y0, x1=x1, y1=y1,
        box=[x0, y0, x1, y1],
        original_text=b['original_text'],
        uzbek_translation=b['uzbek_translation'],
        shape_type="oval"
    )
    speech_bubbles.append(sb)

print(f"Created {len(speech_bubbles)} SpeechBubble objects")
cleaned = engine.clean_page_ink_telea(img, speech_bubbles)
cleaned.save("scratch/test_cleaned_result.png")
print("Saved scratch/test_cleaned_result.png")
