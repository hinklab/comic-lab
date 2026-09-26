import sys, os
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

sys.path.insert(0, r'C:\Users\Ozod\Desktop\comic-lab')
import engine
import bubble_lettering

print("1. Opening samples/page_4.png...")
img_pil = Image.open('samples/page_4.png')
img_cv = cv2.imread('samples/page_4.png')
h, w = img_cv.shape[:2]

print("2. Scanning bubbles with scan_bubbles_ocr...")
bubbles = engine.scan_bubbles_ocr(img_pil)
print(f"Detected {len(bubbles)} bubbles on page_4.")

# Find ghost bubbles (b8 and b9: "THAT'S NOT SPIDER-MAN!" and "THAT'S OTTO OCTAVIUS...")
ghost_b8 = None
ghost_b9 = None
for b in bubbles:
    txt = b.original_text.upper()
    if "SPIDER" in txt and "THAT'S" in txt:
        ghost_b8 = b
    elif "OCTAVIUS" in txt or "DOCTOR" in txt:
        ghost_b9 = b

print(f"Ghost bubble 8: {ghost_b8.original_text if ghost_b8 else 'Not found'}")
print(f"Ghost bubble 9: {ghost_b9.original_text if ghost_b9 else 'Not found'}")

# Assign idiomatic Uzbek translations
if ghost_b8:
    ghost_b8.uzbek_translation = "BU O'RGIMCHAK-ODAM EMAS!"
if ghost_b9:
    ghost_b9.uzbek_translation = "BU OTTO OKTAVIUS, KALBOSH NODON.\nMENING TANAMDAGI DOKTOR OKTOPUS!"

# Other bubbles in panel 3
for b in bubbles:
    txt = b.original_text.upper()
    if "ANYTHING FOR" in txt:
        b.uzbek_translation = "QONUN VA TARTIB KUCHLARI UCHUN NIMA BO'LSA HAM."

print("3. Inpainting page with clean_page_ink_telea...")
cleaned_pil = engine.clean_page_ink_telea(img_pil, bubbles)
cleaned_cv = cv2.cvtColor(np.array(cleaned_pil), cv2.COLOR_RGB2BGR)

print("4. Typesetting lettering with typeset_lettering_on_page...")
fonts = engine.get_available_fonts()
font_path = list(fonts.values())[0] if fonts else "arial.ttf"

lettered_pil = engine.typeset_lettering_on_page(
    cleaned_image=cleaned_pil,
    bubbles=bubbles,
    font_path=font_path,
    max_font_size=26,
    min_font_size=10
)
lettered_cv = cv2.cvtColor(np.array(lettered_pil), cv2.COLOR_RGB2BGR)

# Crop panel 3: [y: 1320..2220, x: 620..1720]
py0, py1 = 1320, 2220
px0, px1 = 620, 1720

p3_orig = img_cv[py0:py1, px0:px1]
p3_clean = cleaned_cv[py0:py1, px0:px1]
p3_lettered = lettered_cv[py0:py1, px0:px1]

# Build a side-by-side composite comparison image: [ORIGINAL | CLEANED (ZERO WHITEOUT) | UZBEK LETTERED]
cw = px1 - px0
ch = py1 - py0
header_h = 60
composite_w = cw * 3
composite_h = ch + header_h

comp = np.full((composite_h, composite_w, 3), (25, 25, 28), dtype=np.uint8)

# Paste crops
comp[header_h:header_h+ch, 0:cw] = p3_orig
comp[header_h:header_h+ch, cw:cw*2] = p3_clean
comp[header_h:header_h+ch, cw*2:cw*3] = p3_lettered

# Add titles using PIL Draw
comp_pil = Image.fromarray(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(comp_pil)
try:
    title_font = ImageFont.truetype(font_path, 28)
except Exception:
    title_font = ImageFont.load_default()

draw.text((cw // 2 - 160, 15), "1. ORIGINAL (TINTED ART)", font=title_font, fill=(200, 200, 210))
draw.text((cw + cw // 2 - 180, 15), "2. CLEANED (ZERO WHITEOUT)", font=title_font, fill=(50, 220, 150))
draw.text((cw * 2 + cw // 2 - 180, 15), "3. UZBEK LETTERED (TINT PRESERVED)", font=title_font, fill=(80, 180, 255))

# Vertical divider lines
draw.line([(cw, 0), (cw, composite_h)], fill=(60, 60, 70), width=3)
draw.line([(cw * 2, 0), (cw * 2, composite_h)], fill=(60, 60, 70), width=3)

artifact_dir = r"C:\Users\Ozod\.gemini\antigravity\brain\b9e18c00-a13c-4536-8eb0-753bbc58e136"
out_comp = os.path.join(artifact_dir, "ghost_panel_before_after.png")
comp_pil.save(out_comp)
print(f"Saved side-by-side deliverable to {out_comp}")

# Also save individual crops in artifact dir
cv2.imwrite(os.path.join(artifact_dir, "ghost_panel_orig.png"), p3_orig)
cv2.imwrite(os.path.join(artifact_dir, "ghost_panel_cleaned.png"), p3_clean)
cv2.imwrite(os.path.join(artifact_dir, "ghost_panel_lettered.png"), p3_lettered)
print("Saved individual panel crops.")
