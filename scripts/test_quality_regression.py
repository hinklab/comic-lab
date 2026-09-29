"""
scripts/test_quality_regression.py - Translates reviewed comic lines to verify quality
and compare against reference translations.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force stdout to UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import engine
from engine import SpeechBubble

test_lines = [
    ("WHAT THE--?!", "BU NIMA BALO?!", "BU NIMA--?!"),
    ("LOOK OUT!", "EHTIYOT BO'L!", "QARANG!"),
    ("DON'T PANIC!", "VAHIMAGA TUSHMA!", "VAHIMA QILMANG!"),
    ("JUST GIVE ME TEN SECONDS!", "MENGA ATIGI O'N SONIYA BER!", "MENGA FAQAT O'N SONIYA BERING!"),
    ("IMPRESSED? DON'T BOTHER...", "QOYIL QOLDINGMI? HOVLIQMA...", "TAASSUROT QOLDIRDINGIZMI? TASHVISHLANMANG..."),
    ("I CAN SEE IT...", "O'ZIM HAM KO'RIB TURIBMAN...", "MEN BUNI KO'RYAPMAN..."),
    ("CRITICAL LIMIT REACHED!", "XAVFLI DARAJA YETDI!", "KRITIK CHEGARA ERISHILDI!"),
    ("HOLD ON TIGHT, SPIDER-MAN IS ON THE JOB!", "MAHKAM USHLA, O'RGIMCHAK-ODAM ISHGA KISHDAGI!", "MAHKAM USHLANG, SPIDER-MAN ISHDA!"),
]

bubbles = [
    SpeechBubble(bubble_id=i+1, x0=10, y0=i*30, x1=200, y1=i*30+25, original_text=t[0], is_active=True)
    for i, t in enumerate(test_lines)
]

translated = engine.translate_bubbles_list(bubbles, use_subprocess=True)

col_id = "ID"
col_src = "Inglizcha Manba"
col_out = "Subprocess NLLB + Spider-Man Tone"
col_ref = "Google / Literal Reference"

print("\n" + "=" * 115)
print(f"{col_id:<4} | {col_src:<32} | {col_out:<42} | {col_ref}")
print("-" * 115)
for b, t in zip(translated, test_lines):
    print(f"{b.bubble_id:<4} | {b.original_text:<32} | {b.uzbek_translation:<42} | {t[2]}")
print("=" * 115)
