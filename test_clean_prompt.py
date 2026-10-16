import gemini_enhancements
from PIL import Image

img_path = r'C:\Users\Ozod\.gemini\antigravity\brain\b9e18c00-a13c-4536-8eb0-753bbc58e136\.user_uploaded\media_1791797877109_bf8766fc.jpg'
img = Image.open(img_path)

import glossary_manager
glossary_instr = glossary_manager.format_glossary_prompt()

prompt_clean = (
    "Extract all dialogue speech bubbles and story captions from this comic page.\n"
    f"{glossary_instr}\n"
    "Return JSON strictly as:\n"
    '{"bubbles": [\n'
    '  {"box_2d": [ymin, xmin, ymax, xmax], "speaker": "str", "original_text": "str", "uzbek_translation": "str", "bubble_type": "speech"}\n'
    ']}\n\n'
    "RULES:\n"
    "1. box_2d: normalized 0-1000 integers [ymin, xmin, ymax, xmax] tightly enclosing EACH bubble border.\n"
    "2. MANDATORY: Extract EVERY authentic spoken dialogue bubble (white bubbles with dark text) across ALL panels from top to bottom, left to right.\n"
    "3. Conjoined / connected bubbles: If a character speaks in two touching/stacked lobes, extract EACH lobe as its own separate JSON object with its own box_2d, original_text, and uzbek_translation.\n"
    "4. Translate each bubble into natural, punchy comic Uzbek.\n"
    "5. IGNORE background environmental signs, building plaques, and sound effects (SFX)."
)

raw = gemini_enhancements.call_gemini_api(prompt_clean, image=img, response_json=True, use_cache=False)
print("RAW LENGTH:", len(raw))
print(raw[:1500])
