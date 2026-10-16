import gemini_enhancements
from PIL import Image

img_path = r'C:\Users\Ozod\.gemini\antigravity\brain\b9e18c00-a13c-4536-8eb0-753bbc58e136\.user_uploaded\media_1791797877109_bf8766fc.jpg'
img = Image.open(img_path)

import glossary_manager
glossary_instr = glossary_manager.format_glossary_prompt()
prompt = (
    "Extract dialogue speech bubbles from this comic page into natural Uzbek.\n"
    f"{glossary_instr}\n"
    f"{gemini_enhancements.UZBEK_QUALITY_PROMPT_INSTRUCTIONS}\n"
    'Output JSON strictly as: {"bubbles": [{"box_2d": [ymin, xmin, ymax, xmax], "speaker": "str", "original_text": "str", "uzbek_translation": "str", "bubble_type": "speech"|"caption"}]}\n'
    "CRITICAL: Each bubble must be an independent JSON object with its OWN keys: 'box_2d', 'speaker', 'original_text', 'uzbek_translation', 'bubble_type'. NEVER omit the keys and NEVER merge multiple bubbles into a single array.\n"
    "Rules:\n"
    "- box_2d: normalized 0-1000 integers [ymin, xmin, ymax, xmax] tightly enclosing EACH bubble border.\n"
    "- MANDATORY: Extract EVERY authentic spoken dialogue bubble (white bubbles with dark text), including short 1-2 word calls (e.g. 'Uff!', 'Look!', 'Yes!', 'Reed!'), even if located in dark shadows or nighttime backgrounds.\n"
    "- MANDATORY CONJOINED BUBBLE RULE: If two or more bubble lobes touch, chain, or connect in a figure-8 / double-lobed / peanut shape (stacked vertically or placed horizontally), you MUST extract EACH LOBE AS ITS OWN SEPARATE OBJECT in the 'bubbles' list with its own box_2d, original_text, and uzbek_translation! NEVER merge conjoined lobes or multi-sentence lobes into a single box! Each lobe holds its own separate sentence/thought and must be an independent bubble.\n"
    "- MANDATORY MULTI-BUBBLE RULE: When a character speaks in multiple distinct bubbles within the same panel, NEVER merge them into one box. Output each distinct bubble as its own separate entry.\n"
    "- Translate into punchy, colloquial Uzbek preserving character voice (Peter Parker respectful 'SIZ', Jameson fiery NYC slang), keep (!, ?, --).\n"
    "- ONLY extract spoken character dialogue bubbles and narrator story captions (e.g. 'Meanwhile...', 'Later...').\n"
    "- IGNORE in-scene environmental text, architectural signs, establishment names, building headers, street plaques, room numbers, storefronts, and logos (e.g. 'NEW YORK INSTITUTE OF TECHNOLOGY', 'EXIT'). These are background artwork elements, NOT dialogue bubbles.\n"
    "- IGNORE in-scene background art, architectural beams, lamps/fixtures, furniture, formulas, and sound effects (SFX: FWUNG, BAM)."
)

raw = gemini_enhancements.call_gemini_api(prompt, image=img, response_json=True, use_cache=False)
print("RAW RESPONSE LENGTH:", len(raw))
print("RAW RESPONSE:")
print(raw)
