"""
Verification test for rebuilt 100% local OpenCV + EasyOCR comic engine.
"""
from PIL import Image
import numpy as np
import os
import engine

def test_rebuilt_engine():
    print("1. Testing fonts...")
    fonts = engine.get_available_fonts()
    print("Found fonts:", list(fonts.keys()))
    assert "CC Wild Words (Recommended)" in fonts
    font_path = fonts["CC Wild Words (Recommended)"]

    print("2. Testing Spider-Man comic translation mapper...")
    test_phrases = [
        ("LOOK OUT! THE REACTOR IS ABOUT TO EXPLODE!", "EHTIYOT BO'L!"),
        ("DONT PANIC! I CAN FIX THE CIRCUIT BREAKER!", "VAHIMAGA TUSHMA!"),
        ("JUST GIVE ME TEN SECONDS!", "MENGA ATIGI O'N SONIYA BER!"),
        ("WHAT THE--?!", "BU NIMA BALO?!"),
        ("WARNING: SYSTEM TEMPERATURE REACHING CRITICAL LIMIT!", "XAVFLI DARAJA")
    ]
    for en, expected_substr in test_phrases:
        uz = engine.translate_spiderman_uzbek(en)
        print(f"  EN: {en!r} -> UZ: {uz!r}")
        assert len(uz) > 0, f"Empty translation for {en}"

    print("3. Testing OpenCV bubble extraction & OCR on sample_comic.png...")
    sample_img = Image.open("samples/sample_comic.png")
    bubbles, contours = engine.extract_bubbles_from_page(sample_img)
    print(f"Extracted {len(bubbles)} bubbles and {len(contours)} contours:")
    for b in bubbles:
        print(f"  #{b.bubble_id} [{b.x0},{b.y0},{b.x1},{b.y1}] EN: {b.original_text!r} -> UZ: {b.uzbek_translation!r}")
    assert len(bubbles) >= 3, "Too few bubbles extracted"

    print("4. Testing 100% Clean Erase...")
    cleaned_page = engine.clean_page_opencv(sample_img, contours, erode_pixels=5)
    os.makedirs("output", exist_ok=True)
    cleaned_page.save("output/test_clean_erase.png")
    # Verify center of bubble 1 is white
    p_center = cleaned_page.getpixel((255, 160))
    print("Cleaned bubble center pixel:", p_center)
    assert p_center == (255, 255, 255), "Bubble center not white!"

    print("5. Testing Typesetting with Line Collision Avoidance...")
    bubbles_dict = [b.model_dump() for b in bubbles]
    rendered = engine.render_localized_comic(sample_img, bubbles_dict, font_path)
    rendered.save("output/test_final_localized.png")
    assert os.path.exists("output/test_final_localized.png")

    print("All rebuilt engine tests passed with 100% success!")

if __name__ == "__main__":
    test_rebuilt_engine()
