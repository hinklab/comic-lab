import os
import unittest
from PIL import Image
import gemini_enhancements
import engine

class TestGeminiIntegration(unittest.TestCase):
    def test_gemini_api_key_loaded(self):
        self.assertTrue(gemini_enhancements.is_gemini_enabled())
        key = gemini_enhancements.get_gemini_api_key()
        self.assertIsNotNone(key)
        self.assertTrue(len(key) > 20)

    def test_gemini_dialogue_translation(self):
        text = "YOU'RE WRONG ABOUT ME, MAYOR JAMESON!"
        uz = gemini_enhancements.gemini_translate_dialogue(text, speaker="Peter Parker")
        self.assertTrue(len(uz) > 0)
        self.assertNotIn("YOU'RE WRONG", uz)
        # Should be translated into Uzbek
        print(f"\n[Test] Orig: {text} -> Uz: {uz}")

    def test_gemini_vision_bubble_scan_panel1(self):
        if os.path.exists("panel1_source.png"):
            img = Image.open("panel1_source.png")
            bubbles = engine.scan_bubbles_ocr(img)
            self.assertGreaterEqual(len(bubbles), 3)
            for b in bubbles:
                self.assertTrue(len(b.original_text) > 0)
                self.assertTrue(len(b.uzbek_translation) > 0)
                print(f"[Test Bubble] {b.speaker}: {b.original_text} -> {b.uzbek_translation}")

if __name__ == "__main__":
    unittest.main()
