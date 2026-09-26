import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine

class TestOCRJunkFilter(unittest.TestCase):
    def test_comic_slang_and_names(self):
        """Test that common comic slang, character names, and short words pass filter."""
        test_cases = [
            ("A HECKUVA", 0.99),
            ("Spidey!", 0.90),
            ("YOU'VE DONE", 0.46),
            ("Job HERE ,", 0.44),
            ("MJ!", 0.95),
            ("JJJ", 0.90),
            ("GONNA GET YA!", 0.85),
            ("WHAT THE HECK", 0.75),
            ("PETER!", 0.90),
            ("OCTAVIUS", 0.95),
            ("I AM READY", 0.90),
        ]
        for text, conf in test_cases:
            is_junk, reason = engine.check_is_sfx_or_junk(text, conf)
            self.assertFalse(is_junk, f"Expected '{text}' to be valid, but got junk: {reason}")

    def test_phonotactic_plausibility(self):
        """Test that novel names/words with plausible phonotactics pass."""
        novel_names = ["Zorblax", "Kraglin", "Gamora", "Drax", "Corvus", "Proxima"]
        for name in novel_names:
            self.assertTrue(engine.is_phonotactically_plausible_word(name), f"{name} should be plausible")

        garbage = ["zzxkp", "qwrt", "mmmm", "hhhh", "jjjj", "cw", "xzq"]
        for g in garbage:
            self.assertFalse(engine.is_phonotactically_plausible_word(g), f"{g} should NOT be plausible")

    def test_sfx_and_watermark_still_rejected(self):
        """Test that actual SFX and watermarks are still filtered out."""
        sfx_cases = ["BOOM", "THWIP", "KRAK", "KOFF", "TEKK", "readallcomics.com", "hoegomig"]
        for item in sfx_cases:
            is_junk, reason = engine.check_is_sfx_or_junk(item, 0.95)
            self.assertTrue(is_junk, f"Expected '{item}' to be filtered, but got valid")

    def test_bubble_paper_protection(self):
        """Test that text on bubble paper is protected unless it's pure SFX or watermark."""
        # A word that might not be in dictionary or phonotactics but is on bubble paper
        is_junk, reason = engine.check_is_sfx_or_junk("Eep!", 0.30, is_on_bubble_paper=True)
        self.assertFalse(is_junk, f"Expected 'Eep!' to be valid on bubble paper, got: {reason}")

if __name__ == '__main__':
    unittest.main()
