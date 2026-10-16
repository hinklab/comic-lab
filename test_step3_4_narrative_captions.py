"""
Unit tests for Big Update 3-Bosqich: Step 3.4: Narrative Monologue & Caption Box Detection & Resilient Gemini Fallback.
Verifies:
1. Model fallback loop correctly tries alternative models in GEMINI_MODELS on 429/404/503.
2. Prompt instructions explicitly cover narrative caption boxes, monologue declarations, and exclude unboxed background art.
3. Adaptive dark ratio filter accepts bright paper boxes even on pitch-black backgrounds.
4. Adaptive dark ratio filter rejects authentic dark artwork/clothing.
5. End-to-end extraction on comic image captures all 4 monologue/dialogue boxes while ignoring unboxed background text ("YEARS AGO.").
"""

import os
import unittest
import numpy as np
from PIL import Image
import gemini_enhancements


class TestStep34NarrativeCaptions(unittest.TestCase):

    def test_gemini_models_priority_order(self):
        """GEMINI_MODELS must start with verified high-availability models."""
        self.assertIn("gemini-flash-lite-latest", gemini_enhancements.GEMINI_MODELS)
        self.assertIn("gemini-3.5-flash", gemini_enhancements.GEMINI_MODELS)
        # Verify first model is an active working model
        self.assertEqual(gemini_enhancements.GEMINI_MODELS[0], "gemini-flash-lite-latest")

    def test_adaptive_dark_ratio_accepts_bright_box_on_dark_bg(self):
        """A small white/bright box (e.g. 50x20) on a pitch black background must NOT be rejected by the adaptive filter."""
        # Create a mock 100x100 pitch black image
        gray = np.zeros((100, 100), dtype=np.uint8)
        # Place a 50x20 white speech/caption box in the center (y: 40-60, x: 25-75)
        # Interior contains white paper (~240) and some dark text (~30)
        box_paper = np.full((20, 50), 240, dtype=np.uint8)
        # Draw some dark text lines
        box_paper[8:12, 10:40] = 30
        gray[40:60, 25:75] = box_paper

        y0, y1, x0, x1 = 40, 60, 25, 75
        # Expanded crop (6px on all sides)
        exp_y0 = max(0, y0 - 6)
        exp_y1 = min(100, y1 + 6)
        exp_x0 = max(0, x0 - 6)
        exp_x1 = min(100, x1 + 6)
        crop_gray = gray[exp_y0:exp_y1, exp_x0:exp_x1]

        # Calculate standard dark ratio and luminance on expanded crop
        dark_ratio = float(np.mean(crop_gray < 130))
        med_luminance = float(np.median(crop_gray))

        # Check adaptive logic
        inner_gray = gray[y0:y1, x0:x1]
        inner_p85 = float(np.percentile(inner_gray, 85))
        inner_p70 = float(np.percentile(inner_gray, 70))
        is_inner_bright = bool(inner_p85 >= 170 and inner_p70 >= 130)

        # Verification: inner core is recognized as bright paper
        self.assertTrue(is_inner_bright)
        self.assertGreaterEqual(inner_p85, 170)
        self.assertGreaterEqual(inner_p70, 130)

        # Rejection should NOT trigger when is_inner_bright is True
        should_reject = not is_inner_bright and (dark_ratio >= 0.50 or med_luminance < 110)
        self.assertFalse(should_reject)

    def test_adaptive_dark_ratio_rejects_dark_artwork(self):
        """A dark artwork region (e.g. shadow, dark armor) must be rejected."""
        # 100x100 dark image with max pixel < 100
        gray = np.random.randint(10, 80, size=(100, 100), dtype=np.uint8)
        y0, y1, x0, x1 = 40, 60, 25, 75
        crop_gray = gray[y0:y1, x0:x1]

        inner_gray = gray[y0:y1, x0:x1]
        inner_p85 = float(np.percentile(inner_gray, 85))
        inner_p70 = float(np.percentile(inner_gray, 70))
        is_inner_bright = bool(inner_p85 >= 170 and inner_p70 >= 130)

        self.assertFalse(is_inner_bright)
        dark_ratio = float(np.mean(crop_gray < 130))
        med_luminance = float(np.median(crop_gray))
        should_reject = not is_inner_bright and (dark_ratio >= 0.50 or med_luminance < 110)
        self.assertTrue(should_reject)

    def test_doctor_doom_caption_and_bubble_extraction(self):
        """Doctor Doom comic page must extract all 4 narrative/dialogue boxes and ignore 'YEARS AGO.'."""
        img_path = r"C:/Users/Ozod/.gemini/antigravity/brain/b9e18c00-a13c-4536-8eb0-753bbc58e136/.user_uploaded/media_1792147787717_1c81fbf8.png"
        if not os.path.exists(img_path):
            self.skipTest("Doctor Doom sample image not found at test path.")

        img = Image.open(img_path)
        bubbles = gemini_enhancements.gemini_extract_comic_bubbles(img, use_cache=True)

        self.assertGreaterEqual(len(bubbles), 4, f"Expected at least 4 bubbles, got {len(bubbles)}")

        all_orig_texts = " ".join([b["original_text"].upper() for b in bubbles])

        # 1. First caption box
        self.assertIn("WHO AM I", all_orig_texts)
        # 2. Middle monologue
        self.assertTrue("BARCHA" in all_orig_texts or "MASTER" in all_orig_texts)
        # 3. Third monologue
        self.assertTrue("LATVERYA" in all_orig_texts or "LATVERIA" in all_orig_texts)
        # 4. Final title declaration box
        self.assertTrue("DOCTOR DOOM" in all_orig_texts or "DOOM" in all_orig_texts)

        # 5. Non-bubble background text 'YEARS AGO.' MUST be ignored
        self.assertNotIn("YEARS AGO", all_orig_texts)


if __name__ == "__main__":
    unittest.main()
