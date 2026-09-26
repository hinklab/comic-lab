import unittest
import numpy as np
import cv2
from PIL import Image
import pickle

import engine
import bubble_lettering
import quality_gates


class TestPage5BubbleFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        cls.img_cv = cv2.imread('samples/page_5.png')
        cls.img_pil = Image.open('samples/page_5.png')
        ocr_path = 'tests/fixtures/page5_ocr.pkl' if os.path.exists('tests/fixtures/page5_ocr.pkl') else 'scratch/page5_ocr.pkl'
        with open(ocr_path, 'rb') as f:
            cls.cached_ocr = pickle.load(f)

        cls._orig_get_ocr_reader = engine.get_ocr_reader
        class MockReader:
            def readtext(self, *args, **kwargs):
                return cls.cached_ocr

        engine.get_ocr_reader = lambda: MockReader()
        try:
            cls.bubbles = engine.scan_bubbles_ocr(cls.img_pil)
        finally:
            engine.get_ocr_reader = cls._orig_get_ocr_reader

    @classmethod
    def tearDownClass(cls):
        engine.get_ocr_reader = cls._orig_get_ocr_reader

    def test_rrphbl_vocalization_not_dropped(self):
        """Verify 'RrphBl?' is preserved on bubble paper and not discarded as junk."""
        is_junk, reason = engine.check_is_sfx_or_junk("RrphBl?", conf=0.52, is_on_bubble_paper=True)
        self.assertFalse(is_junk, f"RrphBl? should not be dropped on bubble paper, got reason: {reason}")

        # Check it exists in detected bubbles
        rrphbl_bubble = next((b for b in self.bubbles if "RRPHBL" in b.original_text.upper()), None)
        self.assertIsNotNone(rrphbl_bubble, "'RRPHBL?' speech bubble must be detected on Page 5")
        self.assertEqual(rrphbl_bubble.shape_type, "oval")

    def test_zero_false_earpiece_rectangles_on_page5(self):
        """Verify no false-positive earpiece rectangles are detected on Page 5 background art."""
        rects = engine.detect_earpiece_rectangles(self.img_cv)
        self.assertEqual(len(rects), 0, f"Expected 0 earpiece rects on Page 5, got {len(rects)}: {rects}")

    def test_tv_news_anchor_speech_bubble_split(self):
        """Verify TV news anchor multi-lobe speech bubble is cleanly split into 2 separate lobes."""
        lobe1 = next((b for b in self.bubbles if "STILL HURTS" in b.original_text.upper()), None)
        lobe2 = next((b for b in self.bubbles if "SOMETHING HE SAID" in b.original_text.upper()), None)
        self.assertIsNotNone(lobe1, "Upper anchor lobe must be detected as a distinct speech bubble")
        self.assertIsNotNone(lobe2, "Lower anchor lobe must be detected as a distinct speech bubble")
        self.assertEqual(lobe1.shape_type, "oval")
        self.assertEqual(lobe2.shape_type, "oval")
        self.assertNotEqual(lobe1.bubble_id, lobe2.bubble_id, "Lobes must have separate bubble IDs")

    def test_bottom_left_panel_sequential_clean_dialogue(self):
        """Verify bottom-left panel has no overlapping duplicate boxes and clean bubble ordering without horns."""
        # Find 'Wish' bubble and 'Crazy-town' bubble
        wish_b = next((b for b in self.bubbles if "WISH" in b.original_text.upper()), None)
        crazy_b = next((b for b in self.bubbles if "BANANA" in b.original_text.upper() and b.box[1] > 2400), None)
        self.assertIsNotNone(wish_b, "'Wish could' bubble must be detected")
        self.assertIsNotNone(crazy_b, "'Crazy-town banana-pants' bubble must be detected")

        # They must not overlap completely (IoU should be small)
        w_box = wish_b.box
        c_box = crazy_b.box
        ix0, iy0 = max(w_box[0], c_box[0]), max(w_box[1], c_box[1])
        ix1, iy1 = min(w_box[2], c_box[2]), min(w_box[3], c_box[3])
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        w_area = (w_box[2] - w_box[0]) * (w_box[3] - w_box[1])
        self.assertLess(inter / float(w_area), 0.35, "Wish and Crazy-town bubbles must be distinct, not duplicate")

        # Verify no sharp horn spike from crazy_b protruding into wish_b (y < 2625 at x < 470)
        pts_crazy = np.array(crazy_b.contour_points)
        spikes = [p for p in pts_crazy if p[0] < 470 and p[1] < 2625]
        self.assertEqual(len(spikes), 0, f"Expected 0 spike points on Crazy-town bubble, got: {spikes}")

    def test_tv_headline_chyron_is_excluded(self):
        """Verify TV news chyron banner 'MENACE NO MORE?' is excluded from speech bubbles."""
        menace_b = next((b for b in self.bubbles if "MENACE" in b.original_text.upper()), None)
        self.assertIsNone(menace_b, "'MENACE NO MORE?' chyron must be excluded from speech bubbles")

    def test_top_connector_tubes_inpainted(self):
        """Verify white connector tubes between linked top bubbles are included in the inpaint mask and cleaned."""
        cleaned = engine.clean_page_ink_telea(self.img_pil, self.bubbles)
        cleaned_cv = cv2.cvtColor(np.array(cleaned), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cleaned_cv, cv2.COLOR_BGR2GRAY)
        # Check that the tube between bubble 1 and 2 is white and cleaned (mean lum > 200)
        # Tube region between [92, 86, 372, 193] and [280, 217, 405, 256]: x in [280, 340], y in [185, 210]
        tube_sample = gray[185:210, 280:340]
        self.assertGreater(float(np.mean(tube_sample)), 200.0, "Connector tube must be cleaned/white")

    def test_bubble5_ah_smooth_top(self):
        """Verify tiny bubble 'AH.' does not have its top sliced flat by gutter detection."""
        ah_b = next((b for b in self.bubbles if b.original_text.strip().upper() == "AH."), None)
        self.assertIsNotNone(ah_b, "'AH.' bubble must be detected")
        pts = np.array(ah_b.contour_points)
        y_min = pts[:, 1].min()
        # Text starts at y=244. Automated top contour is at y <= 240
        self.assertLessEqual(y_min, 240, f"'AH.' top contour at y={y_min} is too low; border was cut")

        # Verify Stage 2 Smart Brush organically expands clipped top to encompass full border
        import bubble_mask_editor
        H, W = self.img_cv.shape[:2]
        brush_stroke = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(brush_stroke, (1080, 225), (1135, 245), 255, -1)
        grown_b = bubble_mask_editor.apply_smart_brush_growth(ah_b, self.img_cv, brush_stroke, tolerance=35.0, min_lum=110.0)
        grown_y_min = min(p[1] for p in grown_b['contour_points'])
        self.assertLessEqual(grown_y_min, 230, f"Smart Brush must expand top to encompass full border (y={grown_y_min})")


if __name__ == '__main__':
    unittest.main()
