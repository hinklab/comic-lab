import unittest
from PIL import Image
import cv2
import numpy as np

import engine
import bubble_lettering


class TestSpectralConjoinedBubbles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img_pil = Image.open('samples/page_4.png')
        cls.img_cv = cv2.imread('samples/page_4.png')
        cls.bubbles = engine.scan_bubbles_ocr(cls.img_pil)
        # Running clean_page_ink_telea samples the ink & sets text_is_spectral on all bubbles
        cls.cleaned_pil = engine.clean_page_ink_telea(cls.img_pil, cls.bubbles)

    def test_page4_total_bubbles_count_14(self):
        """Page 4 must have 14 bubbles with both ghost conjoined bubbles split cleanly."""
        self.assertEqual(len(self.bubbles), 14, f"Expected 14 bubbles on Page 4, got {len(self.bubbles)}")

    def test_spectral_bubble9_figure8_split(self):
        """Verify Peter Parker's ghost bubble 9 ('Otto Octavius' / 'Doctor Octopus') is split into 2 lobes."""
        otto_b = next((b for b in self.bubbles if "OTTO" in b.original_text.upper()), None)
        dock_b = next((b for b in self.bubbles if "OCTOPUS IN" in b.original_text.upper()), None)
        self.assertIsNotNone(otto_b, "'That's Otto Octavius' lobe must exist as separate bubble")
        self.assertIsNotNone(dock_b, "'Doctor Octopus in My Body' lobe must exist as separate bubble")

        # They must have distinct bounding boxes
        self.assertLess(otto_b.box[3], dock_b.box[1] + 15, "Otto lobe must be above Doctor Octopus lobe")

        # Must retain spectral dialogue identity
        self.assertTrue(otto_b.text_is_spectral, "Otto ghost lobe must have text_is_spectral=True")
        self.assertTrue(dock_b.text_is_spectral, "Doctor Octopus ghost lobe must have text_is_spectral=True")

    def test_spectral_bubble12_diagonal_split(self):
        """Verify Peter Parker's diagonal bubble 12 ('Live Body' / 'Crazy-Town') is split into 2 lobes."""
        live_b = next((b for b in self.bubbles if "LIVE BODY" in b.original_text.upper()), None)
        crazy_b = next((b for b in self.bubbles if "CRAZY-" in b.original_text.upper()), None)
        self.assertIsNotNone(live_b, "'Live Body' lobe must exist as separate bubble")
        self.assertIsNotNone(crazy_b, "'Crazy-Town Banana Pants' lobe must exist as separate bubble")

        # Verify diagonal separation: Live Body is top-left, Crazy-Town is bottom-right
        self.assertLess(live_b.box[0], crazy_b.box[0], "Live Body lobe must be to the left of Crazy-Town lobe")
        self.assertLess(live_b.box[1], crazy_b.box[1], "Live Body lobe must be above Crazy-Town lobe")

        # Must retain spectral dialogue identity
        self.assertTrue(live_b.text_is_spectral, "Live Body ghost lobe must have text_is_spectral=True")
        self.assertTrue(crazy_b.text_is_spectral, "Crazy-Town ghost lobe must have text_is_spectral=True")


if __name__ == '__main__':
    unittest.main()
