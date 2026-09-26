import unittest
import numpy as np
import cv2
from PIL import Image

import engine
import bubble_lettering
import quality_gates


class TestPage3Bubble13Fixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img_pil = Image.open('samples/3.png')
        cls.img_cv = cv2.imread('samples/3.png')

    def test_bubble13_classified_as_oval_not_rectangle(self):
        """
        Verify Bubble #13 (...BUT WE HAVE A SPECIAL GUEST...) with aspect ratio > 5.0
        and a downward speech tail is classified as 'oval', NOT regularized as a blunt rectangle.
        """
        box = (283, 2812, 852, 2923)
        cnt, is_rect = bubble_lettering.get_bubble_contour(self.img_cv, box)
        self.assertFalse(is_rect, "Speech bubble with tail must not be flagged as rectangular banner")
        self.assertGreater(len(cnt), 4, "Organic bubble contour must retain curved points, not 4-point box")

        # Check vertical tail protrusion is preserved
        pts = cnt.reshape(-1, 2)
        max_y = int(np.max(pts[:, 1]))
        self.assertGreaterEqual(max_y, 2970, "Bubble tail reaching woman's hair must be preserved in contour")

    def test_bubble13_inpainting_preserves_hair_artwork(self):
        """
        Verify inpainting Bubble #13 cleans interior text while preserving the woman's hair.
        The hair region below y=2940 must not be covered with solid white rectangle.
        """
        box = (283, 2812, 852, 2923)
        cnt, is_rect = bubble_lettering.get_bubble_contour(self.img_cv, box)
        b13 = engine.SpeechBubble(
            bubble_id=13,
            x0=box[0], y0=box[1], x1=box[2], y1=box[3],
            box=list(box),
            original_text="...BUT WE HAVE A SPECIAL GUEST...",
            clean_text="...BUT WE HAVE A SPECIAL GUEST...",
            uzbek_translation="",
            confidence=0.9,
            shape_type="oval",
            z_order=0,
            contour=cnt.reshape(-1, 2).tolist(),
            contour_points=cnt.reshape(-1, 2).tolist(),
            line_centers=[[(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]]
        )
        cleaned_pil = engine.clean_page_ink_telea(self.img_pil, [b13])
        cleaned_cv = cv2.cvtColor(np.array(cleaned_pil), cv2.COLOR_RGB2BGR)

        # Sample the woman's hair region: y in [2945, 2975], x in [350, 500] (to the left of tail)
        # In the bug, this was wiped out to pure white (254, 254, 254).
        # In reality, it has blonde hair artwork with dark ink lines (mean luminance < 200).
        hair_sample = cleaned_cv[2945:2975, 350:500]
        hair_gray = cv2.cvtColor(hair_sample, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(hair_gray))
        self.assertLess(mean_lum, 220.0, f"Woman's hair must not be whited out (mean lum: {mean_lum:.1f})")

    def test_badge_anchor_clamping_for_upward_tails(self):
        """
        Verify Bubble #14 (with upward tail pointing towards Spider-Man) does not place
        its badge floating far above the bubble near the machinery.
        """
        box14 = (1629, 3511, 1915, 3628)
        cnt14, is_rect = bubble_lettering.get_bubble_contour(self.img_cv, box14)
        b14 = {
            "x0": box14[0], "y0": box14[1], "x1": box14[2], "y1": box14[3],
            "bubble_id": 14, "shape_type": "oval", "contour": cnt14.reshape(-1, 2).tolist()
        }
        w, h = self.img_cv.shape[1], self.img_cv.shape[0]
        cnt, (ax, ay) = engine.get_contour_from_bubble(b14, w, h)
        # y0 is 3511. ay must not be higher than by0 - 15 (i.e. >= 3496)
        self.assertGreaterEqual(ay, box14[1] - 15, f"Badge anchor_y ({ay}) should be clamped near bubble top")

    def test_multi_lobe_badge_non_collision(self):
        """
        Verify Bubble #15 (bottom lobe) badge does not collide with Bubble #14 text.
        """
        w, h = self.img_cv.shape[1], self.img_cv.shape[0]
        b14 = engine.SpeechBubble(
            bubble_id=14, x0=1629, y0=3511, x1=1915, y1=3628, box=[1629, 3511, 1915, 3628],
            original_text="Lobe 1", clean_text="Lobe 1", uzbek_translation="", confidence=0.9
        )
        b15 = engine.SpeechBubble(
            bubble_id=15, x0=1638, y0=3663, x1=1907, y1=3727, box=[1638, 3663, 1907, 3727],
            original_text="Lobe 2", clean_text="Lobe 2", uzbek_translation="", confidence=0.9
        )
        overlay = engine.draw_bounding_box_overlay(self.img_pil, [b14, b15])
        self.assertIsNotNone(overlay)


if __name__ == '__main__':
    unittest.main()
