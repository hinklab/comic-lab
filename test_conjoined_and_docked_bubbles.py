import unittest
import numpy as np
import cv2
from PIL import Image
import pickle

import bubble_lettering
import quality_gates
import engine


class TestConjoinedAndDockedBubbles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        cls.img_cv = cv2.imread('samples/page_4.png')
        ocr_path = 'tests/fixtures/page4_ocr.pkl' if os.path.exists('tests/fixtures/page4_ocr.pkl') else 'scratch/page4_ocr.pkl'
        with open(ocr_path, 'rb') as f:
            cls.cached_ocr = pickle.load(f)

    def test_bubble5_jameson_conjoined_waist_separation(self):
        """Verify Jameson's figure-8 conjoined bubble splits horizontally at waist without diagonal cut."""
        jameson_lines = []
        for item in self.cached_ocr:
            bbox, text, conf = item
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            if x0 > 1800 and y1 < 600:
                jameson_lines.append({
                    'text': text,
                    'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                    'cx': (x0 + x1) // 2, 'cy': (y0 + y1) // 2,
                    'conf': conf
                })

        bx0 = min(l['x0'] for l in jameson_lines)
        by0 = min(l['y0'] for l in jameson_lines)
        bx1 = max(l['x1'] for l in jameson_lines)
        by1 = max(l['y1'] for l in jameson_lines)
        glyph_h = int(np.median([l['y1'] - l['y0'] for l in jameson_lines]))

        cnt_page, is_rect = bubble_lettering.get_bubble_contour(
            self.img_cv, (bx0, by0, bx1, by1), approx_glyph_height=glyph_h
        )

        lobe_splits = bubble_lettering.split_multi_lobe_bubble_contour(
            cnt_page, jameson_lines, approx_glyph_height=glyph_h
        )

        self.assertEqual(len(lobe_splits), 2, "Figure-8 bubble must split into exactly 2 lobes")

        lobe0_cnt, lobe0_lines = lobe_splits[0]
        lobe1_cnt, lobe1_lines = lobe_splits[1]

        lobe0_text = " ".join(l['text'] for l in lobe0_lines)
        lobe1_text = " ".join(l['text'] for l in lobe1_lines)

        # Lobe 0 must contain 5 lines and end with 'Youl'
        self.assertEqual(len(lobe0_lines), 5)
        self.assertTrue('Youl' in lobe0_text or 'You' in lobe0_text)

        # Lobe 1 must contain 6 lines and start with 'So On'
        self.assertEqual(len(lobe1_lines), 6)
        self.assertTrue(lobe1_text.startswith('So On'))

        # Lobe 0 must be strictly above Lobe 1 (horizontal waist around y=278)
        b0 = cv2.boundingRect(lobe0_cnt)
        b1 = cv2.boundingRect(lobe1_cnt)
        self.assertLess(b0[1] + b0[3], b1[1] + 25)

    def test_bubble8_docked_gutter_bubble_contour(self):
        """Verify bubble #8 docked against panel gutter extracts organic contour without art leak fallback."""
        b8_box = (1385, 1355, 1683, 1498)
        cnt, is_rect = bubble_lettering.get_bubble_contour(
            self.img_cv, b8_box, approx_glyph_height=24
        )
        self.assertFalse(is_rect)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Must not be clipped rectangle box (solidity >= 0.90)
        self.assertGreaterEqual(solidity, 0.90)
        # Page bounding box must closely match police officer bubble (~1360..1710, 1360..1510)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        self.assertGreater(bx, 1340)
        self.assertLess(bx + bw, 1720)
        self.assertGreater(by, 1340)
        self.assertLess(by + bh, 1530)

    def test_synthetic_ellipse_fallback_smoothness(self):
        """Verify quality gate art leak fallback produces smooth organic ellipse without flat clipped edges."""
        synth, is_rect = quality_gates._resolve_background_art_leak(
            {
                "cx": 500, "cy": 500, "bw": 200, "bh": 100,
                "x0": 400, "y0": 450, "x1": 600, "y1": 550
            },
            {"image_w": 2000, "image_h": 2000}
        )
        self.assertFalse(is_rect)
        self.assertGreaterEqual(len(synth), 20)
        # Check that contour is smooth: polygon approximation with 2% arcLength has > 8 vertices
        peri = cv2.arcLength(synth, True)
        approx = cv2.approxPolyDP(synth, 0.02 * peri, True)
        self.assertGreaterEqual(len(approx), 6, "Synthetic ellipse must have curved oval points, not flat 4-sided box")


if __name__ == '__main__':
    unittest.main()
