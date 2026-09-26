import unittest
import os
import sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
import bubble_lettering
import quality_gates

class TestUniversalConjoinedBubbles(unittest.TestCase):

    def test_synthetic_vertical_peanut_split(self):
        """Tests that a vertical figure-8/peanut contour is cleanly split into top and bottom lobes."""
        # Create a synthetic figure-8 peanut mask
        h, w = 400, 300
        mask = np.zeros((h, w), dtype=np.uint8)
        # Top lobe: ellipse at (150, 120), axes (100, 70)
        cv2.ellipse(mask, (150, 120), (100, 70), 0, 0, 360, 255, -1)
        # Bottom lobe: ellipse at (150, 270), axes (100, 70)
        cv2.ellipse(mask, (150, 270), (100, 70), 0, 0, 360, 255, -1)
        # Neck bridge at y=195, width ~ 90
        cv2.rectangle(mask, (105, 170), (195, 220), 255, -1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.assertEqual(len(cnts), 1)
        peanut_cnt = cnts[0]

        # Fake OCR text lines: 2 lines in top lobe, 2 lines in bottom lobe
        lines = [
            {'x0': 100, 'y0': 90, 'x1': 200, 'y1': 110, 'cx': 150, 'cy': 100, 'text': 'Top line 1'},
            {'x0': 90, 'y0': 120, 'x1': 210, 'y1': 140, 'cx': 150, 'cy': 130, 'text': 'Top line 2'},
            {'x0': 100, 'y0': 240, 'x1': 200, 'y1': 260, 'cx': 150, 'cy': 250, 'text': 'Bot line 1'},
            {'x0': 90, 'y0': 270, 'x1': 210, 'y1': 290, 'cx': 150, 'cy': 280, 'text': 'Bot line 2'}
        ]

        splits = bubble_lettering.split_multi_lobe_bubble_contour(peanut_cnt, lines, approx_glyph_height=20)
        self.assertEqual(len(splits), 2, "Peanut contour should split into exactly 2 lobes")

        top_cnt, top_lines = splits[0]
        bot_cnt, bot_lines = splits[1]

        self.assertEqual(len(top_lines), 2)
        self.assertEqual(len(bot_lines), 2)
        self.assertTrue(all('Top' in l['text'] for l in top_lines))
        self.assertTrue(all('Bot' in l['text'] for l in bot_lines))

    def test_synthetic_diagonal_conjoined_split(self):
        """Tests that diagonally overlapping bubbles are cleanly split into two distinct lobes."""
        h, w = 400, 500
        mask = np.zeros((h, w), dtype=np.uint8)
        # Upper-left lobe
        cv2.ellipse(mask, (150, 150), (90, 60), 0, 0, 360, 255, -1)
        # Lower-right lobe
        cv2.ellipse(mask, (320, 260), (100, 70), 0, 0, 360, 255, -1)
        # Connecting neck
        cv2.line(mask, (180, 170), (280, 240), 255, 35)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.assertEqual(len(cnts), 1)
        diag_cnt = cnts[0]

        lines = [
            {'x0': 110, 'y0': 130, 'x1': 190, 'y1': 150, 'cx': 150, 'cy': 140, 'text': 'Lobe A line 1'},
            {'x0': 100, 'y0': 155, 'x1': 200, 'y1': 175, 'cx': 150, 'cy': 165, 'text': 'Lobe A line 2'},
            {'x0': 270, 'y0': 240, 'x1': 370, 'y1': 260, 'cx': 320, 'cy': 250, 'text': 'Lobe B line 1'},
            {'x0': 260, 'y0': 265, 'x1': 380, 'y1': 285, 'cx': 320, 'cy': 275, 'text': 'Lobe B line 2'}
        ]

        splits = bubble_lettering.split_multi_lobe_bubble_contour(diag_cnt, lines, approx_glyph_height=20)
        self.assertEqual(len(splits), 2, "Diagonal conjoined bubble should split into exactly 2 lobes")

    def test_single_bubble_with_tail_not_split(self):
        """Tests that a single organic bubble with a dialogue tail is NOT falsely split."""
        h, w = 300, 300
        mask = np.zeros((h, w), dtype=np.uint8)
        # Main bubble body
        cv2.ellipse(mask, (150, 130), (90, 60), 0, 0, 360, 255, -1)
        # Tail pointing towards bottom-left
        pts = np.array([[120, 180], [160, 180], [70, 260]], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.assertEqual(len(cnts), 1)
        tailed_cnt = cnts[0]

        # All lines are inside the main bubble body
        lines = [
            {'x0': 110, 'y0': 110, 'x1': 190, 'y1': 130, 'cx': 150, 'cy': 120, 'text': 'Hello world'},
            {'x0': 100, 'y0': 135, 'x1': 200, 'y1': 155, 'cx': 150, 'cy': 145, 'text': 'Spider-Man!'}
        ]

        splits = bubble_lettering.split_multi_lobe_bubble_contour(tailed_cnt, lines, approx_glyph_height=20)
        self.assertEqual(len(splits), 1, "Single bubble with tail must NOT be split")

    def test_quality_gate_conjoined_bubble(self):
        """Tests quality gate evaluation for conjoined bubbles."""
        state = {
            "is_multi_lobe": True,
            "has_split_lines": True,
            "split_clusters": [[{'text': 'L1'}], [{'text': 'L2'}]]
        }
        res = quality_gates.apply_quality_gates("bubble_segmentation", state)
        self.assertEqual(len(res.get("split_clusters", [])), 2)

if __name__ == '__main__':
    unittest.main()
