import unittest
import numpy as np
import cv2

import bubble_lettering
import engine


class TestStep1ConjoinedLobeSplit(unittest.TestCase):
    def test_pinched_waist_narrow_neck_split(self):
        """Verify conjoined bubble with narrow neck (< 12px, e.g. 9.5px) is cleanly split into 2 lobes."""
        # Create a synthetic peanut/horizontal figure-8 conjoined contour
        # Left circle: center (50, 40), r=35 -> x: 15 to 85, y: 5 to 75
        # Right circle: center (130, 40), r=35 -> x: 95 to 165, y: 5 to 75
        # Connected by waist at x=90, y: 35 to 45 (height = 10px, dist = 10px)
        mask = np.zeros((80, 180), dtype=np.uint8)
        cv2.circle(mask, (50, 40), 32, 255, -1)
        cv2.circle(mask, (130, 40), 32, 255, -1)
        # Bridge connecting them
        cv2.rectangle(mask, (45, 36), (135, 44), 255, -1)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.assertTrue(len(cnts) >= 1)
        cnt = max(cnts, key=cv2.contourArea)

        lines = [
            {"x0": 25, "y0": 25, "x1": 75, "y1": 55, "cx": 50, "cy": 40, "text": "First sentence.", "conf": 0.95},
            {"x0": 105, "y0": 25, "x1": 155, "y1": 55, "cx": 130, "cy": 40, "text": "Second sentence.", "conf": 0.95}
        ]

        splits = bubble_lettering.split_multi_lobe_bubble_contour(cnt, lines, approx_glyph_height=20)
        self.assertEqual(len(splits), 2, f"Expected conjoined bubble to split into 2 lobes, got {len(splits)}")

        lobe1_cnt, lobe1_lines = splits[0]
        lobe2_cnt, lobe2_lines = splits[1]

        self.assertEqual(len(lobe1_lines), 1)
        self.assertEqual(len(lobe2_lines), 1)
        self.assertEqual(lobe1_lines[0]['text'], "First sentence.")
        self.assertEqual(lobe2_lines[0]['text'], "Second sentence.")

    def test_horizontal_constriction_fallback_with_clustered_lines(self):
        """Verify secondary constriction fallback splits lines when lines sit near waist boundary."""
        mask = np.zeros((80, 200), dtype=np.uint8)
        cv2.circle(mask, (55, 40), 34, 255, -1)
        cv2.circle(mask, (145, 40), 34, 255, -1)
        cv2.rectangle(mask, (50, 36), (150, 44), 255, -1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)

        split_x, details = bubble_lettering.find_contour_horizontal_constriction(cnt)
        self.assertIsNotNone(split_x)
        self.assertLessEqual(details.get("constriction_ratio", 1.0), 0.65)

    def test_partition_translated_text_for_lobes_preserves_sentences(self):
        """Verify 2-lobe partition keeps Sentence 1 in lobe 1 and Sentence 2 in lobe 2."""
        uz_text = "Keling, 48 soatlik rejalashtiramiz. Keyin uni uyg'otamiz."
        parts = engine.partition_translated_text_for_lobes(uz_text, [1.0, 1.0])
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], "Keling, 48 soatlik rejalashtiramiz.")
        self.assertEqual(parts[1], "Keyin uni uyg'otamiz.")


if __name__ == '__main__':
    unittest.main()
