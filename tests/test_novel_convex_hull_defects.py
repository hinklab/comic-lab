"""
test_novel_convex_hull_defects.py - Regression test for non-monotonous / self-intersecting
contours in cv2.convexityDefects during lobe splitting (found on page_14 splash art).
"""

import unittest
import numpy as np
import cv2
import bubble_lettering


class TestNovelConvexHullDefects(unittest.TestCase):
    def test_self_intersecting_contour_safe_fallback(self):
        """
        Verify that a self-intersecting figure-8 / bowtie contour does not crash
        bubble_lettering.split_conjoined_bubble_lobes with cv2.error (-5:Bad argument).
        """
        # Create a self-intersecting figure-8 contour (bowtie)
        pts = np.array([
            [100, 100], [200, 200], [200, 100], [100, 200],
            [150, 150], [120, 180], [180, 120], [100, 100],
            [105, 105], [195, 195], [195, 105], [105, 195]
        ], dtype=np.int32).reshape(-1, 1, 2)

        lines = [
            {"x0": 110, "y0": 110, "x1": 140, "y1": 130, "text": "LINE 1", "conf": 0.9},
            {"x0": 160, "y0": 160, "x1": 190, "y1": 180, "text": "LINE 2", "conf": 0.9}
        ]

        # Should safely return single or split lobes without raising cv2.error
        try:
            result = bubble_lettering.split_multi_lobe_bubble_contour(pts, lines)
            self.assertIsInstance(result, list)
            self.assertGreaterEqual(len(result), 1)
        except cv2.error as e:
            self.fail(f"split_multi_lobe_bubble_contour raised uncaught cv2.error: {e}")


if __name__ == "__main__":
    unittest.main()
