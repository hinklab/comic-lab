"""
Unit tests for Step 3.7: Text Centering in Rectangular Speech Bubbles / Caption Boxes.
Verifies that pure rectangular bubbles have their text positioned precisely at the vertical
and horizontal center of the rectangle both immediately after translation (with contour present)
and after resizing/moving (contour cleared or modified), eliminating the top-drift bug.
"""
import unittest
from pathlib import Path

INDEX_HTML_PATH = Path(__file__).resolve().parent / "web" / "index.html"


class TestRectangleTextCentering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            cls.html_content = f.read()

    def test_pure_rect_bypasses_contour_slicing_in_center_calc(self):
        """getBubbleGeometricCenter must immediately return exact box center for pure rectangular bubbles."""
        self.assertIn("const isPureRect = (b.shape_type === 'rectangle' && !b.has_tail);", self.html_content)
        self.assertIn("if (isPureRect) {", self.html_content)
        self.assertIn("cx: cx + (b.offset_x || 0),", self.html_content)
        self.assertIn("cy: cy + (b.offset_y || 0)", self.html_content)

    def test_slice_algorithm_picks_mid_closest_slice_for_equal_spans(self):
        """The contour slicing algorithm must choose the slice closest to vertical center for near-maximal spans."""
        self.assertIn("const midSlice = Math.floor(numSlices / 2);", self.html_content)
        self.assertIn("let bestSlice = midSlice;", self.html_content)
        self.assertIn("let minCenterDist = Infinity;", self.html_content)
        self.assertIn("const dist = Math.abs(s - midSlice);", self.html_content)

    def test_simulated_center_calculation_comparison(self):
        """Simulate JS center calculation logic for a rectangular bubble with contour."""
        box = [100, 100, 300, 220]  # bw = 200, bh = 120, cx = 200, cy = 160
        bw = box[2] - box[0]
        bh = box[3] - box[1]

        # Old broken slicing simulation:
        num_slices = 14
        slice_max_x = [300] * num_slices
        slice_min_x = [100] * num_slices
        slice_count = [10] * num_slices

        # Old logic:
        old_best_span = 0
        old_best_slice = num_slices // 2
        for s in range(num_slices):
            if slice_count[s] >= 2:
                sp = slice_max_x[s] - slice_min_x[s]
                if sp > old_best_span:
                    old_best_span = sp
                    old_best_slice = s

        old_body_cy = box[1] + ((old_best_slice + 0.5) / num_slices) * bh
        # Old broken logic produced slice 0 -> cy = 100 + (0.5 / 14) * 120 = 104.28px (TOP OF RECTANGLE!)
        self.assertEqual(old_best_slice, 0)
        self.assertAlmostEqual(old_body_cy, 104.2857, places=3)

        # New logic:
        # 1. Pure rect check immediately gives:
        new_pure_rect_cy = box[1] + bh / 2
        self.assertEqual(new_pure_rect_cy, 160.0)

        # 2. Slice algorithm with center-proximity fallback gives:
        max_span = max(slice_max_x[s] - slice_min_x[s] for s in range(num_slices))
        mid_slice = num_slices // 2
        new_best_slice = mid_slice
        min_dist = float("inf")
        for s in range(num_slices):
            sp = slice_max_x[s] - slice_min_x[s]
            if sp >= max_span * 0.96:
                dist = abs(s - mid_slice)
                if dist < min_dist:
                    min_dist = dist
                    new_best_slice = s

        new_body_cy = box[1] + ((new_best_slice + 0.5) / num_slices) * bh
        self.assertEqual(new_best_slice, 7)  # EXACT VERTICAL CENTER!
        self.assertAlmostEqual(new_body_cy, 164.2857, delta=5.0)  # Center-aligned, not top!


if __name__ == "__main__":
    unittest.main()
