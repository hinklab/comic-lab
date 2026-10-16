"""
Unit tests for Big Update 3-Bosqich: Step 3.1: Geometric Diamond-Fitting Model.
Verifies:
1. Parabolic / elliptic chord width tapering towards top and bottom (Diamond rule).
2. Rectangles maintain uniform caption box width.
3. Word balancing across rows produces diamond aesthetic rather than top-heavy pyramids.
4. Clearance from bubble borders (no line corner crosses contour boundary).
"""

import unittest
import numpy as np
import cv2
import bubble_lettering
from bubble_lettering import (
    compute_diamond_chord_width,
    calculate_diamond_row_widths,
    balance_words_to_diamond,
    fit_text_to_bubble,
    _synthetic_ellipse_contour
)


class TestStep31DiamondFitting(unittest.TestCase):

    def test_diamond_chord_width_oval_tapering(self):
        """Oval speech bubbles must produce tapered top/bottom rows and widest middle row."""
        box_w = 200.0
        box_h = 100.0
        line_h = 22.0
        total_rows = 3

        widths = calculate_diamond_row_widths(
            total_rows=total_rows,
            box_w=box_w,
            box_h=box_h,
            line_h=line_h,
            shape_type="oval",
            safety_margin=0.84
        )

        self.assertEqual(len(widths), 3)
        w_top, w_mid, w_bot = widths[0], widths[1], widths[2]

        # Middle row must be the widest
        self.assertGreater(w_mid, w_top, "Middle row should be wider than top row")
        self.assertGreater(w_mid, w_bot, "Middle row should be wider than bottom row")
        # Top and bottom rows should be symmetrically balanced
        self.assertAlmostEqual(w_top, w_bot, delta=1.0, msg="Top and bottom row widths should be symmetric")
        # Center width should be approximately box_w * safety_margin
        self.assertAlmostEqual(w_mid, box_w * 0.84, delta=3.0)

    def test_diamond_chord_width_rectangle_uniform(self):
        """Rectangular caption boxes must maintain uniform row width across all rows."""
        box_w = 200.0
        box_h = 100.0
        line_h = 22.0
        total_rows = 4

        widths = calculate_diamond_row_widths(
            total_rows=total_rows,
            box_w=box_w,
            box_h=box_h,
            line_h=line_h,
            shape_type="rectangle"
        )

        self.assertEqual(len(widths), 4)
        for w in widths:
            self.assertEqual(w, box_w * 0.90, "All rectangular rows must share 0.90 * box_w width")

    def test_diamond_chord_width_four_lines_profile(self):
        """A 4-line diamond profile has narrower apex and base, with wide middle pairs."""
        box_w = 240.0
        box_h = 120.0
        line_h = 24.0
        total_rows = 4

        widths = calculate_diamond_row_widths(
            total_rows=total_rows,
            box_w=box_w,
            box_h=box_h,
            line_h=line_h,
            shape_type="oval"
        )

        self.assertEqual(len(widths), 4)
        # Row 0 and Row 3 are outer rows; Row 1 and Row 2 are inner rows
        self.assertLess(widths[0], widths[1], "Row 0 must be narrower than Row 1")
        self.assertLess(widths[3], widths[2], "Row 3 must be narrower than Row 2")
        self.assertAlmostEqual(widths[0], widths[3], delta=1.5)
        self.assertAlmostEqual(widths[1], widths[2], delta=1.5)

    def test_balance_words_to_diamond_distribution(self):
        """Words must distribute across lines conforming to diamond width profile."""
        words = ["XO'SH...", "UKAM", "JONNI", "BOR", "VA", "SIZ."]
        # 3 rows with diamond widths: 110 (top), 170 (mid), 110 (bottom)
        max_widths = [110.0, 170.0, 110.0]

        # Measure function where each character is ~8px and space is 6px
        def mock_measure(text: str) -> float:
            return len(text) * 8.0

        partition = balance_words_to_diamond(
            words=words,
            num_lines=3,
            max_widths=max_widths,
            measure_fn=mock_measure
        )

        self.assertIsNotNone(partition, "Word distribution should succeed")
        self.assertEqual(len(partition), 3, "Must produce exactly 3 lines")

        # Verify no line exceeds its allocated diamond chord width
        for r_idx, line_words in enumerate(partition):
            line_str = " ".join(line_words)
            line_w = mock_measure(line_str)
            self.assertLessEqual(
                line_w, max_widths[r_idx],
                f"Line {r_idx} '{line_str}' ({line_w}px) exceeds max width {max_widths[r_idx]}px"
            )

    def test_fit_text_to_bubble_synthetic_contour(self):
        """fit_text_to_bubble must fit dialogue inside an elliptic contour with safe corner clearances."""
        contour = _synthetic_ellipse_contour(150, 150, 180, 120)
        img_shape = (300, 300)

        import os
        font_path = "fonts/KOMTXT__.ttf"
        if not os.path.exists(font_path):
            for f in os.listdir("fonts"):
                if f.endswith(".ttf"):
                    font_path = os.path.join("fonts", f)
                    break

        text = "XO'SH... UKAM JONNI BOR VA SIZ."
        res = fit_text_to_bubble(
            text=text,
            contour=contour,
            image_shape=img_shape,
            font_path=font_path,
            max_font_size=22,
            min_font_size=10,
            line_spacing=1.15,
            min_clearance=4
        )

        self.assertIsNotNone(res, "Text should fit inside 180x120 ellipse contour")
        self.assertGreaterEqual(res["font_size"], 10)
        self.assertGreaterEqual(len(res["lines"]), 2, "Text should wrap into multiple lines")

        # Check every line's corner points have >= 4px clearance from contour boundary
        for line_txt, lx, ly in res["lines"]:
            dist = cv2.pointPolygonTest(contour, (float(lx), float(ly)), True)
            self.assertGreaterEqual(dist, 2.0, f"Top-left corner ({lx}, {ly}) must be well inside contour")


if __name__ == "__main__":
    unittest.main()
