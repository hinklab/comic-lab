"""
test_step1_1_panel_segmentation.py
-----------------------------------
End-to-end verification of Step 1.1:
Kadrlar (Panellar) Segmentatsiyasi va To'g'ri O'qish Zanjiri (Panel Segmentation & Reading Order Graph)
"""

import os
import unittest
import numpy as np
import cv2

import reading_order
import engine
from engine import SpeechBubble


class TestStep1_1_PanelSegmentation(unittest.TestCase):

    def test_multi_panel_segmentation_restaurant_page(self):
        """Verify panel segmentation on standard comic page samples/restaurant_page.png."""
        path = "samples/restaurant_page.png"
        self.assertTrue(os.path.exists(path), f"File {path} must exist.")
        img = cv2.imread(path)
        self.assertIsNotNone(img)

        panels = reading_order.detect_panel_boxes(img)
        self.assertGreaterEqual(len(panels), 4, f"Expected at least 4 panels, got {len(panels)}")
        self.assertLessEqual(len(panels), 8, f"Expected realistic comic panels <= 8, got {len(panels)}")

        # Check topological ordering
        ordered_panels = reading_order._row_major_order(panels)
        self.assertEqual(len(ordered_panels), len(panels))
        # Top panel should have y0 near top of page
        self.assertLess(ordered_panels[0][1], img.shape[0] * 0.4)

    def test_oversegmentation_guard_letter_page23(self):
        """Verify that page 23 with dense text lines does NOT explode into 100+ panels."""
        path = "samples/page_23.png"
        if os.path.exists(path):
            img = cv2.imread(path)
            panels = reading_order.detect_panel_boxes(img)
            self.assertLessEqual(len(panels), 10, f"Over-segmentation bug: got {len(panels)} panels!")

    def test_reading_order_cross_panel_isolation(self):
        """
        Verify that bubbles in earlier panels are ALWAYS read before bubbles in subsequent panels,
        even if a bubble in the earlier panel is vertically lower on the page than a bubble in the next panel.
        """
        # Panel 1: Left column (0, 0, 500, 1000)
        # Panel 2: Right column (550, 0, 1050, 1000)
        panel_boxes = [
            (550, 0, 1050, 1000),  # Given in arbitrary order
            (0, 0, 500, 1000),
        ]

        # Bubble 1: in Panel 1, top
        # Bubble 2: in Panel 1, very bottom (y=800)
        # Bubble 3: in Panel 2, top (y=100) -> Higher than Bubble 2 on the Y axis!
        # Bubble 4: in Panel 2, bottom (y=750)
        bubbles = [
            SpeechBubble(bubble_id=0, x0=600, y0=100, x1=800, y1=180, original_text="Panel2_Top"),
            SpeechBubble(bubble_id=0, x0=50, y0=800, x1=300, y1=880, original_text="Panel1_Bottom"),
            SpeechBubble(bubble_id=0, x0=50, y0=50, x1=250, y1=120, original_text="Panel1_Top"),
            SpeechBubble(bubble_id=0, x0=600, y0=750, x1=850, y1=820, original_text="Panel2_Bottom"),
        ]

        img = np.full((1000, 1050, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (480, 980), (30, 30, 30), -1)
        cv2.rectangle(img, (570, 20), (1030, 980), (30, 30, 30), -1)

        ordered_b, panels_info = reading_order.get_page_panels_and_order(img, bubbles)

        texts = [b.original_text for b in ordered_b]
        # Dialogue MUST NOT jump across panels:
        self.assertEqual(texts, ["Panel1_Top", "Panel1_Bottom", "Panel2_Top", "Panel2_Bottom"])

        # Check panel_id and reading_index annotations
        self.assertEqual(ordered_b[0].panel_id, 1)
        self.assertEqual(ordered_b[1].panel_id, 1)
        self.assertEqual(ordered_b[2].panel_id, 2)
        self.assertEqual(ordered_b[3].panel_id, 2)

        self.assertEqual(ordered_b[0].reading_index, 1)
        self.assertEqual(ordered_b[1].reading_index, 2)
        self.assertEqual(ordered_b[2].reading_index, 3)
        self.assertEqual(ordered_b[3].reading_index, 4)

        # Check panels_info structure
        self.assertEqual(len(panels_info), 2)
        self.assertEqual(panels_info[0]["panel_id"], 1)
        self.assertEqual(panels_info[0]["bubble_count"], 2)
        self.assertEqual(panels_info[1]["panel_id"], 2)
        self.assertEqual(panels_info[1]["bubble_count"], 2)

    def test_speechbubble_model_fields(self):
        """Verify SpeechBubble model has panel_id and reading_index."""
        b = SpeechBubble(
            bubble_id=1,
            x0=10, y0=10, x1=100, y1=50,
            original_text="Test",
            panel_id=2,
            reading_index=5
        )
        self.assertEqual(b.panel_id, 2)
        self.assertEqual(b.reading_index, 5)
        d = b.model_dump()
        self.assertEqual(d["panel_id"], 2)
        self.assertEqual(d["reading_index"], 5)


if __name__ == "__main__":
    unittest.main()
