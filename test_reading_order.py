"""
test_reading_order.py
Unit tests for reading_order.py:
1. Panel-aware reading order vs flat (y, x) interleaving
2. Dict and SpeechBubble object compatibility
3. Location-based SFX vs dialogue classification (is_dialogue_text)
4. Panel assignment with centroid and gutter fallback
5. Synthetic gutter detection (detect_panel_boxes)
"""

import sys
import os
import unittest
import numpy as np
import cv2

import reading_order
from engine import SpeechBubble


class TestReadingOrder(unittest.TestCase):

    def test_panel_reading_order_prevents_interleaving(self):
        """
        Panel A (Left: 0, 0, 500, 500)
        Panel B (Right: 550, 0, 1050, 500)
        
        Bubble A1: (50, 50, 200, 100) -> Panel A, top
        Bubble A2: (50, 350, 200, 420) -> Panel A, bottom (high y!)
        Bubble B1: (600, 80, 800, 140) -> Panel B, top
        Bubble B2: (600, 200, 800, 260) -> Panel B, middle
        
        A flat (y, x) sort would interleave: A1, B1, B2, A2!
        Panel-aware sort MUST yield: A1, A2, B1, B2.
        """
        panel_boxes = [
            (550, 0, 1050, 500),  # Panel B (given first in arbitrary order)
            (0, 0, 500, 500),      # Panel A
        ]

        bubbles = [
            {"id": "B1", "bbox": (600, 80, 800, 140)},
            {"id": "A2", "bbox": (50, 350, 200, 420)},
            {"id": "B2", "bbox": (600, 200, 800, 260)},
            {"id": "A1", "bbox": (50, 50, 200, 100)},
        ]

        ordered = reading_order.order_bubbles_reading_order(bubbles, panel_boxes)
        ordered_ids = [b["id"] for b in ordered]
        self.assertEqual(ordered_ids, ["A1", "A2", "B1", "B2"])

    def test_speech_bubble_objects_compatibility(self):
        """Verify order_bubbles_reading_order works natively with SpeechBubble instances."""
        panel_boxes = [
            (0, 0, 500, 500),
            (550, 0, 1050, 500)
        ]

        sb_a1 = SpeechBubble(bubble_id=0, x0=50, y0=50, x1=200, y1=100, original_text="A1")
        sb_a2 = SpeechBubble(bubble_id=0, x0=50, y0=350, x1=200, y1=420, original_text="A2")
        sb_b1 = SpeechBubble(bubble_id=0, x0=600, y0=80, x1=800, y1=140, original_text="B1")
        sb_b2 = SpeechBubble(bubble_id=0, x0=600, y0=200, x1=800, y1=260, original_text="B2")

        ordered = reading_order.order_bubbles_reading_order([sb_b2, sb_a2, sb_b1, sb_a1], panel_boxes)
        ordered_texts = [b.original_text for b in ordered]
        self.assertEqual(ordered_texts, ["A1", "A2", "B1", "B2"])

    def test_assign_item_to_panel_with_gutter_tail(self):
        """Verify fallback when item centroid or tail is in a gutter."""
        panel_boxes = [
            (0, 0, 400, 500),       # Panel 0: 0..400
            (500, 0, 900, 500),     # Panel 1: 500..900 (gutter is 400..500)
        ]
        # Item in gutter near panel 0 (e.g. x centroid = 410)
        item_box = (405, 100, 415, 150)
        p_idx = reading_order.assign_item_to_panel(item_box, panel_boxes)
        self.assertEqual(p_idx, 0)

        # Item in gutter near panel 1 (e.g. x centroid = 480)
        item_box2 = (470, 100, 490, 150)
        p_idx2 = reading_order.assign_item_to_panel(item_box2, panel_boxes)
        self.assertEqual(p_idx2, 1)

    def test_is_dialogue_text_location_classification(self):
        """
        SFX inside a bubble contour is dialogue (True).
        SFX drawn on background art outside all bubbles is environmental SFX (False).
        """
        # Create circular bubble contour around (200, 200) with radius 50
        angles = np.linspace(0, 2 * np.pi, 36)
        xs = 200 + 50 * np.cos(angles)
        ys = 200 + 50 * np.sin(angles)
        cnt = np.stack([xs, ys], axis=1).astype(np.int32).reshape((-1, 1, 2))
        bubble_contours = [cnt]

        # Case 1: Text box inside bubble (e.g. at (190, 190, 210, 210))
        inside_box = (190, 190, 210, 210)
        self.assertTrue(reading_order.is_dialogue_text(inside_box, bubble_contours))

        # Case 2: Free-floating SFX on art outside bubble (e.g. at (50, 50, 90, 80))
        outside_box = (50, 50, 90, 80)
        self.assertFalse(reading_order.is_dialogue_text(outside_box, bubble_contours))

        # Case 3: Word that sounds like SFX ("BOOM!") inside the bubble
        dialogue_sfx_word = (195, 195, 205, 205)
        self.assertTrue(reading_order.is_dialogue_text(dialogue_sfx_word, bubble_contours))

    def test_detect_panel_boxes_synthetic_grid(self):
        """Test recursive XY-cut panel detector on synthetic 2-panel comic layout."""
        # 1000x800 white canvas
        img = np.full((800, 1000, 3), 255, dtype=np.uint8)
        # Draw dark panel contents inside 2 panel regions with white gutter between them:
        # Panel 1: (50, 50, 450, 750)
        # Gutter: 450..550
        # Panel 2: (550, 50, 950, 750)
        cv2.rectangle(img, (50, 50), (450, 750), (30, 30, 30), -1)
        cv2.rectangle(img, (550, 50), (950, 750), (30, 30, 30), -1)

        panels = reading_order.detect_panel_boxes(img)
        self.assertEqual(len(panels), 2)
        ordered_panels = reading_order._row_major_order(panels)
        self.assertLess(ordered_panels[0][0], ordered_panels[1][0])


    def test_conversational_dialogue_alternation(self):
        """
        Verifies that when Speaker 1 speaks (top-left), Speaker 2 replies
        (middle, slightly lower), and Speaker 1 continues (top-right),
        the reading order is correctly Speaker 1 -> Speaker 2 -> Speaker 1.
        Prevents Speaker 1's bubbles from being grouped consecutively across the panel.
        """
        panel_boxes = [(0, 0, 1000, 1000)]
        bubbles = [
            {"id": "Jonah_Prompt", "bbox": (86, 83, 355, 201)},
            {"id": "Jonah_Continuation", "bbox": (554, 86, 906, 197)},
            {"id": "Spidey_Reply", "bbox": (348, 165, 555, 247)},
        ]
        ordered = reading_order.order_bubbles_reading_order(bubbles, panel_boxes)
        ordered_ids = [b["id"] for b in ordered]
        self.assertEqual(ordered_ids, ["Jonah_Prompt", "Spidey_Reply", "Jonah_Continuation"])


if __name__ == "__main__":
    unittest.main()
