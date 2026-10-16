"""
test_step1_2_tail_vectorization.py
-----------------------------------
End-to-end verification of Step 1.2:
Dum (Tail) Vektorizatsiyasi va Uchi Geometriyasi (Tail Tip & Angle Vector Detection)
"""

import os
import math
import unittest
import numpy as np
import cv2
from PIL import Image

import tail_detector
import engine
from engine import SpeechBubble


class TestStep1_2_TailVectorization(unittest.TestCase):

    def test_synthetic_oval_with_downward_tail(self):
        """Verify that an ellipse with a downward pointy protrusion is reliably detected as a tail."""
        # Create a 400x400 canvas and draw an oval bubble with a tail pointing down
        img = np.zeros((400, 400), dtype=np.uint8)
        cv2.ellipse(img, (200, 150), (100, 60), 0, 0, 360, 255, -1)
        # Tail triangle pointing down towards (200, 290)
        tail_poly = np.array([[180, 200], [220, 200], [200, 290]], dtype=np.int32)
        cv2.fillPoly(img, [tail_poly], 255)

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.assertGreater(len(contours), 0)
        cnt = contours[0]

        tail_info = tail_detector.detect_bubble_tail(cnt, [100, 90, 300, 290], shape_type="oval")
        self.assertTrue(tail_info["has_tail"], "Should detect tail on synthetic speech bubble")
        self.assertEqual(tail_info["tail_direction"], "bottom", "Should detect bottom direction")
        self.assertIsNotNone(tail_info["tail_tip"])
        # Tip should be near (200, 290)
        tx, ty = tail_info["tail_tip"]
        self.assertAlmostEqual(tx, 200, delta=15)
        self.assertAlmostEqual(ty, 290, delta=15)
        # Normalized vector should point downward: uy ~ 1.0, ux ~ 0.0
        ux, uy = tail_info["tail_vector"]
        self.assertAlmostEqual(ux**2 + uy**2, 1.0, delta=0.01)
        self.assertGreater(uy, 0.85)

    def test_caption_box_negative_rejection(self):
        """Verify that a rectangular caption box does NOT falsely detect a tail."""
        img = np.zeros((300, 500), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (450, 150), 255, -1)

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.assertGreater(len(contours), 0)
        cnt = contours[0]

        tail_info = tail_detector.detect_bubble_tail(cnt, [50, 50, 450, 150], shape_type="rectangle")
        self.assertFalse(tail_info["has_tail"], "Rectangular caption box must NOT have a tail")
        self.assertIsNone(tail_info["tail_tip"])
        self.assertEqual(tail_info["tail_direction"], "none")

    def test_vector_normalization_and_angle_coherence(self):
        """Verify that tail_vector is strictly unit-length and matches tail_angle_deg."""
        # Synthetic bubble with top-left tail
        img = np.zeros((400, 400), dtype=np.uint8)
        cv2.ellipse(img, (200, 200), (80, 50), 0, 0, 360, 255, -1)
        tail_poly = np.array([[160, 170], [180, 155], [70, 70]], dtype=np.int32)
        cv2.fillPoly(img, [tail_poly], 255)

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]

        tail_info = tail_detector.detect_bubble_tail(cnt, shape_type="oval")
        self.assertTrue(tail_info["has_tail"])
        ux, uy = tail_info["tail_vector"]
        self.assertAlmostEqual(ux**2 + uy**2, 1.0, delta=0.01)

        # Angle check
        expected_deg = math.degrees(math.atan2(uy, ux))
        self.assertAlmostEqual(tail_info["tail_angle_deg"], expected_deg, delta=0.5)
        self.assertIn(tail_info["tail_direction"], ["top-left", "left", "top"])

    def test_project_tail_to_speaker_region(self):
        """Verify that projection stays strictly clamped within the panel boundary."""
        panel_box = [100, 100, 600, 800]
        tail_tip = [300, 750]
        tail_vector = [0.0, 1.0]  # Pointing straight down

        proj_x, proj_y = tail_detector.project_tail_to_speaker_region(
            tail_tip, tail_vector, panel_box=panel_box, distance_px=200
        )
        # Should project down, but clamp to panel bottom (800 - 5 = 795)
        self.assertEqual(proj_x, 300)
        self.assertEqual(proj_y, 795)

    def test_annotate_bubble_with_tail_helper(self):
        """Verify annotate_bubble_with_tail updates SpeechBubble dataclass and dictionary."""
        # 1. On SpeechBubble
        b = SpeechBubble(
            bubble_id=1,
            x0=100, y0=100, x1=300, y1=250,
            original_text="HELLO WORLD",
            uzbek_translation="SALOM DUNYO",
            contour=[[100, 100], [300, 100], [300, 250], [100, 250]]
        )
        tail = tail_detector.annotate_bubble_with_tail(b)
        self.assertIn("has_tail", tail)
        self.assertTrue(hasattr(b, "has_tail"))
        self.assertTrue(hasattr(b, "tail_direction"))

        # 2. On Dict
        b_dict = {
            "bubble_id": 2,
            "x0": 50, "y0": 50, "x1": 150, "y1": 120,
            "contour": [[50, 50], [150, 50], [150, 120], [50, 120]],
            "shape_type": "rectangle"
        }
        tail_dict = tail_detector.annotate_bubble_with_tail(b_dict)
        self.assertFalse(b_dict["has_tail"])
        self.assertEqual(b_dict["tail_direction"], "none")

    def test_real_comic_page_integration(self):
        """Verify real comic bubbles from samples/sample_comic.png."""
        path = "samples/sample_comic.png"
        if not os.path.exists(path):
            self.skipTest(f"{path} not found")

        img = Image.open(path)
        import unittest.mock
        with unittest.mock.patch.dict(os.environ, {"FORCE_LOCAL_OCR": "1"}):
            bubbles = engine.scan_bubbles_ocr(img)
        self.assertGreater(len(bubbles), 0)

        # Bubble #3 in sample_comic.png is known to have a prominent bottom tail pointing at robot
        tail_detected_count = sum(1 for b in bubbles if getattr(b, "has_tail", False))
        self.assertGreater(tail_detected_count, 0, "At least one speech bubble should have a detected tail")

        b3 = next((b for b in bubbles if b.bubble_id == 3), None)
        if b3:
            self.assertTrue(b3.has_tail, "Bubble #3 must have detected tail")
            self.assertIn(b3.tail_direction, ["bottom", "bottom-left", "bottom-right"])
            self.assertGreater(b3.tail_length, 30.0)


if __name__ == "__main__":
    unittest.main()
