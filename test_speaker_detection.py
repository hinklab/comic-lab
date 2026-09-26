"""
Unit tests for automatic speaker detection (classical CV) and visual signature memory.
"""

import unittest
import numpy as np
import cv2
import os

from speaker_detector import (
    find_tail_tip,
    extract_dominant_colors,
    match_against_signatures,
    merge_signature,
    detect_speaker,
    CONF_THRESHOLD
)
import character_profiles
from character_profiles import CharacterVoiceProfile
import engine
from engine import SpeechBubble


class TestSpeakerDetection(unittest.TestCase):

    def setUp(self):
        self.spiderman_profile = CharacterVoiceProfile(
            name="Superior Spider-Man",
            aliases=["spider-man", "spiderman", "otto"],
            voice_traits=["Formal, scientific"],
            visual_signature={
                "dominant_colors": [[178, 30, 30], [30, 55, 175], [215, 215, 215]],
                "color_weights": [0.45, 0.40, 0.15],
                "sample_count": 1,
                "last_updated": "2026-09-18"
            }
        )

    def test_tail_tip_extraction_protruding_tail(self):
        """A contour with an outward tail tip should be correctly identified."""
        # Define an oval with a pointy tail extending downward to (100, 160)
        contour = [
            [50, 50], [100, 40], [150, 50],
            [160, 80], [140, 110],
            [100, 160],  # tail tip extending far below bbox
            [60, 110], [40, 80]
        ]
        bbox = (40, 40, 160, 120)
        tip = find_tail_tip(contour, bbox)
        self.assertIsNotNone(tip, "Expected to find a tail tip")
        self.assertEqual(tip, (100, 160), "Should match the sharpest protruding vertex")

    def test_tail_tip_extraction_pure_rectangle_caption(self):
        """A plain rectangular box without a tail should return None."""
        rect_contour = [
            [50, 50], [150, 50], [150, 100], [50, 100]
        ]
        bbox = (50, 50, 150, 100)
        tip = find_tail_tip(rect_contour, bbox)
        self.assertIsNone(tip, "Plain rectangle caption should have no tail tip")

    def test_dominant_color_extraction(self):
        """K-means should extract correct dominant colors from synthetic pixel data."""
        # 60 red pixels (BGR: 0, 0, 200 -> RGB: 200, 0, 0), 40 blue pixels (BGR: 200, 0, 0 -> RGB: 0, 0, 200)
        red_pixels = np.zeros((60, 3), dtype=np.uint8)
        red_pixels[:, 2] = 200
        blue_pixels = np.zeros((40, 3), dtype=np.uint8)
        blue_pixels[:, 0] = 200
        pixels_bgr = np.vstack([red_pixels, blue_pixels])

        colors, weights = extract_dominant_colors(pixels_bgr, k=2)
        self.assertEqual(len(colors), 2)
        self.assertAlmostEqual(weights[0], 0.60, places=1)
        # Check first dominant color is Red in RGB
        self.assertGreater(colors[0][0], 180)
        self.assertLess(colors[0][2], 20)

    def test_signature_matching_above_threshold(self):
        """Colors close to Spider-Man suit should match with confidence >= 0.55."""
        # Close to Spider-Man suit: dark red, navy blue, light highlight
        query_colors = [[175, 32, 28], [32, 58, 170]]
        query_weights = [0.55, 0.45]

        matched_name, conf = match_against_signatures(
            query_colors, query_weights, [self.spiderman_profile]
        )
        self.assertEqual(matched_name, "Superior Spider-Man")
        self.assertGreaterEqual(conf, CONF_THRESHOLD)

    def test_signature_matching_below_threshold(self):
        """Colors unrelated to known characters (e.g. bright green) should return None."""
        query_colors = [[20, 220, 30], [200, 200, 40]]
        query_weights = [0.70, 0.30]

        matched_name, conf = match_against_signatures(
            query_colors, query_weights, [self.spiderman_profile]
        )
        self.assertIsNone(matched_name)
        self.assertLess(conf, CONF_THRESHOLD)

    def test_visual_signature_growth_dedup_and_merge(self):
        """Visual signature growth should merge near-identical clusters without unbounded growth."""
        initial_sig = {
            "dominant_colors": [[178, 30, 30], [30, 55, 175]],
            "color_weights": [0.60, 0.40],
            "sample_count": 1,
            "last_updated": "2026-09-18"
        }

        # Observation 1: Very close color (distance < 30) -> should blend, not append new cluster
        new_colors = [[180, 32, 28]]
        new_weights = [1.0]

        updated = merge_signature(initial_sig, new_colors, new_weights)
        self.assertEqual(len(updated["dominant_colors"]), 2, "Near-duplicate cluster should be blended, not added")
        self.assertEqual(updated["sample_count"], 2)

        # Observation 2: Novel color variance (e.g. yellow highlight) -> should append
        novel_colors = [[240, 220, 50]]
        novel_weights = [0.20]

        updated2 = merge_signature(updated, novel_colors, novel_weights)
        self.assertEqual(len(updated2["dominant_colors"]), 3, "Novel color should add a new cluster")
        self.assertEqual(updated2["sample_count"], 3)

    def test_assign_speakers_integration(self):
        """assign_speakers should process SpeechBubble objects and assign confidence."""
        # Create a synthetic image (300x300) with Spider-Man red/blue near bottom of bubble
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        # Fill background with comic art gray/blue
        img[:] = (80, 80, 80)
        # Tail area: red (BGR: 0, 0, 180)
        img[140:220, 80:160] = (25, 25, 178)

        bubble = SpeechBubble(
            bubble_id=1,
            x0=80, y0=50, x1=160, y1=130,
            original_text="YOU WILL LEARN I AM SUPERIOR!",
            contour_points=[[80, 50], [160, 50], [160, 130], [120, 160], [80, 130]]
        )

        assigned = engine.assign_speakers([bubble], img)
        self.assertEqual(len(assigned), 1)
        self.assertIsNotNone(assigned[0].speaker_confidence)
        # If confidence >= threshold, speaker should be assigned
        if assigned[0].speaker_confidence >= CONF_THRESHOLD:
            self.assertEqual(assigned[0].speaker, "Superior Spider-Man")


if __name__ == "__main__":
    unittest.main()
