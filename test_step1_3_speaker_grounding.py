"""
test_step1_3_speaker_grounding.py
----------------------------------
End-to-end verification of Step 1.3:
Kontekstual Qahramon Xaritasi va Grounding (Speaker Identification & Character Mapping)
"""

import os
import unittest
import numpy as np
import cv2
from PIL import Image

import speaker_detector
import tail_detector
from engine import SpeechBubble


class TestStep1_3_SpeakerGrounding(unittest.TestCase):

    def test_caption_box_automatic_narrator_grounding(self):
        """Verify rectangular caption boxes without tails are grounded as Sharhlovchi (Narrator)."""
        caption_bubble = SpeechBubble(
            bubble_id=1,
            x0=50, y0=50, x1=400, y1=150,
            original_text="MEANWHILE, ACROSS THE MULTIVERSE...",
            uzbek_translation="SHU PAYTDA, KO'P OLAMNING NARGI CHEKKASIDA...",
            shape_type="rectangle",
            has_tail=False
        )

        speaker, conf, anchor = speaker_detector.ground_bubble_speaker(caption_bubble)
        self.assertEqual(speaker, "Sharhlovchi", "Caption box must be grounded as Sharhlovchi")
        self.assertGreaterEqual(conf, 0.90)
        self.assertIsNone(anchor)

    def test_sample_character_palette_along_tail_ray(self):
        """Verify sampling along the tail vector extracts non-white character colors."""
        # Create an image with a white bubble at top and a green/silver character below
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        # Background: dark slate
        img[:] = (30, 30, 30)
        # White bubble at (100, 50) to (300, 150)
        cv2.rectangle(img, (100, 50), (300, 150), (255, 255, 255), -1)
        # Green cloak character below at (150, 220) to (250, 350)
        # OpenCV BGR: Green cloak is (30, 120, 40)
        cv2.rectangle(img, (150, 200), (250, 350), (30, 120, 40), -1)

        tail_tip = [200, 150]
        tail_vector = [0.0, 1.0]  # Pointing straight down into the green character

        colors, weights, anchor = speaker_detector.sample_character_palette(
            img, tail_tip, tail_vector, distance_px=70, radius=30
        )

        self.assertIsNotNone(anchor)
        self.assertEqual(anchor[0], 200)
        self.assertEqual(anchor[1], 220)
        self.assertGreater(len(colors), 0, "Should extract dominant colors from character region")
        # Top dominant color should be green: R ~ 40, G ~ 120, B ~ 30
        top_color_rgb = colors[0]
        self.assertGreater(top_color_rgb[1], top_color_rgb[0])  # Green channel > Red
        self.assertGreater(top_color_rgb[1], top_color_rgb[2])  # Green channel > Blue

    def test_signature_matching_against_profiles(self):
        """Verify matching against mock CharacterVoiceProfiles."""
        class MockProfile:
            def __init__(self, name, dominant_colors):
                self.name = name
                self.visual_signature = {
                    "dominant_colors": dominant_colors,
                    "color_weights": [0.7, 0.3]
                }

        profiles = [
            MockProfile("Doctor Doom", [[35, 120, 45], [180, 180, 185]]),  # Green & Silver
            MockProfile("Spider-Man", [[210, 20, 25], [20, 40, 190]])     # Red & Blue
        ]

        # Query palette: predominantly green and silver
        query_colors = [[40, 115, 50], [175, 175, 180]]
        query_weights = [0.75, 0.25]

        matched, conf = speaker_detector.match_against_signatures(query_colors, query_weights, profiles)
        self.assertEqual(matched, "Doctor Doom")
        self.assertGreater(conf, 0.70)

        # Query palette: predominantly red and blue
        query_spidey = [[205, 25, 30], [25, 45, 185]]
        matched_spidey, conf_spidey = speaker_detector.match_against_signatures(query_spidey, query_weights, profiles)
        self.assertEqual(matched_spidey, "Spider-Man")
        self.assertGreater(conf_spidey, 0.70)

    def test_cluster_continuity_preserves_single_speaker(self):
        """Verify conjoined multi-lobe bubble clusters share the same speaker."""
        b1 = SpeechBubble(
            bubble_id=1,
            x0=100, y0=100, x1=200, y1=150,
            original_text="FIRST LOBE",
            uzbek_translation="BIRINCHI QISM",
            cluster_id="cluster_alpha",
            speaker="Doctor Doom",
            speaker_confidence=0.92
        )
        b2 = SpeechBubble(
            bubble_id=2,
            x0=100, y0=160, x1=200, y1=210,
            original_text="SECOND LOBE",
            uzbek_translation="IKKINCHI QISM",
            cluster_id="cluster_alpha",
            speaker=None,
            speaker_confidence=0.0
        )

        grounded = speaker_detector.ground_page_speakers([b1, b2])
        self.assertEqual(grounded[0].speaker, "Doctor Doom")
        self.assertEqual(grounded[1].speaker, "Doctor Doom", "Second lobe must inherit Doctor Doom from cluster")
        self.assertGreater(grounded[1].speaker_confidence, 0.8)

    def test_merge_signature_ema(self):
        """Verify merge_signature EMA blending of dominant colors."""
        sig = {
            "dominant_colors": [[40, 120, 50]],
            "color_weights": [1.0],
            "sample_count": 1
        }
        # Add nearby color
        new_colors = [[44, 116, 52]]
        new_weights = [1.0]

        updated = speaker_detector.merge_signature(sig, new_colors, new_weights)
        self.assertEqual(updated["sample_count"], 2)
        # Color should be blended between 40 and 44
        blended_r = updated["dominant_colors"][0][0]
        self.assertIn(blended_r, [41, 42, 43])

    def test_off_panel_voice_detection(self):
        """Verify bubble with tail pointing at panel boundary is grounded as Off-Panel Voice."""
        panel_box = [100, 100, 600, 800]
        # Bubble near right edge of panel, tail pointing straight right towards border (600)
        bubble = SpeechBubble(
            bubble_id=7,
            x0=300, y0=150, x1=580, y1=240,
            original_text="PLANNING A 48-HOUR SHIFT...",
            uzbek_translation="48 SOATLIK REJALASHTIRIB...",
            has_tail=True,
            tail_tip=[590, 195],       # 10px from right panel border (600)
            tail_vector=[0.98, 0.05],  # pointing right
            tail_direction="right",
            panel_id=1
        )

        speaker, conf, anchor = speaker_detector.ground_bubble_speaker(
            bubble,
            panel_box=panel_box,
            candidate_speaker="Boss"
        )

        self.assertTrue(bubble.is_off_panel, "Bubble must be flagged as is_off_panel")
        self.assertIn("Kadrdan tashqari", speaker)
        self.assertIn("Boss", speaker)
        self.assertIsNone(anchor, "Off-panel voice must NOT project anchor onto in-panel character")


if __name__ == "__main__":
    unittest.main()
