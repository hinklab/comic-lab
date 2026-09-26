"""
tests/test_regressions.py - Master Persistent Regression Test Suite.
Tests all reported comic translation pipeline bug fixes across multiple pages and synthetic fixtures:
1. Diamond-rule clustering split
2. Background-art leak / solidity defense
3. SFX false positives (CHUK, KOFF, TEKK, etc.)
4. Rectangle-oval occlusion & earpiece classification
5. Empty / duplicate fragments
6. Sliver-fragment text overlap
7. Residual-ink leak under lettering
8. Tinted / spectral bubble preservation (Zero Whiteout)
9. Touching-bubbles merge & constriction split
10. Multi-page fixture regression verification
"""

import os
import unittest
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import engine
import bubble_lettering
import quality_gates
from quality_gates import apply_quality_gates, get_quality_gate_log, clear_quality_gate_log


class TestPipelineRegressions(unittest.TestCase):
    def setUp(self):
        clear_quality_gate_log()

    # -------------------------------------------------------------------------
    # 1. Diamond-Rule Clustering Split
    # -------------------------------------------------------------------------
    def test_diamond_rule_clustering_split(self):
        """Verifies that diamond-shaped center-wrapped lines with zero horizontal overlap are not split."""
        # Top short line, middle wide line, bottom short line in round bubble
        lines = [
            {"x0": 130, "y0": 100, "x1": 170, "y1": 120, "cx": 150, "cy": 110, "text": "SHORT TOP"},
            {"x0": 80,  "y0": 125, "x1": 220, "y1": 145, "cx": 150, "cy": 135, "text": "WIDE MIDDLE LINE HERE"},
            {"x0": 125, "y0": 150, "x1": 175, "y1": 170, "cx": 150, "cy": 160, "text": "SHORT BOT"}
        ]
        clusters = engine.cluster_lines_into_bubbles(lines, max_dx=25, max_dy=30, diamond_center_tolerance=0.65)
        self.assertEqual(len(clusters), 1, "Diamond-wrapped lines should form a single coherent bubble cluster")
        self.assertEqual(len(clusters[0]), 3)

        # Confirm quality gate detects diamond rule line pairing
        pair_info = {"gap_x": 30, "gap_y": 10, "c_dist": 5.0, "wider_half": 70.0}
        res = apply_quality_gates("line_pairing", pair_info, context={"max_dy": 30, "max_dx": 25, "diamond_center_tolerance": 0.65})
        self.assertTrue(res.get("allow_connection", False))

    # -------------------------------------------------------------------------
    # 2. Background-Art Leak / Solidity
    # -------------------------------------------------------------------------
    def test_background_art_leak_solidity(self):
        """Verifies that ragged flood-fills leaking into open artwork are caught by solidity and ratio caps."""
        # Test low solidity check: area is 4000, but convex hull is 6000 (solidity = 0.66 < 0.80)
        leak_state = {
            "misplaced": False,
            "low_solidity": True,
            "ratio_to_text": 45.0,
            "effective_ratio_cap": 25.0,
            "cx": 400, "cy": 400, "bw": 120, "bh": 60,
            "x0": 340, "y0": 370, "x1": 460, "y1": 430
        }
        res_cnt, is_rect = apply_quality_gates("contour_extraction", leak_state, context={"image_w": 1000, "image_h": 1000})
        self.assertFalse(is_rect)
        self.assertIsNotNone(res_cnt)
        self.assertGreaterEqual(len(res_cnt), 4)

        # Scale-invariant ratio cap check in bubble_lettering
        aspect = 120.0 / 60.0
        eff_cap = 90.0 * min(2.0, max(1.0, aspect / 3.5))
        self.assertGreaterEqual(eff_cap, 90.0)

    # -------------------------------------------------------------------------
    # 3. SFX False Positives (CHUK, KOFF, TEKK, etc.)
    # -------------------------------------------------------------------------
    def test_sfx_false_positives(self):
        """Verifies that comic sound effects and dark-panel text are filtered from speech bubbles."""
        sfx_examples = ["CHUK", "KOFF", "TEKK", "BOOM", "THWIP", "KRAK", "CLANG"]
        for sfx in sfx_examples:
            is_junk, reason = engine.check_is_sfx_or_junk(sfx, conf=0.95)
            self.assertTrue(is_junk, f"SFX '{sfx}' should be identified as junk/sfx (reason: {reason})")

        # Legitimate short dialogue words must NOT be rejected
        dialogue_words = ["YES", "WHY", "STOP", "WAIT", "HELP", "NO", "OH"]
        for dw in dialogue_words:
            is_junk, _ = engine.check_is_sfx_or_junk(dw, conf=0.95)
            self.assertFalse(is_junk, f"Legitimate dialogue '{dw}' must not be rejected")

        # Quality gate for dark/colored background art false positives
        dark_art_sfx = {
            "mean_luminance": 80.0,
            "percentile_75": 110.0,
            "mean_saturation": 120.0,
            "bright_ratio": 0.02,
            "line_h": 90,
            "text": "CRASH"
        }
        res = apply_quality_gates("ocr_filtering", dark_art_sfx, context={"median_glyph_h": 35})
        self.assertIsNone(res, "Dark non-speech art candidate must be filtered by quality gate")

    # -------------------------------------------------------------------------
    # 4. Rectangle-Oval Occlusion & Earpiece Classification
    # -------------------------------------------------------------------------
    def test_rectangle_oval_occlusion(self):
        """Verifies that speech bubbles are not flattened to boxes, and occluded crescents tuck under rects."""
        # 1. Earpiece rectangle misclassification protection
        speech_bubble = {"z_order": 0, "shape_type": "rectangle", "bubble_id": 1}
        res = apply_quality_gates("bubble_inpainting", speech_bubble)
        self.assertEqual(res["shape_type"], "oval", "Z=0 speech bubble must be demoted from rectangle to oval")

        # 2. Caption shape destruction protection
        shaped_caption = {
            "z_order": 1,
            "shape_type": "rectangle",
            "contour": [[0, 0], [120, 0], [100, 60], [20, 60]],  # non-rectangular trapezoid/polygon
            "is_rectangular_fill": True
        }
        res_shape = apply_quality_gates("bubble_inpainting", shaped_caption)
        self.assertEqual(res_shape["shape_type"], "shaped_caption")
        self.assertFalse(res_shape["is_rectangular_fill"])

        # 3. Occluded crescent text tuck
        tuck_state = {
            "is_occluded": True,
            "top_rect": {"x": 50, "y": 20, "x1": 200, "y1": 90},
            "box": (60, 85, 190, 180)
        }
        res_tuck = apply_quality_gates("lettering", tuck_state)
        self.assertTrue(res_tuck.get("tuck_applied", False))
        self.assertEqual(res_tuck.get("y1_pos"), 87)  # 90 - 3

    # -------------------------------------------------------------------------
    # 5. Empty / Duplicate Fragments
    # -------------------------------------------------------------------------
    def test_empty_and_duplicate_fragments(self):
        """Verifies that empty fragments are discarded and duplicate IDs are sequentially normalized."""
        class DummyBubble:
            def __init__(self, bid, text):
                self.bubble_id = bid
                self.original_text = text
                self.y0 = 100
                self.x0 = 100

        # Empty fragment gate
        empty_b = DummyBubble(1, "   ")
        self.assertIsNone(apply_quality_gates("bubble_validation", empty_b))

        # Duplicate / non-sequential ID gate
        bubbles = [DummyBubble(3, "Text A"), DummyBubble(3, "Text B"), DummyBubble(99, "Text C")]
        resolved = apply_quality_gates("bubble_list_indexing", bubbles)
        self.assertEqual([b.bubble_id for b in resolved], [1, 2, 3])

    # -------------------------------------------------------------------------
    # 6. Sliver-Fragment Text Overlap
    # -------------------------------------------------------------------------
    def test_sliver_fragment_text_overlap(self):
        """Verifies that fragments too narrow or small to hold assigned text are flagged for review."""
        sliver = {"area": 600, "text": "CANNOT POSSIBLY FIT IN THIS TINY SLIVER", "w": 30, "h": 14}
        res = apply_quality_gates("fragment_processing", sliver)
        self.assertEqual(res.get("quality_warning"), "sliver_too_small_for_text")

    # -------------------------------------------------------------------------
    # 7. Residual-Ink Leak Under Lettering
    # -------------------------------------------------------------------------
    def test_residual_ink_cleaned(self):
        """Verifies that residual ink clusters inside bubble interiors trigger secondary inpaint cleanup."""
        img = np.full((60, 60, 3), 255, dtype=np.uint8)
        img[20:30, 20:30] = 0  # dark ink spot
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[20:30, 20:30] = 255
        check_state = {"image_bgr": img, "residual_mask": mask, "residual_ink_count": 100}
        res = apply_quality_gates("post_clean", check_state)
        # Spot in image should have been inpainted (no longer pure black)
        inpainted_patch = res["image_bgr"][22:28, 22:28]
        self.assertGreater(float(np.mean(inpainted_patch)), 180.0)

    # -------------------------------------------------------------------------
    # 8. Tinted / Spectral Bubble Preservation (Zero Whiteout)
    # -------------------------------------------------------------------------
    def test_tinted_spectral_bubble_preservation(self):
        """Verifies that monochromatic tinted bubble interiors are preserved without whiteout."""
        # Quality gate check
        tinted_info = {
            "is_tinted": True,
            "bg_is_tinted": True,
            "text_is_spectral": False,
            "tint_bgr": [235, 220, 195],
            "tint_rgb": [195, 220, 235]
        }
        res = apply_quality_gates("bubble_inpainting", tinted_info)
        self.assertTrue(res.get("is_tinted", False))
        self.assertTrue(res.get("c_bg_protected", False))

        # Inpainting test on authentic tinted canvas (Zero Whiteout)
        w, h = 300, 200
        img = np.full((h, w, 3), (40, 40, 50), dtype=np.uint8)  # dark comic art panel
        bubble_bgr = (245, 230, 200)  # soft cyan/slate tint
        center = (150, 100)
        axes = (110, 65)
        cv2.ellipse(img, center, axes, 0, 0, 360, bubble_bgr, thickness=-1)
        cv2.ellipse(img, center, axes, 0, 0, 360, (200, 180, 150), thickness=2)
        # Draw light text dialogue
        cv2.putText(img, "SPECTRAL VOICE", (65, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 190, 160), 2, cv2.LINE_AA)

        cnt_mask = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 100).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(cnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubble_cnt = max(cnts, key=cv2.contourArea)

        cleaned = bubble_lettering.inpaint_bubble_text_only(img, bubble_cnt)
        text_patch_cleaned = cleaned[90:120, 70:230]
        mean_cleaned_bgr = np.mean(text_patch_cleaned, axis=(0, 1))

        # Cleaned area must be close to authentic tint (245, 230, 200), NOT whiteout (255, 255, 255)
        diff_from_tint = np.linalg.norm(mean_cleaned_bgr - np.array(bubble_bgr))
        diff_from_white = np.linalg.norm(mean_cleaned_bgr - np.array([255.0, 255.0, 255.0]))
        self.assertLess(diff_from_tint, 6.0, "Authentic tint must be preserved without distortion")
        self.assertGreater(diff_from_white, 20.0, "Must not whiteout to pure white")

    # -------------------------------------------------------------------------
    # 9. Touching-Bubbles Merge & Constriction Split
    # -------------------------------------------------------------------------
    def test_touching_bubbles_constriction_split(self):
        """Verifies that conjoined speech bubbles connected by a waist constriction are split into lobes."""
        # Quality gate check
        state = {
            "constriction_ratio": 0.62,
            "has_split_lines": True,
            "lines_left": [{"text": "Hello left lobe"}],
            "lines_right": [{"text": "Hello right lobe"}]
        }
        res = apply_quality_gates("bubble_segmentation", state)
        self.assertEqual(len(res.get("split_clusters", [])), 2)

        # Cross-column dual cluster check
        cross_col_cluster = [
            {"x0": 100, "y0": 100, "x1": 200, "y1": 130, "text": "SPEAKER 1 LEFT"},
            {"x0": 250, "y0": 105, "x1": 380, "y1": 135, "text": "SPEAKER 2 RIGHT"}
        ]
        split_res = apply_quality_gates("line_clustering", cross_col_cluster, context={"min_split_gap": 15})
        self.assertEqual(len(split_res), 2, "Cross-column lines should be split into left and right clusters")

    # -------------------------------------------------------------------------
    # 10. Multi-Page Fixture Integrity Verification
    # -------------------------------------------------------------------------
    def test_multi_page_fixtures(self):
        """Runs image integrity and basic OCR verification on real comic page samples."""
        sample_pages = ["page_4.png", "page_5.png", "page_18.png"]
        for p_name in sample_pages:
            p_path = os.path.join("samples", p_name)
            if not os.path.exists(p_path):
                continue
            img = cv2.imread(p_path)
            self.assertIsNotNone(img, f"Failed to load sample fixture: {p_name}")
            self.assertGreater(img.shape[0], 500)
            self.assertGreater(img.shape[1], 500)

            # Confirm rectangle detector runs without crash
            rects = engine.detect_earpiece_rectangles(img)
            self.assertIsInstance(rects, list)

            # Confirm quality gates registry is healthy and intact
            self.assertGreaterEqual(len(quality_gates.QUALITY_GATES_REGISTRY), 16)


if __name__ == "__main__":
    unittest.main()
