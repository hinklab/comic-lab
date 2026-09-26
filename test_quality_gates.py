"""
test_quality_gates.py - Comprehensive unit test for quality_gates.py
Tests each IssuePattern detect and resolve logic in isolation.
"""

import unittest
import numpy as np
import quality_gates
from quality_gates import (
    apply_quality_gates,
    get_quality_gate_log,
    clear_quality_gate_log,
    quality_gate_context,
    QUALITY_GATES_REGISTRY
)


class TestQualityGates(unittest.TestCase):
    def setUp(self):
        clear_quality_gate_log()

    def test_registry_patterns_loaded(self):
        self.assertGreaterEqual(len(QUALITY_GATES_REGISTRY), 10)
        names = {p.name for p in QUALITY_GATES_REGISTRY}
        expected = {
            "sfx_false_positive",
            "diamond_rule_line_split",
            "adjacent_bubble_merging_cross_column",
            "figure8_double_lobed_bubble",
            "background_art_leak",
            "earpiece_rect_misclassification",
            "empty_fragment",
            "duplicate_bubble_id",
            "sliver_too_small_for_text",
            "residual_ink_under_lettering",
            "conjoined_bubble_constriction_split",
            "translation_sentence_drop"
        }
        self.assertTrue(expected.issubset(names))

    def test_sfx_false_positive_gate(self):
        # Dark non-speech line
        sfx_candidate = {
            "mean_luminance": 110.0,
            "percentile_75": 140.0,
            "mean_saturation": 90.0,
            "bright_ratio": 0.05,
            "line_h": 120,
            "text": "Zwk"
        }
        res = apply_quality_gates("ocr_filtering", sfx_candidate, context={"median_glyph_h": 35})
        self.assertIsNone(res)
        log = get_quality_gate_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["pattern_name"], "sfx_false_positive")
        self.assertEqual(log[0]["action_taken"], "auto_fix_applied")

    def test_diamond_rule_gate(self):
        pair_info = {
            "gap_x": 15,
            "gap_y": 10,
            "c_dist": 20.0,
            "wider_half": 100.0,
        }
        res = apply_quality_gates("line_pairing", pair_info, context={"max_dy": 30, "max_dx": 25, "diamond_center_tolerance": 0.65})
        self.assertTrue(res.get("allow_connection", False))
        log = get_quality_gate_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["pattern_name"], "diamond_rule_line_split")

    def test_adjacent_bubble_merging_cross_column_gate(self):
        cluster = [
            {"x0": 926, "y0": 1242, "x1": 1287, "y1": 1281, "text": "OF EVERYONE HERE"},
            {"x0": 1305, "y0": 1278, "x1": 1704, "y1": 1320, "text": "OuR Stolen SCiEnCe"}
        ]
        res = apply_quality_gates("line_clustering", cluster, context={"min_split_gap": 12})
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0][0]["text"], "OF EVERYONE HERE")
        self.assertEqual(res[1][0]["text"], "OuR Stolen SCiEnCe")
        log = get_quality_gate_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["pattern_name"], "adjacent_bubble_merging_cross_column")

    def test_background_art_leak_gate(self):
        leak_state = {
            "misplaced": False,
            "low_solidity": True,  # Ragged edge leak
            "ratio_to_text": 35.0,
            "effective_ratio_cap": 20.0,
            "cx": 500, "cy": 500, "bw": 100, "bh": 50,
            "x0": 450, "y0": 475, "x1": 550, "y1": 525
        }
        res_cnt, is_rect = apply_quality_gates("contour_extraction", leak_state, context={"image_w": 2000, "image_h": 3000})
        self.assertFalse(is_rect)
        self.assertIsNotNone(res_cnt)
        log = get_quality_gate_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["pattern_name"], "background_art_leak")

    def test_earpiece_rect_misclassification_gate(self):
        bubble_info = {"z_order": 0, "shape_type": "rectangle", "bubble_id": 8}
        res = apply_quality_gates("bubble_inpainting", bubble_info)
        self.assertEqual(res["shape_type"], "oval")
        log = get_quality_gate_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["pattern_name"], "earpiece_rect_misclassification")

    def test_empty_fragment_gate(self):
        class DummyBubble:
            original_text = "   "
        res = apply_quality_gates("bubble_validation", DummyBubble())
        self.assertIsNone(res)
        log = get_quality_gate_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["pattern_name"], "empty_fragment")

    def test_duplicate_bubble_id_gate(self):
        class DummyBubble:
            def __init__(self, bid, y, x):
                self.bubble_id = bid
                self.y0 = y
                self.x0 = x

        bubbles = [DummyBubble(5, 500, 100), DummyBubble(5, 200, 100), DummyBubble(1, 100, 100)]
        res = apply_quality_gates("bubble_list_indexing", bubbles)
        self.assertEqual([b.bubble_id for b in res], [1, 2, 3])
        log = get_quality_gate_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["pattern_name"], "duplicate_bubble_id")

    def test_sliver_too_small_gate(self):
        frag_info = {
            "area": 800,
            "text": "Hello world this text is long",
            "w": 50,
            "h": 12
        }
        res = apply_quality_gates("fragment_processing", frag_info)
        self.assertEqual(res.get("quality_warning"), "sliver_too_small_for_text")
        log = get_quality_gate_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["pattern_name"], "sliver_too_small_for_text")
        self.assertEqual(log[0]["severity"], "flag_for_review")

    def test_residual_ink_gate(self):
        img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        img[20:30, 20:30] = 0  # dark ink
        residual_mask = np.zeros((50, 50), dtype=np.uint8)
        residual_mask[20:30, 20:30] = 255
        clean_check = {
            "image_bgr": img,
            "residual_mask": residual_mask,
            "residual_ink_count": 100
        }
        res = apply_quality_gates("post_clean", clean_check)
        self.assertIn("image_bgr", res)
        log = get_quality_gate_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["pattern_name"], "residual_ink_under_lettering")
        self.assertEqual(log[0]["action_taken"], "auto_fix_applied")

    def test_conjoined_bubble_constriction_gate(self):
        bubble_state = {
            "constriction_ratio": 0.60,
            "has_split_lines": True,
            "lines_left": [{"text": "Hello"}],
            "lines_right": [{"text": "World"}]
        }
        res = apply_quality_gates("bubble_segmentation", bubble_state)
        self.assertEqual(len(res.get("split_clusters", [])), 2)
        log = get_quality_gate_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["pattern_name"], "conjoined_bubble_constriction_split")
    def test_caption_shape_destruction_gate(self):
        # Non-rectangular contour with low fill ratio (e.g. wings/tails)
        # Bounding box 100x100, but triangular/wing shape with area ~ 3000 (ratio 0.30)
        pts = np.array([[0, 0], [100, 0], [50, 100]], dtype=np.int32)
        bubble_info = {
            "z_order": 1,
            "shape_type": "rectangle",
            "contour": pts.tolist(),
            "bubble_id": 1
        }
        res = apply_quality_gates("bubble_inpainting", bubble_info)
        self.assertEqual(res["shape_type"], "shaped_caption")
        self.assertFalse(res.get("is_rectangular_fill", True))
        log = get_quality_gate_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["pattern_name"], "caption_shape_destruction")
        self.assertEqual(log[0]["action_taken"], "auto_fix_applied")


    def test_untranslated_english_residue_gate(self):
        # Hybrid English-Uzbek words or failed translation
        bad_text = "BU OTTO OKTAVIUS, YOU FLAAT-TOPPED FINKA DOKTOR OKTOPUSNING BRAINI BILAN"
        has_res, words = quality_gates.check_untranslated_english_residue(bad_text)
        self.assertTrue(has_res)
        self.assertTrue(any("BRAINI" in w or "FINKA" in w or "YOU" in w for w in words))

        # Clean Uzbek text with permitted character names
        good_text = "BU YERDA AJOYIB ISH QILDINGIZ, SPIDEY! MER J. JONA JEYMSON XOTIRA FONDI NOMIDAN."
        has_res_clean, words_clean = quality_gates.check_untranslated_english_residue(good_text)
        self.assertFalse(has_res_clean)
        self.assertEqual(len(words_clean), 0)


    def test_tinted_spectral_bubble_gate(self):
        bubble_info = {
            "is_tinted": True,
            "bg_is_tinted": True,
            "text_is_spectral": False,
            "tint_bgr": [245, 230, 200],
            "tint_rgb": [200, 230, 245],
            "bubble_id": 4
        }
        res = apply_quality_gates("bubble_inpainting", bubble_info)
        self.assertTrue(res.get("is_tinted", False))
        self.assertTrue(res.get("c_bg_protected", False))
        log = get_quality_gate_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["pattern_name"], "tinted_spectral_bubble_preservation")
        self.assertEqual(log[0]["action_taken"], "auto_fix_applied")

    def test_occluded_crescent_tuck_gate(self):
        lettering_state = {
            "is_occluded": True,
            "top_rect": {"x": 100, "y": 50, "x1": 250, "y1": 120},
            "box": (110, 110, 240, 220)
        }
        res = apply_quality_gates("lettering", lettering_state)
        self.assertTrue(res.get("tuck_applied", False))
        self.assertEqual(res.get("y1_pos"), 117)  # 120 - 3
        log = get_quality_gate_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["pattern_name"], "occluded_crescent_text_tuck")
        self.assertEqual(log[0]["action_taken"], "auto_fix_applied")

    def test_translation_sentence_drop_gate(self):
        # Bubble with 2 sentences in English dropped to 1 in Uzbek
        drop_case = {
            "source_text": "I'M NOT DOC OCK. I'M SPIDER-MAN!",
            "translated_text": "MEN O'RGIMCHAK-ODAMMAN!"
        }
        res = apply_quality_gates("translation_validation", drop_case)
        self.assertTrue(res.get("needs_review", False))
        self.assertEqual(res.get("quality_warning"), "translation_sentence_drop")
        log = get_quality_gate_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["pattern_name"], "translation_sentence_drop")
        self.assertEqual(log[0]["severity"], "flag_for_review")

        # Post-translation validation criterion 5 check
        val = quality_gates.post_translation_validation(drop_case["translated_text"], drop_case["source_text"])
        self.assertTrue(val.get("needs_review"))
        self.assertIn("Tarjimada gap tushib qolgan", val.get("reason", ""))

        # Clean translation preserving both sentences
        clear_quality_gate_log()
        clean_case = {
            "source_text": "I'M NOT DOC OCK. I'M SPIDER-MAN!",
            "translated_text": "MEN DOKTOR OKTOPUS EMASMAN. MEN O'RGIMCHAK-ODAMMAN!"
        }
        res_clean = apply_quality_gates("translation_validation", clean_case)
        self.assertFalse(res_clean.get("needs_review", False))
        log_clean = get_quality_gate_log()
        self.assertEqual(len(log_clean), 0)

        val_clean = quality_gates.post_translation_validation(clean_case["translated_text"], clean_case["source_text"])
        self.assertFalse(val_clean.get("needs_review"))


if __name__ == "__main__":
    unittest.main()

