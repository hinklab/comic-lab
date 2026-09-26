"""
test_bubble_mask_editor.py
--------------------------
Comprehensive unit test suite for Stage 2 manual bubble-mask editing tools:
1. Tool 1 -- "Qaychi" (Scissors):
   - Clean 2-component split of merged bubbles
   - Rejection on incomplete cut (1 component)
   - Rejection on excessive cut (>2 components)
   - Spatial containment line assignment
   - Ambiguous borderline line detection
2. Tool 2 -- "Aqlli mo'yqalam" (Smart Brush):
   - Tolerance-filtered candidate addition
   - Strict rejection of dark line art and contrasting background art
   - Safe contour and bounding box expansion
3. Persistent Learning & Quality Gates:
   - Event logging into mask_correction_memory.json
   - QualityGateEvent integration
   - Dynamic threshold and tolerance tuning
4. Real-world fixture tests:
   - samples/page_20.png (Spider-Man/ghost merged bubble)
   - samples/page_4.png (Jameson panel clipped bubble edge)
"""

import os
import json
import unittest
import numpy as np
import cv2

import quality_gates
import bubble_mask_editor
import engine


class TestBubbleMaskEditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_memory_file = "test_mask_correction_memory.json"
        bubble_mask_editor.MEMORY_FILE = cls.test_memory_file

    @classmethod
    def tearDownClass(cls):
        bubble_mask_editor.MEMORY_FILE = "mask_correction_memory.json"
        quality_gates.reset_learned_thresholds()

    def setUp(self):
        quality_gates.clear_quality_gate_log()
        if os.path.exists(self.test_memory_file):
            try:
                os.remove(self.test_memory_file)
            except Exception:
                pass

    def tearDown(self):
        if os.path.exists(self.test_memory_file):
            try:
                os.remove(self.test_memory_file)
            except Exception:
                pass

    # --------------------------------------------------------------------------
    # Helper & Mask Conversion Tests
    # --------------------------------------------------------------------------

    def test_create_bubble_mask(self):
        img_shape = (200, 200)
        # Empty contour
        mask_empty = bubble_mask_editor.create_bubble_mask(None, img_shape)
        self.assertEqual(mask_empty.shape, img_shape)
        self.assertEqual(np.count_nonzero(mask_empty), 0)

        # Valid polygon contour
        cnt = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.int32)
        mask = bubble_mask_editor.create_bubble_mask(cnt, img_shape)
        self.assertEqual(mask.shape, img_shape)
        self.assertGreater(np.count_nonzero(mask), 9000)
        self.assertEqual(mask[100, 100], 255)
        self.assertEqual(mask[10, 10], 0)

    def test_rasterize_cut_stroke_formats(self):
        mask_shape = (200, 200)

        # 1. Point list
        pts = [(50, 100), (150, 100)]
        m_pts = bubble_mask_editor.rasterize_cut_stroke(pts, mask_shape, thickness=4)
        self.assertEqual(m_pts.shape, mask_shape)
        self.assertGreater(np.count_nonzero(m_pts), 0)
        self.assertEqual(m_pts[100, 100], 255)

        # 2. Line segments
        segs = [[50, 50, 150, 150]]
        m_segs = bubble_mask_editor.rasterize_cut_stroke(segs, mask_shape, thickness=4)
        self.assertGreater(np.count_nonzero(m_segs), 0)
        self.assertEqual(m_segs[100, 100], 255)

        # 3. RGBA numpy array (st_canvas output format)
        rgba = np.zeros((200, 200, 4), dtype=np.uint8)
        rgba[95:105, 50:150, 3] = 255  # Alpha channel
        m_rgba = bubble_mask_editor.rasterize_cut_stroke(rgba, mask_shape, thickness=4)
        self.assertGreater(np.count_nonzero(m_rgba), 0)
        self.assertEqual(m_rgba[100, 100], 255)

    def test_sample_fill_color_c_bg(self):
        page = np.full((150, 150, 3), (250, 248, 245), dtype=np.uint8)
        mask = np.zeros((150, 150), dtype=np.uint8)
        cv2.circle(mask, (75, 75), 40, 255, -1)

        c_bg = bubble_mask_editor.sample_fill_color_c_bg(page, mask)
        self.assertEqual(len(c_bg), 3)
        self.assertAlmostEqual(c_bg[0], 250, delta=5)
        self.assertAlmostEqual(c_bg[1], 248, delta=5)
        self.assertAlmostEqual(c_bg[2], 245, delta=5)

    # --------------------------------------------------------------------------
    # TOOL 1: "Qaychi" (Scissors) Tests
    # --------------------------------------------------------------------------

    def _create_synthetic_merged_bubble(self):
        H, W = 400, 500
        page = np.full((H, W, 3), (255, 255, 255), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.uint8)

        # Left lobe at (160, 200), r=70
        cv2.circle(mask, (160, 200), 70, 255, -1)
        # Right lobe at (290, 200), r=70
        cv2.circle(mask, (290, 200), 70, 255, -1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = cnts[0]

        bubble = {
            "bubble_id": 1,
            "x0": 90, "y0": 130, "x1": 360, "y1": 270,
            "box": [90, 130, 360, 270],
            "original_text": "LEFT BUBBLE RIGHT BUBBLE",
            "clean_text": "LEFT BUBBLE RIGHT BUBBLE",
            "uzbek_translation": "",
            "confidence": 0.95,
            "shape_type": "oval",
            "contour": cnt.reshape(-1, 2).tolist(),
            "lines": [
                {'text': 'LEFT BUBBLE', 'x0': 120, 'y0': 190, 'x1': 190, 'y1': 210, 'cx': 155, 'cy': 200},
                {'text': 'RIGHT BUBBLE', 'x0': 260, 'y0': 190, 'x1': 330, 'y1': 210, 'cx': 295, 'cy': 200}
            ]
        }
        return page, bubble

    def test_scissors_clean_two_component_split(self):
        page, bubble = self._create_synthetic_merged_bubble()
        H, W = page.shape[:2]

        # Draw vertical cutting line through the waist at x=225 from y=100 to y=300
        stroke_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.line(stroke_mask, (225, 100), (225, 300), 255, 6)

        success, msg, new_bubbles, amb_lines = bubble_mask_editor.apply_scissors_split(
            bubble, stroke_mask, page
        )

        self.assertTrue(success, f"Scissors split failed: {msg}")
        self.assertEqual(len(new_bubbles), 2, "Must produce exactly 2 sub-bubbles")
        self.assertEqual(len(amb_lines), 0, "No ambiguous lines expected here")

        b1, b2 = new_bubbles[0], new_bubbles[1]
        self.assertIn("LEFT", b1["original_text"])
        self.assertIn("RIGHT", b2["original_text"])

        # Validate bounding boxes are separated
        self.assertLess(b1["x0"], b2["x0"])
        self.assertLess(b1["x1"], b2["x1"])

        # Validate contours are non-empty
        self.assertGreater(len(b1["contour"]), 5)
        self.assertGreater(len(b2["contour"]), 5)

    def test_scissors_rejection_incomplete_cut(self):
        page, bubble = self._create_synthetic_merged_bubble()
        H, W = page.shape[:2]

        # Cut ends halfway inside bubble (from top y=100 to middle y=190, doesn't exit bottom)
        stroke_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.line(stroke_mask, (225, 100), (225, 190), 255, 6)

        success, msg, new_bubbles, amb_lines = bubble_mask_editor.apply_scissors_split(
            bubble, stroke_mask, page
        )

        self.assertFalse(success, "Incomplete cut must be rejected")
        self.assertIn("faqat 1 ta bo'lak", msg)
        self.assertIsNone(new_bubbles)

    def test_scissors_rejection_multi_cut(self):
        page, bubble = self._create_synthetic_merged_bubble()
        H, W = page.shape[:2]

        # Two cuts dividing bubble into 3 pieces (at x=170 and x=280)
        stroke_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.line(stroke_mask, (170, 100), (170, 300), 255, 6)
        cv2.line(stroke_mask, (280, 100), (280, 300), 255, 6)

        success, msg, new_bubbles, amb_lines = bubble_mask_editor.apply_scissors_split(
            bubble, stroke_mask, page
        )

        self.assertFalse(success, "Multi-cut (>2 pieces) must be rejected")
        self.assertIn("2 tadan ko'p bo'lak", msg)
        self.assertIsNone(new_bubbles)

    def test_scissors_ambiguous_line_flagging(self):
        page, bubble = self._create_synthetic_merged_bubble()
        H, W = page.shape[:2]

        # Add a borderline line right at the cutting seam (cx=225)
        bubble["lines"].append({
            'text': 'BORDERLINE LINE',
            'x0': 210, 'y0': 195, 'x1': 240, 'y1': 215,
            'cx': 225, 'cy': 205
        })

        stroke_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.line(stroke_mask, (225, 100), (225, 300), 255, 6)

        success, msg, new_bubbles, amb_lines = bubble_mask_editor.apply_scissors_split(
            bubble, stroke_mask, page
        )

        self.assertTrue(success)
        self.assertEqual(len(amb_lines), 1, "The borderline line must be flagged as ambiguous")
        self.assertEqual(amb_lines[0]['text'], 'BORDERLINE LINE')

    # --------------------------------------------------------------------------
    # TOOL 2: "Aqlli mo'yqalam" (Smart Brush) Tests
    # --------------------------------------------------------------------------

    def test_smart_brush_preview_acceptance_and_rejection(self):
        H, W = 250, 250
        # Background page: light bubble area (245, 245, 245), dark line art (20, 20, 20),
        # and non-white background scenery (30, 80, 180)
        page = np.full((H, W, 3), (30, 80, 180), dtype=np.uint8)  # Blue/brown scenery
        # Bubble body + clipped tail area (white fill)
        page[50:180, 50:180] = [245, 245, 245]
        # Dark black ink border around bubble
        cv2.rectangle(page, (48, 48), (182, 182), (20, 20, 20), 3)

        # Existing clipped bubble mask only covers (50:140, 50:180) - bottom tail missed
        bubble_mask = np.zeros((H, W), dtype=np.uint8)
        bubble_mask[50:140, 50:180] = 255

        # User brush paints over missed tail (140:175, 80:120) and also spills over
        # the dark border (180:190) into the blue background (190:220)
        brush_mask = np.zeros((H, W), dtype=np.uint8)
        brush_mask[135:210, 75:130] = 255

        c_bg = [245, 245, 245]
        updated_mask, candidate_addition = bubble_mask_editor.compute_smart_brush_preview(
            page, bubble_mask, brush_mask, c_bg_bgr=c_bg, tolerance=30.0, min_lum=120.0
        )

        # 1. Accepted: missed tail white pixels (145:175, 80:120) must be included
        self.assertGreater(np.count_nonzero(candidate_addition[145:175, 80:120]), 500)

        # 2. Rejected: dark ink border at row 181 must NOT be in candidate addition
        self.assertEqual(np.count_nonzero(candidate_addition[181:184, 80:120]), 0)

        # 3. Rejected: blue scenery art (195:210, 80:120) must NOT be in candidate addition
        self.assertEqual(np.count_nonzero(candidate_addition[195:210, 80:120]), 0)

    def test_smart_brush_growth_application(self):
        H, W = 300, 300
        page = np.full((H, W, 3), (250, 250, 250), dtype=np.uint8)

        # Original circular bubble mask at (120, 120), r=50
        orig_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(orig_mask, (120, 120), 50, 255, -1)
        cnts, _ = cv2.findContours(orig_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        orig_cnt = cnts[0]

        bubble = {
            "bubble_id": 1,
            "x0": 70, "y0": 70, "x1": 170, "y1": 170,
            "box": [70, 70, 170, 170],
            "original_text": "GROW TEST",
            "clean_text": "GROW TEST",
            "confidence": 1.0,
            "contour": orig_cnt.reshape(-1, 2).tolist(),
            "tint_bgr": [250, 250, 250]
        }

        # Brush stroke extending downward to simulate adding a missed tail
        brush_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(brush_mask, (120, 190), 30, 255, -1)

        updated_bubble = bubble_mask_editor.apply_smart_brush_growth(
            bubble, page, brush_mask, tolerance=35.0, min_lum=110.0
        )

        # Verify bubble mask grew vertically
        self.assertGreater(updated_bubble["y1"], bubble["y1"])
        self.assertGreaterEqual(updated_bubble["y1"], 210)

        # Verify contour was updated
        new_cnt = np.array(updated_bubble["contour"], dtype=np.int32)
        self.assertGreater(cv2.contourArea(new_cnt), cv2.contourArea(orig_cnt))

    def test_quick_selection_single_click_growth(self):
        """Single click inside a missed tail grows to fill the entire region and stops at ink edges."""
        H, W = 250, 250
        page = np.full((H, W, 3), (40, 60, 140), dtype=np.uint8)  # Background art

        # Bubble interior + missed tail: pure white (255, 255, 255)
        page[50:180, 50:180] = [255, 255, 255]
        # Solid dark ink border
        cv2.rectangle(page, (48, 48), (182, 182), (15, 15, 15), 3)

        # Existing clipped bubble mask only covers top portion
        bubble_mask = np.zeros((H, W), dtype=np.uint8)
        bubble_mask[50:130, 50:180] = 255

        # Single click dot at (x=100, y=155) - exactly 3x3 pixels
        click_mask = np.zeros((H, W), dtype=np.uint8)
        click_mask[154:157, 99:102] = 255

        c_bg = [255, 255, 255]
        updated_mask, candidate_addition = bubble_mask_editor.compute_smart_brush_preview(
            page, bubble_mask, click_mask, c_bg_bgr=c_bg, tolerance=30.0, min_lum=110.0
        )

        # 1. Whole missed tail (130:180, 52:178) should be auto-grown from the single click!
        grown_area = np.count_nonzero(candidate_addition[132:178, 52:178])
        self.assertGreater(grown_area, 1800, "Single click must auto-expand to fill entire edge-bounded tail")

        # 2. Must stop strictly at the ink border (row 181..183) without leaking
        self.assertEqual(np.count_nonzero(candidate_addition[181:185, :]), 0)

        # 3. Outside scenery must remain 100% unselected
        self.assertEqual(np.count_nonzero(candidate_addition[190:, :]), 0)

    def test_quick_selection_drag_multiple_seeds(self):
        """A rough drag samples multiple seeds along the path and unions their edge-bounded regions."""
        H, W = 300, 300
        page = np.full((H, W, 3), (50, 70, 90), dtype=np.uint8)

        # Two adjacent white bubble chambers separated by white neck
        page[50:200, 50:130] = [250, 250, 250]
        page[120:200, 130:220] = [250, 250, 250]
        cv2.rectangle(page, (48, 48), (222, 202), (20, 20, 20), 3)

        # Mask only covers chamber 1
        bubble_mask = np.zeros((H, W), dtype=np.uint8)
        bubble_mask[50:200, 50:130] = 255

        # Rough diagonal drag stroke into chamber 2
        stroke_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.line(stroke_mask, (135, 140), (200, 180), 255, 3)

        updated_mask, candidate_addition = bubble_mask_editor.compute_smart_brush_preview(
            page, bubble_mask, stroke_mask, c_bg_bgr=[250, 250, 250], tolerance=30.0, seed_spacing=15
        )

        # Chamber 2 should be substantially filled
        ch2_area = np.count_nonzero(candidate_addition[125:198, 132:218])
        self.assertGreater(ch2_area, 2500, "Rough drag must union edge-bounded growth from multiple seeds")

        # Must not cross the dark rectangle border
        self.assertEqual(np.count_nonzero(candidate_addition[203:, :]), 0)

    # --------------------------------------------------------------------------
    # Persistent Learning & Quality Gates Integration
    # --------------------------------------------------------------------------

    def test_record_mask_correction_and_threshold_tuning(self):
        initial_scissors_thresh = quality_gates.get_learned_scissors_thresholds()["constriction_ratio"]
        initial_brush_tol = quality_gates.get_learned_brush_tolerances()["color_tolerance"]

        # 1. Record Scissors Correction
        bubble_mask_editor.record_mask_correction(
            page_name="test_page.png",
            tool_name="scissors",
            bubble_id=5,
            original_contour=[[10, 10], [50, 10], [50, 50], [10, 50]],
            corrected_contours=[
                [[10, 10], [25, 10], [25, 50], [10, 50]],
                [[30, 10], [50, 10], [50, 50], [30, 50]]
            ],
            metadata={"reason": "merged_two_speakers"}
        )

        # Verify saved in JSON file
        corrections = bubble_mask_editor.load_mask_corrections(self.test_memory_file)
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["tool_used"], "scissors")
        self.assertEqual(corrections[0]["bubble_id"], 5)

        # Verify quality gate event dispatched
        log = quality_gates.get_quality_gate_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["pattern_name"], "touching_bubbles_merged")
        self.assertEqual(log[0]["action_taken"], "manual_scissors_applied")

        # Verify scissors threshold tuned dynamically
        tuned_scissors_thresh = quality_gates.get_learned_scissors_thresholds()["constriction_ratio"]
        self.assertGreaterEqual(tuned_scissors_thresh, initial_scissors_thresh)

        # 2. Record Smart Brush Correction
        bubble_mask_editor.record_mask_correction(
            page_name="test_page.png",
            tool_name="smart_brush",
            bubble_id=8,
            original_contour=[[10, 10], [50, 10], [50, 50], [10, 50]],
            corrected_contours=[[10, 10], [60, 10], [60, 50], [10, 50]],
            metadata={"tolerance": 38.0}
        )

        corrections2 = bubble_mask_editor.load_mask_corrections(self.test_memory_file)
        self.assertEqual(len(corrections2), 2)
        self.assertEqual(corrections2[1]["tool_used"], "smart_brush")

        log2 = quality_gates.get_quality_gate_log()
        self.assertEqual(len(log2), 2)
        self.assertEqual(log2[1]["pattern_name"], "clipped_bubble_edge")
        self.assertEqual(log2[1]["action_taken"], "manual_smart_brush_applied")

        # Verify brush tolerance tuned dynamically
        tuned_brush_tol = quality_gates.get_learned_brush_tolerances()["color_tolerance"]
        self.assertGreaterEqual(tuned_brush_tol, initial_brush_tol)

    # --------------------------------------------------------------------------
    # Real-world Image Fixture Tests
    # --------------------------------------------------------------------------

    def test_fixture_page4_jameson_bubble_smart_brush(self):
        page4_path = "samples/page_4.png"
        if not os.path.exists(page4_path):
            self.skipTest(f"{page4_path} not found")

        page_bgr = cv2.imread(page4_path)
        self.assertIsNotNone(page_bgr)
        H, W = page_bgr.shape[:2]

        # Jameson top-right bubble approximate location on page 4
        # Create a clipped seed contour inside the bubble
        seed_cnt = np.array([
            [1880, 150], [2020, 150], [2020, 250], [1880, 250]
        ], dtype=np.int32)
        seed_bubble = {
            "bubble_id": 1,
            "x0": 1880, "y0": 150, "x1": 2020, "y1": 250,
            "box": [1880, 150, 2020, 250],
            "original_text": "JAMESON DIALOGUE",
            "clean_text": "JAMESON DIALOGUE",
            "confidence": 0.98,
            "contour": seed_cnt.tolist(),
            "tint_bgr": [250, 250, 250]
        }

        # Brush stroke expanding toward the authentic bubble edge
        brush_stroke = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(brush_stroke, (1850, 200), 40, 255, -1)

        grown_bubble = bubble_mask_editor.apply_smart_brush_growth(
            seed_bubble, page_bgr, brush_stroke, tolerance=40.0, min_lum=110.0
        )

        # The bounding box should expand to the left without error
        self.assertLessEqual(grown_bubble["x0"], seed_bubble["x0"])

    def test_fixture_page20_ghost_bubble_split(self):
        page20_path = "samples/page_20.png"
        if not os.path.exists(page20_path):
            self.skipTest(f"{page20_path} not found")

        page_bgr = cv2.imread(page20_path)
        self.assertIsNotNone(page_bgr)
        H, W = page_bgr.shape[:2]

        # Synthetic merged bubble over a real page region (overlapping lobes)
        test_cx, test_cy = W // 2, H // 3
        merged_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(merged_mask, (test_cx - 50, test_cy), (70, 50), 0, 0, 360, 255, -1)
        cv2.ellipse(merged_mask, (test_cx + 50, test_cy), (70, 50), 0, 0, 360, 255, -1)

        cnts, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        merged_cnt = max(cnts, key=cv2.contourArea)

        bx, by, bw, bh = cv2.boundingRect(merged_cnt)
        bubble = {
            "bubble_id": 2,
            "box": [bx, by, bx + bw, by + bh],
            "x0": bx, "y0": by, "x1": bx + bw, "y1": by + bh,
            "original_text": "FIRST SPEAKER SECOND SPEAKER",
            "clean_text": "FIRST SPEAKER SECOND SPEAKER",
            "confidence": 0.99,
            "contour": merged_cnt.reshape(-1, 2).tolist(),
            "lines": [
                {'text': 'FIRST SPEAKER', 'x0': test_cx - 100, 'y0': test_cy - 10, 'x1': test_cx - 20, 'y1': test_cy + 10,
                 'cx': test_cx - 50, 'cy': test_cy},
                {'text': 'SECOND SPEAKER', 'x0': test_cx + 20, 'y0': test_cy - 10, 'x1': test_cx + 100, 'y1': test_cy + 10,
                 'cx': test_cx + 50, 'cy': test_cy}
            ]
        }

        # Cut stroke between the two lobes
        stroke_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.line(stroke_mask, (test_cx, test_cy - 70), (test_cx, test_cy + 70), 255, 6)

        success, msg, split_bubbles, amb = bubble_mask_editor.apply_scissors_split(
            bubble, stroke_mask, page_bgr
        )

        self.assertTrue(success, f"Split on page 20 fixture failed: {msg}")
        self.assertEqual(len(split_bubbles), 2)
        self.assertIn("FIRST", split_bubbles[0]["original_text"])
        self.assertIn("SECOND", split_bubbles[1]["original_text"])

    def test_smart_brush_prevents_runaway_panel_leak(self):
        """Verify that smart brush with tolerance=30 and size=30 does not leak into panel artwork."""
        H, W = 800, 800
        # Comic page with light-colored panel background (e.g. wall at [210, 220, 205])
        page = np.full((H, W, 3), (210, 220, 205), dtype=np.uint8)
        # Panel black border
        cv2.rectangle(page, (50, 50), (750, 750), (10, 10, 10), 4)

        # Bubble at (200, 200) radius 60 with pure white fill
        cv2.circle(page, (200, 200), 60, (255, 255, 255), -1)
        cv2.circle(page, (200, 200), 60, (15, 15, 15), 3)

        # Bubble mask (slightly clipped on the right)
        bubble_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(bubble_mask, (190, 200), 50, 255, -1)
        cnts, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        orig_cnt = cnts[0]
        bx, by, bw, bh = cv2.boundingRect(orig_cnt)

        bubble = {
            "bubble_id": 9,
            "x0": bx, "y0": by, "x1": bx + bw, "y1": by + bh,
            "box": [bx, by, bx + bw, by + bh],
            "original_text": "MJ DIALOGUE",
            "contour": orig_cnt.reshape(-1, 2).tolist(),
            "tint_bgr": [255, 255, 255]
        }

        # Brush stroke of size 30 near the bubble edge (240, 200)
        brush_stroke = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(brush_stroke, (235, 200), 15, 255, -1)

        updated_bubble = bubble_mask_editor.apply_smart_brush_growth(
            bubble, page, brush_stroke, tolerance=30.0, min_lum=110.0
        )

        # Bubble should have grown slightly to include the white circle
        new_cnt = np.array(updated_bubble["contour"], dtype=np.int32)
        new_area = cv2.contourArea(new_cnt)
        self.assertGreater(new_area, cv2.contourArea(orig_cnt))

        # BUT must NEVER have leaked across the panel!
        # Entire panel area is 700x700 = 490,000. Bounded bubble area should be < 25,000!
        self.assertLess(new_area, 25000, "Mask must not leak across panel art!")
        self.assertLess(updated_bubble["x1"], 350, "Bounding box must stay strictly local to the bubble!")
        self.assertLess(updated_bubble["y1"], 350, "Bounding box must stay strictly local to the bubble!")


if __name__ == "__main__":
    unittest.main()
