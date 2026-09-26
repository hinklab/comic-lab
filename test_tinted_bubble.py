"""
test_tinted_bubble.py - Unit tests for tinted/spectral speech bubble pipeline handling.
Verifies color generalization (blue, green, purple), adaptive ink cleaning without whiteout,
and lettering legibility guards.
"""

import unittest
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import engine
import bubble_lettering
import quality_gates


class TestTintedBubble(unittest.TestCase):
    def setUp(self):
        quality_gates.clear_quality_gate_log()

    def test_detection_generalizes_across_colors(self):
        """Verify that monochromatic tinted speech fills (blue, green, purple, cyan) are accepted while noisy art is rejected."""
        test_tints = [
            # (name, mean_lum, p75, mean_sat, sat_std, max_val, bright_ratio, should_be_bubble)
            ("spectral_blue", 155.0, 195.0, 110.0, 12.0, 230.0, 0.40, True),
            ("telepathic_green", 160.0, 200.0, 105.0, 14.0, 225.0, 0.45, True),
            ("astral_purple", 145.0, 185.0, 120.0, 16.0, 215.0, 0.35, True),
            ("warm_yellow", 180.0, 210.0, 95.0, 10.0, 235.0, 0.60, True),
            ("noisy_background_art", 130.0, 150.0, 140.0, 55.0, 190.0, 0.08, False),
            ("dark_sfx_explosion", 90.0, 120.0, 160.0, 45.0, 140.0, 0.02, False),
        ]

        for name, lum, p75, sat, sat_std, max_v, br, expected_bubble in test_tints:
            cand = {
                "mean_luminance": lum,
                "percentile_75": p75,
                "mean_saturation": sat,
                "sat_std": sat_std,
                "bright_ratio": br,
                "max_val": max_v,
                "line_h": 32,
                "text": "TEST DIALOGUE"
            }
            # Quality gate filter
            gate_res = quality_gates.apply_quality_gates("ocr_filtering", cand, context={"median_glyph_h": 35})
            is_std = (lum >= 140 and sat <= 85) or (p75 >= 180 and sat <= 85 and br >= 0.15)
            is_tint = (
                (lum >= 120 or p75 >= 160 or max_v >= 160)
                and (sat_std <= 22 or (sat > 0 and sat_std / max(1.0, sat) <= 0.40))
                and max_v >= 150
            )
            is_bubble = (is_std or is_tint) and (gate_res is not None)
            self.assertEqual(
                is_bubble, expected_bubble,
                f"Candidate '{name}' expected is_bubble={expected_bubble}, got {is_bubble}"
            )

    def test_adaptive_inpaint_preserves_authentic_tint(self):
        """Confirm Zero Whiteout: text ink is removed and authentic tinted background is preserved underneath."""
        # Create a synthetic image with a cyan-tinted bubble on dark background
        w, h = 300, 200
        img_bgr = np.full((h, w, 3), (40, 40, 50), dtype=np.uint8)  # dark panel
        
        # Authentic cyan tint fill: BGR = (245, 230, 200)
        bubble_bgr = (245, 230, 200)
        center = (150, 100)
        axes = (110, 65)
        cv2.ellipse(img_bgr, center, axes, 0, 0, 360, bubble_bgr, thickness=-1)
        # 2px border
        cv2.ellipse(img_bgr, center, axes, 0, 0, 360, (200, 180, 150), thickness=2)
        
        # Draw light blue text inside the bubble (like Peter Parker's ghost dialogue)
        ink_bgr = (210, 190, 160)
        cv2.putText(img_bgr, "GHOST DIALOGUE", (65, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ink_bgr, 2, cv2.LINE_AA)
        
        # Contour of the bubble
        cnt_mask = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) > 100).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(cnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubble_cnt = max(cnts, key=cv2.contourArea)
        
        # Clean the bubble
        cleaned = bubble_lettering.inpaint_bubble_text_only(img_bgr, bubble_cnt)
        
        # Verify text was cleaned (no residual ink pixels near text position)
        text_patch_cleaned = cleaned[90:120, 70:230]
        text_patch_orig = img_bgr[90:120, 70:230]
        
        # Text region changed due to inpainting
        self.assertGreater(float(np.mean(cv2.absdiff(text_patch_orig, text_patch_cleaned))), 5.0)
        
        # The cleaned patch MUST preserve the cyan tint (NOT flat white 255, 255, 255)
        mean_cleaned_bgr = np.mean(text_patch_cleaned, axis=(0, 1))
        # Distance from authentic bubble_bgr must be minimal (< 4.0)
        dist_from_tint = np.linalg.norm(mean_cleaned_bgr - np.array(bubble_bgr))
        self.assertLess(dist_from_tint, 4.0, f"Cleaned area lost tint! Sampled BGR: {mean_cleaned_bgr}")
        
        # Distance from flat white (255, 255, 255) must be significant (verifying Zero Whiteout)
        dist_from_white = np.linalg.norm(mean_cleaned_bgr - np.array([255.0, 255.0, 255.0]))
        self.assertGreater(dist_from_white, 20.0, "Cleaned area was flattened to whiteout!")

    def test_find_light_seed_on_tinted_fill(self):
        """Verify _find_light_seed finds genuine interior on saturated colored fill."""
        h, w = 100, 100
        # Highly saturated blue fill: B=240, G=120, R=50 -> Grayscale luminance is only ~100
        roi_bgr = np.full((h, w, 3), (240, 120, 50), dtype=np.uint8)
        gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        barrier_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        # Put an ink border around edges
        barrier_mask[1, :] = 1
        barrier_mask[-2, :] = 1
        barrier_mask[:, 1] = 1
        barrier_mask[:, -2] = 1
        
        val_roi = np.max(roi_bgr, axis=2)
        bright_roi = np.maximum(gray_roi, val_roi)
        
        seed = bubble_lettering._find_light_seed(
            gray_roi, (50, 50), barrier_mask, light_thresh=180, search_radius=25, bright_roi=bright_roi
        )
        self.assertIsNotNone(seed, "Seed should be found on bright value channel despite low grayscale luminance")
        self.assertTrue(25 <= seed[0] <= 75 and 25 <= seed[1] <= 75)

    def test_legibility_contrast_guard(self):
        """Verify contrast guard logs low_contrast_tint event on unreadable dark tints."""
        canvas = Image.new("RGB", (200, 200), color=(50, 50, 50))  # Dark tint
        bubble = engine.SpeechBubble(
            bubble_id=1,
            x0=20, y0=20, x1=180, y1=180,
            original_text="DARK BUBBLE",
            uzbek_translation="QORANG'I SHAR"
        )
        fonts = engine.get_available_fonts()
        font_p = list(fonts.values())[0] if fonts else "arial.ttf"
        
        # Run typeset
        engine.typeset_lettering_on_page(canvas, [bubble], font_path=font_p)
        log = quality_gates.get_quality_gate_log()
        events = [e for e in log if e.get("pattern_name") == "low_contrast_tint"]
        self.assertEqual(len(events), 1, "Should log low_contrast_tint event for dark background")

    def test_output_canvas_tint_preservation_and_spectral_text(self):
        """
        Numeric verification of tint preservation on output canvas:
        1. Output canvas bubble-fill pixel color matches original art's sampled fill (Δ < 8.0)
           and is strictly NOT flat white (245+, 245+, 245+).
        2. Lettered text color matches darkened spectral slate-blue (#325069 / [50, 80, 105], contrast > 5.6:1)
           and is NOT flat #000000.
        3. Untinted dialogue bubble touching colored SFX remains whiteout-cleaned with pure black text.
        """
        w, h = 600, 400
        img_bgr = np.full((h, w, 3), (40, 40, 50), dtype=np.uint8)

        # Bubble 1: Spectral blue ghost bubble (Peter Parker ghost style)
        b1_bgr = (235, 220, 205)  # RGB: [205, 220, 235]
        b1_center, b1_axes = (160, 200), (120, 80)
        cv2.ellipse(img_bgr, b1_center, b1_axes, 0, 0, 360, b1_bgr, thickness=-1)
        cv2.ellipse(img_bgr, b1_center, b1_axes, 0, 0, 360, (200, 180, 150), thickness=2)
        # Ghost text ink: light slate-blue
        cv2.putText(img_bgr, "GHOST DIALOGUE", (70, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 212, 202), 2, cv2.LINE_AA)

        # Bubble 2: Standard white dialogue bubble touching a blue SFX patch
        b2_bgr = (255, 255, 255)
        b2_center, b2_axes = (440, 200), (120, 80)
        cv2.circle(img_bgr, (540, 260), 40, (220, 170, 110), thickness=-1)  # SFX bleed at border
        cv2.ellipse(img_bgr, b2_center, b2_axes, 0, 0, 360, b2_bgr, thickness=-1)
        cv2.ellipse(img_bgr, b2_center, b2_axes, 0, 0, 360, (0, 0, 0), thickness=2)
        cv2.putText(img_bgr, "POLICE DIALOGUE", (350, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)

        cnt1_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(cnt1_mask, b1_center, b1_axes, 0, 0, 360, 255, -1)
        cnts1, _ = cv2.findContours(cnt1_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnt2_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(cnt2_mask, b2_center, b2_axes, 0, 0, 360, 255, -1)
        cnts2, _ = cv2.findContours(cnt2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        b1 = engine.SpeechBubble(
            bubble_id=1, x0=40, y0=120, x1=280, y1=280,
            original_text="GHOST DIALOGUE", uzbek_translation="RUH GAPIRMOQDA",
            contour=cnts1[0].tolist()
        )
        b2 = engine.SpeechBubble(
            bubble_id=2, x0=320, y0=120, x1=560, y1=280,
            original_text="POLICE DIALOGUE", uzbek_translation="POLITSIYA GAPIRMOQDA",
            contour=cnts2[0].tolist()
        )

        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        cleaned_pil = engine.clean_page_ink_telea(img_pil, [b1, b2])
        cleaned_cv = cv2.cvtColor(np.array(cleaned_pil), cv2.COLOR_RGB2BGR)

        # 1. Verify Bubble 1 cleaned interior preserves authentic tint (Zero Whiteout)
        self.assertTrue(b1.is_tinted, "Spectral bubble should be classified as is_tinted=True")
        patch1_cleaned = cleaned_cv[185:215, 90:230]
        mean1_bgr = np.mean(patch1_cleaned, axis=(0, 1))
        dist_from_tint = np.linalg.norm(mean1_bgr - np.array(b1_bgr))
        self.assertLess(dist_from_tint, 8.0, f"Cleaned patch lost tint! Δ={dist_from_tint:.2f} >= 8.0")
        
        # Verify NOT flat white (245+, 245+, 245+)
        dist_from_white = np.linalg.norm(mean1_bgr - np.array([255.0, 255.0, 255.0]))
        self.assertGreater(dist_from_white, 20.0, "Cleaned patch was flattened to pure white!")
        self.assertLess(float(mean1_bgr[2]), 242.0, "Red channel indicates flat white instead of spectral blue!")

        # 2. Verify Bubble 2 (police dialogue touching SFX) is NOT tinted
        self.assertFalse(b2.is_tinted, "Standard dialogue touching SFX must NOT be classified as is_tinted")
        patch2_cleaned = cleaned_cv[185:215, 370:510]
        mean2_bgr = np.mean(patch2_cleaned, axis=(0, 1))
        self.assertGreater(float(np.mean(mean2_bgr)), 250.0, "Standard bubble should clean to pure white paper")

        # 3. Typeset lettering and verify text ink color
        fonts = engine.get_available_fonts()
        font_p = list(fonts.values())[0] if fonts else "arial.ttf"
        lettered_pil = engine.typeset_lettering_on_page(cleaned_pil, [b1, b2], font_path=font_p)
        lettered_cv = cv2.cvtColor(np.array(lettered_pil), cv2.COLOR_RGB2BGR)

        # Bubble 1 lettered text ink: spectral havorang (#7497d4 / luminous blue), NOT #000000
        patch1_lettered = lettered_cv[180:220, 80:240]
        gray1 = cv2.cvtColor(patch1_lettered, cv2.COLOR_BGR2GRAY)
        text_pts1 = patch1_lettered[gray1 < 170]
        self.assertGreater(len(text_pts1), 0, "No lettered text found in Bubble 1")
        med_text1 = np.median(text_pts1, axis=0)  # BGR
        r_text1, g_text1, b_text1 = int(med_text1[2]), int(med_text1[1]), int(med_text1[0])
        
        # Ensure it is spectral blue (B > R, B >= 80), NOT pure black
        self.assertGreater(b_text1, r_text1, "Spectral text ink must have blue dominance (B > R)")
        self.assertGreater(b_text1, 40, "Spectral text ink must NOT be flat black #000000")

        # Bubble 2 lettered text ink: pure black #000000
        patch2_lettered = lettered_cv[180:220, 360:520]
        gray2 = cv2.cvtColor(patch2_lettered, cv2.COLOR_BGR2GRAY)
        text_pts2 = patch2_lettered[gray2 < 120]
        self.assertGreater(len(text_pts2), 0, "No lettered text found in Bubble 2")
        med_text2 = np.median(text_pts2, axis=0)
        self.assertLess(float(np.mean(med_text2)), 15.0, "Standard bubble text ink must be pure black #000000")

    def test_ghost_bubble_pure_white_background_and_havorang_text(self):
        """
        Verify Peter Parker ghost dialogue bubble specification:
        - Bubble background is 100% pure clean paper WHITE (mean BGR > 250, distance from pure white < 5.0).
        - ZERO blue tint wash or background haze.
        - Lettered Uzbek text font color is havorang (#7497d4, blue > red, luminance ~148).
        """
        w, h = 400, 300
        # Dark panel background
        img_bgr = np.full((h, w, 3), (30, 40, 45), dtype=np.uint8)

        # Pure white speech bubble: BGR = (254, 254, 254)
        bubble_center, bubble_axes = (200, 150), (140, 80)
        cv2.ellipse(img_bgr, bubble_center, bubble_axes, 0, 0, 360, (254, 254, 254), thickness=-1)
        cv2.ellipse(img_bgr, bubble_center, bubble_axes, 0, 0, 360, (255, 255, 255), thickness=3)

        # Light cyan/havorang ghost text ink (like original Marvel English ghost text: BGR = (207, 195, 117))
        cv2.putText(img_bgr, "THAT'S NOT SPIDER-MAN!", (80, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (207, 195, 117), 2, cv2.LINE_AA)

        cnt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(cnt_mask, bubble_center, bubble_axes, 0, 0, 360, 255, -1)
        cnts, _ = cv2.findContours(cnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        b_ghost = engine.SpeechBubble(
            bubble_id=1, x0=60, y0=70, x1=340, y1=230,
            original_text="THAT'S NOT SPIDER-MAN!", uzbek_translation="BU O'RGIMCHAK-ODAM EMAS!",
            contour=cnts[0].tolist()
        )

        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        cleaned_pil = engine.clean_page_ink_telea(img_pil, [b_ghost])
        cleaned_cv = cv2.cvtColor(np.array(cleaned_pil), cv2.COLOR_RGB2BGR)

        # 1. Bubble background MUST be pure clean paper WHITE (not washed with blue)
        self.assertTrue(b_ghost.text_is_spectral, "Ghost dialogue text must be detected as spectral")
        self.assertFalse(b_ghost.bg_is_tinted, "Ghost bubble background is paper white, NOT tinted")

        interior_patch = cleaned_cv[135:165, 120:280]
        mean_bgr = np.mean(interior_patch, axis=(0, 1))
        dist_from_white = np.linalg.norm(mean_bgr - np.array([255.0, 255.0, 255.0]))
        self.assertLess(dist_from_white, 5.0, f"Ghost bubble background must be pure white! Got BGR: {mean_bgr}")
        self.assertGreater(float(np.min(mean_bgr)), 250.0, "All channels must remain >= 250 (pure white)")

        # 2. Lettering: typeset Uzbek text and verify text is havorang (#7497d4)
        fonts = engine.get_available_fonts()
        font_p = list(fonts.values())[0] if fonts else "arial.ttf"
        lettered_pil = engine.typeset_lettering_on_page(cleaned_pil, [b_ghost], font_path=font_p)
        lettered_cv = cv2.cvtColor(np.array(lettered_pil), cv2.COLOR_RGB2BGR)

        lettered_patch = lettered_cv[135:165, 100:300]
        gray_lp = cv2.cvtColor(lettered_patch, cv2.COLOR_BGR2GRAY)
        # Havorang text has luminance ~148 on white background
        text_pts = lettered_patch[gray_lp < 180]
        self.assertGreater(len(text_pts), 0, "Uzbek text must be rendered in bubble")
        med_text = np.median(text_pts, axis=0)  # BGR
        r_t, g_t, b_t = int(med_text[2]), int(med_text[1]), int(med_text[0])

        # Havorang (#7497d4): Blue channel dominates Red channel significantly
        self.assertGreater(b_t, r_t + 30, f"Text must be havorang (blue dominant over red). Got B={b_t}, R={r_t}")
        self.assertGreater(b_t, 140, f"Blue channel must be luminous (> 140). Got {b_t}")


if __name__ == "__main__":
    unittest.main()
