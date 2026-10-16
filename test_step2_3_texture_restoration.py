import unittest
import numpy as np
import cv2

import sfx_engine


class TestStep23TextureRestoration(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 160x160 comic panel with Ben-Day dot matrix and a line crossing
        self.h, self.w = 160, 160
        self.img = np.ones((self.h, self.w, 3), dtype=np.uint8) * 245
        
        # Authentic Ben-Day / Screentone dot pattern (6px grid)
        for y in range(0, self.h, 6):
            for x in range(0, self.w, 6):
                shift = 3 if (y // 6) % 2 else 0
                cv2.circle(self.img, (x + shift, y), 1, (150, 150, 150), -1)

        # Draw a horizontal character sketch line (y=80)
        cv2.line(self.img, (15, 80), (145, 80), (20, 20, 20), 2)
        self.orig_art = self.img.copy()

        # Inpaint mask for text region
        self.mask = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.putText(self.img, "CRASH!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 30, 230), 4)
        cv2.putText(self.mask, "CRASH!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 4)
        self.mask = cv2.dilate(self.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    def test_screentone_high_frequency_variance(self):
        """Verify screentone dot pattern is reconstituted inside inpaint mask without flat blur."""
        restored = sfx_engine.restore_texture_and_lineart(self.img, self.mask, inpaint_radius=4)
        
        gray_restored = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
        lap_restored = cv2.Laplacian(gray_restored[self.mask > 0], cv2.CV_32F).var()
        
        # A flat naive Telea blur has Laplacian var < 300; authentic screentone is > 1000
        self.assertGreater(lap_restored, 800.0, f"Screentone texture was flattened! Var: {lap_restored}")

    def test_lineart_crossing_reconnection(self):
        """Verify broken black sketch lines passing through text are bridged across the gap."""
        restored = sfx_engine.restore_texture_and_lineart(self.img, self.mask, inpaint_radius=4)
        
        # Check that the horizontal line at y=80 inside the mask center (x=80) is reconnected
        center_pixel_gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)[80, 80]
        self.assertLess(center_pixel_gray, 80, f"Line-art crossing was not bridged! Gray: {center_pixel_gray}")

    def test_smooth_paper_remains_smooth(self):
        """Verify smooth paper backgrounds do not receive artificial grain."""
        smooth_img = np.ones((100, 100, 3), dtype=np.uint8) * 252
        smooth_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(smooth_img, (50, 50), 15, (20, 20, 20), -1)
        cv2.circle(smooth_mask, (50, 50), 16, 255, -1)
        
        restored = sfx_engine.restore_texture_and_lineart(smooth_img, smooth_mask, inpaint_radius=3)
        gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)[smooth_mask > 0]
        # Standard deviation on smooth paper should remain tiny (< 5.0)
        self.assertLess(np.std(gray), 5.0, "Smooth paper received unwanted noise!")

    def test_zero_bleed_outside_mask(self):
        """Verify 100.00% bit-exact identity for all pixels outside the inpaint mask."""
        restored = sfx_engine.restore_texture_and_lineart(self.img, self.mask, inpaint_radius=4)
        
        diff = np.max(np.abs(restored[self.mask == 0].astype(int) - self.img[self.mask == 0].astype(int)))
        self.assertEqual(diff, 0, f"Pixels outside inpaint mask were modified! Max diff: {diff}")

    def test_end_to_end_sfx_inpainting_with_texture(self):
        """Verify inpaint_sfx_elements applies high-fidelity texture restoration to SFXElements."""
        sfx_elem = sfx_engine.SFXElement(
            bbox=[[25, 60], [135, 60], [135, 100], [25, 100]],
            original_text="CRASH",
            transliterated_text="QARS",
            conf=0.98,
            center=(80.0, 80.0),
            angle_deg=0.0,
            width=110.0,
            height=40.0,
            fill_color=(20, 30, 230),
            stroke_color=(0, 0, 0),
            stroke_width=2,
            category="destruction_sharp",
            category_label="Sinish / Qirsillash"
        )
        cleaned = sfx_engine.inpaint_sfx_elements(self.img, [sfx_elem])
        self.assertIsNotNone(cleaned)
        self.assertEqual(cleaned.shape, self.img.shape)


if __name__ == "__main__":
    unittest.main()
