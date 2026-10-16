import unittest
import numpy as np
import cv2

import bubble_lettering


class TestStep21ZeroBleedMasking(unittest.TestCase):
    def setUp(self):
        # Create a synthetic comic panel (140x160) with rich background art
        self.h, self.w = 140, 160
        self.img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        # Background art: gradient and character line art
        for y in range(self.h):
            for x in range(self.w):
                self.img[y, x] = [int((x * 1.5) % 255), int((y * 1.8) % 255), 160]

        # Draw a character's face/hair silhouette right next to where the bubble will be
        cv2.circle(self.img, (25, 110), 20, (30, 40, 180), -1)
        cv2.line(self.img, (10, 80), (45, 130), (10, 10, 10), 3)

        # Bubble geometry: Ellipse center (85, 60), axes (45, 28) with a tail pointing to (35, 105)
        self.bubble_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.ellipse(self.bubble_mask, (85, 60), (45, 28), 0, 0, 360, 255, -1)
        
        # Add a pointed tail: polygon from bubble bottom to character
        tail_poly = np.array([[70, 84], [85, 84], [35, 105]], dtype=np.int32)
        cv2.fillPoly(self.bubble_mask, [tail_poly], 255)

        cnts, _ = cv2.findContours(self.bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cnt = max(cnts, key=cv2.contourArea)

        # Draw authentic white interior
        self.img[self.bubble_mask > 0] = (252, 252, 252)

        # Draw dark artist stroke (2px)
        self.stroke_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.drawContours(self.stroke_mask, [self.cnt], -1, 255, thickness=2)
        self.img[self.stroke_mask > 0] = (15, 15, 15)

        # Add comic dialogue text inside the bubble
        cv2.putText(self.img, "STOP RIGHT", (58, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (12, 12, 12), 1)
        cv2.putText(self.img, "THERE!", (68, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (12, 12, 12), 1)

    def test_zero_bleed_outside_bubble(self):
        """Verify 100.00% bit-exact identity for all pixels outside the bubble."""
        cleaned = bubble_lettering.inpaint_batch_bubbles(self.img, [self.cnt])
        
        outside = (self.bubble_mask == 0)
        diff_outside = np.abs(cleaned[outside].astype(int) - self.img[outside].astype(int))
        max_diff = np.max(diff_outside) if diff_outside.size > 0 else 0
        self.assertEqual(max_diff, 0, f"Background art was modified! Max pixel diff: {max_diff}")

    def test_stroke_integrity_preservation(self):
        """Verify the artist's black stroke boundary is 100.00% bit-exact locked."""
        cleaned = bubble_lettering.inpaint_batch_bubbles(self.img, [self.cnt])
        
        diff_stroke = np.abs(cleaned[self.stroke_mask > 0].astype(int) - self.img[self.stroke_mask > 0].astype(int))
        max_diff = np.max(diff_stroke) if diff_stroke.size > 0 else 0
        self.assertEqual(max_diff, 0, f"Bubble border stroke was modified! Max pixel diff: {max_diff}")

    def test_interior_text_removal(self):
        """Verify dark text is cleanly eliminated from the safe interior."""
        cleaned = bubble_lettering.inpaint_batch_bubbles(self.img, [self.cnt])
        
        # Check interior region (inside bubble and away from stroke)
        k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        eroded = cv2.erode(self.bubble_mask, k_in)
        
        interior_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)[eroded > 0]
        dark_specks = np.count_nonzero(interior_gray < 160)
        self.assertEqual(dark_specks, 0, f"Residual dark text ink remained in interior: {dark_specks} pixels")

    def test_tail_tip_and_speaker_art_preservation(self):
        """Verify pointed tail tip and speaker artwork touching the tip remain untouched."""
        cleaned = bubble_lettering.inpaint_batch_bubbles(self.img, [self.cnt])
        
        # Character face region at (25, 110)
        char_region_orig = self.img[90:130, 10:50]
        char_region_clean = cleaned[90:130, 10:50]
        
        diff = np.max(np.abs(char_region_clean.astype(int) - char_region_orig.astype(int)))
        self.assertEqual(diff, 0, f"Speaker character art at tail tip was touched! Max diff: {diff}")

    def test_tinted_bubble_color_fidelity(self):
        """Verify vintage tinted paper bubbles retain authentic paper background color."""
        tinted_img = self.img.copy()
        # Parchment tint: B=215, G=238, R=248
        parchment_bgr = (215, 238, 248)
        k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded = cv2.erode(self.bubble_mask, k_in)
        tinted_img[eroded > 0] = parchment_bgr
        
        # Add text on parchment
        cv2.putText(tinted_img, "VINTAGE", (62, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (15, 15, 15), 1)
        
        cleaned = bubble_lettering.inpaint_batch_bubbles(tinted_img, [self.cnt])
        
        # Verify inpainted text area has parchment luminance
        int_clean = cleaned[eroded > 0]
        med_clean = np.median(int_clean, axis=0)
        color_err = np.linalg.norm(med_clean - np.array(parchment_bgr))
        self.assertLess(color_err, 10.0, f"Tinted paper background deviated: {color_err}")


if __name__ == "__main__":
    unittest.main()
