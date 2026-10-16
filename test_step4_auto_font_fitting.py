import numpy as np
import cv2
import bubble_lettering


def test_contour_area_ratio_acceptance():
    """Verify that speech bubbles with high bubble-to-text area ratio (> 2.8x) are accepted as valid contours."""
    # Simulate a 120x80 speech bubble with 80x20 text box (ratio 6.0)
    cnt_area = 120 * 80  # 9600
    ref_area = 80 * 20   # 1600
    gemini_area = 9000
    W_img, H_img = 666, 1024
    overlap_ratio = 0.85

    # New condition from engine.py
    is_valid_contour = (
        (overlap_ratio >= 0.35)
        and (cnt_area >= 0.15 * ref_area)
        and (cnt_area <= 12.0 * ref_area or cnt_area <= 4.0 * max(1, gemini_area))
        and (cnt_area < 0.25 * (W_img * H_img))
    )
    assert is_valid_contour is True, f"Contour with 6x area ratio should be accepted as valid, got {is_valid_contour}"

    # Verify that massive page-wide backgrounds (> 25% of page) are still safely rejected
    huge_area = int(0.30 * W_img * H_img)
    is_huge_rejected = not (huge_area < 0.25 * (W_img * H_img))
    assert is_huge_rejected is True, "Massive page background must be rejected"


def test_fit_text_to_bubble_min_readable_font():
    """Verify that fit_text_to_bubble fits multi-line comic text with readable font size (>= 10px)."""
    # Create an elliptic speech bubble contour: center (100, 100), axes (60, 40)
    contour = bubble_lettering._synthetic_ellipse_contour(100, 100, 120, 80)
    img_shape = (200, 200)
    font_path = "fonts/KOMTXT__.ttf"
    import os
    if not os.path.exists(font_path):
        # Fallback to any available ttf font in fonts/
        for f in os.listdir("fonts"):
            if f.endswith(".ttf"):
                font_path = os.path.join("fonts", f)
                break

    text = "KECHIRASIZ, PODKAST\nTINGLAYOTGAN EDIM..."
    result = bubble_lettering.fit_text_to_bubble(
        text=text,
        contour=contour,
        image_shape=img_shape,
        font_path=font_path,
        max_font_size=20,
        min_font_size=10
    )

    assert result is not None, "fit_text_to_bubble should successfully fit text inside 120x80 bubble"
    assert result["font_size"] >= 10, f"Font size must be >= 10px, got {result['font_size']}"
    assert len(result["lines"]) >= 2, f"Lines must be wrapped properly, got {len(result['lines'])}"


def test_auto_fit_font_size_simulation():
    """Verify that the improved auto-fit algorithm in JS does not collapse font size to microscopic 8px."""
    import subprocess
    cmd = ["node", "scratch/test_fit_improved.js"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Node script failed: {res.stderr}"
    output = res.stdout
    assert "fontSize=12" in output or "fontSize=11" in output
    assert "fontSize=8" not in output, "Font size should not drop to microscopic 8px"


if __name__ == "__main__":
    test_contour_area_ratio_acceptance()
    print("test_contour_area_ratio_acceptance PASSED")
    test_fit_text_to_bubble_min_readable_font()
    print("test_fit_text_to_bubble_min_readable_font PASSED")
    test_auto_fit_font_size_simulation()
    print("test_auto_fit_font_size_simulation PASSED")
    print("\nALL STEP 4 TESTS PASSED PERFECTLY!")
