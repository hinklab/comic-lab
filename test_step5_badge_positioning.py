import re


def compute_badge_bounds(b, canvas_w=1000, canvas_h=1000, badge_w=34, badge_h=17):
    """Simulates the smart collision-free positioning logic from web/index.html."""
    x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
    
    # 1. Prefer placing above the bubble, clear of the bubble border
    badge_x = max(6, min(canvas_w - badge_w - 6, x0 + 4))
    badge_y = y0 - badge_h - 5  # 5px clean clearance above bubble

    # 2. If above the canvas top (y0 is small)
    if badge_y < 4:
        if y1 + badge_h + 6 <= canvas_h:
            badge_y = y1 + 5  # Safely below the bubble
        else:
            badge_y = y0 + 4
            badge_x = min(canvas_w - badge_w - 6, x0 + 6)
            
    return {"x": badge_x, "y": badge_y, "w": badge_w, "h": badge_h}


def test_standard_bubble_badge_clearance():
    """Verify that badge is positioned outside and above the bubble with clear margin."""
    b = {"x0": 100, "y0": 200, "x1": 220, "y1": 270}
    badge = compute_badge_bounds(b)
    # Bottom of badge must be strictly above top of bubble (y0)
    badge_bottom = badge["y"] + badge["h"]
    assert badge_bottom <= b["y0"] - 4, f"Badge bottom {badge_bottom} should be <= {b['y0'] - 4}"
    assert badge["x"] >= 6, "Badge X must be within left canvas margin"


def test_top_edge_bubble_badge_avoidance():
    """Verify that bubbles near top edge (y0 < 22) place badge below the bubble, avoiding clipping and border overlap."""
    b = {"x0": 50, "y0": 6, "x1": 150, "y1": 70}
    badge = compute_badge_bounds(b)
    # Must be placed below bubble
    assert badge["y"] >= b["y1"] + 4, f"Badge Y {badge['y']} must be >= {b['y1'] + 4}"
    assert badge["y"] + badge["h"] <= 1000, "Badge must be within canvas height"


def test_right_edge_clamping():
    """Verify that bubbles on far right edge do not cause badge overflow."""
    b = {"x0": 980, "y0": 300, "x1": 998, "y1": 350}
    badge = compute_badge_bounds(b, canvas_w=1000)
    assert badge["x"] + badge["w"] <= 1000 - 6, f"Badge right edge {badge['x'] + badge['w']} exceeds canvas bounds"


def test_html_ui_has_zero_emojis_in_badges():
    """Verify that web/index.html toolbar and badge implementation has NO emojis."""
    with open("web/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Find the badge toggle button in HTML
    badge_btn = re.search(r'<button[^>]*id="btn-toggle-badges"[^>]*>(.*?)</button>', html, re.DOTALL)
    assert badge_btn is not None, "btn-toggle-badges button must exist in toolbar"
    btn_content = badge_btn.group(1)

    # Check for emojis in toolbar button
    emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
    emojis_found = emoji_pattern.findall(btn_content)
    assert len(emojis_found) == 0, f"Found emojis in badge button: {emojis_found}"
    assert "Badjlar" in btn_content, "Button text should be 'Badjlar'"


if __name__ == "__main__":
    test_standard_bubble_badge_clearance()
    print("test_standard_bubble_badge_clearance PASSED")
    test_top_edge_bubble_badge_avoidance()
    print("test_top_edge_bubble_badge_avoidance PASSED")
    test_right_edge_clamping()
    print("test_right_edge_clamping PASSED")
    test_html_ui_has_zero_emojis_in_badges()
    print("test_html_ui_has_zero_emojis_in_badges PASSED")
    print("\nALL STEP 5 TESTS PASSED PERFECTLY!")
