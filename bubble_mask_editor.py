"""
bubble_mask_editor.py
---------------------
Manual bubble-mask editing engine for comic translation pipeline (Stage 2: Tarjima & Tahrir).
Zero AI API Keys (free/local translation model); online services for data storage are fine.

Provides two interactive precision tools:
1. Tool 1 -- "Qaychi" (Scissors):
   Splits a merged bubble into two separate bubbles along a user-drawn cutting stroke.
   Subtracts the cut stroke, validates exactly 2 connected components, and re-assigns
   OCR text lines using spatial containment (_extract_oval_fragments pattern), flagging
   ambiguous borderline lines for user resolution.

2. Tool 2 -- "Aqlli mo'yqalam" (Smart Brush):
   Grows a bubble's mask to include clipped regions (e.g. missed tails or shoulders).
   Constrained by magic-wand / color-luminance tolerance to the bubble's sampled fill
   color (c_bg), strictly rejecting dark line art and background art.

3. Persistent Learning:
   Records every manual correction into Supabase mask_corrections table (with local
   mask_correction_memory.json fallback) and feeds events into quality_gates.py to
   dynamically tune detection thresholds.
"""

import os
import json
import time
from typing import Optional, Tuple, List, Dict, Any, Union
import cv2
import numpy as np

import quality_gates


MEMORY_FILE = "mask_correction_memory.json"




# --------------------------------------------------------------------------
# Mask & Contour Conversion Helpers
# --------------------------------------------------------------------------

def create_bubble_mask(contour: Any, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Rasterizes a contour polygon into a binary mask of shape (H, W).
    contour can be a numpy array, list of [x, y], or list of [[x, y]].
    """
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    if contour is None:
        return mask

    cnt_arr = np.array(contour, dtype=np.int32)
    if cnt_arr.size == 0:
        return mask

    if cnt_arr.ndim == 2 and cnt_arr.shape[1] == 2:
        cnt_arr = cnt_arr.reshape(-1, 1, 2)
    elif cnt_arr.ndim == 1:
        cnt_arr = cnt_arr.reshape(-1, 1, 2)

    cv2.drawContours(mask, [cnt_arr], -1, 255, -1)
    return mask


def rasterize_cut_stroke(
    stroke_input: Any,
    mask_shape: Tuple[int, int],
    thickness: int = 4
) -> np.ndarray:
    """
    Rasterizes user drawing input into a binary cutting line mask of shape (H, W).
    Handles:
    - RGBA numpy array from st_canvas (alpha channel)
    - List of points [(x, y), ...]
    - List of stroke lines [[x0, y0, x1, y1], ...]
    """
    H, W = mask_shape[:2]
    cut_mask = np.zeros((H, W), dtype=np.uint8)

    if stroke_input is None:
        return cut_mask

    if isinstance(stroke_input, np.ndarray):
        if stroke_input.ndim == 3 and stroke_input.shape[2] >= 4:
            # RGBA from st_canvas: alpha channel > 20 is stroke
            alpha = stroke_input[:, :, 3]
            stroke_bin = (alpha > 20).astype(np.uint8) * 255
            # Resize if dimensions differ from target mask
            if stroke_bin.shape[:2] != (H, W):
                stroke_bin = cv2.resize(stroke_bin, (W, H), interpolation=cv2.INTER_NEAREST)
            # Dilate to ensure continuous cutting barrier
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, thickness), max(3, thickness)))
            cut_mask = cv2.dilate(stroke_bin, kernel, iterations=1)
            return cut_mask
        elif stroke_input.ndim == 2:
            stroke_bin = (stroke_input > 0).astype(np.uint8) * 255
            if stroke_bin.shape[:2] != (H, W):
                stroke_bin = cv2.resize(stroke_bin, (W, H), interpolation=cv2.INTER_NEAREST)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, thickness), max(3, thickness)))
            cut_mask = cv2.dilate(stroke_bin, kernel, iterations=1)
            return cut_mask

    elif isinstance(stroke_input, list):
        pts = []
        for item in stroke_input:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pts.append((int(item[0]), int(item[1])))
            elif isinstance(item, (list, tuple)) and len(item) == 4:
                # Line segment [x0, y0, x1, y1]
                cv2.line(cut_mask, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), 255, thickness)

        if len(pts) >= 2:
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(cut_mask, [pts_arr], False, 255, thickness)

    return cut_mask


def sample_fill_color_c_bg(page_bgr: np.ndarray, bubble_mask: np.ndarray) -> List[int]:
    """
    Samples authentic background fill color c_bg inside the bubble mask,
    reusing the tinted-bubble sampling logic. Returns [B, G, R].
    """
    if page_bgr is None or bubble_mask is None or np.count_nonzero(bubble_mask) == 0:
        return [255, 255, 255]

    int_pts = page_bgr[bubble_mask > 0]
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    int_gray = gray[bubble_mask > 0]

    if int_pts.size == 0 or int_gray.size == 0:
        return [255, 255, 255]

    # Non-text background analysis (brightest 60% of interior)
    p40_g = np.percentile(int_gray, 40)
    bg_pts = int_pts[int_gray >= p40_g]

    int_hsv = cv2.cvtColor(int_pts.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)[0]
    bg_hsv = int_hsv[int_gray >= p40_g]
    bg_sat = bg_hsv[:, 1] if len(bg_hsv) > 0 else []
    bg_med_sat = np.median(bg_sat) if len(bg_sat) > 0 else 0
    bg_is_tinted = bool(bg_med_sat >= 12 or (len(bg_sat) > 0 and np.percentile(bg_sat, 60) >= 12))

    if bg_is_tinted and len(bg_pts) > 0:
        c_bg = np.median(bg_pts, axis=0)
    else:
        p80_gray = np.percentile(int_gray, 80)
        bg_sample = int_pts[int_gray >= p80_gray]
        c_bg = np.median(bg_sample, axis=0) if len(bg_sample) > 0 else np.median(int_pts, axis=0)

    return [int(round(float(c_bg[0]))), int(round(float(c_bg[1]))), int(round(float(c_bg[2])))]


# --------------------------------------------------------------------------
# TOOL 1: "Qaychi" (Scissors) Implementation
# --------------------------------------------------------------------------

def apply_scissors_split(
    bubble: Any,
    stroke_mask: np.ndarray,
    page_bgr: np.ndarray,
    all_page_lines: Optional[List[Dict[str, Any]]] = None
) -> Tuple[bool, str, Optional[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Applies scissors cut across bubble's binary mask.

    Returns:
        (success: bool, message: str, new_bubbles: Optional[List[Dict]], ambiguous_lines: List[Dict])

    If success is True:
      - If ambiguous_lines is empty: the 2 new bubble entries are fully resolved.
      - If ambiguous_lines is not empty: lines that could not be confidently assigned
        are returned for manual user choice.
    If success is False:
      - Returns rejection message explaining why (e.g. didn't cross mask, or cut into 3+ pieces).
    """
    H, W = page_bgr.shape[:2]

    # Extract bubble attributes
    cnt = bubble.get("contour") if isinstance(bubble, dict) else getattr(bubble, "contour", None)
    if cnt is None:
        cnt = bubble.get("contour_points") if isinstance(bubble, dict) else getattr(bubble, "contour_points", None)

    if cnt is None:
        # Fallback to bounding box contour
        box = bubble.get("box") if isinstance(bubble, dict) else getattr(bubble, "box", None)
        if not box:
            x0 = bubble.get("x0") if isinstance(bubble, dict) else getattr(bubble, "x0", 0)
            y0 = bubble.get("y0") if isinstance(bubble, dict) else getattr(bubble, "y0", 0)
            x1 = bubble.get("x1") if isinstance(bubble, dict) else getattr(bubble, "x1", 100)
            y1 = bubble.get("y1") if isinstance(bubble, dict) else getattr(bubble, "y1", 100)
            box = [x0, y0, x1, y1]
        cnt = np.array([
            [[box[0], box[1]]], [[box[2], box[1]]],
            [[box[2], box[3]]], [[box[0], box[3]]]
        ], dtype=np.int32)

    bubble_mask = create_bubble_mask(cnt, (H, W))
    bubble_area = int(np.count_nonzero(bubble_mask))
    if bubble_area < 50:
        return False, "Pufak maydoni juda kichik yoki mavjud emas.", None, []

    # Subtract stroke from mask
    cut_mask = cv2.bitwise_and(bubble_mask, cv2.bitwise_not(stroke_mask))

    # Connected components analysis
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cut_mask, connectivity=8)

    # Filter out tiny residual dust/sliver fragments (area < 4% of original bubble or < 120 px)
    min_comp_area = max(120, int(0.04 * bubble_area))
    valid_labels = []
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_comp_area:
            valid_labels.append(lbl)

    if len(valid_labels) < 2:
        return (
            False,
            "Kesish chizig'i pufakni to'liq kesib o'tmadi (faqat 1 ta bo'lak aniqlandi). "
            "Iltimos, pufakning bir chetidan ikkinchi chetiga qadar to'liq chizing.",
            None,
            []
        )

    if len(valid_labels) > 2:
        return (
            False,
            f"Kesish natijasida 2 tadan ko'p bo'lak hosil bo'ldi ({len(valid_labels)} ta bo'lak). "
            "Iltimos, bitta to'g'ri kesuvchi chiziq torting.",
            None,
            []
        )

    # Exactly 2 components: extract masks & contours
    lbl1, lbl2 = valid_labels[0], valid_labels[1]
    comp1_mask = (labels == lbl1).astype(np.uint8) * 255
    comp2_mask = (labels == lbl2).astype(np.uint8) * 255

    # Dilate each component slightly (1px) to restore the cutting seam without gap
    seam_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    comp1_mask = cv2.bitwise_and(cv2.dilate(comp1_mask, seam_kernel), bubble_mask)
    comp2_mask = cv2.bitwise_and(cv2.dilate(comp2_mask, seam_kernel), bubble_mask)

    cnts1, _ = cv2.findContours(comp1_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, _ = cv2.findContours(comp2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts1 or not cnts2:
        return False, "Bo'laklar konturini ajratib bo'lmadi.", None, []

    c1 = max(cnts1, key=cv2.contourArea)
    c2 = max(cnts2, key=cv2.contourArea)

    bx1, by1, bw1, bh1 = cv2.boundingRect(c1)
    bx2, by2, bw2, bh2 = cv2.boundingRect(c2)

    # Sort components in natural reading order: top-to-bottom, then left-to-right
    # (using baseline binning of 40px)
    if (by2 // 40 < by1 // 40) or (by2 // 40 == by1 // 40 and bx2 < bx1):
        c1, c2 = c2, c1
        comp1_mask, comp2_mask = comp2_mask, comp1_mask
        bx1, by1, bw1, bh1, bx2, by2, bw2, bh2 = bx2, by2, bw2, bh2, bx1, by1, bw1, bh1

    # -----------------------------------------------------------------------
    # Text Handling: Spatial containment re-assignment
    # -----------------------------------------------------------------------
    # Retrieve constituent lines of the original bubble
    orig_lines = bubble.get("lines") if isinstance(bubble, dict) else getattr(bubble, "lines", None)
    if not orig_lines and all_page_lines:
        # Match lines from page OCR falling inside original bubble bounding box
        orig_box = bubble.get("box") if isinstance(bubble, dict) else getattr(bubble, "box", None)
        if orig_box:
            ox0, oy0, ox1, oy1 = orig_box
            orig_lines = [
                l for l in all_page_lines
                if (ox0 - 20 <= (l['x0'] + l['x1']) // 2 <= ox1 + 20) and
                   (oy0 - 20 <= (l['y0'] + l['y1']) // 2 <= oy1 + 20)
            ]

    # If still no detailed lines, construct synthetic lines from words or original text
    orig_text = (bubble.get("original_text") or bubble.get("clean_text", "")) if isinstance(bubble, dict) else (getattr(bubble, "original_text", "") or getattr(bubble, "clean_text", ""))
    if not orig_lines and orig_text:
        words = orig_text.split()
        if len(words) >= 2:
            # Synthetic line items for left and right
            mid = max(1, len(words) // 2)
            orig_lines = [
                {'text': " ".join(words[:mid]), 'x0': bx1, 'y0': by1, 'x1': bx1 + bw1, 'y1': by1 + bh1,
                 'cx': bx1 + bw1 // 2, 'cy': by1 + bh1 // 2},
                {'text': " ".join(words[mid:]), 'x0': bx2, 'y0': by2, 'x1': bx2 + bw2, 'y1': by2 + bh2,
                 'cx': bx2 + bw2 // 2, 'cy': by2 + bh2 // 2}
            ]
        else:
            orig_lines = [{'text': orig_text, 'x0': bx1, 'y0': by1, 'x1': bx1 + bw1, 'y1': by1 + bh1,
                           'cx': bx1 + bw1 // 2, 'cy': by1 + bh1 // 2}]

    lines1 = []
    lines2 = []
    ambiguous_lines = []

    for l in orig_lines:
        lcx = int(l.get('cx', (l['x0'] + l['x1']) // 2))
        lcy = int(l.get('cy', (l['y0'] + l['y1']) // 2))
        lcx = min(W - 1, max(0, lcx))
        lcy = min(H - 1, max(0, lcy))

        # Check mask containment
        in_m1 = comp1_mask[lcy, lcx] > 0
        in_m2 = comp2_mask[lcy, lcx] > 0

        # Check point polygon distance
        d1 = cv2.pointPolygonTest(c1, (float(lcx), float(lcy)), True)
        d2 = cv2.pointPolygonTest(c2, (float(lcx), float(lcy)), True)

        # Confident assignment: clearly inside one mask and outside the other
        if in_m1 and not in_m2 and d1 >= 0:
            lines1.append(l)
        elif in_m2 and not in_m1 and d2 >= 0:
            lines2.append(l)
        elif d1 > 8 and d2 < -4:
            lines1.append(l)
        elif d2 > 8 and d1 < -4:
            lines2.append(l)
        else:
            # Ambiguous: center falls directly on cut seam or equidistant
            ambiguous_lines.append(l)

    # Construct the two split bubble dictionaries
    c_bg = sample_fill_color_c_bg(page_bgr, bubble_mask)
    cur_spk = bubble.get("speaker") if isinstance(bubble, dict) else getattr(bubble, "speaker", None)
    cur_conf = 1.0
    if isinstance(bubble, dict):
        if bubble.get("confidence") is not None:
            cur_conf = float(bubble["confidence"])
    elif hasattr(bubble, "confidence") and bubble.confidence is not None:
        cur_conf = float(bubble.confidence)
    is_tinted = bubble.get("is_tinted", False) if isinstance(bubble, dict) else getattr(bubble, "is_tinted", False)

    text1 = " ".join(l['text'] for l in lines1).strip()
    text2 = " ".join(l['text'] for l in lines2).strip()

    bubble1 = {
        "bubble_id": 1,
        "x0": int(bx1), "y0": int(by1), "x1": int(bx1 + bw1), "y1": int(by1 + bh1),
        "box": [int(bx1), int(by1), int(bx1 + bw1), int(by1 + bh1)],
        "original_text": text1,
        "clean_text": text1,
        "uzbek_translation": "",
        "confidence": float(cur_conf),
        "shape_type": "oval",
        "contour": c1.reshape(-1, 2).tolist(),
        "contour_points": c1.reshape(-1, 2).tolist(),
        "lines": lines1,
        "is_active": True,
        "enabled": True,
        "speaker": cur_spk,
        "tint_bgr": c_bg,
        "tint_rgb": c_bg[::-1],
        "is_tinted": is_tinted
    }

    bubble2 = {
        "bubble_id": 2,
        "x0": int(bx2), "y0": int(by2), "x1": int(bx2 + bw2), "y1": int(by2 + bh2),
        "box": [int(bx2), int(by2), int(bx2 + bw2), int(by2 + bh2)],
        "original_text": text2,
        "clean_text": text2,
        "uzbek_translation": "",
        "confidence": float(cur_conf),
        "shape_type": "oval",
        "contour": c2.reshape(-1, 2).tolist(),
        "contour_points": c2.reshape(-1, 2).tolist(),
        "lines": lines2,
        "is_active": True,
        "enabled": True,
        "speaker": cur_spk,
        "tint_bgr": c_bg,
        "tint_rgb": c_bg[::-1],
        "is_tinted": is_tinted
    }

    return True, "OK", [bubble1, bubble2], ambiguous_lines


# --------------------------------------------------------------------------
# TOOL 2: "Tezkor Tanlash" (Quick Selection / Smart Brush) Implementation
# --------------------------------------------------------------------------

def extract_seed_points(
    stroke_mask: np.ndarray,
    json_data: Optional[Dict[str, Any]] = None,
    disp_shape: Optional[Tuple[int, int]] = None,
    spacing: int = 20
) -> List[Tuple[int, int]]:
    """
    Extracts seed points (x, y) from a user's single click or rough drag.

    - Single click (compact cluster): yields 1 seed point at the centroid.
    - Drag / line: samples points along the drawn polyline spaced by `spacing` px.
    """
    H, W = stroke_mask.shape[:2]
    seeds = []

    # 1. Parse Fabric.js path objects from canvas json_data if available
    if json_data and isinstance(json_data, dict) and "objects" in json_data:
        objects = json_data.get("objects", [])
        for obj in objects:
            if obj.get("type") == "path" and "path" in obj:
                path_cmds = obj.get("path", [])
                pts = []
                for cmd in path_cmds:
                    if len(cmd) >= 3 and cmd[0] in ("M", "L"):
                        pts.append((float(cmd[1]), float(cmd[2])))
                    elif len(cmd) >= 5 and cmd[0] == "Q":
                        pts.append((float(cmd[3]), float(cmd[4])))

                if pts:
                    scale_x = float(W) / float(disp_shape[1]) if disp_shape and disp_shape[1] > 0 else 1.0
                    scale_y = float(H) / float(disp_shape[0]) if disp_shape and disp_shape[0] > 0 else 1.0

                    scaled_pts = [(int(round(px * scale_x)), int(round(py * scale_y))) for px, py in pts]

                    if len(scaled_pts) <= 2:
                        seeds.append(scaled_pts[0])
                    else:
                        last_pt = scaled_pts[0]
                        seeds.append(last_pt)
                        accum_dist = 0.0
                        for pt in scaled_pts[1:]:
                            d = float(np.hypot(pt[0] - last_pt[0], pt[1] - last_pt[1]))
                            accum_dist += d
                            if accum_dist >= spacing:
                                seeds.append(pt)
                                accum_dist = 0.0
                            last_pt = pt
                        if np.hypot(scaled_pts[-1][0] - seeds[-1][0], scaled_pts[-1][1] - seeds[-1][1]) > (spacing / 2):
                            seeds.append(scaled_pts[-1])

    if seeds:
        valid_seeds = []
        for sx, sy in seeds:
            valid_seeds.append((min(W - 1, max(0, sx)), min(H - 1, max(0, sy))))
        return valid_seeds

    # 2. Fallback: extract from binary stroke_mask
    if stroke_mask is None or np.count_nonzero(stroke_mask) == 0:
        return []

    y_indices, x_indices = np.where(stroke_mask > 0)
    if len(y_indices) == 0:
        return []

    min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
    min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))
    bw, bh = max_x - min_x, max_y - min_y

    # Single click detection: small bounding box or small pixel cluster
    if (bw <= 20 and bh <= 20) or len(y_indices) < 90:
        cx = int(round(np.mean(x_indices)))
        cy = int(round(np.mean(y_indices)))
        return [(min(W - 1, max(0, cx)), min(H - 1, max(0, cy)))]

    # Drag path: distance-spaced sampling
    sampled = [(int(x_indices[0]), int(y_indices[0]))]
    sp_sq = float(spacing * spacing)

    for i in range(1, len(x_indices)):
        px, py = int(x_indices[i]), int(y_indices[i])
        too_close = False
        for sx, sy in sampled:
            if (px - sx) ** 2 + (py - sy) ** 2 < sp_sq:
                too_close = True
                break
        if not too_close:
            sampled.append((px, py))

    return [(min(W - 1, max(0, x)), min(H - 1, max(0, y))) for x, y in sampled]


def compute_quick_selection_mask(
    roi_bgr: np.ndarray,
    bubble_mask_local: np.ndarray,
    seed_points: List[Tuple[int, int]],
    c_bg_bgr: Optional[List[int]] = None,
    tolerance: float = 30.0,
    min_lum: float = 110.0,
    edge_low: int = 40,
    edge_high: int = 120,
    close_kernel_size: int = 5,
    stroke_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Photoshop Quick Selection Tool engine:
    Reuses bubble_lettering.py Canny edge detection + dark ink suppression barrier.
    Automatically floods outward from each seed point, stopping cleanly at ink lines
    and artwork boundaries without leaking into the page art.
    Supports direct stroke area inclusion, text hole-filling, and robust edge snapping.
    """
    rh, rw = roi_bgr.shape[:2]
    if bubble_mask_local.shape[:2] != (rh, rw):
        bubble_mask_local = cv2.resize(bubble_mask_local, (rw, rh), interpolation=cv2.INTER_NEAREST)

    if stroke_mask is not None and stroke_mask.shape[:2] != (rh, rw):
        stroke_mask = cv2.resize(stroke_mask, (rw, rh), interpolation=cv2.INTER_NEAREST)

    if not seed_points and (stroke_mask is None or np.count_nonzero(stroke_mask) == 0):
        return bubble_mask_local.copy(), np.zeros((rh, rw), dtype=np.uint8)

    if c_bg_bgr is None:
        c_bg_bgr = sample_fill_color_c_bg(roi_bgr, bubble_mask_local)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Build Edge & Ink Barrier (Canny + Closing + Dark Ink)
    edges = cv2.Canny(gray, edge_low, edge_high)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    k = max(3, close_kernel_size)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )

    ink_barrier_thresh = min(85, int(min_lum - 20))
    ink_wall = gray < ink_barrier_thresh
    barrier = (edges > 0) | ink_wall

    # 2. Flood-fill from each seed point
    union_filled = np.zeros((rh, rw), dtype=np.uint8)
    tol_int = max(8, int(round(tolerance)))
    tol_bgr = (tol_int, tol_int, tol_int)

    c_bg_arr = np.array(c_bg_bgr, dtype=np.float32)
    color_dist = np.linalg.norm(roi_bgr.astype(np.float32) - c_bg_arr, axis=2)

    for sx, sy in seed_points:
        sx = min(rw - 1, max(0, sx))
        sy = min(rh - 1, max(0, sy))

        # If seed is on dark ink or off-color boundary, search 8px radius for bright interior
        if barrier[sy, sx] or color_dist[sy, sx] > tolerance * 1.5:
            found_alt = False
            for r_chk in range(1, 8):
                for dy in range(-r_chk, r_chk + 1):
                    for dx in range(-r_chk, r_chk + 1):
                        nx, ny = sx + dx, sy + dy
                        if 0 <= nx < rw and 0 <= ny < rh:
                            if not barrier[ny, nx] and color_dist[ny, nx] <= tolerance * 1.2:
                                sx, sy = nx, ny
                                found_alt = True
                                break
                    if found_alt:
                        break
                if found_alt:
                    break
            if not found_alt:
                continue

        flood_mask = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
        flood_mask[1:-1, 1:-1][barrier] = 1

        # Use FLOODFILL_FIXED_RANGE to compare strictly to seed pixel and prevent gradient drift
        cv2.floodFill(
            roi_bgr, flood_mask, (sx, sy), (255, 255, 255),
            loDiff=tol_bgr, upDiff=tol_bgr,
            flags=4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
        )

        cur_fill = (flood_mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255
        union_filled = cv2.bitwise_or(union_filled, cur_fill)

    # 3. Filter candidate additions:
    # - Must be within tolerance of c_bg
    # - Must not be dark ink borders
    # - Must not already be part of existing bubble mask
    valid_fill = (
        (union_filled > 0) &
        (color_dist <= (tolerance * 1.35)) &
        (gray >= ink_barrier_thresh) &
        (bubble_mask_local == 0)
    )
    candidate_addition = valid_fill.astype(np.uint8) * 255

    # Direct stroke area: include area painted by user matching tolerance and not dark ink
    if stroke_mask is not None and np.count_nonzero(stroke_mask) > 0:
        stroke_direct = (
            (stroke_mask > 0) &
            (color_dist <= (tolerance * 1.35)) &
            (gray >= ink_barrier_thresh) &
            (bubble_mask_local == 0)
        )
        candidate_addition = cv2.bitwise_or(candidate_addition, stroke_direct.astype(np.uint8) * 255)

    if np.count_nonzero(candidate_addition) > 0:
        clean_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        candidate_addition = cv2.morphologyEx(candidate_addition, cv2.MORPH_CLOSE, clean_k)

        # Retain only connected components that touch the bubble or stroke
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_addition, connectivity=8)
        touch_target = cv2.dilate(bubble_mask_local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        if stroke_mask is not None and np.count_nonzero(stroke_mask) > 0:
            touch_target = cv2.bitwise_or(touch_target, stroke_mask)

        connected_addition = np.zeros_like(candidate_addition)
        for lbl in range(1, num_labels):
            comp = (labels == lbl)
            if np.any(touch_target[comp] > 0):
                connected_addition[comp] = 255
        candidate_addition = connected_addition

    # 4. Seamless hole filling for comic lettering inside the bubble:
    # Only fill small enclosed holes (lettering like 'O', 'A', etc.), NEVER outer panels or artwork
    merged_pre = cv2.bitwise_or(bubble_mask_local, candidate_addition)
    if np.count_nonzero(merged_pre) > 0:
        inv_mask = cv2.bitwise_not(merged_pre)
        hole_cnts, _ = cv2.findContours(inv_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for h_cnt in hole_cnts:
            area = cv2.contourArea(h_cnt)
            if 0 < area < 1500:
                hx, hy, hw, hh = cv2.boundingRect(h_cnt)
                # Ensure it's fully enclosed inside the ROI without touching image outer edges
                if hx > 0 and hy > 0 and (hx + hw) < rw and (hy + hh) < rh:
                    cv2.drawContours(merged_pre, [h_cnt], -1, 255, thickness=cv2.FILLED)

        # Re-apply edge boundary barrier and ink wall
        merged_pre[edges > 0] = 0
        merged_pre[ink_wall] = 0
        merged_pre = cv2.bitwise_or(merged_pre, bubble_mask_local)
        candidate_addition = cv2.bitwise_and(merged_pre, cv2.bitwise_not(bubble_mask_local))

    updated_mask = cv2.bitwise_or(bubble_mask_local, candidate_addition)
    return updated_mask, candidate_addition


def compute_smart_brush_preview(
    page_bgr: np.ndarray,
    bubble_mask: np.ndarray,
    brush_mask: np.ndarray,
    c_bg_bgr: Optional[List[int]] = None,
    tolerance: float = 35.0,
    min_lum: float = 110.0,
    json_data: Optional[Dict[str, Any]] = None,
    disp_shape: Optional[Tuple[int, int]] = None,
    seed_spacing: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Photoshop Quick Selection preview:
    Extracts seed points from click / rough drag, then grows edge-bounded flood fill within a local ROI.
    """
    H, W = page_bgr.shape[:2]
    if brush_mask.shape[:2] != (H, W):
        brush_mask = cv2.resize(brush_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    if bubble_mask.shape[:2] != (H, W):
        bubble_mask = cv2.resize(bubble_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    seed_points = extract_seed_points(
        brush_mask, json_data=json_data, disp_shape=disp_shape, spacing=seed_spacing
    )

    if not seed_points and np.count_nonzero(brush_mask) == 0:
        return bubble_mask.copy(), np.zeros((H, W), dtype=np.uint8)

    # Local Bounding ROI: confine computation strictly around bubble and stroke footprint
    active_pts = np.argwhere((bubble_mask > 0) | (brush_mask > 0))
    if len(active_pts) == 0:
        return bubble_mask.copy(), np.zeros((H, W), dtype=np.uint8)

    y_min, x_min = active_pts.min(axis=0)
    y_max, x_max = active_pts.max(axis=0)

    pad = max(60, int(tolerance * 2.5))
    roi_x0 = max(0, int(x_min) - pad)
    roi_y0 = max(0, int(y_min) - pad)
    roi_x1 = min(W, int(x_max) + pad + 1)
    roi_y1 = min(H, int(y_max) + pad + 1)

    roi_bgr = page_bgr[roi_y0:roi_y1, roi_x0:roi_x1]
    roi_bubble = bubble_mask[roi_y0:roi_y1, roi_x0:roi_x1]
    roi_brush = brush_mask[roi_y0:roi_y1, roi_x0:roi_x1]

    roi_seeds = [
        (sx - roi_x0, sy - roi_y0)
        for sx, sy in seed_points
        if roi_x0 <= sx < roi_x1 and roi_y0 <= sy < roi_y1
    ]

    roi_updated, roi_candidate = compute_quick_selection_mask(
        roi_bgr, roi_bubble, roi_seeds,
        c_bg_bgr=c_bg_bgr, tolerance=tolerance, min_lum=min_lum,
        stroke_mask=roi_brush
    )

    updated_mask = bubble_mask.copy()
    candidate_added = np.zeros((H, W), dtype=np.uint8)
    updated_mask[roi_y0:roi_y1, roi_x0:roi_x1] = roi_updated
    candidate_added[roi_y0:roi_y1, roi_x0:roi_x1] = roi_candidate

    return updated_mask, candidate_added


def apply_smart_brush_growth(
    bubble: Any,
    page_bgr: np.ndarray,
    brush_mask: np.ndarray,
    tolerance: float = 35.0,
    min_lum: float = 110.0,
    json_data: Optional[Dict[str, Any]] = None,
    disp_shape: Optional[Tuple[int, int]] = None,
    seed_spacing: int = 20
) -> Dict[str, Any]:
    """
    Grows bubble mask using Photoshop Quick Selection edge-bounded flood-fill and recomputes contour and bbox.
    """
    H, W = page_bgr.shape[:2]
    cnt = bubble.get("contour") if isinstance(bubble, dict) else getattr(bubble, "contour", None)
    bubble_mask = create_bubble_mask(cnt, (H, W))

    c_bg = bubble.get("tint_bgr") if isinstance(bubble, dict) else getattr(bubble, "tint_bgr", None)
    if not c_bg:
        c_bg = sample_fill_color_c_bg(page_bgr, bubble_mask)

    updated_mask, candidate_added = compute_smart_brush_preview(
        page_bgr, bubble_mask, brush_mask,
        c_bg_bgr=c_bg, tolerance=tolerance, min_lum=min_lum,
        json_data=json_data, disp_shape=disp_shape, seed_spacing=seed_spacing
    )

    def _dump_bubble(b):
        if isinstance(b, dict):
            return dict(b)
        return b.model_dump() if hasattr(b, "model_dump") else b.dict()

    cnts, _ = cv2.findContours(updated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return _dump_bubble(bubble)

    # If multiple contours exist (e.g. brushed tail/extension with a 1-2px microscopic gap),
    # bridge them using morphological closure so the addition is never discarded
    if len(cnts) > 1:
        bridge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        bridged_mask = cv2.morphologyEx(updated_mask, cv2.MORPH_CLOSE, bridge_k)
        cnts_b, _ = cv2.findContours(bridged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_b:
            cnts = cnts_b

    # Pick the contour that best overlaps or encompasses the original bubble
    best_cnt = None
    best_overlap = -1
    for c in cnts:
        m_c = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(m_c, [c], -1, 255, thickness=cv2.FILLED)
        ov = int(np.count_nonzero(cv2.bitwise_and(m_c, bubble_mask)))
        if ov > best_overlap:
            best_overlap = ov
            best_cnt = c

    new_cnt = best_cnt if best_cnt is not None else max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(new_cnt, True)
    new_cnt = cv2.approxPolyDP(new_cnt, 0.0025 * peri, True)
    bx, by, bw, bh = cv2.boundingRect(new_cnt)

    new_b = _dump_bubble(bubble)
    new_b["contour"] = new_cnt.reshape(-1, 2).tolist()
    new_b["contour_points"] = new_cnt.reshape(-1, 2).tolist()
    new_b["box"] = [int(bx), int(by), int(bx + bw), int(by + bh)]
    new_b["x0"] = int(bx)
    new_b["y0"] = int(by)
    new_b["x1"] = int(bx + bw)
    new_b["y1"] = int(by + bh)
    new_b["tint_bgr"] = c_bg
    new_b["tint_rgb"] = c_bg[::-1]

    return new_b


# --------------------------------------------------------------------------
# PERSISTENT LEARNING & QUALITY GATES WIRING
# --------------------------------------------------------------------------

def load_mask_corrections(memory_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads recorded mask corrections from persistent storage (Supabase with local fallback)."""
    path = memory_path or MEMORY_FILE
    if memory_path is None and MEMORY_FILE == "mask_correction_memory.json":
        try:
            import supabase_db
            db_records = supabase_db.fetch_mask_corrections()
            if db_records is not None:
                return db_records
        except Exception as e:
            print(f"[CORRECTION_MEMORY_WARN] Supabase fetch failed: {e}. Falling back to local file.")

    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def save_mask_corrections(corrections: List[Dict[str, Any]], memory_path: Optional[str] = None):
    """Saves mask corrections to persistent storage."""
    path = memory_path or MEMORY_FILE
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(corrections, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[CORRECTION_MEMORY] Error saving memory: {e}")


def record_mask_correction(
    page_name: str,
    tool_name: str,
    bubble_id: int,
    original_contour: Any,
    corrected_contours: Any,
    metadata: Optional[Dict[str, Any]] = None,
    memory_path: Optional[str] = None
):
    """
    Records a manual mask correction event:
    1. Persists to Supabase mask_corrections table (or local mask_correction_memory.json)
    2. Dispatches QualityGateEvent into quality_gates.py
    3. Tunes threshold models for touching_bubbles_merged or clipped_bubble_edge
    """
    path = memory_path or MEMORY_FILE

    # Format contours
    orig_cnt_list = np.array(original_contour, dtype=np.int32).reshape(-1, 2).tolist() if original_contour is not None else []
    if isinstance(corrected_contours, list) and len(corrected_contours) > 0 and isinstance(corrected_contours[0], (list, np.ndarray)):
        if isinstance(corrected_contours[0][0], (list, tuple, np.ndarray)):
            corr_cnt_list = [np.array(c, dtype=np.int32).reshape(-1, 2).tolist() for c in corrected_contours]
        else:
            corr_cnt_list = [np.array(corrected_contours, dtype=np.int32).reshape(-1, 2).tolist()]
    else:
        corr_cnt_list = []

    event_record = {
        "page": page_name,
        "timestamp": time.time(),
        "tool_used": tool_name,  # 'scissors' | 'smart_brush' | 'quick_selection'
        "bubble_id": bubble_id,
        "original_contour": orig_cnt_list,
        "corrected_contours": corr_cnt_list,
        "metadata": metadata or {}
    }

    # Always persist locally to path (supports test suites and local fallback)
    corrections = load_mask_corrections(path)
    corrections.append(event_record)
    save_mask_corrections(corrections, path)

    # Persist to Supabase if in production mode
    if memory_path is None and MEMORY_FILE == "mask_correction_memory.json":
        try:
            import supabase_db
            supabase_db.insert_mask_correction(event_record)
        except Exception as e:
            print(f"[CORRECTION_MEMORY_WARN] Supabase error: {e}. Falling back to local file.")

    # Quality Gate pattern mapping
    pattern_name = "touching_bubbles_merged" if tool_name == "scissors" else "clipped_bubble_edge"
    quality_gates.log_quality_gate_event(
        pattern_name=pattern_name,
        stage="bubble_review_edit",
        severity="auto_fix",
        action_taken=f"manual_{tool_name}_applied",
        details={
            "tool_used": tool_name,
            "bubble_id": bubble_id,
            "components_count": len(corr_cnt_list),
            "metadata": metadata or {}
        },
        bubble_id=bubble_id,
        page=page_name
    )

    # Dynamic threshold tuning: notify quality_gates
    quality_gates.tune_pattern_from_correction(pattern_name, event_record)

