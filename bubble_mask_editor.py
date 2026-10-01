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
import history_manager


MEMORY_FILE = "mask_correction_memory.json"


def patch_streamlit_image_to_url():
    """
    Compatibility shim for Streamlit >= 1.29 / 1.60+ where image_to_url was relocated
    from streamlit.elements.image to streamlit.elements.lib.image_utils and its
    signature updated to require LayoutConfig.
    """
    try:
        import streamlit.elements.image as st_image
        if not hasattr(st_image, "image_to_url"):
            from streamlit.elements.lib import image_utils
            try:
                from streamlit.elements.image import create_layout_config
            except Exception:
                def create_layout_config(width=None):
                    class _Cfg:
                        def __init__(self, w):
                            self.width = w
                            self.height = None
                            self.text_alignment = None
                    return _Cfg(width)

            def _compat_image_to_url(image, width_or_layout_config, clamp=True, channels='RGB', output_format='PNG', image_id=''):
                if isinstance(width_or_layout_config, int):
                    layout_config = create_layout_config(width=width_or_layout_config)
                else:
                    layout_config = width_or_layout_config
                return image_utils.image_to_url(image, layout_config, clamp, channels, output_format, image_id)

            st_image.image_to_url = _compat_image_to_url
    except Exception:
        pass

# Apply compatibility shim immediately upon import
patch_streamlit_image_to_url()


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


# --------------------------------------------------------------------------
# FIGMA-STYLE FLOATING TOOLBAR STYLES & BEHAVIOR
# --------------------------------------------------------------------------

def inject_floating_toolbar_styles_and_js(active_mode: str = "view", brush_size: int = 30):
    """
    Injects CSS and vanilla JS for Figma-style persistent, draggable, minimizable floating toolbar,
    and Photoshop-grade circular selection brush cursor that dynamically scales with brush size (pixels).
    - Position: fixed in viewport, default bottom-center.
    - Draggable anywhere on screen via header grab bar.
    - Quick Select mode: Photoshop circular dashed ring with center crosshair following mouse.
    - Dynamically resizes when brush pixel size changes (or via [ ] keyboard shortcuts).
    """
    import streamlit as st
    import streamlit.components.v1 as components

    css = """
    <style>
    /* ============================================================
       FIGMA-STYLE COMPACT BOTTOM TOOLBAR
       Thin single-row bar at bottom center, like Figma's toolbar.
       ============================================================ */

    .st-key-figma_stage2_toolbar {
        position: fixed !important;
        bottom: 24px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        z-index: 999999 !important;
        background: rgba(20, 20, 24, 0.96) !important;
        backdrop-filter: blur(20px) saturate(1.4) !important;
        -webkit-backdrop-filter: blur(20px) saturate(1.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 14px !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.7),
                    0 0 0 1px rgba(255, 255, 255, 0.08) !important;
        padding: 5px 8px !important;
        width: fit-content !important;
        min-width: unset !important;
        max-width: calc(100vw - 24px) !important;
        box-sizing: border-box !important;
        transition: box-shadow 0.2s ease !important;
    }

    /* While dragging */
    .st-key-figma_stage2_toolbar.is-dragging {
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.9),
                    0 0 0 1.5px rgba(99, 102, 241, 0.6) !important;
        user-select: none !important;
    }

    /* Drag handle - thin top bar */
    .figma-header-bar {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        cursor: grab !important;
        margin-bottom: 3px !important;
        user-select: none !important;
        height: 6px !important;
        border-bottom: none !important;
    }
    .figma-header-bar:active { cursor: grabbing !important; }

    /* Drag pill indicator (like Figma's subtle drag area) */
    .figma-drag-grip {
        width: 28px !important;
        height: 3px !important;
        background: rgba(255, 255, 255, 0.25) !important;
        border-radius: 2px !important;
        display: block !important;
        font-size: 0 !important;
        color: transparent !important;
        overflow: hidden !important;
        margin: 0 auto !important;
        transition: background 0.15s !important;
    }
    .figma-header-bar:hover .figma-drag-grip {
        background: rgba(255, 255, 255, 0.5) !important;
    }
    /* Hide the old title & min-toggle from header */
    .figma-tool-title { display: none !important; }
    .figma-min-toggle { display: none !important; }

    /* ---- HORIZONTAL ROW & COLUMNS: Exact tight Figma packing ---- */
    .st-key-figma_stage2_toolbar [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
        flex-wrap: nowrap !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .st-key-figma_stage2_toolbar [data-testid="column"] {
        padding: 0 !important;
        margin: 0 !important;
        width: auto !important;
        min-width: unset !important;
        max-width: unset !important;
        flex: 0 0 auto !important;
    }

    /* ---- MODE & HISTORY BUTTONS: Generous, crisp icon-style pills ---- */
    .st-key-figma_stage2_toolbar .st-key-btn_mode_view,
    .st-key-figma_stage2_toolbar .st-key-btn_mode_scissors,
    .st-key-figma_stage2_toolbar .st-key-btn_mode_quick_select,
    .st-key-figma_stage2_toolbar .st-key-btn_history_undo,
    .st-key-figma_stage2_toolbar .st-key-btn_history_redo {
        width: 42px !important;
        min-width: 42px !important;
        max-width: 42px !important;
    }
    .st-key-figma_stage2_toolbar .st-key-btn_mode_view button,
    .st-key-figma_stage2_toolbar .st-key-btn_mode_scissors button,
    .st-key-figma_stage2_toolbar .st-key-btn_mode_quick_select button,
    .st-key-figma_stage2_toolbar .st-key-btn_history_undo button,
    .st-key-figma_stage2_toolbar .st-key-btn_history_redo button {
        width: 42px !important;
        min-width: 42px !important;
        max-width: 42px !important;
        height: 38px !important;
        min-height: 38px !important;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 9px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        position: relative !important;
    }

    /* Active Mode Button (primary kind) - High contrast & prominent */
    .st-key-figma_stage2_toolbar [data-testid="stBaseButton-primary"],
    .st-key-figma_stage2_toolbar [data-testid="baseButton-primary"],
    .st-key-figma_stage2_toolbar button[kind="primary"] {
        background: #18181b !important;
        border: 2px solid #10b981 !important;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.45) !important;
        color: #ffffff !important;
    }
    /* Inactive Mode Button (secondary kind) - Crisp, bright, high-contrast */
    .st-key-figma_stage2_toolbar [data-testid="stBaseButton-secondary"],
    .st-key-figma_stage2_toolbar [data-testid="baseButton-secondary"],
    .st-key-figma_stage2_toolbar button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
    }
    .st-key-figma_stage2_toolbar [data-testid="stBaseButton-secondary"]:hover,
    .st-key-figma_stage2_toolbar [data-testid="baseButton-secondary"]:hover,
    .st-key-figma_stage2_toolbar button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.22) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
        color: #ffffff !important;
    }

    /* Disabled state for undo/redo buttons */
    .st-key-figma_stage2_toolbar button:disabled {
        opacity: 0.28 !important;
        cursor: not-allowed !important;
        pointer-events: none !important;
        filter: grayscale(1) !important;
    }

    /* Hide text labels inside icon buttons */
    .st-key-figma_stage2_toolbar .st-key-btn_mode_view button p,
    .st-key-figma_stage2_toolbar .st-key-btn_mode_scissors button p,
    .st-key-figma_stage2_toolbar .st-key-btn_mode_quick_select button p,
    .st-key-figma_stage2_toolbar .st-key-btn_history_undo button p,
    .st-key-figma_stage2_toolbar .st-key-btn_history_redo button p,
    .st-key-figma_stage2_toolbar .st-key-btn_mode_view button [data-testid="stMarkdownContainer"],
    .st-key-figma_stage2_toolbar .st-key-btn_mode_scissors button [data-testid="stMarkdownContainer"],
    .st-key-figma_stage2_toolbar .st-key-btn_mode_quick_select button [data-testid="stMarkdownContainer"],
    .st-key-figma_stage2_toolbar .st-key-btn_history_undo button [data-testid="stMarkdownContainer"],
    .st-key-figma_stage2_toolbar .st-key-btn_history_redo button [data-testid="stMarkdownContainer"] {
        display: none !important;
    }

    /* SVG icons directly inside buttons - ALWAYS 100% white */
    .st-key-figma_stage2_toolbar .st-key-btn_mode_view button svg,
    .st-key-figma_stage2_toolbar .st-key-btn_mode_scissors button svg,
    .st-key-figma_stage2_toolbar .st-key-btn_mode_quick_select button svg,
    .st-key-figma_stage2_toolbar .st-key-btn_history_undo button svg,
    .st-key-figma_stage2_toolbar .st-key-btn_history_redo button svg {
        width: 24px !important;
        height: 24px !important;
        display: block !important;
        margin: auto !important;
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    .st-key-figma_stage2_toolbar .st-key-btn_mode_quick_select button svg path,
    .st-key-figma_stage2_toolbar .st-key-btn_mode_quick_select button svg circle {
        fill: #ffffff !important;
    }

    /* SVG icon pseudo-elements via CSS mask (fallback before JS injection) */
    .st-key-figma_stage2_toolbar .st-key-btn_mode_view button:not(:has(svg))::after,
    .st-key-figma_stage2_toolbar .st-key-btn_mode_scissors button:not(:has(svg))::after,
    .st-key-figma_stage2_toolbar .st-key-btn_mode_quick_select button:not(:has(svg))::after,
    .st-key-figma_stage2_toolbar .st-key-btn_history_undo button:not(:has(svg))::after,
    .st-key-figma_stage2_toolbar .st-key-btn_history_redo button:not(:has(svg))::after {
        content: "" !important;
        display: block !important;
        width: 22px !important;
        height: 22px !important;
        background-color: #ffffff !important;
        pointer-events: none !important;
        margin: auto !important;
    }

    /* 1. Cursor SVG from desktop cursor.svg */
    .st-key-figma_stage2_toolbar .st-key-btn_mode_view button:not(:has(svg))::after {
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='m9.448 3.487 10.887 8.989a1.82 1.82 0 0 1-1.168 3.224h-3.91a3.67 3.67 0 0 0-2.927 1.454l-2.356 3.117a1.83 1.83 0 0 1-3.288-1.007l-.686-14.064a2.1 2.1 0 0 1 3.448-1.713z'/%3E%3C/svg%3E") no-repeat center / contain !important;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='m9.448 3.487 10.887 8.989a1.82 1.82 0 0 1-1.168 3.224h-3.91a3.67 3.67 0 0 0-2.927 1.454l-2.356 3.117a1.83 1.83 0 0 1-3.288-1.007l-.686-14.064a2.1 2.1 0 0 1 3.448-1.713z'/%3E%3C/svg%3E") no-repeat center / contain !important;
    }

    /* 2. Scissors SVG from desktop scissors.svg */
    .st-key-figma_stage2_toolbar .st-key-btn_mode_scissors button:not(:has(svg))::after {
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'%3E%3Cpath d='M512,107.275c-23.658-33.787-70.696-42.691-104.489-19.033L233.753,209.907l-63.183-44.246c23.526-40.618,12.46-93.179-26.71-120.603c-41.364-28.954-98.355-18.906-127.321,22.45c-28.953,41.358-18.913,98.361,22.452,127.327c28.384,19.874,64.137,21.364,93.129,6.982l77.388,54.185l-77.381,54.179c-28.992-14.375-64.743-12.885-93.129,6.982c-41.363,28.966-51.404,85.963-22.452,127.32c28.966,41.363,85.963,51.411,127.32,22.457c39.165-27.424,50.229-79.985,26.71-120.603l63.183-44.246l173.751,121.658c33.793,23.665,80.831,14.755,104.489-19.033l-212.41-148.715L512,107.275z M91.627,167.539c-26.173,0-47.392-21.219-47.392-47.392s21.22-47.392,47.392-47.392c26.179,0,47.392,21.219,47.392,47.392S117.806,167.539,91.627,167.539z M91.627,439.253c-26.173,0-47.392-21.219-47.392-47.392c0-26.173,21.219-47.392,47.392-47.392c26.179,0,47.392,21.219,47.392,47.392C139.019,418.033,117.806,439.253,91.627,439.253z'/%3E%3C/svg%3E") no-repeat center / contain !important;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'%3E%3Cpath d='M512,107.275c-23.658-33.787-70.696-42.691-104.489-19.033L233.753,209.907l-63.183-44.246c23.526-40.618,12.46-93.179-26.71-120.603c-41.364-28.954-98.355-18.906-127.321,22.45c-28.953,41.358-18.913,98.361,22.452,127.327c28.384,19.874,64.137,21.364,93.129,6.982l77.388,54.185l-77.381,54.179c-28.992-14.375-64.743-12.885-93.129,6.982c-41.363,28.966-51.404,85.963-22.452,127.32c28.966,41.363,85.963,51.411,127.32,22.457c39.165-27.424,50.229-79.985,26.71-120.603l63.183-44.246l173.751,121.658c33.793,23.665,80.831,14.755,104.489-19.033l-212.41-148.715L512,107.275z M91.627,167.539c-26.173,0-47.392-21.219-47.392-47.392s21.22-47.392,47.392-47.392c26.179,0,47.392,21.219,47.392,47.392S117.806,167.539,91.627,167.539z M91.627,439.253c-26.173,0-47.392-21.219-47.392-47.392c0-26.173,21.219-47.392,47.392-47.392c26.179,0,47.392,21.219,47.392,47.392C139.019,418.033,117.806,439.253,91.627,439.253z'/%3E%3C/svg%3E") no-repeat center / contain !important;
    }

    /* 3. Task (Target / Reticle) SVG from desktop task.svg */
    .st-key-figma_stage2_toolbar .st-key-btn_mode_quick_select button:not(:has(svg))::after {
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 477.867 477.867'%3E%3Cpath fill='%23fff' d='M238.933,0C106.974,0,0,106.974,0,238.933s106.974,238.933,238.933,238.933s238.933-106.974,238.933-238.933C477.726,107.033,370.834,0.141,238.933,0z M238.933,443.733c-113.108,0-204.8-91.692-204.8-204.8s91.692-204.8,204.8-204.8s204.8,91.692,204.8,204.8C443.611,351.991,351.991,443.611,238.933,443.733z'/%3E%3Ccircle cx='238.933' cy='238.933' r='136.533' fill='%23fff'/%3E%3C/svg%3E") no-repeat center / contain !important;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 477.867 477.867'%3E%3Cpath fill='%23fff' d='M238.933,0C106.974,0,0,106.974,0,238.933s106.974,238.933,238.933,238.933s238.933-106.974,238.933-238.933C477.726,107.033,370.834,0.141,238.933,0z M238.933,443.733c-113.108,0-204.8-91.692-204.8-204.8s91.692-204.8,204.8-204.8s204.8,91.692,204.8,204.8C443.611,351.991,351.991,443.611,238.933,443.733z'/%3E%3Ccircle cx='238.933' cy='238.933' r='136.533' fill='%23fff'/%3E%3C/svg%3E") no-repeat center / contain !important;
    }

    /* 4. Undo SVG (curved arrow pointing left) */
    .st-key-figma_stage2_toolbar .st-key-btn_history_undo button:not(:has(svg))::after {
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23fff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9 14 4 9l5-5'/%3E%3Cpath d='M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5v0a5.5 5.5 0 0 1-5.5 5.5H11'/%3E%3C/svg%3E") no-repeat center / contain !important;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23fff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9 14 4 9l5-5'/%3E%3Cpath d='M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5v0a5.5 5.5 0 0 1-5.5 5.5H11'/%3E%3C/svg%3E") no-repeat center / contain !important;
    }

    /* 5. Redo SVG (curved arrow pointing right) */
    .st-key-figma_stage2_toolbar .st-key-btn_history_redo button:not(:has(svg))::after {
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23fff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m15 14 5-5-5-5'/%3E%3Cpath d='M20 9H9.5A5.5 5.5 0 0 0 4 14.5v0A5.5 5.5 0 0 0 9.5 20H13'/%3E%3C/svg%3E") no-repeat center / contain !important;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23fff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m15 14 5-5-5-5'/%3E%3Cpath d='M20 9H9.5A5.5 5.5 0 0 0 4 14.5v0A5.5 5.5 0 0 0 9.5 20H13'/%3E%3C/svg%3E") no-repeat center / contain !important;
    }

    /* Hide any tooltip icons (? icon) inside toolbar */
    .st-key-figma_stage2_toolbar [data-testid="stTooltipIcon"] {
        display: none !important;
    }

    /* ---- Tool controls row: comfortable spacing ---- */
    .st-key-figma_stage2_toolbar [data-testid="stSlider"] {
        margin: 0 !important;
        padding: 0 4px !important;
        width: 120px !important;
        min-width: 115px !important;
        max-width: 130px !important;
    }
    .st-key-figma_stage2_toolbar [data-testid="stSlider"] > div {
        width: 100% !important;
    }
    .st-key-figma_stage2_toolbar [data-testid="stSlider"] [data-baseweb="slider"] {
        width: 100% !important;
        min-width: 110px !important;
        margin: 2px 0 0 0 !important;
    }
    .st-key-figma_stage2_toolbar [data-testid="stSlider"] label {
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #f4f4f5 !important;
        margin: 0 0 1px 0 !important;
        padding: 0 !important;
        line-height: 1.1 !important;
        white-space: nowrap !important;
    }
    .st-key-figma_stage2_toolbar [data-testid="stSlider"] label p {
        font-size: 11px !important;
        margin: 0 !important;
        white-space: nowrap !important;
    }
    /* Hide bottom min/max numbers to keep slider compact and unclipped */
    .st-key-figma_stage2_toolbar [data-testid="stSlider"] [data-baseweb="slider"] ~ div,
    .st-key-figma_stage2_toolbar [data-testid="stSlider"] [data-testid="stTickBar"] {
        display: none !important;
    }

    /* Action buttons (Kesish, Qo'llash) - wide enough so text is NEVER clipped */
    .st-key-figma_stage2_toolbar .st-key-floating_apply_brush,
    .st-key-figma_stage2_toolbar .st-key-floating_apply_scissors {
        width: 110px !important;
        min-width: 105px !important;
        max-width: 115px !important;
    }
    .st-key-figma_stage2_toolbar .st-key-floating_apply_brush button,
    .st-key-figma_stage2_toolbar .st-key-floating_apply_scissors button {
        width: 110px !important;
        min-width: 105px !important;
        height: 38px !important;
        min-height: 38px !important;
        padding: 0 8px !important;
        font-size: 12.5px !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        border-radius: 9px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-sizing: border-box !important;
    }

    /* Caption text inside toolbar */
    .st-key-figma_stage2_toolbar [data-testid="stCaptionContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    .st-key-figma_stage2_toolbar [data-testid="stCaptionContainer"] p {
        font-size: 10px !important;
        color: #71717a !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }

    /* Remove default Streamlit block spacing inside toolbar */
    .st-key-figma_stage2_toolbar [data-testid="stVerticalBlock"] {
        gap: 2px !important;
    }

    /* Minimized state - compact capsule */
    .st-key-figma_stage2_toolbar.is-minimized {
        width: auto !important;
        min-width: 100px !important;
        padding: 4px 16px !important;
        border-radius: 20px !important;
        cursor: pointer !important;
    }
    .st-key-figma_stage2_toolbar.is-minimized .figma-header-bar {
        margin-bottom: 0 !important;
    }
    .st-key-figma_stage2_toolbar.is-minimized > div:not(:first-child) {
        display: none !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    js = f"""
    <script>
    (function() {{
        const ACTIVE_MODE = "{active_mode}";
        const BRUSH_SIZE = {int(brush_size)};

        function getCursorDataUri(size) {{
            const d = Math.max(8, Math.min(128, Math.round(size)));
            const r = Math.round(d / 2);
            const r_circ = Math.max(1, r - 1.5);

            const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + d + '" height="' + d + '" viewBox="0 0 ' + d + ' ' + d + '">' +
                '<circle cx="' + r + '" cy="' + r + '" r="' + r_circ + '" fill="rgba(16, 185, 129, 0.18)" stroke="#000000" stroke-width="2.5" opacity="0.8"/>' +
                '<circle cx="' + r + '" cy="' + r + '" r="' + r_circ + '" fill="none" stroke="#10b981" stroke-width="1.5" stroke-dasharray="3,2"/>' +
                '<line x1="' + (r - 3) + '" y1="' + r + '" x2="' + (r + 3) + '" y2="' + r + '" stroke="#000000" stroke-width="2"/>' +
                '<line x1="' + r + '" y1="' + (r - 3) + '" x2="' + r + '" y2="' + (r + 3) + '" stroke="#000000" stroke-width="2"/>' +
                '<line x1="' + (r - 3) + '" y1="' + r + '" x2="' + (r + 3) + '" y2="' + r + '" stroke="#ffffff" stroke-width="1"/>' +
                '<line x1="' + r + '" y1="' + (r - 3) + '" x2="' + r + '" y2="' + (r + 3) + '" stroke="#ffffff" stroke-width="1"/>' +
                '</svg>';

            return 'url("data:image/svg+xml,' + encodeURIComponent(svg) + '") ' + r + ' ' + r + ', crosshair';
        }}

        function getCursorCssRule(size) {{
            const uri = getCursorDataUri(size);
            return '.upper-canvas, .canvas-container {{ cursor: ' + uri + ' !important; }}';
        }}

        function setupToolbar(panel) {{
            if (!panel) return;

            // Purge old static position caches
            try {{
                window.parent.sessionStorage.removeItem('figma_toolbar_pos');
                window.parent.sessionStorage.removeItem('figma_toolbar_minimized');
                window.parent.sessionStorage.removeItem('figma_panel_pos');
                window.parent.sessionStorage.removeItem('figma_panel_pos_v2');
            }} catch(e) {{}}

            // Restore center-anchored position from sessionStorage (v3)
            const savedPos = window.parent.sessionStorage.getItem('figma_panel_pos_v3');
            if (savedPos) {{
                try {{
                    const pos = JSON.parse(savedPos);
                    const docW = window.parent.innerWidth;
                    const docH = window.parent.innerHeight;
                    const pW = panel.offsetWidth || 500;
                    const pH = panel.offsetHeight || 50;
                    if (pos.centerX >= 0 && pos.centerX <= docW) {{
                        let left = Math.round(pos.centerX - pW / 2);
                        left = Math.max(12, Math.min(docW - pW - 12, left));
                        let top = Math.round(pos.top);
                        if (top >= 10 && top <= docH - pH - 10) {{
                            panel.style.top = top + 'px';
                            panel.style.bottom = 'auto';
                        }} else {{
                            panel.style.bottom = '24px';
                            panel.style.top = 'auto';
                        }}
                        panel.style.transform = 'none';
                        panel.style.left = left + 'px';
                    }}
                }} catch(e) {{
                    window.parent.sessionStorage.removeItem('figma_panel_pos_v3');
                }}
            }} else {{
                panel.style.bottom = '24px';
                panel.style.top = 'auto';
                panel.style.left = '50%';
                panel.style.transform = 'translateX(-50%)';
            }}

            // Restore minimize state (v2)
            const isMin = window.parent.sessionStorage.getItem('figma_panel_min_v2') === 'true';
            if (isMin) {{
                panel.classList.add('is-minimized');
                const mb = panel.querySelector('.figma-min-toggle');
                if (mb) mb.innerText = '◻';
            }}

            // Setup Dragging
            const header = panel.querySelector('.figma-header-bar');
            if (header && !header.dataset.dragAttached) {{
                header.dataset.dragAttached = 'true';
                let isDragging = false;
                let startX = 0, startY = 0;
                let origLeft = 0, origTop = 0;

                header.addEventListener('mousedown', (e) => {{
                    if (e.target.closest('.figma-min-toggle')) return;
                    isDragging = true;
                    panel.classList.add('is-dragging');
                    const rect = panel.getBoundingClientRect();
                    startX = e.clientX;
                    startY = e.clientY;
                    origLeft = rect.left;
                    origTop = rect.top;

                    panel.style.bottom = 'auto';
                    panel.style.transform = 'none';
                    panel.style.left = origLeft + 'px';
                    panel.style.top = origTop + 'px';
                    e.preventDefault();
                }});

                window.parent.document.addEventListener('mousemove', (e) => {{
                    if (!isDragging) return;
                    const dx = e.clientX - startX;
                    const dy = e.clientY - startY;
                    const docW = window.parent.innerWidth;
                    const docH = window.parent.innerHeight;
                    const pW = panel.offsetWidth;
                    const pH = panel.offsetHeight;

                    let newLeft = Math.max(10, Math.min(docW - pW - 10, origLeft + dx));
                    let newTop = Math.max(10, Math.min(docH - pH - 10, origTop + dy));
                    panel.style.left = newLeft + 'px';
                    panel.style.top = newTop + 'px';
                }});

                window.parent.document.addEventListener('mouseup', () => {{
                    if (!isDragging) return;
                    isDragging = false;
                    panel.classList.remove('is-dragging');
                    const rect = panel.getBoundingClientRect();
                    window.parent.sessionStorage.setItem('figma_panel_pos_v3', JSON.stringify({{
                        centerX: Math.round(rect.left + rect.width / 2),
                        top: Math.round(rect.top)
                    }}));
                }});

                // Double click resets toolbar back to bottom-center
                header.addEventListener('dblclick', () => {{
                    window.parent.sessionStorage.removeItem('figma_panel_pos_v3');
                    panel.style.top = 'auto';
                    panel.style.bottom = '24px';
                    panel.style.left = '50%';
                    panel.style.transform = 'translateX(-50%)';
                }});
            }}

            // Setup Minimize Button
            const minBtn = panel.querySelector('.figma-min-toggle');
            if (minBtn && !minBtn.dataset.minAttached) {{
                minBtn.dataset.minAttached = 'true';
                minBtn.addEventListener('click', (e) => {{
                    e.stopPropagation();
                    panel.classList.toggle('is-minimized');
                    const min = panel.classList.contains('is-minimized');
                    minBtn.innerText = min ? '◻' : '—';
                    minBtn.title = min ? 'Kattalashtirish (Ochish)' : 'Kichraytirish';
                    window.parent.sessionStorage.setItem('figma_panel_min_v2', min ? 'true' : 'false');
                }});
            }}

            // Header click when minimized expands back
            if (header && !header.dataset.expandAttached) {{
                header.dataset.expandAttached = 'true';
                header.addEventListener('click', (e) => {{
                    if (panel.classList.contains('is-minimized') && !e.target.closest('.figma-min-toggle')) {{
                        panel.classList.remove('is-minimized');
                        const mb = panel.querySelector('.figma-min-toggle');
                        if (mb) mb.innerText = '—';
                        window.parent.sessionStorage.setItem('figma_panel_min_v2', 'false');
                    }}
                }});
            }}

            // Inject inline SVGs into mode buttons - 100% Crisp White
            const svgMap = {{
                'btn_mode_view': {{
                    svg: '<svg viewBox="0 0 24 24" width="24" height="24" fill="#ffffff" style="display:block;margin:auto;"><path fill="#ffffff" d="m9.448 3.487 10.887 8.989a1.82 1.82 0 0 1-1.168 3.224h-3.91a3.67 3.67 0 0 0-2.927 1.454l-2.356 3.117a1.83 1.83 0 0 1-3.288-1.007l-.686-14.064a2.1 2.1 0 0 1 3.448-1.713z"/></svg>',
                    title: "Ko'rish / Kursor rejimi (V)"
                }},
                'btn_mode_scissors': {{
                    svg: '<svg viewBox="0 0 512 512" width="24" height="24" fill="#ffffff" style="display:block;margin:auto;"><path fill="#ffffff" d="M512,107.275c-23.658-33.787-70.696-42.691-104.489-19.033L233.753,209.907l-63.183-44.246c23.526-40.618,12.46-93.179-26.71-120.603c-41.364-28.954-98.355-18.906-127.321,22.45c-28.953,41.358-18.913,98.361,22.452,127.327c28.384,19.874,64.137,21.364,93.129,6.982l77.388,54.185l-77.381,54.179c-28.992-14.375-64.743-12.885-93.129,6.982c-41.363,28.966-51.404,85.963-22.452,127.32c28.966,41.363,85.963,51.411,127.32,22.457c39.165-27.424,50.229-79.985,26.71-120.603l63.183-44.246l173.751,121.658c33.793,23.665,80.831,14.755,104.489-19.033l-212.41-148.715L512,107.275z M91.627,167.539c-26.173,0-47.392-21.219-47.392-47.392s21.22-47.392,47.392-47.392c26.179,0,47.392,21.219,47.392,47.392S117.806,167.539,91.627,167.539z M91.627,439.253c-26.173,0-47.392-21.219-47.392-47.392c0-26.173,21.219-47.392,47.392-47.392c26.179,0,47.392,21.219,47.392,47.392C139.019,418.033,117.806,439.253,91.627,439.253z"/></svg>',
                    title: "Qaychi / Kesish rejimi (C)"
                }},
                'btn_mode_quick_select': {{
                    svg: '<svg viewBox="0 0 477.867 477.867" width="24" height="24" fill="#ffffff" style="display:block;margin:auto;"><path fill="#ffffff" d="M238.933,0C106.974,0,0,106.974,0,238.933s106.974,238.933,238.933,238.933s238.933-106.974,238.933-238.933C477.726,107.033,370.834,0.141,238.933,0z M238.933,443.733c-113.108,0-204.8-91.692-204.8-204.8s91.692-204.8,204.8-204.8s204.8,91.692,204.8,204.8C443.611,351.991,351.991,443.611,238.933,443.733z"/><circle cx="238.933" cy="238.933" r="136.533" fill="#ffffff"/></svg>',
                    title: "Photoshop Quick Select / Tezkor tanlash (W)"
                }},
                'btn_history_undo': {{
                    svg: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#ffffff" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" style="display:block;margin:auto;"><path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5v0a5.5 5.5 0 0 1-5.5 5.5H11"/></svg>',
                    title: "Bekor qilish (Ctrl+Z)"
                }},
                'btn_history_redo': {{
                    svg: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#ffffff" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" style="display:block;margin:auto;"><path d="m15 14 5-5-5-5"/><path d="M20 9H9.5A5.5 5.5 0 0 0 4 14.5v0A5.5 5.5 0 0 0 9.5 20H13"/></svg>',
                    title: "Qaytarish (Ctrl+Y)"
                }}
            }};
            let injectedCount = 0;
            for (const [k, item] of Object.entries(svgMap)) {{
                const c = panel.querySelector('.st-key-' + k);
                if (c) {{
                    const btn = c.querySelector('button');
                    if (btn) {{
                        if (btn.dataset.svgInjected !== k) {{
                            btn.innerHTML = item.svg;
                            btn.title = item.title;
                            btn.dataset.svgInjected = k;
                        }}
                        injectedCount++;
                    }}
                }}
            }}
            panel.dataset.svgInjectedCount = injectedCount;
        }}

        // Setup Dynamic Photoshop-Grade Round Cursor on Canvas
        function syncCanvasCursorSize(size) {{
            try {{
                const iframes = window.parent.document.querySelectorAll('iframe');
                iframes.forEach(ifr => {{
                    let doc;
                    try {{ doc = ifr.contentDocument || ifr.contentWindow.document; }} catch(e) {{ return; }}
                    if (!doc || !doc.head) return;

                    const upperCanvas = doc.querySelector('.upper-canvas');
                    if (!upperCanvas) return;

                    let styleEl = doc.getElementById('comic-lab-canvas-cursor');
                    if (!styleEl) {{
                        styleEl = doc.createElement('style');
                        styleEl.id = 'comic-lab-canvas-cursor';
                        doc.head.appendChild(styleEl);
                    }}

                    if (ACTIVE_MODE === 'quick_select') {{
                        styleEl.textContent = getCursorCssRule(size);
                        upperCanvas.style.setProperty('cursor', getCursorDataUri(size), 'important');
                    }} else if (ACTIVE_MODE === 'scissors') {{
                        styleEl.textContent = '.upper-canvas, .canvas-container {{ cursor: crosshair !important; }}';
                        upperCanvas.style.setProperty('cursor', 'crosshair', 'important');
                    }} else {{
                        styleEl.textContent = '.upper-canvas, .canvas-container {{ cursor: default !important; }}';
                        upperCanvas.style.setProperty('cursor', 'default', 'important');
                    }}
                }});
            }} catch(e) {{}}
        }}

        function setupCanvasCursor() {{
            try {{
                const iframes = window.parent.document.querySelectorAll('iframe');
                let foundCanvas = false;
                iframes.forEach(ifr => {{
                    let doc;
                    try {{
                        doc = ifr.contentDocument || ifr.contentWindow.document;
                    }} catch(e) {{ return; }}
                    if (!doc || !doc.head) return;

                    const upperCanvas = doc.querySelector('.upper-canvas');
                    if (!upperCanvas) return;

                    foundCanvas = true;

                    let styleEl = doc.getElementById('comic-lab-canvas-cursor');
                    if (!styleEl) {{
                        styleEl = doc.createElement('style');
                        styleEl.id = 'comic-lab-canvas-cursor';
                        doc.head.appendChild(styleEl);
                    }}

                    const currentSize = parseInt(window.parent.document.querySelector('.st-key-brush_size_slider input[type="range"]')?.value, 10) || BRUSH_SIZE;

                    if (ACTIVE_MODE === 'quick_select') {{
                        const curRule = getCursorCssRule(currentSize);
                        if (styleEl.textContent !== curRule) {{
                            styleEl.textContent = curRule;
                        }}
                        upperCanvas.style.setProperty('cursor', getCursorDataUri(currentSize), 'important');

                        if (!upperCanvas.dataset.psCursorBound) {{
                            upperCanvas.dataset.psCursorBound = 'true';
                            upperCanvas.addEventListener('pointerenter', () => {{
                                if (ACTIVE_MODE === 'quick_select') {{
                                    const s = parseInt(window.parent.document.querySelector('.st-key-brush_size_slider input[type="range"]')?.value, 10) || BRUSH_SIZE;
                                    styleEl.textContent = getCursorCssRule(s);
                                    upperCanvas.style.setProperty('cursor', getCursorDataUri(s), 'important');
                                }}
                            }});
                        }}
                    }} else if (ACTIVE_MODE === 'scissors') {{
                        styleEl.textContent = '.upper-canvas, .canvas-container {{ cursor: crosshair !important; }}';
                        upperCanvas.style.setProperty('cursor', 'crosshair', 'important');
                    }} else {{
                        styleEl.textContent = '.upper-canvas, .canvas-container {{ cursor: default !important; }}';
                        upperCanvas.style.setProperty('cursor', 'default', 'important');
                    }}
                }});
                return foundCanvas;
            }} catch(e) {{ return false; }}
        }}

        // Setup Photoshop [ and ] keyboard shortcuts, Mode keys (V, C, W), and Undo/Redo (Ctrl+Z, Ctrl+Y, Ctrl+Shift+Z)
        function setupKeyboardShortcuts() {{
            function handleKeyDown(e) {{
                // Skip shortcuts if user is typing in any text box or editable element
                if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable)) return;

                const isCtrlOrCmd = e.ctrlKey || e.metaKey;
                if (isCtrlOrCmd) {{
                    const k = e.key.toLowerCase();
                    if (k === 'z') {{
                        if (e.shiftKey) {{
                            // Redo (Ctrl+Shift+Z / Cmd+Shift+Z)
                            e.preventDefault();
                            const rBtn = window.parent.document.querySelector('.st-key-btn_history_redo button') || window.parent.document.querySelector('.st-key-panel_redo_st2 button') || window.parent.document.querySelector('.st-key-panel_redo_st3 button');
                            if (rBtn && !rBtn.disabled) {{
                                rBtn.click();
                            }}
                            return;
                        }} else {{
                            // Undo (Ctrl+Z / Cmd+Z)
                            e.preventDefault();
                            const uBtn = window.parent.document.querySelector('.st-key-btn_history_undo button') || window.parent.document.querySelector('.st-key-panel_undo_st2 button') || window.parent.document.querySelector('.st-key-panel_undo_st3 button');
                            if (uBtn && !uBtn.disabled) {{
                                uBtn.click();
                            }}
                            return;
                        }}
                    }} else if (k === 'y') {{
                        // Redo (Ctrl+Y / Cmd+Y)
                        e.preventDefault();
                        const rBtn = window.parent.document.querySelector('.st-key-btn_history_redo button') || window.parent.document.querySelector('.st-key-panel_redo_st2 button') || window.parent.document.querySelector('.st-key-panel_redo_st3 button');
                        if (rBtn && !rBtn.disabled) {{
                            rBtn.click();
                        }}
                        return;
                    }}
                }}

                // Tool mode hotkeys: V (View), C (Scissors), W (Quick Select)
                if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {{
                    const k = e.key.toLowerCase();
                    if (k === 'v') {{
                        const btnV = window.parent.document.querySelector('.st-key-btn_mode_view button');
                        if (btnV) btnV.click();
                        return;
                    }} else if (k === 'c') {{
                        const btnC = window.parent.document.querySelector('.st-key-btn_mode_scissors button');
                        if (btnC) btnC.click();
                        return;
                    }} else if (k === 'w') {{
                        const btnW = window.parent.document.querySelector('.st-key-btn_mode_quick_select button');
                        if (btnW) btnW.click();
                        return;
                    }}
                }}

                // Brush sizing shortcuts '[' and ']'
                if (ACTIVE_MODE !== 'quick_select') return;
                if (e.key === '[' || e.key === ']') {{
                    const slider = window.parent.document.querySelector('.st-key-brush_size_slider input[type="range"]');
                    if (slider) {{
                        const cur = parseInt(slider.value, 10) || 30;
                        const delta = e.key === '[' ? -5 : 5;
                        const nextVal = Math.max(10, Math.min(100, cur + delta));
                        if (nextVal !== cur) {{
                            // Instantly update canvas cursor at 0ms latency
                            syncCanvasCursorSize(nextVal);

                            const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                            nativeSetter.call(slider, nextVal);
                            slider.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            slider.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        }}
                    }}
                }}
            }}
            if (!window.parent._psShortcutsBound) {{
                window.parent._psShortcutsBound = true;
                window.parent.document.addEventListener('keydown', handleKeyDown);
            }}

            // Also attach live input listener to slider for real-time cursor resizing on mouse drag
            try {{
                const slider = window.parent.document.querySelector('.st-key-brush_size_slider input[type="range"]');
                if (slider && !slider.dataset.cursorSyncBound) {{
                    slider.dataset.cursorSyncBound = 'true';
                    slider.addEventListener('input', (e) => {{
                        const v = parseInt(e.target.value, 10);
                        if (v && !isNaN(v)) {{
                            syncCanvasCursorSize(v);
                        }}
                    }});
                }}
            }} catch(err) {{}}
        }}

        let toolbarConfigured = false;
        let canvasConfigured = false;

        function checkAndApply() {{
            try {{
                if (!toolbarConfigured) {{
                    const panels = window.parent.document.querySelectorAll('.st-key-figma_stage2_toolbar');
                    if (panels.length > 0) {{
                        let allDone = true;
                        panels.forEach(p => {{
                            setupToolbar(p);
                            if (!p.dataset.svgInjectedCount || parseInt(p.dataset.svgInjectedCount) < 5) {{
                                allDone = false;
                            }}
                        }});
                        if (allDone) toolbarConfigured = true;
                    }}
                }}

                if (!canvasConfigured) {{
                    const found = setupCanvasCursor();
                    if (found) canvasConfigured = true;
                }}

                setupKeyboardShortcuts();

                if (toolbarConfigured && canvasConfigured) {{
                    if (window.parent._stage2Observer) {{
                        try {{ window.parent._stage2Observer.disconnect(); }} catch(e) {{}}
                        window.parent._stage2Observer = null;
                    }}
                }}
            }} catch(e) {{}}
        }}

        // Clean up any previously running observer
        if (window.parent._stage2Observer) {{
            try {{ window.parent._stage2Observer.disconnect(); }} catch(e) {{}}
            window.parent._stage2Observer = null;
        }}

        // Staggered lightweight initialization
        [0, 50, 100, 200, 400, 750, 1500, 3000].forEach(ms => {{
            setTimeout(() => {{ if (!toolbarConfigured || !canvasConfigured) checkAndApply(); }}, ms);
        }});

        // Debounced self-terminating observer
        let rafId = null;
        try {{
            const obs = new MutationObserver(() => {{
                if (toolbarConfigured && canvasConfigured) {{
                    obs.disconnect();
                    return;
                }}
                if (rafId) return;
                rafId = window.parent.requestAnimationFrame(() => {{
                    rafId = null;
                    checkAndApply();
                }});
            }});
            obs.observe(window.parent.document.body, {{ childList: true, subtree: true }});
            window.parent._stage2Observer = obs;

            // Failsafe auto-disconnect after 4 seconds
            setTimeout(() => {{
                try {{ obs.disconnect(); }} catch(e) {{}}
                if (window.parent._stage2Observer === obs) {{
                    window.parent._stage2Observer = null;
                }}
            }}, 4000);
        }} catch(e) {{}}
    }})();
    </script>
    """
    components.html(js, height=0, width=0)


# --------------------------------------------------------------------------
# STAGE 2 INTERACTIVE UI COMPONENT
# --------------------------------------------------------------------------

def render_stage2_mask_editor(
    annotated_page_image: Any,
    page_bgr: np.ndarray,
    page_name: str = "current_page.png"
):
    """
    Renders Stage 2 interactive bubble-mask editing toolbar and canvas:
    - Tab 1: 👁️ Sahifa Ko'rinishi (Full annotated page)
    - Tab 2: ✂️ Qaychi (Scissors - Split merged bubble with floating toolbar)
    - Tab 3: 🖌️ Aqlli Mo'yqalam (Smart Brush - Grow missed regions with floating toolbar)
    """
    import streamlit as st
    from PIL import Image
    patch_streamlit_image_to_url()
    from streamlit_drawable_canvas import st_canvas
    import engine

    bubbles = st.session_state.get("bubbles", [])
    if not bubbles:
        st.info("Pufaklar mavjud emas.")
        return

    H, W = page_bgr.shape[:2]

    # Ensure tool mode state
    if "stage2_tool_mode" not in st.session_state:
        st.session_state.stage2_tool_mode = "view"
    active_mode = st.session_state.stage2_tool_mode

    # Brush size state
    brush_size = st.session_state.get("brush_size_slider", 30)

    # Inject Figma-style floating toolbar CSS & JS (Photoshop round cursor + dynamic size)
    inject_floating_toolbar_styles_and_js(active_mode=active_mode, brush_size=int(brush_size))

    st.markdown("### Sahifa va Pufak Tahriri")

    # Handle pending ambiguous line resolution
    pending = st.session_state.get("pending_scissors_split")
    if pending is not None:
        sel_idx, sub_b1, sub_b2, amb_lines = pending
        if amb_lines:
            cur_line = amb_lines[0]
            st.warning(
                f"⚠️ **Chegara ustidagi satr**: '{cur_line.get('text', '')}' satri qaysi pufakka tegishli?"
            )
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if st.button(f"1-pufakka biriktirish ({sub_b1['original_text'][:20]}...)", use_container_width=True):
                    sub_b1["lines"].append(cur_line)
                    sub_b1["original_text"] = " ".join(l['text'] for l in sub_b1["lines"]).strip()
                    amb_lines.pop(0)
                    if not amb_lines:
                        _finalize_split(sel_idx, sub_b1, sub_b2, page_name)
                    else:
                        st.session_state.pending_scissors_split = (sel_idx, sub_b1, sub_b2, amb_lines)
                        st.rerun()
            with col_c2:
                if st.button(f"2-pufakka biriktirish ({sub_b2['original_text'][:20]}...)", use_container_width=True):
                    sub_b2["lines"].append(cur_line)
                    sub_b2["original_text"] = " ".join(l['text'] for l in sub_b2["lines"]).strip()
                    amb_lines.pop(0)
                    if not amb_lines:
                        _finalize_split(sel_idx, sub_b1, sub_b2, page_name)
                    else:
                        st.session_state.pending_scissors_split = (sel_idx, sub_b1, sub_b2, amb_lines)
                        st.rerun()
            return

    # Scale annotated full page image for display
    if isinstance(annotated_page_image, Image.Image):
        bg_pil = annotated_page_image.convert("RGB")
    else:
        bg_pil = Image.fromarray(cv2.cvtColor(annotated_page_image, cv2.COLOR_BGR2RGB))

    disp_w = min(W, 800)
    canvas_scale = disp_w / float(W)
    disp_h = int(round(H * canvas_scale))
    bg_disp_pil = bg_pil.resize((disp_w, disp_h), Image.Resampling.LANCZOS)

    # Configure canvas parameters depending on active mode
    if active_mode == "scissors":
        draw_mode = "freedraw"
        stroke_color = "#ef4444"
        stroke_width = max(3, int(round(4 * canvas_scale)))
        fill_color = "rgba(0, 0, 0, 0)"
    elif active_mode == "quick_select":
        draw_mode = "freedraw"
        stroke_color = "#10b981"
        stroke_width = max(8, int(brush_size))
        fill_color = "rgba(0, 0, 0, 0)"
    else:  # "view"
        draw_mode = "freedraw"
        stroke_color = "rgba(0, 0, 0, 0)"
        stroke_width = 0
        fill_color = "rgba(0, 0, 0, 0)"

    canvas_version = st.session_state.get("analysis_version", 0)
    canvas_key = f"stage2_main_page_canvas_{canvas_version}"

    if active_mode in ("scissors", "quick_select"):
        canvas_res = st_canvas(
            fill_color=fill_color,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=bg_disp_pil,
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="freedraw",
            key=canvas_key
        )
    else:
        st.image(bg_disp_pil, use_container_width=True)
        canvas_res = None

    # Check for drawn stroke on canvas
    has_drawn_stroke = (
        canvas_res is not None and
        canvas_res.image_data is not None and
        np.count_nonzero(canvas_res.image_data[:, :, 3] > 20) >= 4
    )

    # Optional auto-detect target bubble from stroke position
    if has_drawn_stroke:
        stroke_disp_chk = (canvas_res.image_data[:, :, 3] > 20).astype(np.uint8) * 255
        stroke_page_chk = cv2.resize(stroke_disp_chk, (W, H), interpolation=cv2.INTER_NEAREST)
        best_overlap = 0
        best_idx = None
        for idx_chk, b_chk in enumerate(bubbles):
            cnt_chk = b_chk.get("contour") or b_chk.get("contour_points")
            if cnt_chk:
                m_chk = create_bubble_mask(cnt_chk, (H, W))
                m_chk_dil = cv2.dilate(m_chk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
                ov = int(np.count_nonzero(cv2.bitwise_and(m_chk_dil, stroke_page_chk)))
                if ov > best_overlap:
                    best_overlap = ov
                    best_idx = idx_chk
        if best_idx is not None and best_overlap > 10:
            if active_mode == "scissors":
                if st.session_state.get("last_auto_scissors_idx") != best_idx:
                    st.session_state.scissors_bubble_select = best_idx
                    st.session_state.last_auto_scissors_idx = best_idx
            elif active_mode == "quick_select":
                if st.session_state.get("last_auto_brush_idx") != best_idx:
                    st.session_state.brush_bubble_select = best_idx
                    st.session_state.last_auto_brush_idx = best_idx

    # ----------------------------------------------------------------------
    # FIGMA-STYLE COMPACT FLOATING TOOLBAR
    # Thin horizontal bar at bottom center, like Figma's toolbar.
    # Drag pill on top; mode toggles + tool controls in one row.
    # ----------------------------------------------------------------------
    floating_box = st.container(border=True, key="figma_stage2_toolbar")
    with floating_box:
        # Drag pill handle
        st.markdown("""
        <div class="figma-header-bar" id="figma-header-stage2">
            <span class="figma-drag-grip" title="Siljitish uchun ushlab torting">⠿</span>
            <span class="figma-tool-title"></span>
            <button class="figma-min-toggle" type="button" title="Kichraytirish">—</button>
        </div>
        """, unsafe_allow_html=True)

        can_undo = history_manager.can_undo()
        can_redo = history_manager.can_redo()
        undo_desc = history_manager.get_undo_description()
        undo_tooltip = f"Bekor qilish: {undo_desc} (Ctrl+Z)" if undo_desc else "Bekor qilish (Ctrl+Z)"
        redo_tooltip = "Qaytarish (Ctrl+Y)"

        # --- ALL controls in a compact horizontal row ---
        if active_mode == "view":
            # View mode: 3 mode buttons + undo/redo buttons strictly fitting icons
            c1, c2, c3, cu, cr = st.columns([1, 1, 1, 1, 1], gap="small")
            with c1:
                if st.button(" ", type="primary", use_container_width=True, key="btn_mode_view"):
                    st.session_state.stage2_tool_mode = "view"
                    st.rerun()
            with c2:
                if st.button(" ", type="secondary", use_container_width=True, key="btn_mode_scissors"):
                    st.session_state.stage2_tool_mode = "scissors"
                    st.rerun()
            with c3:
                if st.button(" ", type="secondary", use_container_width=True, key="btn_mode_quick_select"):
                    st.session_state.stage2_tool_mode = "quick_select"
                    st.rerun()
            with cu:
                if st.button(" ", type="secondary", disabled=not can_undo, use_container_width=True, key="btn_history_undo", help=undo_tooltip):
                    succ, desc = history_manager.undo_action()
                    if succ:
                        st.toast(f"↩️ Bekor qilindi: {desc}", icon=":material/undo:")
                        st.rerun()
            with cr:
                if st.button(" ", type="secondary", disabled=not can_redo, use_container_width=True, key="btn_history_redo", help=redo_tooltip):
                    succ, desc = history_manager.redo_action()
                    if succ:
                        st.toast(f"↪️ Qaytarildi: {desc}", icon=":material/redo:")
                        st.rerun()

        elif active_mode == "scissors":
            # Scissors mode: 3 mode buttons + undo/redo | apply button
            c1, c2, c3, cu, cr, c4 = st.columns([1, 1, 1, 1, 1, 2.5], gap="small")
            with c1:
                if st.button(" ", type="secondary", use_container_width=True, key="btn_mode_view"):
                    st.session_state.stage2_tool_mode = "view"
                    st.rerun()
            with c2:
                if st.button(" ", type="primary", use_container_width=True, key="btn_mode_scissors"):
                    st.session_state.stage2_tool_mode = "scissors"
                    st.rerun()
            with c3:
                if st.button(" ", type="secondary", use_container_width=True, key="btn_mode_quick_select"):
                    st.session_state.stage2_tool_mode = "quick_select"
                    st.rerun()
            with cu:
                if st.button(" ", type="secondary", disabled=not can_undo, use_container_width=True, key="btn_history_undo", help=undo_tooltip):
                    succ, desc = history_manager.undo_action()
                    if succ:
                        st.toast(f"↩️ Bekor qilindi: {desc}", icon=":material/undo:")
                        st.rerun()
            with cr:
                if st.button(" ", type="secondary", disabled=not can_redo, use_container_width=True, key="btn_history_redo", help=redo_tooltip):
                    succ, desc = history_manager.redo_action()
                    if succ:
                        st.toast(f"↪️ Qaytarildi: {desc}", icon=":material/redo:")
                        st.rerun()
            with c4:
                if st.button("✂️ Kesish", type="primary", use_container_width=True, key="floating_apply_scissors"):
                    if not has_drawn_stroke or canvas_res is None or canvas_res.image_data is None or np.count_nonzero(canvas_res.image_data[:, :, 3] > 20) < 10:
                        st.toast("⚠️ Avval pufak ustidan chiziq torting!", icon=":material/warning:")
                    else:
                        stroke_disp = (canvas_res.image_data[:, :, 3] > 20).astype(np.uint8) * 255
                        stroke_page = cv2.resize(stroke_disp, (W, H), interpolation=cv2.INTER_NEAREST)
                        stroke_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
                        stroke_page = cv2.dilate(stroke_page, stroke_kernel, iterations=1)

                        # Auto-detect target bubble from stroke overlap
                        target_idx = None
                        best_overlap = 0
                        for idx_chk, b_chk in enumerate(bubbles):
                            cnt_chk = b_chk.get("contour") or b_chk.get("contour_points")
                            if cnt_chk:
                                m_chk = create_bubble_mask(cnt_chk, (H, W))
                                m_chk_dil = cv2.dilate(m_chk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
                                ov = int(np.count_nonzero(cv2.bitwise_and(m_chk_dil, stroke_page)))
                                if ov > best_overlap:
                                    best_overlap = ov
                                    target_idx = idx_chk

                        if target_idx is None or best_overlap < 10:
                            st.toast("⚠️ Chiziq birorta ham pufakka tegmayapti!", icon=":material/warning:")
                        else:
                            target_b = bubbles[target_idx]
                            success, msg, sub_bubbles, amb_lines = apply_scissors_split(
                                target_b,
                                stroke_page,
                                page_bgr,
                                all_page_lines=st.session_state.get("raw_ocr_lines", [])
                            )

                            if not success:
                                st.toast(f"❌ {msg}", icon=":material/error:")
                            elif amb_lines:
                                st.session_state.pending_scissors_split = (target_idx, sub_bubbles[0], sub_bubbles[1], amb_lines)
                                st.rerun()
                            else:
                                _finalize_split(target_idx, sub_bubbles[0], sub_bubbles[1], page_name)

        elif active_mode == "quick_select":
            # Quick Select mode: 3 mode buttons + undo/redo | size slider | tolerance slider | apply button
            brush_size = st.session_state.get("brush_size_slider", 30)
            tol = st.session_state.get("brush_tol_slider", 30)

            c1, c2, c3, cu, cr, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 2.4, 2.4, 2.3], gap="small")
            with c1:
                if st.button(" ", type="secondary", use_container_width=True, key="btn_mode_view"):
                    st.session_state.stage2_tool_mode = "view"
                    st.rerun()
            with c2:
                if st.button(" ", type="secondary", use_container_width=True, key="btn_mode_scissors"):
                    st.session_state.stage2_tool_mode = "scissors"
                    st.rerun()
            with c3:
                if st.button(" ", type="primary", use_container_width=True, key="btn_mode_quick_select"):
                    st.session_state.stage2_tool_mode = "quick_select"
                    st.rerun()
            with cu:
                if st.button(" ", type="secondary", disabled=not can_undo, use_container_width=True, key="btn_history_undo", help=undo_tooltip):
                    succ, desc = history_manager.undo_action()
                    if succ:
                        st.toast(f"↩️ Bekor qilindi: {desc}", icon=":material/undo:")
                        st.rerun()
            with cr:
                if st.button(" ", type="secondary", disabled=not can_redo, use_container_width=True, key="btn_history_redo", help=redo_tooltip):
                    succ, desc = history_manager.redo_action()
                    if succ:
                        st.toast(f"↪️ Qaytarildi: {desc}", icon=":material/redo:")
                        st.rerun()
            with c4:
                brush_size = st.slider(f"O'lcham: {int(brush_size)}px", 10, 100, int(brush_size), 2, key="brush_size_slider")
            with c5:
                tol = st.slider(f"Sezgirlik: {int(tol)}", 10, 80, int(tol), 2, key="brush_tol_slider")
            with c6:
                if st.button("✨ Qo'llash", type="primary", use_container_width=True, key="floating_apply_brush"):
                    if not has_drawn_stroke or canvas_res is None or canvas_res.image_data is None:
                        st.toast("⚠️ Avval sohaga bosing yoki chiziq torting!", icon=":material/warning:")
                    else:
                        stroke_disp_b = (canvas_res.image_data[:, :, 3] > 20).astype(np.uint8) * 255

                        # Single click expansion to circular brush footprint of radius brush_size/2
                        num_pts = np.count_nonzero(stroke_disp_b)
                        if 0 < num_pts < (brush_size * brush_size * 0.25):
                            r_footprint = max(4, int(brush_size // 2))
                            k_footprint = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_footprint * 2 + 1, r_footprint * 2 + 1))
                            stroke_disp_b = cv2.dilate(stroke_disp_b, k_footprint, iterations=1)

                        stroke_page_b = cv2.resize(stroke_disp_b, (W, H), interpolation=cv2.INTER_NEAREST)

                        # Auto-detect target bubble from stroke overlap/proximity
                        target_idx_b = None
                        best_overlap = 0
                        for idx_chk, b_chk in enumerate(bubbles):
                            cnt_chk = b_chk.get("contour") or b_chk.get("contour_points")
                            if cnt_chk:
                                m_chk = create_bubble_mask(cnt_chk, (H, W))
                                m_chk_dil = cv2.dilate(m_chk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40)), iterations=1)
                                ov = int(np.count_nonzero(cv2.bitwise_and(m_chk_dil, stroke_page_b)))
                                if ov > best_overlap:
                                    best_overlap = ov
                                    target_idx_b = idx_chk

                        if target_idx_b is None or best_overlap < 5:
                            st.toast("⚠️ Chiziq birorta ham pufak yaqinida emas!", icon=":material/warning:")
                        else:
                            seed_spacing = st.session_state.get("brush_spacing_slider", 20)
                            target_b_brush = bubbles[target_idx_b]
                            orig_cnt = target_b_brush.get("contour") or target_b_brush.get("contour_points")
                            updated_bubble = apply_smart_brush_growth(
                                target_b_brush, page_bgr, stroke_page_b, tolerance=float(tol),
                                json_data=canvas_res.json_data,
                                disp_shape=(disp_h, disp_w),
                                seed_spacing=int(seed_spacing)
                            )

                            # Push undo snapshot before applying brush expansion
                            b_id_label = target_b_brush.get("bubble_id", target_idx_b + 1)
                            history_manager.push_undo_snapshot(f"✨ Mo'yqalam: Pufak #{b_id_label} ni kengaytirish")

                            st.session_state.bubbles[target_idx_b] = updated_bubble

                            # Persistent Learning Record
                            record_mask_correction(
                                page_name=page_name,
                                tool_name="quick_selection",
                                bubble_id=target_b_brush.get("bubble_id", target_idx_b + 1),
                                original_contour=orig_cnt,
                                corrected_contours=updated_bubble["contour"],
                                metadata={"tolerance": tol, "brush_size": brush_size, "tool_mode": "quick_selection"}
                            )

                            # Clear cleaning and overlay cache to trigger clean rebuild
                            st.session_state.cleaned_page = None
                            st.session_state.rendered_image = None
                            st.session_state.analysis_version = st.session_state.get("analysis_version", 0) + 1
                            for k in list(st.session_state.keys()):
                                if k.startswith("overlay_st2_"):
                                    del st.session_state[k]

                            st.toast("✨ Pufak kengaytirildi!", icon=":material/check_circle:")
                            import time
                            time.sleep(0.3)
                            st.rerun()


def _finalize_split(sel_idx: int, sub_b1: Dict[str, Any], sub_b2: Dict[str, Any], page_name: str):
    """Replaces original bubble with two split bubbles, translates them, and logs correction."""
    import streamlit as st
    import engine

    orig_b = st.session_state.bubbles[sel_idx]
    orig_cnt = orig_b.get("contour") or orig_b.get("contour_points")
    orig_uzbek = orig_b.get("uzbek_translation", "")

    # Translate or partition Uzbek dialogue
    t1 = sub_b1.get("original_text", "")
    t2 = sub_b2.get("original_text", "")
    w1 = max(1, len(t1.split()))
    w2 = max(1, len(t2.split()))

    if orig_uzbek and len(orig_uzbek.split()) >= 3:
        sub_uz = engine.partition_translated_text_for_lobes(orig_uzbek, [w1, w2])
        sub_b1["uzbek_translation"] = sub_uz[0] if len(sub_uz) > 0 else orig_uzbek
        sub_b2["uzbek_translation"] = sub_uz[1] if len(sub_uz) > 1 else ""
    else:
        sub_b1["uzbek_translation"] = engine.translate_spiderman_uzbek(t1, speaker=sub_b1.get("speaker")) if t1 else ""
        sub_b2["uzbek_translation"] = engine.translate_spiderman_uzbek(t2, speaker=sub_b2.get("speaker")) if t2 else ""

    # Replace in bubbles list
    new_bubbles = []
    for i, b in enumerate(st.session_state.bubbles):
        if i == sel_idx:
            new_bubbles.append(sub_b1)
            new_bubbles.append(sub_b2)
        else:
            new_bubbles.append(b)

    # Re-index 1..N
    for new_id, b in enumerate(new_bubbles, 1):
        if isinstance(b, dict):
            b["bubble_id"] = new_id
        else:
            b.bubble_id = new_id

    # Push undo snapshot before applying split
    b_lbl = orig_b.get("bubble_id", sel_idx + 1)
    history_manager.push_undo_snapshot(f"✂️ Qaychi: Pufak #{b_lbl} ni ikkiga ajratish")

    st.session_state.bubbles = new_bubbles

    # Record persistent learning event
    record_mask_correction(
        page_name=page_name,
        tool_name="scissors",
        bubble_id=orig_b.get("bubble_id", sel_idx + 1),
        original_contour=orig_cnt,
        corrected_contours=[sub_b1["contour"], sub_b2["contour"]],
        metadata={"sub_text1": t1, "sub_text2": t2}
    )

    # Reset pending state and clear caches
    if "pending_scissors_split" in st.session_state:
        del st.session_state.pending_scissors_split

    st.session_state.cleaned_page = None
    st.session_state.rendered_image = None
    st.session_state.analysis_version = st.session_state.get("analysis_version", 0) + 1

    st.toast("✅ Pufak muvaffaqiyatli ikkiga ajratildi!", icon=":material/content_cut:")
    st.rerun()

