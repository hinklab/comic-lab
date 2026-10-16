"""
reading_order.py
-----------------
Two fixes for the comic lettering pipeline:

1. Location-based SFX vs. dialogue classification: an SFX word is only
   translated when it sits inside a detected speech/caption bubble; free-
   floating SFX drawn directly on the art must stay completely untouched.
   Uses bubble_lettering.is_point_in_bubble() -- position, not a word list,
   decides the classification, which is far more robust than dictionary
   heuristics (a word that happens to look like an SFX but is genuinely
   inside a bubble is still real dialogue, and vice versa).

2. Panel-aware reading order: numbering bubbles by a single flat (y, x)
   sort across the whole page breaks as soon as a panel has a tall/low
   bubble, because a bubble still inside an earlier panel can end up with
   a larger y than the first bubble of the next panel. The fix is a
   two-level sort: order panels first (row-major), then order the bubbles
   inside each panel the same way -- panels are always exhausted before
   moving to the next one.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import cv2

import bubble_lettering


# --------------------------------------------------------------------------
# Helper: Universal Bounding Box Extractor
# --------------------------------------------------------------------------

def _extract_bbox(item: Any) -> Tuple[int, int, int, int]:
    """
    Extracts (x0, y0, x1, y1) bounding box coordinates from:
    - SpeechBubble object (or any object with x0, y0, x1, y1 or box attributes)
    - Dictionary with 'bbox', 'box', or ('x0', 'y0', 'x1', 'y1') keys
    - Tuple/List of (x0, y0, x1, y1)
    """
    if hasattr(item, "x0") and hasattr(item, "y0"):
        x0 = int(item.x0)
        y0 = int(item.y0)
        x1 = int(getattr(item, "x1", x0 + 100))
        y1 = int(getattr(item, "y1", y0 + 30))
        return (x0, y0, x1, y1)
    if hasattr(item, "box") and getattr(item, "box") is not None:
        b = getattr(item, "box")
        return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    if isinstance(item, dict):
        if "bbox" in item:
            b = item["bbox"]
            return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        if "box" in item and item["box"] is not None:
            b = item["box"]
            return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        if "x0" in item and "y0" in item:
            x0 = int(item["x0"])
            y0 = int(item["y0"])
            x1 = int(item.get("x1", x0 + 100))
            y1 = int(item.get("y1", y0 + 30))
            return (x0, y0, x1, y1)
    if isinstance(item, (list, tuple)) and len(item) == 4:
        return (int(item[0]), int(item[1]), int(item[2]), int(item[3]))
    raise ValueError(f"Cannot extract (x0, y0, x1, y1) bounding box from {type(item)}: {item}")


# --------------------------------------------------------------------------
# 1. Location-based SFX vs. dialogue classification
# --------------------------------------------------------------------------

def is_dialogue_text(
    text_bbox: Union[Tuple[int, int, int, int], List[int]],
    bubble_contours: List[Any]
) -> bool:
    """
    True if the OCR text's center falls inside ANY detected bubble contour
    -- meaning it's real dialogue/caption and should be translated. False
    means it's environmental SFX drawn on the art itself: skip translation
    AND skip cleanup/inpainting for it entirely, leaving those pixels
    exactly as scanned.

    Args:
        text_bbox: (x0, y0, x1, y1) of the OCR text, full-image coords.
        bubble_contours: contours of every bubble already detected on this
            page (from get_bubble_contour), so an SFX sitting between two
            bubbles doesn't get wrongly claimed by a distant one -- it only
            counts if its center is truly inside one of them.
    """
    if not bubble_contours:
        return False

    box = _extract_bbox(text_bbox)
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2

    for c in bubble_contours:
        if c is None:
            continue
        cnt = np.array(c, dtype=np.int32) if not isinstance(c, np.ndarray) else c
        if cnt.ndim == 2 and cnt.shape[1] == 2:
            cnt = cnt.reshape((-1, 1, 2))
        if bubble_lettering.is_point_in_bubble(cnt, cx, cy):
            return True

    return False


# --------------------------------------------------------------------------
# 2. Panel-aware reading order
# --------------------------------------------------------------------------

def _row_major_order(
    items: List[Any],
    bbox_fn=_extract_bbox,
    row_overlap_thresh: float = 0.25
) -> List[Any]:
    """
    Sort items (panels OR bubbles) into standard reading order: group into
    horizontal 'rows' by VERTICAL OVERLAP between bounding boxes (not a
    raw y-sort -- that's what breaks on panels/bubbles of uneven height),
    then sort each row left-to-right and stack rows top-to-bottom.

    Incorporates conversational flow refinement: prevents jumping over an
    intermediate speaker's bubble (e.g. Speaker 1 -> Speaker 2 -> Speaker 1).
    """
    if not items:
        return []

    seeded = sorted(items, key=lambda it: bbox_fn(it)[1])  # seed by top-y
    rows: List[Dict[str, Any]] = []  # each: {'y_range': (y0, y1), 'items': [...]}
    for it in seeded:
        x0, y0, x1, y1 = bbox_fn(it)
        placed = False
        for row in rows:
            ry0, ry1 = row["y_range"]
            overlap = max(0, min(ry1, y1) - max(ry0, y0))
            span = min(ry1 - ry0, y1 - y0)
            if span > 0 and overlap / span >= row_overlap_thresh:
                row["items"].append(it)
                row["y_range"] = (min(ry0, y0), max(ry1, y1))
                placed = True
                break
        if not placed:
            rows.append({"y_range": (y0, y1), "items": [it]})

    # Conversational flow refinement:
    # If bubble B is horizontally situated between bubbles in an earlier row (e.g. A ... C),
    # and its vertical position is part of that dialogue tier (by0 < ry1 + 25),
    # pull it into the earlier row to preserve alternating conversational back-and-forth flow.
    changed = True
    while changed:
        changed = False
        for r_idx in range(len(rows) - 1):
            row = rows[r_idx]
            rx0 = min(bbox_fn(it)[0] for it in row["items"])
            rx1 = max(bbox_fn(it)[2] for it in row["items"])
            ry1 = row["y_range"][1]

            to_move = []
            for other_idx in range(r_idx + 1, len(rows)):
                other_row = rows[other_idx]
                for it in list(other_row["items"]):
                    bx0, by0, bx1, by1 = bbox_fn(it)
                    if rx0 <= bx0 and bx1 <= rx1 and by0 < ry1 + 25:
                        to_move.append((other_idx, it))

            for other_idx, it in to_move:
                rows[other_idx]["items"].remove(it)
                row["items"].append(it)
                by0, by1 = bbox_fn(it)[1], bbox_fn(it)[3]
                row["y_range"] = (min(row["y_range"][0], by0), max(row["y_range"][1], by1))
                changed = True

            rows = [r for r in rows if len(r["items"]) > 0]
            if changed:
                break

    rows.sort(key=lambda r: r["y_range"][0])
    ordered: List[Any] = []
    for row in rows:
        ordered.extend(sorted(row["items"], key=lambda it: (bbox_fn(it)[0], bbox_fn(it)[1])))
    return ordered


def assign_item_to_panel(
    item_bbox: Union[Tuple[int, int, int, int], List[int], Any],
    panel_boxes: List[Any]
) -> int:
    """
    Index of the panel whose box contains the item's centroid. Falls back
    to the nearest panel center if the centroid lands in a gutter/border
    (e.g. a bubble tail poking slightly outside its panel's box).
    """
    if not panel_boxes:
        return 0

    box = _extract_bbox(item_bbox)
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0

    parsed_panels = [_extract_bbox(p) for p in panel_boxes]

    for i, p in enumerate(parsed_panels):
        if p[0] <= cx <= p[2] and p[1] <= cy <= p[3]:
            return i

    def dist(p: Tuple[int, int, int, int]) -> float:
        pcx, pcy = (p[0] + p[2]) / 2.0, (p[1] + p[3]) / 2.0
        return (pcx - cx) ** 2 + (pcy - cy) ** 2

    return min(range(len(parsed_panels)), key=lambda i: dist(parsed_panels[i]))


def order_bubbles_reading_order(
    bubbles: List[Any],
    panel_boxes: Optional[List[Any]] = None
) -> List[Any]:
    """
    The main fix. Orders bubbles so panels are always fully read before
    the next panel starts, regardless of any individual bubble's raw
    y-position on the page.

    Args:
        bubbles: list of SpeechBubble objects or dicts, each with at least
            (x0, y0, x1, y1) / 'bbox'. Any other keys/attributes (id, text,
            contour...) are preserved and carried along in the returned order.
        panel_boxes: list of (x0, y0, x1, y1) panel bounding boxes, in any
            order -- either from your own panel layout data (preferred),
            or from detect_panel_boxes().

    Returns:
        The same bubbles, reordered into correct panel reading order.
    """
    if not bubbles:
        return []

    if not panel_boxes:
        return _row_major_order(bubbles, bbox_fn=_extract_bbox, row_overlap_thresh=0.25)

    ordered_panels = _row_major_order(panel_boxes, bbox_fn=_extract_bbox, row_overlap_thresh=0.5)
    buckets: List[List[Any]] = [[] for _ in ordered_panels]

    for b in bubbles:
        idx = assign_item_to_panel(b, ordered_panels)
        p_id = idx + 1
        if hasattr(b, "panel_id"):
            setattr(b, "panel_id", p_id)
        elif isinstance(b, dict):
            b["panel_id"] = p_id
        buckets[idx].append(b)

    result: List[Any] = []
    seq_idx = 1
    for p_idx, bucket in enumerate(buckets, start=1):
        ordered_bucket = _row_major_order(bucket, bbox_fn=_extract_bbox, row_overlap_thresh=0.25)
        for b in ordered_bucket:
            if hasattr(b, "reading_index"):
                setattr(b, "reading_index", seq_idx)
            elif isinstance(b, dict):
                b["reading_index"] = seq_idx

            if hasattr(b, "panel_id") and getattr(b, "panel_id", None) is None:
                setattr(b, "panel_id", p_idx)
            elif isinstance(b, dict) and "panel_id" not in b:
                b["panel_id"] = p_idx
            seq_idx += 1
        result.extend(ordered_bucket)
    return result


# --------------------------------------------------------------------------
# Multi-strategy panel auto-detection & Reading Order Graph
# --------------------------------------------------------------------------

def _sample_gutter_color(image_bgr: np.ndarray, border: int = 6) -> np.ndarray:
    """Samples median gutter/border color from outer margins of page."""
    h, w = image_bgr.shape[:2]
    b = min(border, h // 10, w // 10)
    if b < 1:
        return np.array([255, 255, 255], dtype=np.uint8)
    px = np.concatenate([
        image_bgr[:b, :].reshape(-1, 3),
        image_bgr[-b:, :].reshape(-1, 3),
        image_bgr[:, :b].reshape(-1, 3),
        image_bgr[:, -b:].reshape(-1, 3),
    ])
    return np.median(px, axis=0)


def _find_gutter_runs(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """Finds continuous index intervals where mask is True with minimum thickness."""
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if (i - start) >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and (len(mask) - start) >= min_len:
        runs.append((start, len(mask)))
    return runs


def _content_segments(gutter_runs: List[Tuple[int, int]], total_len: int, min_content_len: int) -> List[Tuple[int, int]]:
    """Extracts content intervals between consecutive gutters that satisfy minimum length."""
    segments: List[Tuple[int, int]] = []
    prev = 0
    for a, b in gutter_runs:
        if (a - prev) >= min_content_len:
            segments.append((prev, a))
        prev = b
    if (total_len - prev) >= min_content_len:
        segments.append((prev, total_len))
    return segments


def _detect_bordered_contours_fallback(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Fallback for bordered comic panels (black rectangular strokes) when gutters
    are bleed, artwork, or non-uniform.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = h * w * 0.04
    min_w = int(w * 0.10)
    min_h = int(h * 0.08)

    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if (bw * bh) >= min_area and bw >= min_w and bh >= min_h:
            boxes.append((x, y, x + bw, y + bh))

    if len(boxes) > 1:
        # Filter near-duplicate and fully enclosed child boxes
        filtered: List[Tuple[int, int, int, int]] = []
        for b in sorted(boxes, key=lambda it: (it[2] - it[0]) * (it[3] - it[1]), reverse=True):
            overlap = False
            for f in filtered:
                ix0, iy0 = max(b[0], f[0]), max(b[1], f[1])
                ix1, iy1 = min(b[2], f[2]), min(b[3], f[3])
                if ix1 > ix0 and iy1 > iy0:
                    i_area = (ix1 - ix0) * (iy1 - iy0)
                    b_area = (b[2] - b[0]) * (b[3] - b[1])
                    if i_area / b_area > 0.75:
                        overlap = True
                        break
            if not overlap:
                filtered.append(b)
        if len(filtered) > 1:
            return filtered
    return []


def detect_panel_boxes(
    image_bgr: np.ndarray,
    gutter_tol: int = 20,
    min_gutter_frac: float = 0.95,
    min_panel_frac: float = 0.08
) -> List[Tuple[int, int, int, int]]:
    """
    Robust recursive XY-cut panel splitter with size regularization:
    1. Samples gutter color from outer margins with white/black gutter fallback.
    2. Enforces minimum gutter thickness to avoid false cuts on internal text or art lines.
    3. Enforces page-relative minimum panel dimensions (min_panel_w, min_panel_h).
    4. Limits recursion depth (max 5 levels) to prevent catastrophic over-segmentation.
    5. Falls back to rectangular border contour analysis if gutter cut detects only 1 panel.
    """
    h, w = image_bgr.shape[:2]
    min_panel_w = max(40, int(w * min_panel_frac))
    min_panel_h = max(40, int(h * min_panel_frac))
    min_gutter_thick = max(3, int(min(h, w) * 0.003))

    bg = _sample_gutter_color(image_bgr)
    diff = np.abs(image_bgr.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
    is_gutter = diff < gutter_tol

    # Check if margin sampling found reasonable gutter coverage
    row_gutter_pts = (is_gutter.mean(axis=1) >= min_gutter_frac).sum()
    col_gutter_pts = (is_gutter.mean(axis=0) >= min_gutter_frac).sum()

    # If margin sampling found near zero gutters (e.g. edge art bleed), check standard white and black gutters
    if (row_gutter_pts + col_gutter_pts) < 4:
        white_diff = np.abs(image_bgr.astype(np.int16) - 255).sum(axis=2)
        white_is_gutter = white_diff < gutter_tol
        w_score = (white_is_gutter.mean(axis=1) >= min_gutter_frac).sum() + (white_is_gutter.mean(axis=0) >= min_gutter_frac).sum()

        black_diff = np.abs(image_bgr.astype(np.int16)).sum(axis=2)
        black_is_gutter = black_diff < gutter_tol
        b_score = (black_is_gutter.mean(axis=1) >= min_gutter_frac).sum() + (black_is_gutter.mean(axis=0) >= min_gutter_frac).sum()

        if w_score >= 4 and w_score >= b_score:
            is_gutter = white_is_gutter
        elif b_score >= 4:
            is_gutter = black_is_gutter

    def split(x0: int, y0: int, x1: int, y1: int, depth: int = 0) -> List[Tuple[int, int, int, int]]:
        rw = x1 - x0
        rh = y1 - y0
        if depth >= 5 or rw < int(min_panel_w * 1.5) or rh < int(min_panel_h * 1.5):
            return [(x0, y0, x1, y1)]

        region = is_gutter[y0:y1, x0:x1]

        # Horizontal gutters (splits into rows)
        row_frac = region.mean(axis=1) >= min_gutter_frac
        row_runs = _find_gutter_runs(row_frac, min_gutter_thick)
        if row_runs:
            row_segments = _content_segments(row_runs, rh, min_panel_h)
            if len(row_segments) > 1:
                panels: List[Tuple[int, int, int, int]] = []
                for sy0, sy1 in row_segments:
                    panels.extend(split(x0, y0 + sy0, x1, y0 + sy1, depth + 1))
                if panels:
                    return panels

        # Vertical gutters (splits into columns)
        col_frac = region.mean(axis=0) >= min_gutter_frac
        col_runs = _find_gutter_runs(col_frac, min_gutter_thick)
        if col_runs:
            col_segments = _content_segments(col_runs, rw, min_panel_w)
            if len(col_segments) > 1:
                panels = []
                for sx0, sx1 in col_segments:
                    panels.extend(split(x0 + sx0, y0, x0 + sx1, y1, depth + 1))
                if panels:
                    return panels

        return [(x0, y0, x1, y1)]

    raw_panels = split(0, 0, w, h)
    valid_panels = [p for p in raw_panels if (p[2] - p[0]) >= min_panel_w and (p[3] - p[1]) >= min_panel_h]

    # If recursive XY-cut found only 1 panel, try contour border detection fallback
    if len(valid_panels) <= 1:
        bordered = _detect_bordered_contours_fallback(image_bgr)
        if len(bordered) > 1:
            return bordered

    return valid_panels if valid_panels else [(0, 0, w, h)]


def get_page_panels_and_order(
    image_bgr: np.ndarray,
    bubbles: List[Any]
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Unified Step 1.1 Panel Segmentation & Bubble Ordering pipeline:
    1. Detects panel bounding boxes using multi-strategy recursive XY-cut + contour fallback.
    2. Sorts panels into topological reading order.
    3. Assigns each bubble to its panel, setting bubble.panel_id.
    4. Orders bubbles panel-by-panel, setting sequential bubble.reading_index.
    5. Returns (ordered_bubbles, panels_info).
    """
    panel_boxes = detect_panel_boxes(image_bgr)
    ordered_panels = _row_major_order(panel_boxes, bbox_fn=_extract_bbox, row_overlap_thresh=0.5)

    ordered_bubbles = order_bubbles_reading_order(bubbles, ordered_panels)

    # Build panels_info
    panels_info: List[Dict[str, Any]] = []
    for p_idx, p_box in enumerate(ordered_panels, start=1):
        box_tuple = _extract_bbox(p_box)
        p_bubbles = [
            b for b in ordered_bubbles
            if (getattr(b, "panel_id", None) == p_idx) or (isinstance(b, dict) and b.get("panel_id") == p_idx)
        ]
        b_ids = [
            getattr(b, "bubble_id", None) if hasattr(b, "bubble_id") else b.get("bubble_id")
            for b in p_bubbles
        ]
        panels_info.append({
            "panel_id": p_idx,
            "box": list(box_tuple),
            "reading_order": p_idx,
            "bubble_count": len(p_bubbles),
            "bubble_ids": [bid for bid in b_ids if bid is not None]
        })

    return ordered_bubbles, panels_info


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "page.png"
    page_bgr = cv2.imread(img_path)
    if page_bgr is not None:
        panels = detect_panel_boxes(page_bgr)
        print(f"Detected {len(panels)} panels in {img_path}:")
        for i, p in enumerate(_row_major_order(panels), start=1):
            print(f"  Panel #{i}: {p}")
