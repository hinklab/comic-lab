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
        buckets[idx].append(b)

    result: List[Any] = []
    for bucket in buckets:
        result.extend(_row_major_order(bucket, bbox_fn=_extract_bbox, row_overlap_thresh=0.25))
    return result


# --------------------------------------------------------------------------
# Optional: panel auto-detection (only if you don't already have panel boxes)
# --------------------------------------------------------------------------

def _sample_gutter_color(image_bgr: np.ndarray, border: int = 5) -> np.ndarray:
    """Samples median gutter/border color from outer margins of page."""
    h, w = image_bgr.shape[:2]
    px = np.concatenate([
        image_bgr[:border, :].reshape(-1, 3),
        image_bgr[-border:, :].reshape(-1, 3),
        image_bgr[:, :border].reshape(-1, 3),
        image_bgr[:, -border:].reshape(-1, 3),
    ])
    return np.median(px, axis=0)


def _find_gutter_runs(frac: np.ndarray, min_frac: float) -> List[Tuple[int, int]]:
    """Finds continuous indices where fraction exceeds minimum gutter threshold."""
    mask = frac >= min_frac
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _content_segments(gutter_runs: List[Tuple[int, int]], total_len: int) -> List[Tuple[int, int]]:
    """Extracts content intervals between consecutive gutter intervals."""
    segments: List[Tuple[int, int]] = []
    prev = 0
    for a, b in gutter_runs:
        if a > prev:
            segments.append((prev, a))
        prev = b
    if prev < total_len:
        segments.append((prev, total_len))
    return segments


def detect_panel_boxes(
    image_bgr: np.ndarray,
    gutter_tol: int = 18,
    min_gutter_frac: float = 0.985,
    min_panel_frac: float = 0.03
) -> List[Tuple[int, int, int, int]]:
    """
    Best-effort recursive XY-cut panel splitter: finds rows/columns that
    are almost entirely the page's background color (real gutters) and
    carves the page into panel rectangles along them, recursing into each
    piece. Returns a list of (x0, y0, x1, y1) in no particular order --
    feed straight into order_bubbles_reading_order(), which sorts them.
    """
    h, w = image_bgr.shape[:2]
    bg = _sample_gutter_color(image_bgr)
    diff = np.abs(image_bgr.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
    is_gutter = diff < gutter_tol

    def split(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int, int, int]]:
        region = is_gutter[y0:y1, x0:x1]
        rh, rw = region.shape
        if rh < 10 or rw < 10:
            return [(x0, y0, x1, y1)]

        row_runs = _find_gutter_runs(region.mean(axis=1), min_gutter_frac)
        row_segments = _content_segments(row_runs, rh)
        if len(row_segments) > 1:
            panels: List[Tuple[int, int, int, int]] = []
            for sy0, sy1 in row_segments:
                if (sy1 - sy0) / rh < min_panel_frac:
                    continue
                panels.extend(split(x0, y0 + sy0, x1, y0 + sy1))
            if panels:
                return panels

        col_runs = _find_gutter_runs(region.mean(axis=0), min_gutter_frac)
        col_segments = _content_segments(col_runs, rw)
        if len(col_segments) > 1:
            panels = []
            for sx0, sx1 in col_segments:
                if (sx1 - sx0) / rw < min_panel_frac:
                    continue
                panels.extend(split(x0 + sx0, y0, x0 + sx1, y1))
            if panels:
                return panels

        return [(x0, y0, x1, y1)]

    return split(0, 0, w, h)


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "page.png"
    page_bgr = cv2.imread(img_path)
    if page_bgr is not None:
        panels = detect_panel_boxes(page_bgr)
        print(f"Detected {len(panels)} panels in {img_path}:")
        for i, p in enumerate(_row_major_order(panels), start=1):
            print(f"  Panel #{i}: {p}")
