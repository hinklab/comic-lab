"""
bubble_lettering.py
--------------------
Contour-aware speech/caption bubble isolation, text-only inpainting, and
safe text layout for an automated comic lettering pipeline (EasyOCR + OpenCV
+ Pillow). CPU-only, no deep-learning segmentation.

Pipeline usage per detected OCR text box:

    contour, is_rect = get_bubble_contour(
        page_bgr, text_bbox,
        neighbor_seeds=[centroid(b) for b in other_boxes_on_page],
    )
    cleaned = inpaint_bubble_text_only(page_bgr, contour)
    safe    = calculate_safe_text_area(contour, page_bgr.shape)
    # ... draw translated text into safe['inscribed_rect'] or use
    #     fit_text_to_bubble() for a fully contour-hugging "diamond" layout.

Why bounding-box masking fails
-------------------------------
An OCR box is a rectangle around *glyphs*, not around the bubble. Filling it
solid white either clips the bubble outline/tail (box smaller than the
bubble) or paints over background art (box padded to be safe). Every
function below instead works from the bubble's own *contour*, recovered by
flood-filling the bubble's fill color and stopping at its own ink border --
with the border kept separate from the dark text ink sitting inside it.
"""

import re
from typing import Optional, Tuple, List, Dict, Any, Union
import cv2
import numpy as np
from PIL import ImageFont
from quality_gates import apply_quality_gates


# --------------------------------------------------------------------------
# Rule 6 Formatting Enforcement: Quote Stripping & Sentence Separation
# --------------------------------------------------------------------------

def strip_dialogue_quotes(text: str) -> str:
    """
    Strips leading, trailing, and enclosing quotation marks:
    - Double quotes: ", “, ”, „, ‟, «, », ‹, ›, ＂
    - Outer single quotes wrapping sentences/lines
    Strictly preserves Uzbek apostrophes within words (O', G', and tutruq belgisi).
    """
    if not text:
        return ""
    quote_chars = ['"', '“', '”', '„', '‟', '«', '»', '‹', '›', '＂']
    for q in quote_chars:
        text = text.replace(q, "")

    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        l = line.strip()
        l = re.sub(r"^['`´]+", "", l)
        l = re.sub(r"['`´]+$", "", l)
        l = re.sub(r"(?<=\s)['`´]+(?=[A-Za-z0-9])", "", l)
        l = re.sub(r"(?<=[.!?…])['`´]+(?=\s|$)", "", l)
        cleaned_lines.append(l.strip())
    return "\n".join(cleaned_lines).strip()


def format_dialogue_sentences(text: str) -> str:
    """
    Enforces Rule 6 formatting:
    1. Strips all extraneous quotation marks.
    2. Separates each complete sentence onto its own line (\n).
    Preserves initials (e.g. J. in J. JONA JEYMSON) and standard abbreviations.
    """
    cleaned = strip_dialogue_quotes(text)
    if not cleaned:
        return ""

    def replace_boundary(match):
        punct = match.group(1)
        return punct + "\n"

    pattern = r"(?<!\b[A-Za-z])(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bProf)(?<!\bvs)([.!?]+|…)\s+(?=[A-Za-z0-9O'G'])"
    formatted = re.sub(pattern, replace_boundary, cleaned, flags=re.IGNORECASE)

    out_lines = [re.sub(r'[^\S\r\n]+', ' ', l).strip() for l in formatted.split('\n') if l.strip()]
    return "\n".join(out_lines)


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _adaptive_roi_scale(bw, bh, base_scale):
    """
    Small/dense text boxes need proportionally *more* surrounding context
    than large ones: a fixed multiplier leaves almost no margin around a
    tight two-word balloon, which is exactly what causes false 'leak'
    flags on legitimate tight bubbles. Boost tapers off for larger boxes,
    which already have plenty of absolute margin at scale=1.
    """
    small_dim = min(bw, bh)
    if small_dim < 20:
        boost = 2.4
    elif small_dim < 40:
        boost = 1.7
    elif small_dim < 70:
        boost = 1.25
    else:
        boost = 1.0
    return base_scale * boost


def _suppress_text_for_barrier(gray_roi, approx_glyph_px, ink_thresh=120):
    """
    Erase glyph-scale dark connected components from a throwaway grayscale
    copy, used ONLY to build the Canny/closing barrier -- never written
    back to the real image. This is what stops dense internal text from
    being mistaken for bubble border during edge detection: letters are
    consistently small relative to approx_glyph_px; the true border is one
    stroke that spans (and usually loops around) the whole bubble, so it
    survives the size filter untouched.
    """
    dark = (gray_roi < ink_thresh).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    clean = gray_roi.copy()
    max_glyph_dim = max(6, int(approx_glyph_px * 2.2))
    suppress_mask = np.zeros_like(gray_roi, dtype=np.uint8)
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        if max(w, h) <= max_glyph_dim:
            comp = labels[y:y + h, x:x + w] == i
            suppress_mask[y:y + h, x:x + w][comp] = 255
    if np.any(suppress_mask > 0):
        k_sz = 5 if max_glyph_dim > 20 else 3
        suppress_mask = cv2.dilate(suppress_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz)))
        clean[suppress_mask > 0] = 255
    return clean


def _find_light_seed(gray_roi, seed_xy, barrier_mask, light_thresh=180, search_radius=30, bright_roi=None):
    """
    Letter-proof seed search. Rather than scanning for the first light pixel
    (which fails when tight text chops the fill into small pockets), this:
      1. Builds a distance transform of 'non-wall' space from the (text-
         suppressed) barrier mask, so every free pixel knows how far it is
         from the nearest ink/border wall.
      2. Restricts the search to a window around the OCR centroid, so it
         can't wander into a neighboring, already-merged bubble.
      3. Picks the *deepest interior* light pixel in that window -- the
         point farthest from any wall -- the most reliable genuine-fill
         seed even inside a small, text-dense bubble.
      4. If no light pixel exists in the LOCAL window, progressively relaxes
         the lightness threshold first -- still staying near the OCR text --
         before ever widening the search.
    Supports bright_roi (e.g. value channel or max-channel) to accurately seed
    tinted (spectral/telepathic) bubbles where grayscale luminance alone is low.
    """
    h, w = gray_roi.shape
    barrier_bin = barrier_mask[1:-1, 1:-1] > 0
    free = np.where(barrier_bin, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)

    b_map = bright_roi if bright_roi is not None else gray_roi

    sx, sy = seed_xy
    y0, y1 = max(0, sy - search_radius), min(h, sy + search_radius + 1)
    x0, x1 = max(0, sx - search_radius), min(w, sx + search_radius + 1)

    thresholds = [light_thresh - 15 * i for i in range(5)]

    def _best_in_pool(pool):
        if not pool.any():
            return None
        masked_dist = np.where(pool, dist, -1)
        y, x = np.unravel_index(np.argmax(masked_dist), masked_dist.shape)
        if masked_dist[y, x] > 0:
            return int(x), int(y)
        return None

    # Pass 1: local window only, relaxing the threshold before ever widening.
    for thresh in thresholds:
        light = b_map >= thresh
        candidate = light & (~barrier_bin)
        window = np.zeros_like(candidate)
        window[y0:y1, x0:x1] = candidate[y0:y1, x0:x1]
        found = _best_in_pool(window)
        if found is not None:
            return found

    # Pass 2: last resort -- widen to the full ROI, still trying the
    # strictest threshold first so a genuine bright fill elsewhere is
    # preferred over a barely-lit false one.
    for thresh in thresholds:
        light = b_map >= thresh
        candidate = light & (~barrier_bin)
        found = _best_in_pool(candidate)
        if found is not None:
            return found

    return None


def _dist_sq_to_target(xx, yy, target, rx0, ry0):
    if len(target) == 4:
        x0, y0, x1, y1 = target[0] - rx0, target[1] - ry0, target[2] - rx0, target[3] - ry0
        dx = np.maximum(0, np.maximum(x0 - xx, xx - x1))
        dy = np.maximum(0, np.maximum(y0 - yy, yy - y1))
        return dx * dx + dy * dy
    else:
        x, y = target[0] - rx0, target[1] - ry0
        return (xx - x) ** 2 + (yy - y) ** 2


def _clip_touching_bubbles(filled, own_target, neighbor_seeds_full, rx0, ry0):
    """
    When two bubbles are physically tangent or joined (such as a figure-8 / double-lobed
    bubble) with no visible seam, flood-fill can spill from one into the other.
    Splits the region using distance to each bubble's text bounding box / line boxes (or point seeds)
    so each bubble keeps its natural half, parting cleanly at the constriction/neck
    without cutting into either bubble's text or creating horn spikes.
    """
    rh, rw = filled.shape
    yy, xx = np.mgrid[0:rh, 0:rw]
    cur_x = xx + rx0
    cur_y = yy + ry0

    def _dist_sq_to_target(target):
        if isinstance(target, (list, tuple)) and len(target) > 0 and isinstance(target[0], (list, tuple)):
            dists = []
            for b in target:
                if len(b) == 4:
                    dx = np.maximum(0, np.maximum(b[0] - cur_x, cur_x - b[2]))
                    dy = np.maximum(0, np.maximum(b[1] - cur_y, cur_y - b[3]))
                    dists.append(dx * dx + dy * dy)
                elif len(b) == 2:
                    dists.append((cur_x - b[0]) ** 2 + (cur_y - b[1]) ** 2)
            return np.min(dists, axis=0) if dists else np.zeros((rh, rw), dtype=np.float32)
        elif len(target) == 4:
            dx = np.maximum(0, np.maximum(target[0] - cur_x, cur_x - target[2]))
            dy = np.maximum(0, np.maximum(target[1] - cur_y, cur_y - target[3]))
            return dx * dx + dy * dy
        else:
            return (cur_x - target[0]) ** 2 + (cur_y - target[1]) ** 2

    dist_own = _dist_sq_to_target(own_target)

    if isinstance(own_target, (list, tuple)) and len(own_target) > 0 and isinstance(own_target[0], (list, tuple)):
        first = own_target[0]
        ox = int((first[0] + first[2]) / 2) - rx0 if len(first) == 4 else int(first[0]) - rx0
        oy = int((first[1] + first[3]) / 2) - ry0 if len(first) == 4 else int(first[1]) - ry0
    elif len(own_target) == 4:
        ox = int((own_target[0] + own_target[2]) / 2) - rx0
        oy = int((own_target[1] + own_target[3]) / 2) - ry0
    else:
        ox = int(own_target[0]) - rx0
        oy = int(own_target[1]) - ry0

    # Filter to only neighbor bubbles that were ACTUALLY swallowed by this flood-fill.
    # If the neighbor was not swallowed, the natural ink border held, so clipping would
    # destructively amputate natural bubble shoulder curves.
    swallowed_neighbors = []
    for n_item in neighbor_seeds_full or []:
        if isinstance(n_item, (list, tuple)) and len(n_item) > 0 and isinstance(n_item[0], (list, tuple)):
            for b in n_item:
                if len(b) == 4:
                    nx0, ny0 = b[0] - rx0, b[1] - ry0
                    nx1, ny1 = b[2] - rx0, b[3] - ry0
                    cx0, cy0 = max(0, nx0), max(0, ny0)
                    cx1, cy1 = min(rw, nx1), min(rh, ny1)
                    if cx1 > cx0 and cy1 > cy0:
                        crop = filled[cy0:cy1, cx0:cx1]
                        if int(np.count_nonzero(crop)) / max(1, (nx1 - nx0) * (ny1 - ny0)) > 0.25:
                            swallowed_neighbors.append(n_item)
                            break
        elif len(n_item) == 4:
            nx0, ny0 = n_item[0] - rx0, n_item[1] - ry0
            nx1, ny1 = n_item[2] - rx0, n_item[3] - ry0
            cx0, cy0 = max(0, nx0), max(0, ny0)
            cx1, cy1 = min(rw, nx1), min(rh, ny1)
            if cx1 > cx0 and cy1 > cy0:
                n_area = (nx1 - nx0) * (ny1 - ny0)
                crop = filled[cy0:cy1, cx0:cx1]
                covered = int(np.count_nonzero(crop))
                if covered / max(1, n_area) > 0.25:
                    swallowed_neighbors.append(n_item)
        else:
            px, py = int(n_item[0]) - rx0, int(n_item[1]) - ry0
            if 0 <= px < rw and 0 <= py < rh:
                y_min, y_max = max(0, py - 10), min(rh, py + 10)
                x_min, x_max = max(0, px - 10), min(rw, px + 10)
                crop = filled[y_min:y_max, x_min:x_max]
                if crop.size > 0 and np.mean(crop > 0) > 0.60:
                    swallowed_neighbors.append(n_item)

    if not swallowed_neighbors:
        return filled

    keep = np.ones((rh, rw), dtype=bool)
    has_cut = False
    for n_item in swallowed_neighbors:
        dist_n = _dist_sq_to_target(n_item)
        keep &= (dist_own <= dist_n)
        has_cut = True

    if not has_cut:
        return filled

    clipped = filled.copy()
    clipped[~keep] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clipped)
    if 0 <= oy < rh and 0 <= ox < rw and labels[oy, ox] > 0:
        clipped = (labels == labels[oy, ox]).astype(np.uint8) * 255
    elif num_labels > 1:
        best_lbl = 1
        best_d = float('inf')
        for lbl in range(1, num_labels):
            cx_lbl, cy_lbl = centroids[lbl]
            d = (cx_lbl - ox) ** 2 + (cy_lbl - oy) ** 2
            if d < best_d:
                best_d = d
                best_lbl = lbl
        clipped = (labels == best_lbl).astype(np.uint8) * 255

    return clipped



def _synthetic_ellipse_contour(cx, cy, bw, bh, width_pad=1.25, height_pad=1.35):
    """
    Last-resort fallback when no seed/fill succeeds at all (near-solid dark
    fill, or a panel too degraded to read any border). Returns a plausible
    oval sized from typical comic bubble padding around its text, instead
    of a blunt tight rectangle, so downstream inpainting/text-fitting still
    reads as a bubble rather than a patch.
    """
    axes = (max(6, int(bw * width_pad / 2)), max(6, int(bh * height_pad / 2)))
    pts = cv2.ellipse2Poly((int(cx), int(cy)), axes, 0, 0, 360, 6)
    return pts.reshape(-1, 1, 2).astype(np.int32)


def _is_rectangular(contour, rect_fill_thresh=0.94, max_vertices=5):
    """
    Distinguishes a crisp caption box from an organic oval bubble.
    A true rectangle fills nearly its entire bounding box (>=0.94) and simplifies
    to 4 or 5 vertices at 0.02*perimeter; an oval or docked curved bubble has
    rounded corners/curved edges and fills noticeably less.
    """
    area = cv2.contourArea(contour)
    if area <= 0:
        return False
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = max(1, w * h)
    fill_ratio = area / rect_area
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    return fill_ratio >= rect_fill_thresh and len(approx) <= max_vertices


def _largest_rectangle_in_histogram(heights):
    """Classic stack-based 'largest rectangle in a histogram'. O(n)."""
    stack = []  # (start_index, height)
    best = (0, 0, 0, 0)  # area, left, height, width
    n = len(heights)
    for i in range(n + 1):
        cur_h = heights[i] if i < n else 0
        start = i
        while stack and stack[-1][1] > cur_h:
            idx, h = stack.pop()
            width = i - idx
            area = h * width
            if area > best[0]:
                best = (area, idx, h, width)
            start = idx
        stack.append((start, cur_h))
    return best  # area, left, height, width


def _largest_inscribed_rectangle(mask_bin):
    """
    Maximal axis-aligned rectangle fully inside a binary mask (255=inside),
    via the standard 'maximal rectangle in a binary matrix' technique: run
    the histogram algorithm above once per row, using each row's running
    count of consecutive filled pixels above it as the histogram. O(H*W),
    trivial cost for a bubble-sized crop -- no ML needed.
    """
    h, w = mask_bin.shape
    heights = np.zeros(w, dtype=np.int32)
    best_overall = (0, 0, 0, 0, 0)  # area, x, y, w, h
    for row in range(h):
        row_on = mask_bin[row] > 0
        heights = np.where(row_on, heights + 1, 0).astype(np.int32)
        area, left, height, width = _largest_rectangle_in_histogram(heights.tolist())
        if area > best_overall[0]:
            top = row - height + 1
            best_overall = (area, left, top, width, height)
    _, x, y, w_, h_ = best_overall
    return int(x), int(y), int(w_), int(h_)


def _close_docked_panel_boundaries(
    edges: np.ndarray,
    tx0: int,
    ty0: int,
    tx1: int,
    ty1: int,
    gray: Optional[np.ndarray] = None,
    bridge_collinear: bool = False
) -> np.ndarray:
    """
    Detects flat/docked panel boundaries (where a speech bubble rests directly
    on a panel border or gutter with no dark stroke between the white bubble
    interior and the white gutter/frame).
    1. Detects white gutters (near-white horizontal/vertical bands) across the full ROI.
    2. Bridges collinear docked panel border segments when leak recovery is active.
    """
    rh, rw = edges.shape[:2]
    edges = edges.copy()
    bw = max(1, tx1 - tx0)
    bh = max(1, ty1 - ty0)

    # 1. Gutter barrier detection: check for solid/near-white paper bands separating panels
    if gray is not None:
        # Check above text for horizontal gutter (scan from slightly inside text margin up into context)
        for r in range(min(rh - 1, ty0 + 15), max(-1, ty0 - 150), -1):
            span_local = gray[r, max(0, tx0 - 30):min(rw, tx1 + 30)]
            if len(span_local) > 0 and np.mean(span_local) > 240 and np.min(span_local) > 180:
                edges[r, max(0, tx0 - 60):min(rw, tx1 + 60)] = 255
            else:
                span_full = gray[r, :]
                if len(span_full) > 0 and np.mean(span_full) > 245 and np.min(span_full) > 180:
                    edges[r, :] = 255

        # Check below text for horizontal gutter
        for r in range(max(0, ty1 - 15), min(rh, ty1 + 150)):
            span_local = gray[r, max(0, tx0 - 30):min(rw, tx1 + 30)]
            if len(span_local) > 0 and np.mean(span_local) > 240 and np.min(span_local) > 180:
                edges[r, max(0, tx0 - 60):min(rw, tx1 + 60)] = 255
            else:
                span_full = gray[r, :]
                if len(span_full) > 0 and np.mean(span_full) > 245 and np.min(span_full) > 180:
                    edges[r, :] = 255

        # Check left of text for vertical gutter/margin
        for c in range(min(rw - 1, tx0 + 15), max(-1, tx0 - 150), -1):
            span_local = gray[max(0, ty0 - 30):min(rh, ty1 + 30), c]
            if len(span_local) > 0 and np.mean(span_local) > 240 and np.min(span_local) > 180:
                edges[max(0, ty0 - 60):min(rh, ty1 + 60), c] = 255
            else:
                span_full = gray[:, c]
                if len(span_full) > 0 and np.mean(span_full) > 245 and np.min(span_full) > 180:
                    edges[:, c] = 255

        # Check right of text for vertical gutter/margin
        for c in range(max(0, tx1 - 15), min(rw, tx1 + 150)):
            span_local = gray[max(0, ty0 - 30):min(rh, ty1 + 30), c]
            if len(span_local) > 0 and np.mean(span_local) > 240 and np.min(span_local) > 180:
                edges[max(0, tx0 - 60):min(rw, tx1 + 60), c] = 255
            else:
                span_full = gray[:, c]
                if len(span_full) > 0 and np.mean(span_full) > 245 and np.min(span_full) > 180:
                    edges[:, c] = 255

    if not bridge_collinear:
        return edges

    # 2. Straight panel border line detection and bridging
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=25, maxLineGap=12)
    if lines is None:
        return edges

    h_lines = []
    v_lines = []
    for l in lines:
        pts = l[0] if l.ndim == 2 or (l.ndim == 3 and l.shape[0] == 1) else l
        if len(pts) != 4:
            continue
        x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
        if abs(y1 - y2) <= 3:
            h_lines.append((min(x1, x2), max(x1, x2), (y1 + y2) // 2))
        elif abs(x1 - x2) <= 3:
            v_lines.append((min(y1, y2), max(y1, y2), (x1 + x2) // 2))

    max_gap_h = max(450, int(bw * 1.5))
    max_gap_v = max(450, int(bh * 1.5))

    # Bridge collinear horizontal frame lines above text across open bubble gaps
    top_candidates = [l for l in h_lines if l[2] <= ty0 + 20]
    for l1 in top_candidates:
        for l2 in top_candidates:
            if abs(l1[2] - l2[2]) <= 4 and 0 < l2[0] - l1[1] < max_gap_h:
                if l1[1] <= tx1 + 60 and l2[0] >= tx0 - 60:
                    cv2.line(edges, (l1[1], l1[2]), (l2[0], l2[2]), 255, 3)

    # Bridge collinear horizontal frame lines below text across open bubble gaps
    bot_candidates = [l for l in h_lines if l[2] >= ty1 - 20]
    for l1 in bot_candidates:
        for l2 in bot_candidates:
            if abs(l1[2] - l2[2]) <= 4 and 0 < l2[0] - l1[1] < max_gap_h:
                if l1[1] <= tx1 + 60 and l2[0] >= tx0 - 60:
                    cv2.line(edges, (l1[1], l1[2]), (l2[0], l2[2]), 255, 3)

    # Right barrier (vertical collinear panel frame lines):
    r_candidates = [l for l in v_lines if l[2] >= tx1 - 20]
    for l1 in r_candidates:
        for l2 in r_candidates:
            if abs(l1[2] - l2[2]) <= 4 and 0 < l2[0] - l1[1] < max_gap_v:
                if l1[1] <= ty1 + 60 and l2[0] >= ty0 - 60:
                    cv2.line(edges, (l1[2], l1[1]), (l2[2], l2[0]), 255, 3)

    # Left barrier (vertical collinear panel frame lines):
    l_candidates = [l for l in v_lines if l[2] <= tx0 + 20]
    for l1 in l_candidates:
        for l2 in l_candidates:
            if abs(l1[2] - l2[2]) <= 4 and 0 < l2[0] - l1[1] < max_gap_v:
                if l1[1] <= ty1 + 60 and l2[0] >= ty0 - 60:
                    cv2.line(edges, (l1[2], l1[1]), (l2[2], l2[0]), 255, 3)

    return edges


# --------------------------------------------------------------------------
# 1. True contour extraction from an OCR seed
# --------------------------------------------------------------------------

def get_bubble_contour(image_bgr, text_bbox, roi_scale=4.0, max_expansions=2,
                        edge_low=50, edge_high=150, close_kernel_size=5,
                        flood_tolerance=12, light_thresh=165,
                        seed_search_radius=30, ink_suppression_thresh=120,
                        approx_glyph_height=None, neighbor_seeds=None,
                        max_bubble_to_text_ratio=90.0, roi_leak_ratio=0.75,
                        hard_occluders=None, own_lines=None):
    """
    Recover the bubble's true organic contour (round bubble + tail, or crisp
    caption box) from a single OCR text bounding box.

    Strategy, all CPU / classical CV:
      1. Crop a local ROI around the text, sized adaptively -- small/dense
         boxes get proportionally more surrounding context. A broken border
         can then only leak into a bounded area, never across the page.
      2. Erase glyph-scale dark components from a throwaway grayscale copy
         (see _suppress_text_for_barrier), THEN run Canny + closing on that
         copy to build the flood-fill barrier. This keeps dense internal
         text from acting as a false border, while the true border (a much
         larger connected stroke) survives.
      3. Find a seed via _find_light_seed: the interior point farthest from
         any wall near the OCR centroid, on the ORIGINAL (non-suppressed)
         grayscale, so it's always genuine bubble fill.
      4. Flood-fill from that seed on the real image, confined by the
         barrier, written to the flood-fill mask only.
      5. If another bubble's centroid got swallowed by this fill (tangent
         bubbles with no visible seam), Voronoi-clip the fill back to its
         own side via _clip_touching_bubbles.
      6. Check for leaks with a SCALE-INVARIANT ratio: filled_area versus
         the text_bbox's own area, not the ROI's. A genuine leak into open
         panel art blows this far past any real bubble-to-text padding
         ratio, regardless of ROI size. Retry with a larger closing kernel
         / ROI on suspected leaks; after max_expansions, return a synthetic
         ellipse rather than a blunt tight box.

    Args:
        image_bgr: full page image (cv2.imread, BGR).
        text_bbox: (x0, y0, x1, y1) OCR box in full-image pixel coordinates.
        approx_glyph_height: px height of one line of text, used to size
            the glyph-suppression filter. Defaults to the text_bbox height,
            which is correct for per-line OCR boxes (e.g. EasyOCR). Pass
            explicitly if you've merged multi-line boxes upstream.
        neighbor_seeds: optional list of (x, y) full-image points -- text
            centroids of *other* nearby OCR boxes on the page. Enables
            touching-bubble splitting; cheap to always pass.
        max_bubble_to_text_ratio: hard cap on filled_area / text_bbox_area.
            ~3-8x is typical for a real tight bubble; a leak into open art
            routinely blows past 40-100x.

    Returns:
        (contour, is_rectangle)
        contour: (N, 1, 2) int32 array in FULL image coordinates, directly
            usable with cv2.drawContours / cv2.pointPolygonTest.
        is_rectangle: True if classified as a crisp caption box. False for
            organic bubbles AND for the synthetic-ellipse fallback.
    """
    H, W = image_bgr.shape[:2]
    x0, y0, x1, y1 = text_bbox
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    bbox_area = bw * bh
    glyph_px = approx_glyph_height or bh

    # Wide, short text (typical of a caption box spanning most of a panel, with only
    # one or two sparse lines of text) naturally has a much bigger bubble-area-to-
    # text-area ratio than a squarish word balloon -- a fixed cap flags the correct,
    # fully-filled box as a "leak" and drops it to a tiny fallback ellipse sized from
    # the text alone. Scale the cap up (modestly) with the box's own aspect ratio so
    # a genuinely wide/sparse caption isn't punished for its natural shape. Kept
    # deliberately mild -- the containment check below is what actually guards
    # against real leaks, so this no longer has to do that job by itself.
    aspect = max(bw, bh) / max(1, min(bw, bh))
    effective_ratio_cap = max_bubble_to_text_ratio * min(2.0, max(1.0, aspect / 3.5))

    for expansion in range(max_expansions + 1):
        eff_scale = _adaptive_roi_scale(bw, bh, roi_scale) * (1.5 ** expansion)
        half_w, half_h = int(bw * eff_scale / 2), int(bh * eff_scale / 2)
        rx0, ry0 = max(0, cx - half_w), max(0, cy - half_h)
        rx1, ry1 = min(W, cx + half_w), min(H, cy + half_h)
        roi = image_bgr[ry0:ry1, rx0:rx1]
        rh, rw = roi.shape[:2]
        if rh < 3 or rw < 3:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clean_gray = _suppress_text_for_barrier(gray, glyph_px, ink_thresh=ink_suppression_thresh)

        edges = cv2.Canny(clean_gray, edge_low, edge_high)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        k = close_kernel_size + 2 * expansion
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )

        tx0, ty0 = max(0, x0 - rx0), max(0, y0 - ry0)
        tx1, ty1 = min(rw, x1 - rx0), min(rh, y1 - ry0)

        edges = _close_docked_panel_boundaries(edges, tx0, ty0, tx1, ty1, gray=gray, bridge_collinear=False)

        flood_mask = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
        flood_mask[1:-1, 1:-1][edges > 0] = 1  # pre-mark (text-free) border as a wall

        roi_occ = None
        if hard_occluders is not None:
            if isinstance(hard_occluders, np.ndarray) and hard_occluders.shape[:2] == (H, W):
                roi_occ = hard_occluders[ry0:ry1, rx0:rx1]
            else:
                roi_occ = np.zeros((rh, rw), dtype=np.uint8)
                for occ in hard_occluders:
                    if isinstance(occ, (list, tuple)) and len(occ) == 4:
                        ox0, oy0, ow, oh = occ
                        cv2.rectangle(roi_occ, (max(0, ox0 - rx0), max(0, oy0 - ry0)),
                                      (min(rw, ox0 + ow - rx0), min(rh, oy0 + oh - ry0)), 255, -1)
                    elif isinstance(occ, np.ndarray):
                        if occ.ndim == 2 and occ.shape == (H, W):
                            # Full-page mask: slice the ROI region
                            roi_occ[occ[ry0:ry1, rx0:rx1] > 0] = 255
                        elif occ.ndim >= 2 and occ.shape[1] == 2:
                            # Contour point array: shift coordinates to ROI space
                            occ_shifted = occ - np.array([rx0, ry0])
                            cv2.drawContours(roi_occ, [occ_shifted], -1, 255, -1)
                        elif occ.ndim >= 3:
                            # Already a contour in cv2 format (N,1,2)
                            occ_pts = occ.reshape(-1, 2) - np.array([rx0, ry0])
                            cv2.drawContours(roi_occ, [occ_pts], -1, 255, -1)
            if roi_occ is not None and np.any(roi_occ > 0):
                flood_mask[1:-1, 1:-1][roi_occ > 0] = 1  # hard occluder is an impassable wall

        val_roi = np.max(roi, axis=2) if roi.ndim == 3 else clean_gray
        seed = _find_light_seed(
            clean_gray, (cx - rx0, cy - ry0), flood_mask,
            light_thresh=light_thresh, search_radius=seed_search_radius,
            bright_roi=np.maximum(clean_gray, val_roi)
        )
        if seed is None:
            seed = _find_light_seed(
                gray, (cx - rx0, cy - ry0), flood_mask,
                light_thresh=light_thresh, search_radius=seed_search_radius,
                bright_roi=np.maximum(gray, val_roi)
            )
        if seed is None:
            continue  # try a larger ROI next iteration

        tol = max(flood_tolerance, 25)
        tol_bgr = (tol,) * 3
        clean_bgr = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2BGR)
        cv2.floodFill(
            clean_bgr, flood_mask, seed, (255, 255, 255),
            loDiff=tol_bgr, upDiff=tol_bgr,
            flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8),
        )
        filled = (flood_mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255

        if roi_occ is not None and np.any(roi_occ > 0):
            filled = cv2.bitwise_and(filled, cv2.bitwise_not(roi_occ))

        if neighbor_seeds:
            filled = _clip_touching_bubbles(filled, own_lines or text_bbox, neighbor_seeds, rx0, ry0)

        filled_area = int(np.count_nonzero(filled))
        roi_area = rh * rw
        border_touch = (
            np.any(filled[0, :]) or np.any(filled[-1, :]) or
            np.any(filled[:, 0]) or np.any(filled[:, -1])
        )
        ratio_to_text = filled_area / max(1, bbox_area)
        ratio_to_roi = filled_area / max(1, roi_area)

        # Containment check: a genuine bubble/caption ALWAYS substantially
        # surrounds its own OCR text, since the text is what sits inside it.
        # If the fill barely overlaps the text_bbox, the seed wandered onto
        # an unrelated bright region elsewhere in the panel (a light source,
        # a smoke cloud) rather than the bubble that actually holds this
        # text -- regardless of how "reasonable" the resulting blob's size
        # looks on its own. This is the primary defense against position-
        # drifted leaks; the ratio checks below catch oversized-but-
        # correctly-placed leaks.
        tx0, ty0 = max(0, x0 - rx0), max(0, y0 - ry0)
        tx1, ty1 = min(rw, x1 - rx0), min(rh, y1 - ry0)
        if tx1 > tx0 and ty1 > ty0:
            text_area_px = (tx1 - tx0) * (ty1 - ty0)
            covered_px = int(np.count_nonzero(filled[ty0:ty1, tx0:tx1]))
            text_coverage = covered_px / max(1, text_area_px)
        else:
            text_coverage = 0.0
        misplaced = text_coverage < 0.20

        # Solidity check: a real bubble/caption fill is a smooth, nearly
        # convex blob (round bubble, oval + tail, or crisp rectangle). A fill
        # that has escaped into surrounding art -- a smoke cloud, a dust
        # burst, a lantern's glow, jagged spider-web background -- follows
        # that art's ragged edges instead, so its area comes out noticeably
        # smaller than its own convex hull's.
        # However, compact bubbles with speech connectors or tails naturally
        # have lower solidity (0.72 - 0.79) while staying well within 3.5x text area.
        # Genuine background art leaks balloon outward well past 3.5x text area.
        low_solidity = False
        if filled_area > bbox_area * 1.5:  # only check once fill has ballooned out significantly past text
            _leak_contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if _leak_contours:
                _c = max(_leak_contours, key=cv2.contourArea)
                _hull = cv2.convexHull(_c)
                _hull_area = cv2.contourArea(_hull)
                solidity = (cv2.contourArea(_c) / _hull_area) if _hull_area > 0 else 1.0
                low_solidity = solidity < 0.80

        # Scale-invariant primary check, plus the old ROI-relative check as a
        # secondary corroborating signal (now requires BOTH border-touch and
        # a high ratio, so a snug bubble that happens to nearly fill a small
        # adaptive ROI isn't mistaken for a leak).
        contour_leak_state = {
            "misplaced": misplaced,
            "low_solidity": low_solidity,
            "ratio_to_text": ratio_to_text,
            "effective_ratio_cap": effective_ratio_cap,
            "border_touch": border_touch,
            "ratio_to_roi": ratio_to_roi,
            "roi_leak_ratio": roi_leak_ratio,
            "cx": cx, "cy": cy, "bw": bw, "bh": bh,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1
        }
        suspected_leak = misplaced or low_solidity or (ratio_to_text > effective_ratio_cap) or (
            border_touch and ratio_to_roi > roi_leak_ratio
        )

        if suspected_leak:
            # Check if this leak was caused by an open bubble resting against a panel border/frame
            edges_docked = _close_docked_panel_boundaries(edges.copy(), tx0, ty0, tx1, ty1, gray=gray, bridge_collinear=True)
            flood_mask_docked = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
            flood_mask_docked[1:-1, 1:-1][edges_docked > 0] = 1
            seed_d = _find_light_seed(clean_gray, (cx - rx0, cy - ry0), flood_mask_docked)
            if seed_d is not None:
                clean_bgr_d = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2BGR)
                cv2.floodFill(
                    clean_bgr_d, flood_mask_docked, seed_d, (255, 255, 255),
                    loDiff=tol_bgr, upDiff=tol_bgr,
                    flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8),
                )
                filled_d = (flood_mask_docked[1:-1, 1:-1] == 255).astype(np.uint8) * 255
                d_cnts, _ = cv2.findContours(filled_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if d_cnts:
                    c_d = max(d_cnts, key=cv2.contourArea)
                    ha_d = cv2.contourArea(cv2.convexHull(c_d))
                    sol_d = cv2.contourArea(c_d) / ha_d if ha_d > 0 else 0
                    area_d = int(np.count_nonzero(filled_d))
                    ratio_d = area_d / max(1, bbox_area)
                    if sol_d >= 0.80 and ratio_d <= 10.0:
                        filled = filled_d
                        filled_area = area_d
                        suspected_leak = False

        if suspected_leak:
            continue

        if filled_area < 4:
            continue

        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        largest = cv2.approxPolyDP(largest, 0.003 * peri, True)  # light denoise only
        largest = largest + np.array([[rx0, ry0]], dtype=np.int32)  # -> full-image coords

        # Ultra-wide banners, news chyrons, or captions (aspect ratio >= 4.5 and width >= 400):
        # In comic layout, these are strictly rectangular TV headline bars without speech tails.
        # Regularize contour to its bounding box so TV headline banners are crisp rectangles.
        # Speech bubbles (which have vertical tails or organic curved bodies) MUST NEVER be forced
        # into rectangles!
        rx, ry, rw_c, rh_c = cv2.boundingRect(largest)
        cnt_aspect_w = rw_c / max(1.0, float(rh_c))
        cnt_aspect_h = rh_c / max(1.0, float(rw_c))
        pts_c = largest.reshape(-1, 2)
        min_y_c, max_y_c = int(np.min(pts_c[:, 1])), int(np.max(pts_c[:, 1]))
        has_vertical_protrusion = (y0 - min_y_c >= 18) or (max_y_c - y1 >= 18)
        is_wide_banner = (
            ((cnt_aspect_w >= 4.5 and rw_c >= 400) or (cnt_aspect_h >= 4.5 and rh_c >= 400))
            and not has_vertical_protrusion
        )
        if is_wide_banner:
            rect_cnt = np.array([
                [[rx, ry]],
                [[rx + rw_c, ry]],
                [[rx + rw_c, ry + rh_c]],
                [[rx, ry + rh_c]]
            ], dtype=np.int32)
            return rect_cnt, True

        return largest, _is_rectangular(largest)

    # --- Total failure: apply quality gate policy for background art leak fallback ---
    synth, is_rect = apply_quality_gates(
        "contour_extraction",
        {
            "misplaced": True, "low_solidity": True, "ratio_to_text": 999.0,
            "effective_ratio_cap": effective_ratio_cap, "border_touch": True,
            "ratio_to_roi": 1.0, "roi_leak_ratio": roi_leak_ratio,
            "cx": cx, "cy": cy, "bw": bw, "bh": bh,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1
        },
        context={"image_w": W, "image_h": H}
    )
    return synth, is_rect


def split_multi_lobe_bubble_contour(
    cnt: np.ndarray,
    lines: List[Dict[str, Any]],
    approx_glyph_height: int = 25,
    min_defect_depth: float = 18.0,
    max_depth: int = 2
) -> List[Tuple[np.ndarray, List[Dict[str, Any]]]]:
    """
    Universally detects and cleanly separates conjoined and multi-lobe speech bubbles
    (vertical figure-8/peanut balloons, horizontal conjoined twins, diagonal chained bubbles).

    Algorithm:
    1. Computes 2D Convexity Defects on the bubble contour mask to identify inward constrictions.
    2. Identifies opposing deep defects (depth >= max(min_defect_depth, approx_glyph_height * 0.65)).
    3. Finds the narrowest internal chord (neck/waist) connecting opposing defects.
    4. Slices the mask along this neck line to separate connected components into individual lobes.
    5. Partitions dialogue OCR text lines between lobes via pointPolygonTest.
    6. STRICT INVARIANT: Validates that EACH resulting lobe has at least one dialogue text line
       and substantial area. Tails, wings, or art protrusions without dialogue are safely rejected.
    7. Supports recursive splitting for 3+ chained lobes.

    Returns:
        List of (lobe_contour_page_coords, lobe_lines_sorted)
    """
    if cnt is None or len(cnt) < 10 or not lines or len(lines) < 2:
        return [(cnt, lines)]

    cnt = np.asarray(cnt, dtype=np.int32).reshape(-1, 1, 2)
    bx, by, bw, bh = cv2.boundingRect(cnt)
    parent_area = float(cv2.contourArea(cnt))
    if parent_area < 3500:
        return [(cnt, lines)]

    pad = 15
    mask = np.zeros((bh + 2 * pad, bw + 2 * pad), dtype=np.uint8)
    shifted_cnt = cnt - np.array([[bx - pad, by - pad]], dtype=np.int32)
    cv2.drawContours(mask, [shifted_cnt], -1, 255, -1)

    hull = cv2.convexHull(shifted_cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return [(cnt, lines)]

    defects = cv2.convexityDefects(shifted_cnt, hull)
    if defects is None or len(defects) < 2:
        return [(cnt, lines)]

    dyn_thresh = max(14.0, min(min_defect_depth, float(approx_glyph_height) * 0.50))
    deep_defects = []
    for i in range(len(defects)):
        row = defects[i].flatten()
        s, e, f, d = row[0], row[1], row[2], row[3]
        depth = d / 256.0
        far_pt = tuple(shifted_cnt[f][0])
        if depth >= dyn_thresh:
            deep_defects.append((depth, far_pt))

    if len(deep_defects) < 2:
        return [(cnt, lines)]

    deep_defects.sort(key=lambda x: x[0], reverse=True)

    # Helper to verify chord does not slice through core text
    def _chord_intersects_text(p1_pg, p2_pg, text_lines, margin=None):
        dist = float(np.hypot(p2_pg[0] - p1_pg[0], p2_pg[1] - p1_pg[1]))
        num_pts = max(10, int(dist / 4))
        xs = np.linspace(p1_pg[0], p2_pg[0], num_pts)
        ys = np.linspace(p1_pg[1], p2_pg[1], num_pts)
        for tl in text_lines:
            lx0 = tl.get('x0', 0)
            ly0 = tl.get('y0', 0)
            lx1 = tl.get('x1', 0)
            ly1 = tl.get('y1', 0)
            lh = max(1, ly1 - ly0)
            lcy = tl.get('cy', (ly0 + ly1) // 2)
            core_half_h = max(3.0, float(lh) * 0.25)
            core_y0 = lcy - core_half_h
            core_y1 = lcy + core_half_h
            core_x0 = lx0 + 4
            core_x1 = lx1 - 4
            if core_x1 <= core_x0 or core_y1 <= core_y0:
                continue
            inside = (xs >= core_x0) & (xs <= core_x1) & (ys >= core_y0) & (ys <= core_y1)
            if np.any(inside):
                return True
        return False

    # Check arrangement of lines (predominantly vertical stack vs horizontal)
    line_ys = [l.get('cy', (l.get('y0', 0) + l.get('y1', 0)) // 2) for l in lines]
    line_xs = [l.get('cx', (l.get('x0', 0) + l.get('x1', 0)) // 2) for l in lines]
    is_vertical_stack = (max(line_ys) - min(line_ys)) >= (max(line_xs) - min(line_xs)) * 0.8

    # Search through candidate defect pairs to find the best valid internal neck that separates dialogue lines
    candidate_defects = deep_defects[:12]
    best_results = None
    min_neck_dist = float("inf")

    for i in range(len(candidate_defects)):
        for j in range(i + 1, len(candidate_defects)):
            p1 = candidate_defects[i][1]
            p2 = candidate_defects[j][1]
            dist = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
            # Valid neck is between 12px and 95% of smallest dimension
            if not (12 <= dist < min(bw, bh) * 0.95):
                continue
            # Tilt angle check: vertically stacked lobes must have roughly horizontal neck chord
            if is_vertical_stack and abs(p1[1] - p2[1]) > max(55.0, bh * 0.35):
                continue
            # Horizontally arranged lobes must have roughly vertical neck chord
            if not is_vertical_stack and abs(p1[0] - p2[0]) > max(55.0, bw * 0.35):
                continue

            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            # Midpoint must lie inside the bubble mask (internal chord)
            if not (0 <= mid[1] < mask.shape[0] and 0 <= mid[0] < mask.shape[1] and mask[mid[1], mid[0]] > 0):
                continue

            p1_pg = (p1[0] + bx - pad, p1[1] + by - pad)
            p2_pg = (p2[0] + bx - pad, p2[1] + by - pad)
            if _chord_intersects_text(p1_pg, p2_pg, lines):
                continue

            # Create surgical cut line
            cut_mask = mask.copy()
            cv2.line(cut_mask, p1, p2, 0, thickness=3)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cut_mask)
            if num_labels - 1 < 2:
                continue

            # Extract valid candidate lobes
            candidate_lobes = []
            for lbl in range(1, num_labels):
                area = float(stats[lbl, cv2.CC_STAT_AREA])
                if area < parent_area * 0.10 or area < 1200:
                    continue
                l_mask = (labels == lbl).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(l_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    c_page = c + np.array([[bx - pad, by - pad]], dtype=np.int32)
                    candidate_lobes.append(c_page)

            if len(candidate_lobes) < 2:
                continue

            # Sort lobes in reading order (top-to-bottom, left-to-right)
            def reading_order_key(c):
                b = cv2.boundingRect(c)
                return (b[1], b[0])
            candidate_lobes.sort(key=reading_order_key)

            # Partition OCR lines across candidate lobes
            lobe_line_map = {li: [] for li in range(len(candidate_lobes))}
            unassigned_lines = []

            for l in lines:
                cx, cy = l['cx'], l['cy']
                best_li = None
                max_dist = -float('inf')
                for li, c_page in enumerate(candidate_lobes):
                    dist = cv2.pointPolygonTest(c_page, (float(cx), float(cy)), True)
                    if dist > max_dist:
                        max_dist = dist
                        best_li = li
                if best_li is not None and max_dist >= -15:
                    lobe_line_map[best_li].append(l)
                else:
                    unassigned_lines.append(l)

            # Check invariant: EVERY lobe must contain at least 1 dialogue line
            if any(len(lobe_line_map[li]) == 0 for li in range(len(candidate_lobes))):
                continue

            # Distribute any unassigned lines to the nearest lobe
            for l in unassigned_lines:
                cx, cy = l['cx'], l['cy']
                best_li = min(range(len(candidate_lobes)), key=lambda li: -cv2.pointPolygonTest(candidate_lobes[li], (float(cx), float(cy)), True))
                lobe_line_map[best_li].append(l)

            # Compile split results
            pair_results = []
            for li, c_page in enumerate(candidate_lobes):
                l_sorted = sorted(lobe_line_map[li], key=lambda x: (x['y0'], x['x0']))
                # Optional recursive splitting for 3+ conjoined lobes
                if max_depth > 1 and len(l_sorted) >= 4:
                    sub_res = split_multi_lobe_bubble_contour(
                        c_page, l_sorted,
                        approx_glyph_height=approx_glyph_height,
                        min_defect_depth=min_defect_depth,
                        max_depth=max_depth - 1
                    )
                    pair_results.extend(sub_res)
                else:
                    pair_results.append((c_page, l_sorted))

            if dist < min_neck_dist:
                min_neck_dist = dist
                best_results = pair_results

    if best_results is not None:
        return best_results

    return [(cnt, lines)]


def find_contour_horizontal_constriction(cnt: np.ndarray, min_neck_ratio: float = 0.75) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Analyzes a 2D contour to detect if it has a horizontal waist/neck constriction
    connecting two lobes (like a peanut or conjoined twin speech bubbles).
    
    Returns:
        (best_x_split, details_dict) where best_x_split is the x coordinate of the neck.
    """
    if cnt is None or len(cnt) < 10:
        return None, {}

    bx, by, bw, bh = cv2.boundingRect(cnt)
    # Must be horizontally elongated to have two horizontal lobes
    if bw < 1.35 * bh or bw < 80:
        return None, {}

    # Create binary mask of contour
    mask = np.zeros((bh + 4, bw + 4), dtype=np.uint8)
    shifted_cnt = cnt - np.array([[bx - 2, by - 2]], dtype=np.int32)
    cv2.drawContours(mask, [shifted_cnt], -1, 255, -1)

    # Measure vertical span h(x) for each x column
    heights = []
    top_ys = []
    bot_ys = []
    for x in range(2, bw + 2):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) > 0:
            heights.append(int(ys[-1] - ys[0]))
            top_ys.append(int(ys[0]))
            bot_ys.append(int(ys[-1]))
        else:
            heights.append(0)
            top_ys.append(0)
            bot_ys.append(0)

    if len(heights) < 20:
        return None, {}

    # Search for neck between 25% and 75% of width
    w_quarter = len(heights) // 4
    w_three_quarters = 3 * len(heights) // 4

    left_lobe_max = max(heights[:w_quarter + 10])
    right_lobe_max = max(heights[w_three_quarters - 10:])
    lobe_peak = max(left_lobe_max, right_lobe_max)
    if lobe_peak == 0:
        return None, {}

    mid_heights = heights[w_quarter:w_three_quarters]
    min_h = min(mid_heights)
    min_idx = w_quarter + mid_heights.index(min_h)

    ratio = min_h / float(lobe_peak)

    # Check if there is significant indentation from top and/or bottom
    top_dip = top_ys[min_idx] - min(top_ys[:w_quarter] + top_ys[w_three_quarters:])
    bot_dip = max(bot_ys[:w_quarter] + bot_ys[w_three_quarters:]) - bot_ys[min_idx]

    if ratio <= min_neck_ratio and (top_dip >= 6 or bot_dip >= 6 or ratio <= 0.65):
        split_x = bx + (min_idx - 2)
        return split_x, {
            "neck_x": int(split_x),
            "neck_height": int(min_h),
            "left_max": int(left_lobe_max),
            "right_max": int(right_lobe_max),
            "constriction_ratio": round(float(ratio), 2),
            "top_dip": int(top_dip),
            "bot_dip": int(bot_dip)
        }

    return None, {}


# --------------------------------------------------------------------------
# 2. Zero-box ink removal
# --------------------------------------------------------------------------

def inpaint_bubble_text_only(image_bgr, contour, ink_luminance_threshold=165,
                              border_margin=4, ink_dilate=2, inpaint_radius=3):
    """
    Remove only the dark ink pixels inside `contour`, leaving everything
    else -- including the bubble's own outline and all art outside it --
    byte-for-byte untouched.

    Two safeguards make this "zero-box":
      - `border_margin` erodes the bubble mask inward before ink detection,
        so the bubble's own black outline is never mistaken for text ink
        and inpainted away.
      - cv2.inpaint only ever rewrites pixels where the mask is nonzero; the
        mask here is the intersection of (dark pixels) AND (bubble interior),
        so pixels outside the bubble are provably never modified regardless
        of inpaint_radius.

    Returns a new image (original is not mutated).
    """
    if contour is None:
        return image_bgr.copy()

    if not isinstance(contour, np.ndarray):
        try:
            contour = np.array(contour, dtype=np.int32)
        except Exception:
            return image_bgr.copy()

    if contour.ndim == 2 and contour.shape[1] == 2:
        contour = contour.reshape(-1, 1, 2)
    elif contour.ndim != 3:
        return image_bgr.copy()

    if _is_rectangular(contour):
        border_margin = min(border_margin, 2)
        ink_luminance_threshold = max(ink_luminance_threshold, 215)
        ink_dilate = max(ink_dilate, 3)

    h, w = image_bgr.shape[:2]
    bubble_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(bubble_mask, [contour], -1, 255, thickness=cv2.FILLED)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * border_margin + 1, 2 * border_margin + 1)
    )
    interior_mask = cv2.erode(bubble_mask, kernel, iterations=1)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ink_mask = np.zeros((h, w), dtype=np.uint8)

    # Adaptive background-aware ink detection:
    # Samples authentic fill color c_bg inside interior_mask
    int_pts = image_bgr[interior_mask > 0]
    int_gray = gray[interior_mask > 0]
    is_tinted = False
    bg_is_tinted = False
    text_is_spectral = False
    c_bg_bgr = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    if int_pts.size > 0 and int_gray.size > 0:
        int_hsv = cv2.cvtColor(int_pts.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)[0]
        sat = int_hsv[:, 1]

        # 1. Text ink analysis
        bg_ref_lum = np.percentile(int_gray, 75)
        ink_cand_mask = (int_gray < bg_ref_lum - 12) | (sat >= 10)
        if np.count_nonzero(ink_cand_mask) >= 10:
            ink_pts = int_pts[ink_cand_mask]
            ink_gray = int_gray[ink_cand_mask]
            ink_hsv = int_hsv[ink_cand_mask]
        else:
            p10_g = np.percentile(int_gray, 10)
            ink_pts = int_pts[int_gray <= p10_g]
            ink_gray = int_gray[int_gray <= p10_g]
            ink_hsv = int_hsv[int_gray <= p10_g]

        ink_sat = ink_hsv[:, 1] if len(ink_hsv) > 0 else []
        ink_med_bgr = np.median(ink_pts, axis=0) if len(ink_pts) > 0 else [0, 0, 0]
        ink_lum = np.median(ink_gray) if len(ink_gray) > 0 else 0
        b_r_text_diff = int(ink_med_bgr[0]) - int(ink_med_bgr[2])

        # Spectral text check: text is light tinted ink (Peter Parker ghost dialogue)
        # Slate blue / havorang ink: luminance > 120, sat >= 8 or abs(B - R) >= 12
        text_is_spectral = bool(ink_lum > 120 and (len(ink_sat) > 0 and np.median(ink_sat) >= 8 or abs(b_r_text_diff) >= 12))

        # 2. Non-text background analysis (brightest 60% of interior)
        p40_g = np.percentile(int_gray, 40)
        bg_pts = int_pts[int_gray >= p40_g]
        bg_hsv = int_hsv[int_gray >= p40_g]
        bg_sat = bg_hsv[:, 1] if len(bg_hsv) > 0 else []
        bg_med_sat = np.median(bg_sat) if len(bg_sat) > 0 else 0
        bg_is_tinted = bool(bg_med_sat >= 12 or (len(bg_sat) > 0 and np.percentile(bg_sat, 60) >= 12))

        is_tinted = bool(text_is_spectral or bg_is_tinted)

        if bg_is_tinted:
            c_bg_bgr = np.median(bg_pts, axis=0) if len(bg_pts) > 0 else np.median(int_pts, axis=0)
        else:
            p80_gray = np.percentile(int_gray, 80)
            bg_sample = int_pts[int_gray >= p80_gray]
            c_bg_bgr = np.median(bg_sample, axis=0) if len(bg_sample) > 0 else np.median(int_pts, axis=0)

        c_bg_rgb = [round(float(c_bg_bgr[2]), 1), round(float(c_bg_bgr[1]), 1), round(float(c_bg_bgr[0]), 1)]
        print(f"[INPAINT_SAMPLING] Bubble interior c_bg: BGR={[int(v) for v in c_bg_bgr]}, RGB={c_bg_rgb}, is_tinted={is_tinted}, bg_is_tinted={bg_is_tinted}, text_is_spectral={text_is_spectral}")

        bg_gray = float(0.114 * c_bg_bgr[0] + 0.587 * c_bg_bgr[1] + 0.299 * c_bg_bgr[2])
        color_diff = np.linalg.norm(image_bgr.astype(np.float32) - c_bg_bgr, axis=2)

        hsv_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        sat_full = hsv_full[:, :, 1]

        if text_is_spectral and not bg_is_tinted:
            # Ghost/spectral text on white paper:
            # The background paper is pure clean white. All colored/anti-aliased text ink
            # departing from the white paper is masked without leaving halos.
            is_ink = ((gray < 246) | (sat_full > 6) | (color_diff > 8)) & (interior_mask > 0)
            ink_mask[is_ink] = 255
            ink_dilate = max(ink_dilate, 2)
        elif bg_is_tinted:
            # Genuinely tinted fill (e.g. yellow or colored box):
            # Mask pixels differing in luminance/chroma from the background fill
            is_ink = ((gray < 205) | (color_diff > 12)) & (interior_mask > 0)
            ink_mask[is_ink] = 255
        else:
            is_ink = ((color_diff > 18) | (gray < bg_gray - 12) | (gray < ink_luminance_threshold)) & (interior_mask > 0)
            ink_mask[is_ink] = 255
    else:
        ink_mask[(gray < ink_luminance_threshold) & (interior_mask > 0)] = 255

    if ink_dilate > 0:
        d_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * ink_dilate + 1, 2 * ink_dilate + 1)
        )
        ink_mask = cv2.dilate(ink_mask, d_kernel, iterations=1)
        # re-clip after dilation so it can't creep back onto the border stroke
        ink_mask = cv2.bitwise_and(ink_mask, interior_mask)

    if not np.any(ink_mask):
        return image_bgr.copy()

    cleaned = cv2.inpaint(image_bgr, ink_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)

    if bg_is_tinted:
        # Harmonize ONLY if the background fill itself is genuinely tinted (e.g. yellow box)
        cur_int = cleaned[ink_mask > 0].astype(np.float32)
        if cur_int.size > 0:
            cur_int = 0.40 * cur_int + 0.60 * c_bg_bgr
            cleaned[ink_mask > 0] = np.clip(cur_int, 0, 255).astype(np.uint8)

    # Post-clean quality gate: check for residual ink clusters remaining in interior
    clean_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    residual_mask = ((clean_gray < (140 if is_tinted else ink_luminance_threshold)) & (interior_mask > 0)).astype(np.uint8) * 255
    res_count = int(np.count_nonzero(residual_mask))
    if res_count > 0:
        check_info = {
            "image_bgr": cleaned,
            "residual_mask": residual_mask,
            "residual_ink_count": res_count
        }
        res_info = apply_quality_gates("post_clean", check_info)
        if isinstance(res_info, dict) and "image_bgr" in res_info:
            cleaned = res_info["image_bgr"]

    return cleaned


# --------------------------------------------------------------------------
# 3. Contour-aware text fitting
# --------------------------------------------------------------------------

def calculate_safe_text_area(contour, image_shape, safety_padding=8):
    """
    Compute the safe layout region inside a bubble: an eroded interior mask,
    the largest axis-aligned rectangle that fits entirely inside it (for
    simple rectangular text blocks), and helpers for organic ("diamond")
    per-row wrapping and point-in-bubble checks.

    Args:
        contour: bubble contour, full-image coordinates (from get_bubble_contour).
        image_shape: image_bgr.shape (used to size the mask correctly).
        safety_padding: inward margin in pixels from the bubble's ink border.

    Returns dict:
        'eroded_mask'       -- uint8 mask, 255 = safe to draw on
        'inscribed_rect'    -- (x, y, w, h), max rectangle fully inside the safe area
        'row_width_at(y)'   -- (start_x, end_x) safe horizontal span at row y, or None
        'point_in_safe_area(x, y)' -- bool
    """
    h, w = image_shape[:2]

    # Fast localized bounding box crop (110x speedup over full-page 4K processing)
    bx, by, bw, bh = cv2.boundingRect(contour)
    pad = max(10, int(safety_padding) * 2)
    x0, y0 = max(0, bx - pad), max(0, by - pad)
    x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)
    h_c, w_c = y1 - y0, x1 - x0

    cnt_c = contour - np.array([[[x0, y0]]], dtype=np.int32)
    mask_c = np.zeros((h_c, w_c), dtype=np.uint8)
    cv2.drawContours(mask_c, [cnt_c], -1, 255, thickness=cv2.FILLED)

    k = max(1, int(safety_padding))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    eroded_c = cv2.erode(mask_c, kernel, iterations=1)
    if not np.any(eroded_c):
        eroded_c = cv2.erode(mask_c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    rx, ry, rw, rh = _largest_inscribed_rectangle(eroded_c)

    # Precompute row spans table for O(1) row_width_at lookup (zero np.where inside typesetting loops)
    row_spans = {}
    non_zero_rows = np.where(np.any(eroded_c > 0, axis=1))[0]
    for r in non_zero_rows:
        xs = np.where(eroded_c[r] > 0)[0]
        if xs.size > 0:
            row_spans[int(r + y0)] = (int(xs[0] + x0), int(xs[-1] + x0))

    eroded_full = np.zeros((h, w), dtype=np.uint8)
    eroded_full[y0:y1, x0:x1] = eroded_c

    def row_width_at(row_y):
        return row_spans.get(int(row_y), None)

    def point_in_safe_area(px, py):
        if px < 0 or py < 0 or px >= w or py >= h:
            return False
        return bool(eroded_full[int(py), int(px)] > 0)

    return {
        "eroded_mask": eroded_full,
        "inscribed_rect": (rx + x0, ry + y0, rw, rh),
        "row_width_at": row_width_at,
        "point_in_safe_area": point_in_safe_area,
    }


def is_point_in_bubble(contour, x, y):
    """
    Point-in-polygon check against the raw bubble contour (not the eroded
    safe area) -- useful while laying out individual glyphs with PIL to
    confirm a glyph's corner hasn't crossed the true bubble edge.
    """
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def fit_text_to_bubble(text, contour, image_shape, font_path, safety_padding=12,
                        max_font_size=26, min_font_size=8, line_spacing=1.15, min_clearance=4):
    """
    Contour-aware comic lettering:
    Binary-searches descending font size to fit `text` inside the bubble contour.
    Vertically centers the text block inside the bubble and uses a 'diamond rule'
    wrap so each line's width conforms strictly to the bubble's contour at that row.
    Guarantees that all corners of every line have at least `min_clearance` pixels
    distance from the bubble boundary (zero overflow, zero touching borders).

    Returns {'font_size': int, 'lines': [(text, x, y), ...]} with (x, y) as
    the top-left draw position for each line, or None if it won't fit.
    """
    if contour is None or not (text or "").strip():
        return None

    # Mandatory Rule 6: strip extraneous quotation marks and enforce sentence-per-line formatting
    text = format_dialogue_sentences(text)
    if not text.strip():
        return None

    if not isinstance(contour, np.ndarray):
        try:
            contour = np.array(contour, dtype=np.int32)
        except Exception:
            return None

    if contour.ndim == 2 and contour.shape[1] == 2:
        contour = contour.reshape(-1, 1, 2)

    dynamic_pad = max(1, int(safety_padding))

    safe = calculate_safe_text_area(contour, image_shape, safety_padding=dynamic_pad)
    row_width_at = safe["row_width_at"]
    mask = safe["eroded_mask"]
    ys = np.where(np.any(mask > 0, axis=1))[0]
    if ys.size == 0:
        return None
    top, bottom = int(ys[0]), int(ys[-1])
    bubble_cy = (top + bottom) // 2

    # Inscribed rectangle and protrusion / antenna wing detection
    irx, iry, irw, irh = safe["inscribed_rect"]
    cnt_x, cnt_y, cnt_w, cnt_h = cv2.boundingRect(contour)
    if irh > 20 and ((irx - cnt_x > 20) or ((cnt_x + cnt_w) - (irx + irw) > 20)):
        # Asymmetric wing / tail protrusion: center vertically on main text body
        bubble_cy = iry + irh // 2
        top = max(top, iry)
        bottom = min(bottom, iry + irh)

    paragraphs = [p.strip().split() for p in text.strip().split('\n') if p.strip()]
    if not paragraphs:
        return None

    for size in range(max_font_size, min_font_size - 1, -1):
        try:
            font = ImageFont.truetype(font_path, size)
        except Exception:
            font = ImageFont.load_default()

        line_h = int(size * line_spacing)
        if line_h <= 0 or (bottom - top) < line_h:
            continue

        max_lines = max(1, (bottom - top) // line_h)

        for num_lines in range(1, max_lines + 1):
            total_h = num_lines * line_h
            if total_h > (bottom - top):
                continue

            center_y = max(top, min(bottom - total_h, bubble_cy - (total_h // 2)))
            # Focused center-first vertical positioning (fast local search)
            cand_start_ys = [center_y]
            max_jitter = min(20, max(0, (bottom - top - total_h) // 2))
            for step in range(3, max_jitter + 1, 3):
                if center_y + step + total_h <= bottom:
                    cand_start_ys.append(center_y + step)
                if center_y - step >= top:
                    cand_start_ys.append(center_y - step)

            for start_y in cand_start_ys:
                lines = []
                para_idx = 0
                word_idx = 0
                possible = True

                for line_idx in range(num_lines):
                    if para_idx >= len(paragraphs):
                        possible = False
                        break

                    line_y = start_y + line_idx * line_h
                    y_top = line_y
                    y_mid = line_y + line_h // 2
                    y_bot = min(bottom, line_y + line_h)

                    span_top = row_width_at(y_top)
                    span_mid = row_width_at(y_mid)
                    span_bot = row_width_at(y_bot)

                    if span_top is None or span_mid is None or span_bot is None:
                        possible = False
                        break

                    min_x = max(span_top[0], span_mid[0], span_bot[0])
                    max_x = min(span_top[1], span_mid[1], span_bot[1])
                    if irw > 30 and (irx - cnt_x > 20):
                        min_x = max(min_x, irx)
                    if irw > 30 and ((cnt_x + cnt_w) - (irx + irw) > 20):
                        max_x = min(max_x, irx + irw)
                    max_w = max_x - min_x
                    if max_w <= 10:
                        possible = False
                        break

                    cur_line = ""
                    cur_para = paragraphs[para_idx]
                    while word_idx < len(cur_para):
                        trial = (cur_line + " " + cur_para[word_idx]).strip()
                        bbox = font.getbbox(trial)
                        tw = bbox[2] - bbox[0]
                        if tw <= max_w:
                            cur_line = trial
                            word_idx += 1
                        else:
                            break

                    if not cur_line:
                        possible = False
                        break

                    lines.append((cur_line, min_x, max_x, max_w, line_y))
                    if word_idx >= len(cur_para):
                        para_idx += 1
                        word_idx = 0

                if possible and para_idx >= len(paragraphs):
                    placed = []
                    fits_all_corners = True
                    for l_txt, l_min_x, l_max_x, l_max_w, l_y in lines:
                        bbox = font.getbbox(l_txt)
                        lw = bbox[2] - bbox[0]
                        lh = bbox[3] - bbox[1]
                        cx = l_min_x + max(0, (l_max_w - lw) // 2)

                        pts = [
                            (cx, l_y),
                            (cx + lw, l_y),
                            (cx, l_y + lh),
                            (cx + lw, l_y + lh),
                            (cx + lw // 2, l_y + lh)
                        ]
                        for px, py in pts:
                            dist = cv2.pointPolygonTest(contour, (float(px), float(py)), True)
                            if dist < min_clearance:
                                fits_all_corners = False
                                break
                        if not fits_all_corners:
                            break
                        placed.append((l_txt, cx, l_y))

                    if fits_all_corners and len(placed) == len(lines):
                        return {"font_size": size, "lines": placed}

    # Fallback to inscribed rectangle if needed
    rx, ry, rw, rh = safe["inscribed_rect"]
    if rw > 20 and rh > 20:
        for size in range(min(max_font_size, 18), min_font_size - 1, -1):
            try:
                fb_font = ImageFont.truetype(font_path, size)
            except Exception:
                fb_font = ImageFont.load_default()
            line_h = int(size * line_spacing)
            fb_lines = []
            fits = True
            for cur_para in paragraphs:
                cur = []
                for w in cur_para:
                    cand = " ".join(cur + [w])
                    bbox = fb_font.getbbox(cand)
                    if bbox[2] - bbox[0] <= rw:
                        cur.append(w)
                    else:
                        if cur:
                            fb_lines.append(" ".join(cur))
                            cur = [w]
                            if fb_font.getbbox(w)[2] - fb_font.getbbox(w)[0] > rw:
                                fits = False
                                break
                        else:
                            fits = False
                            break
                if cur:
                    fb_lines.append(" ".join(cur))
                if not fits:
                    break
            if not fits:
                continue
            tot_h = len(fb_lines) * line_h
            if tot_h <= rh:
                sy = ry + (rh - tot_h) // 2
                placed = []
                for i, fl in enumerate(fb_lines):
                    bbox = fb_font.getbbox(fl)
                    lw = bbox[2] - bbox[0]
                    cx = rx + max(0, (rw - lw) // 2)
                    placed.append((fl, cx, sy + i * line_h))
                return {"font_size": size, "lines": placed}

    return None


# --------------------------------------------------------------------------
# Example integration with an EasyOCR pass over a full page
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import easyocr  # pip install easyocr

    page_path = "page.png"
    page_bgr = cv2.imread(page_path)
    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(page_path)  # [(box, text, conf), ...]

    def to_bbox(box):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    all_bboxes = [to_bbox(box) for box, _text, _conf in results]
    all_centroids = [((b[0] + b[2]) // 2, (b[1] + b[3]) // 2) for b in all_bboxes]

    output = page_bgr.copy()
    for idx, (box, ocr_text, conf) in enumerate(results):
        text_bbox = all_bboxes[idx]
        neighbor_seeds = [c for j, c in enumerate(all_centroids) if j != idx]

        contour, is_rect = get_bubble_contour(page_bgr, text_bbox, neighbor_seeds=neighbor_seeds)
        output = inpaint_bubble_text_only(output, contour)

        translated = ocr_text  # <- replace with your translation call
        layout = fit_text_to_bubble(translated, contour, page_bgr.shape,
                                     font_path="CC-Wild-Words.ttf")
        # draw `layout['lines']` onto `output` with PIL here.

    cv2.imwrite("lettered_page.png", output)