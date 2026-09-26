"""
speaker_detector.py - Classical CV Speaker Detection (zero GPU, zero neural nets).

Pipeline per bubble:
  1. Find the bubble tail tip from its contour polygon.
  2. Sample a region of the source image near the tail tip.
  3. Extract dominant colors (k-means, k=3) from that region.
  4. Compare against all stored visual signatures -> best match above threshold.

All deps already in the project: numpy, opencv-python.
"""

from __future__ import annotations

import math
import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from character_profiles import CharacterVoiceProfile

# Tuning constants
CONF_THRESHOLD   = 0.55   # minimum confidence to assign a speaker
SAMPLE_RADIUS    = 65     # half-size of sampling window around tail tip (px)
KMEANS_K         = 3      # dominant color clusters
MAX_COLOR_DIST   = 160.0  # max RGB distance used for confidence normalisation
MERGE_DIST       = 30.0   # Euclidean distance below which new sig clusters are merged


# ---- Tail tip extraction -----------------------------------------------------

def _inner_bbox(contour_pts, shrink=0.70):
    xs = [p[0] for p in contour_pts]
    ys = [p[1] for p in contour_pts]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    hw = (x1 - x0) * shrink / 2
    hh = (y1 - y0) * shrink / 2
    return int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)


def find_tail_tip(contour_pts, bubble_bbox):
    """
    Find the tail-tip point - the contour vertex that protrudes significantly
    further outside the bubble bounding box than the typical contour vertex.
    Returns (x, y) or None if no clear tail (e.g. caption rectangle).
    """
    if not contour_pts or len(contour_pts) < 4:
        return None

    bx0, by0, bx1, by1 = bubble_bbox
    dists = []
    for p in contour_pts:
        px, py = int(p[0]), int(p[1])
        dx = max(0, bx0 - px, px - bx1)
        dy = max(0, by0 - py, py - by1)
        dists.append(math.hypot(dx, dy))

    dists = np.array(dists)
    median_d = np.median(dists)
    max_idx = int(np.argmax(dists))
    max_d = dists[max_idx]

    # Require that the tail protrudes at least 20px further than the median boundary
    if max_d - median_d < 20.0:
        return None

    best_pt = contour_pts[max_idx]
    return (int(best_pt[0]), int(best_pt[1]))



# ---- Region sampling ---------------------------------------------------------

def _bubble_fill_mask(region_bgr):
    """Bool mask: pixels that are NOT bubble fill (white) or ink (black)."""
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    return (gray < 220) & (gray > 35)


def sample_region_near_tip(image_bgr, tip_xy, radius=SAMPLE_RADIUS):
    """
    Crop window around tail tip, return non-white/non-black pixels as Nx3 BGR array.
    Returns None if < 50 usable pixels.
    """
    h, w = image_bgr.shape[:2]
    tx, ty = tip_xy
    x0 = max(0, tx - radius)
    y0 = max(0, ty - radius)
    x1 = min(w, tx + radius)
    y1 = min(h, ty + radius)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = image_bgr[y0:y1, x0:x1]
    mask = _bubble_fill_mask(crop)
    pixels = crop[mask]
    return pixels if len(pixels) >= 50 else None


def sample_region_near_bbox(image_bgr, bubble_bbox, radius=SAMPLE_RADIUS):
    """Fallback: sample just above the bubble when no tail is found."""
    bx0, by0, bx1, by1 = bubble_bbox
    cx = (bx0 + bx1) // 2
    tip = (cx, max(0, by0 - radius // 2))
    return sample_region_near_tip(image_bgr, tip, radius=radius)


# ---- Dominant color extraction -----------------------------------------------

def extract_dominant_colors(pixels_bgr, k=KMEANS_K):
    """
    k-means dominant colors from an Nx3 BGR pixel array.
    Returns (colors_rgb_list, weights_list).
    """
    if pixels_bgr is None or len(pixels_bgr) == 0:
        return [], []
    data = np.float32(pixels_bgr)
    actual_k = min(k, len(data))
    if actual_k < 1:
        return [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        data, actual_k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    label_counts = np.bincount(labels.flatten(), minlength=actual_k)
    total = max(1, label_counts.sum())
    weights = (label_counts / total).tolist()
    # BGR -> RGB
    colors = [[int(c[2]), int(c[1]), int(c[0])] for c in centers]
    # Sort by weight desc
    paired = sorted(zip(weights, colors), reverse=True)
    return [p[1] for p in paired], [p[0] for p in paired]


# ---- Signature comparison ----------------------------------------------------

def _rgb_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _weighted_signature_distance(query_colors, query_weights, sig_colors, sig_weights):
    if not sig_colors or not query_colors:
        return MAX_COLOR_DIST
    total_dist = 0.0
    for qc, qw in zip(query_colors, query_weights):
        min_d = min(_rgb_dist(qc, sc) for sc in sig_colors)
        total_dist += qw * min_d
    return total_dist


def match_against_signatures(query_colors, query_weights, profiles):
    """
    Compare query dominant colors against all character visual signatures.
    Returns (character_name_or_None, confidence_float).
    """
    best_name = None
    best_conf = 0.0
    for profile in profiles:
        sig = getattr(profile, "visual_signature", None)
        if not sig or not sig.get("dominant_colors"):
            continue
        sig_colors  = sig["dominant_colors"]
        sig_weights = sig.get("color_weights", [1.0 / len(sig_colors)] * len(sig_colors))
        dist = _weighted_signature_distance(query_colors, query_weights, sig_colors, sig_weights)
        conf = max(0.0, 1.0 - dist / MAX_COLOR_DIST)
        if conf > best_conf:
            best_conf = conf
            best_name = profile.name
    if best_conf >= CONF_THRESHOLD:
        return best_name, best_conf
    return None, best_conf


# ---- Public API --------------------------------------------------------------

def detect_speaker(contour_pts, bubble_bbox, image_bgr, profiles):
    """
    Detect the speaker for a single bubble.

    Returns:
        (speaker_name_or_None, confidence, dominant_colors, color_weights)
    """
    tip = find_tail_tip(contour_pts, bubble_bbox)
    pixels = (sample_region_near_tip(image_bgr, tip)
              if tip is not None
              else sample_region_near_bbox(image_bgr, bubble_bbox))
    if pixels is None:
        return None, 0.0, None, None
    colors, weights = extract_dominant_colors(pixels, k=KMEANS_K)
    if not colors:
        return None, 0.0, None, None
    speaker, confidence = match_against_signatures(colors, weights, profiles)
    return speaker, confidence, colors, weights


# ---- Signature growth helper -------------------------------------------------

def should_merge_color(new_color, existing_colors, threshold=MERGE_DIST):
    return any(_rgb_dist(new_color, ec) < threshold for ec in existing_colors)


def merge_signature(existing_sig, new_colors, new_weights):
    """
    Merge new dominant colors into an existing visual signature using EMA.
    Near-duplicate clusters (within MERGE_DIST) are blended, not added.
    """
    import copy
    sig = copy.deepcopy(existing_sig)
    ex_colors  = sig.get("dominant_colors", [])
    ex_weights = sig.get("color_weights", [])
    n_samples  = sig.get("sample_count", 0)

    alpha = 1.0 / (n_samples + 1)   # EMA factor
    added_any = False

    for nc, nw in zip(new_colors, new_weights):
        if should_merge_color(nc, ex_colors):
            dists = [_rgb_dist(nc, ec) for ec in ex_colors]
            idx   = int(np.argmin(dists))
            ec    = ex_colors[idx]
            ex_colors[idx]  = [int(ec[i] * (1 - alpha) + nc[i] * alpha) for i in range(3)]
            ex_weights[idx] = ex_weights[idx] * (1 - alpha) + nw * alpha
        else:
            ex_colors.append(nc)
            ex_weights.append(nw * alpha)
        added_any = True

    if added_any:
        total_w = sum(ex_weights) or 1.0
        sig["dominant_colors"] = ex_colors
        sig["color_weights"]   = [w / total_w for w in ex_weights]
        sig["sample_count"]    = n_samples + 1
        sig["last_updated"]    = datetime.date.today().isoformat()

    return sig
