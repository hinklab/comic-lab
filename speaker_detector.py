"""
speaker_detector.py - Contextual Character Grounding & Speaker Identification.
-------------------------------------------------------------------------------
Integrates Tail Vector Projection (Step 1.2) with Character Color Signatures &
Multimodal Dialogue Roles (Step 1.3) to resolve comic speech bubble speakers.

Pipeline per bubble:
  1. Determine bubble role: Rectangular captions without tails -> "Sharhlovchi" (Narrator).
  2. For speech bubbles: Follow tail unit vector outward into the panel to locate
     the speaker's probable anchor point (face/upper body).
  3. Sample character region around the anchor, filtering out bubble white fill & panel lines.
  4. Extract dominant k-means color palette (costume/skin/hair).
  5. Cross-reference against persistent Character Voice Profiles (character_memory.json).
  6. Ground the bubble to the verified character with confidence score.
  7. If confirmed, dynamically update and grow character visual memory.
"""

from __future__ import annotations

import math
import datetime
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

import cv2
import numpy as np

import tail_detector

if TYPE_CHECKING:
    from character_profiles import CharacterVoiceProfile

# Tuning constants
CONF_THRESHOLD   = 0.50   # minimum confidence to assign a speaker from palette match
SAMPLE_RADIUS    = 45     # radius of sampling window around speaker anchor (px)
KMEANS_K         = 3      # dominant color clusters
MAX_COLOR_DIST   = 160.0  # max RGB distance used for confidence normalisation
MERGE_DIST       = 30.0   # Euclidean distance below which new sig clusters are merged


# ---- Tail tip extraction -----------------------------------------------------

def find_tail_tip(contour_pts: Any, bubble_bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
    """
    Find the tail-tip point - the contour vertex that protrudes significantly
    further outside the bubble bounding box than the typical contour vertex.
    Returns (x, y) or None if no clear tail (e.g. caption rectangle).
    """
    if contour_pts is None or len(contour_pts) < 4:
        return None

    bx0, by0, bx1, by1 = bubble_bbox
    dists = []
    for p in contour_pts:
        px, py = int(p[0]), int(p[1])
        dx = max(0, bx0 - px, px - bx1)
        dy = max(0, by0 - py, py - by1)
        dists.append(math.hypot(dx, dy))

    dists_arr = np.array(dists)
    median_d = float(np.median(dists_arr))
    max_idx = int(np.argmax(dists_arr))
    max_d = float(dists_arr[max_idx])

    # Require that the tail protrudes at least 20px further than the median boundary
    if max_d - median_d < 20.0:
        return None

    best_pt = contour_pts[max_idx]
    return (int(best_pt[0]), int(best_pt[1]))


# ---- Color Distance & Signature Matching -------------------------------------

def _rgb_dist(a: List[int], b: List[int]) -> float:
    return math.sqrt(sum((int(x) - int(y)) ** 2 for x, y in zip(a, b)))


def _weighted_signature_distance(
    query_colors: List[List[int]],
    query_weights: List[float],
    sig_colors: List[List[int]],
    sig_weights: List[float]
) -> float:
    if not sig_colors or not query_colors:
        return MAX_COLOR_DIST
    total_dist = 0.0
    for qc, qw in zip(query_colors, query_weights):
        min_d = min(_rgb_dist(qc, sc) for sc in sig_colors)
        total_dist += qw * min_d
    return total_dist


def match_against_signatures(
    query_colors: List[List[int]],
    query_weights: List[float],
    profiles: List[Any]
) -> Tuple[Optional[str], float]:
    """
    Compare query dominant colors against all character visual signatures.
    Returns (character_name_or_None, confidence_float).
    """
    best_name = None
    best_conf = 0.0

    for profile in profiles:
        sig = getattr(profile, "visual_signature", None) if not isinstance(profile, dict) else profile.get("visual_signature")
        name = getattr(profile, "name", None) if not isinstance(profile, dict) else profile.get("name")
        if not sig or not sig.get("dominant_colors"):
            continue

        sig_colors  = sig["dominant_colors"]
        sig_weights = sig.get("color_weights", [1.0 / len(sig_colors)] * len(sig_colors))
        dist = _weighted_signature_distance(query_colors, query_weights, sig_colors, sig_weights)
        conf = max(0.0, 1.0 - dist / MAX_COLOR_DIST)
        if conf > best_conf:
            best_conf = conf
            best_name = name

    if best_conf >= CONF_THRESHOLD:
        return best_name, round(best_conf, 3)
    return None, round(best_conf, 3)


# ---- Region Sampling & Dominant Colors ---------------------------------------

def _character_pixel_mask(region_bgr: np.ndarray) -> np.ndarray:
    """
    Boolean mask: selects pixels belonging to character skin, hair, and costume.
    Excludes white/light bubble fill (V > 220, S < 35) and dark line art (V < 30).
    """
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    # Filter out pure white bubble interior or empty gutters
    is_bubble_white = (v > 220) & (s < 38)
    # Filter out black borders/lines
    is_black_line = (v < 32)
    return ~(is_bubble_white | is_black_line)


def sample_character_palette(
    image_bgr: np.ndarray,
    tail_tip: List[int],
    tail_vector: List[float],
    panel_box: Optional[List[int]] = None,
    page_size: Optional[Tuple[int, int]] = None,
    distance_px: int = 80,
    radius: int = SAMPLE_RADIUS
) -> Tuple[List[List[int]], List[float], Optional[Tuple[int, int]]]:
    """
    Follows the tail unit vector into the comic panel to project the speaker's
    body/face anchor, and extracts dominant k-means colors of the character.
    Returns: (colors_rgb_list, weights_list, anchor_point_xy).
    """
    if image_bgr is None or tail_tip is None or tail_vector is None:
        return [], [], None

    h, w = image_bgr.shape[:2]
    if page_size is None:
        page_size = (w, h)

    # Project along tail direction
    anchor_x, anchor_y = tail_detector.project_tail_to_speaker_region(
        tail_tip=tail_tip,
        tail_vector=tail_vector,
        panel_box=panel_box,
        page_size=page_size,
        distance_px=distance_px
    )

    x0 = max(0, anchor_x - radius)
    y0 = max(0, anchor_y - radius)
    x1 = min(w, anchor_x + radius)
    y1 = min(h, anchor_y + radius)

    if x1 <= x0 or y1 <= y0:
        return [], [], (anchor_x, anchor_y)

    crop = image_bgr[y0:y1, x0:x1]
    mask = _character_pixel_mask(crop)
    pixels = crop[mask]

    if len(pixels) < 30:
        # Fallback: if masked pixels are few (e.g. monochromatic scene), take middle non-white
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        fallback_mask = (gray < 225) & (gray > 25)
        pixels = crop[fallback_mask]
        if len(pixels) < 20:
            return [], [], (anchor_x, anchor_y)

    colors, weights = extract_dominant_colors(pixels, k=KMEANS_K)
    return colors, weights, (anchor_x, anchor_y)


def extract_dominant_colors(
    pixels_bgr: np.ndarray,
    k: int = KMEANS_K
) -> Tuple[List[List[int]], List[float]]:
    """
    Extracts k-means dominant colors from an Nx3 BGR pixel array.
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
    # Sort by weight descending
    paired = sorted(zip(weights, colors), reverse=True)
    return [p[1] for p in paired], [round(p[0], 3) for p in paired]


# ---- Contextual Grounding Core ------------------------------------------------

def ground_bubble_speaker(
    bubble: Any,
    image_bgr: Optional[np.ndarray] = None,
    panel_box: Optional[List[int]] = None,
    profiles: Optional[List[Any]] = None,
    candidate_speaker: Optional[str] = None
) -> Tuple[Optional[str], float, Optional[Tuple[int, int]]]:
    """
    Grounds a single bubble to its character persona.
    Returns: (speaker_name, confidence, speaker_anchor_xy)
    """
    shape_type = getattr(bubble, "shape_type", "oval") if not isinstance(bubble, dict) else bubble.get("shape_type", "oval")
    has_tail = getattr(bubble, "has_tail", False) if not isinstance(bubble, dict) else bubble.get("has_tail", False)
    orig_text = str(getattr(bubble, "original_text", "") or "" if not isinstance(bubble, dict) else bubble.get("original_text", "")).strip()

    # Rule 1: Rectangular caption boxes without tails are Narrator / Story Captions
    if shape_type == "rectangle" and not has_tail:
        return "Sharhlovchi", 0.95, None

    # Rule 1.5: Off-Panel / Off-Screen Voice (Tail points towards or outside panel border)
    is_off_panel = getattr(bubble, "is_off_panel", False) if not isinstance(bubble, dict) else bubble.get("is_off_panel", False)
    tip = getattr(bubble, "tail_tip", None) if not isinstance(bubble, dict) else bubble.get("tail_tip")
    vec = getattr(bubble, "tail_vector", None) if not isinstance(bubble, dict) else bubble.get("tail_vector")

    if not is_off_panel and has_tail and tip and vec:
        h, w = image_bgr.shape[:2] if image_bgr is not None else (None, None)
        psize = (w, h) if w and h else None
        is_off_panel = tail_detector.is_tail_off_panel(tip, vec, panel_box=panel_box, page_size=psize)
        if isinstance(bubble, dict):
            bubble["is_off_panel"] = is_off_panel
        else:
            bubble.is_off_panel = is_off_panel

    if is_off_panel:
        # Off-Panel voice! MUST NOT be attributed to the in-panel character sitting in the chair!
        cleaned_candidate = candidate_speaker.strip() if candidate_speaker else None
        if cleaned_candidate and cleaned_candidate.lower() not in ("unknown", "noma'lum", "none", "shakl", "pufak"):
            if "kadrdan" not in cleaned_candidate.lower() and "off-screen" not in cleaned_candidate.lower() and "o.s." not in cleaned_candidate.lower():
                return f"{cleaned_candidate} (Kadrdan tashqari)", 0.90, None
            return cleaned_candidate, 0.90, None
        return "Kadrdan tashqari (Off-Screen)", 0.90, None

    # Rule 2: If candidate speaker is already recognized (e.g. from Gemini Vision)
    cleaned_candidate = candidate_speaker.strip() if candidate_speaker else None
    if cleaned_candidate and cleaned_candidate.lower() not in ("unknown", "noma'lum", "none", "shakl", "pufak"):
        # Strip old bogus seed profiles if erroneously assigned
        if cleaned_candidate.lower() in ("otto octavius", "otto oktavius", "woman"):
            cleaned_candidate = None
        else:
            # Verified candidate
            anchor = None
            if has_tail and image_bgr is not None:
                if tip and vec:
                    colors, weights, anchor = sample_character_palette(
                        image_bgr, tip, vec, panel_box=panel_box, distance_px=75
                    )
            return cleaned_candidate, 0.92, anchor

    # Rule 3: Visual Signature Matching using Tail Direction
    if has_tail and image_bgr is not None and profiles:
        tip = getattr(bubble, "tail_tip", None) if not isinstance(bubble, dict) else bubble.get("tail_tip")
        vec = getattr(bubble, "tail_vector", None) if not isinstance(bubble, dict) else bubble.get("tail_vector")
        if tip and vec:
            colors, weights, anchor = sample_character_palette(
                image_bgr, tip, vec, panel_box=panel_box, distance_px=80
            )
            if colors:
                matched_name, conf = match_against_signatures(colors, weights, profiles)
                if matched_name and conf >= CONF_THRESHOLD:
                    return matched_name, conf, anchor

    return None, 0.0, None


def ground_page_speakers(
    bubbles: List[Any],
    image_bgr: Optional[np.ndarray] = None,
    panel_boxes: Optional[List[Any]] = None,
    profiles: Optional[List[Any]] = None
) -> List[Any]:
    """
    Grounds all bubbles across a full comic page:
    - Associates bubbles with their respective panels.
    - Resolves multi-lobe cluster consistency (conjoined bubbles share speaker).
    - Assigns b.speaker, b.speaker_confidence, and b.speaker_anchor.
    """
    if not bubbles:
        return bubbles

    if profiles is None:
        try:
            import character_profiles
            store = character_profiles.CharacterMemoryStore.get_instance()
            profiles = list(store._profiles.values())
        except Exception:
            profiles = []

    # Map bubbles to panel boxes
    panel_box_map = {}
    if panel_boxes:
        for idx, pbox in enumerate(panel_boxes, start=1):
            panel_box_map[idx] = list(pbox) if not isinstance(pbox, list) else pbox

    for b in bubbles:
        p_id = getattr(b, "panel_id", None) if not isinstance(b, dict) else b.get("panel_id")
        pbox = panel_box_map.get(p_id) if p_id else None

        existing_spk = getattr(b, "speaker", None) if not isinstance(b, dict) else b.get("speaker")
        spk_name, conf, anchor = ground_bubble_speaker(
            b,
            image_bgr=image_bgr,
            panel_box=pbox,
            profiles=profiles,
            candidate_speaker=existing_spk
        )

        if isinstance(b, dict):
            b["speaker"] = spk_name or ("Sharhlovchi" if (b.get("shape_type") == "rectangle" and not b.get("has_tail")) else None)
            b["speaker_confidence"] = conf
            if anchor:
                b["speaker_anchor"] = list(anchor)
        else:
            b.speaker = spk_name or ("Sharhlovchi" if (getattr(b, "shape_type", "oval") == "rectangle" and not getattr(b, "has_tail", False)) else None)
            b.speaker_confidence = conf
            if anchor and hasattr(b, "speaker_anchor"):
                b.speaker_anchor = list(anchor)

    # Continuity Pass: multi-lobe clusters must share speaker
    cluster_speakers: Dict[str, Tuple[Optional[str], float]] = {}
    for b in bubbles:
        c_id = getattr(b, "cluster_id", None) if not isinstance(b, dict) else b.get("cluster_id")
        spk = getattr(b, "speaker", None) if not isinstance(b, dict) else b.get("speaker")
        conf = getattr(b, "speaker_confidence", 0.0) if not isinstance(b, dict) else b.get("speaker_confidence", 0.0)
        if c_id and spk:
            if c_id not in cluster_speakers or conf > cluster_speakers[c_id][1]:
                cluster_speakers[c_id] = (spk, conf)

    for b in bubbles:
        c_id = getattr(b, "cluster_id", None) if not isinstance(b, dict) else b.get("cluster_id")
        if c_id and c_id in cluster_speakers:
            c_spk, c_conf = cluster_speakers[c_id]
            if isinstance(b, dict):
                b["speaker"] = c_spk
                b["speaker_confidence"] = c_conf
            else:
                b.speaker = c_spk
                b.speaker_confidence = c_conf

    return bubbles


# ---- Backward Compatibility & Signature Growth -------------------------------

def should_merge_color(new_color: List[int], existing_colors: List[List[int]], threshold: float = MERGE_DIST) -> bool:
    return any(_rgb_dist(new_color, ec) < threshold for ec in existing_colors)


def merge_signature(
    existing_sig: Dict[str, Any],
    new_colors: List[List[int]],
    new_weights: List[float]
) -> Dict[str, Any]:
    """
    Merge new dominant colors into an existing visual signature using EMA.
    Near-duplicate clusters (within MERGE_DIST) are blended, not added.
    """
    import copy
    sig = copy.deepcopy(existing_sig)
    ex_colors  = sig.get("dominant_colors", [])
    ex_weights = sig.get("color_weights", [])
    n_samples  = sig.get("sample_count", 0)

    alpha = 1.0 / (n_samples + 1)
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
        sig["color_weights"]   = [round(w / total_w, 3) for w in ex_weights]
        sig["sample_count"]    = n_samples + 1
        sig["last_updated"]    = datetime.date.today().isoformat()

    return sig


def detect_speaker(
    contour_pts: Any,
    bubble_bbox: List[int],
    image_bgr: np.ndarray,
    profiles: List[Any]
) -> Tuple[Optional[str], float, Optional[List[List[int]]], Optional[List[float]]]:
    """Backward-compatible helper function."""
    import tail_detector
    tail_info = tail_detector.detect_bubble_tail(contour_pts, bubble_bbox)
    if tail_info.get("has_tail"):
        tip = tail_info.get("tail_tip")
        vec = tail_info.get("tail_vector")
        colors, weights, _ = sample_character_palette(image_bgr, tip, vec)
        if colors:
            speaker, confidence = match_against_signatures(colors, weights, profiles)
            return speaker, confidence, colors, weights
    return None, 0.0, None, None
