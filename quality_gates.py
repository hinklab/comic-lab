"""
quality_gates.py - Self-diagnosing runtime policy layer for comic translation pipeline.

Runs automatically inside engine.py and bubble_lettering.py on every page.
Maintains persistent memory of known failure patterns, their numeric signatures,
and verified resolutions, logging every gate firing into a structured report.
"""

from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Optional, Tuple
import threading
import time
import cv2
import numpy as np


@dataclass
class QualityGateEvent:
    """Represents a detected failure shape and the policy applied to resolve or flag it."""
    pattern_name: str
    stage: str
    severity: str  # 'auto_fix' | 'flag_for_review'
    action_taken: str
    details: Dict[str, Any] = field(default_factory=dict)
    bubble_id: Optional[int] = None
    page: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page": self.page,
            "bubble_id": self.bubble_id,
            "stage": self.stage,
            "pattern_name": self.pattern_name,
            "severity": self.severity,
            "action_taken": self.action_taken,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class IssuePattern:
    """A known failure pattern with its detection criteria and resolution policy."""
    name: str
    stage: str
    description: str
    severity: str  # 'auto_fix' | 'flag_for_review'
    detect: Callable[[Any, Dict[str, Any]], bool]
    resolve: Callable[[Any, Dict[str, Any]], Any]


class QualityGateLogger:
    """Thread-safe collector for quality gate events across pipeline stages."""
    def __init__(self):
        self._lock = threading.Lock()
        self._events: List[QualityGateEvent] = []
        self._current_page: Optional[str] = None

    def set_current_page(self, page_name: Optional[str]):
        with self._lock:
            self._current_page = page_name

    def record_event(self, event: QualityGateEvent):
        with self._lock:
            if event.page is None:
                event.page = self._current_page
            self._events.append(event)

    def get_log(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [e.to_dict() for e in self._events]

    def clear(self):
        with self._lock:
            self._events.clear()


# Global logger instance
_GLOBAL_LOGGER = QualityGateLogger()


class quality_gate_context:
    """Context manager to scope quality gate logging to a specific page/job."""
    def __init__(self, page: Optional[str] = None, clear_existing: bool = False):
        self.page = page
        self.clear_existing = clear_existing

    def __enter__(self):
        if self.clear_existing:
            _GLOBAL_LOGGER.clear()
        _GLOBAL_LOGGER.set_current_page(self.page)
        return _GLOBAL_LOGGER

    def __exit__(self, exc_type, exc_val, exc_tb):
        _GLOBAL_LOGGER.set_current_page(None)


def get_quality_gate_log() -> List[Dict[str, Any]]:
    """Retrieve full list of structured quality gate events recorded so far."""
    return _GLOBAL_LOGGER.get_log()


def clear_quality_gate_log():
    """Clear all recorded quality gate events and reset learned thresholds."""
    _GLOBAL_LOGGER.clear()
    reset_learned_thresholds()


def log_quality_gate_event(pattern_name: str, stage: str, severity: str, action_taken: str,
                           details: Optional[Dict[str, Any]] = None,
                           bubble_id: Optional[int] = None, page: Optional[str] = None):
    """Explicitly record a quality gate event into in-memory logger and Supabase (with local fallback)."""
    event = QualityGateEvent(
        pattern_name=pattern_name,
        stage=stage,
        severity=severity,
        action_taken=action_taken,
        details=details or {},
        bubble_id=bubble_id,
        page=page
    )
    _GLOBAL_LOGGER.record_event(event)

    # Persist to Supabase quality_gate_events table (or local quality_gate_log.json fallback)
    try:
        import supabase_db
        supabase_db.insert_quality_gate_event(event.to_dict())
    except Exception:
        pass


# ==============================================================================
# PATTERN DETECT & RESOLVE DEFINITIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. SFX False Positive (Stage: ocr_filtering)
# ------------------------------------------------------------------------------
def _detect_sfx_false_positive(candidate_info: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """
    Detects non-dialogue sound-effect or background art text lines:
    - Dark background: mean_luminance < 140 AND percentile_75 < 180
    - Highly saturated background on bright pixels: mean_sat > 85 (unless uniform monochromatic tint)
    - Exceptionally giant glyphs (> 3.0x median line height on page)
    """
    mean_lum = candidate_info.get("mean_luminance", 255.0)
    p75 = candidate_info.get("percentile_75", 255.0)
    mean_sat = candidate_info.get("mean_saturation", 0.0)
    sat_std = candidate_info.get("sat_std", 0.0)
    bright_ratio = candidate_info.get("bright_ratio", 1.0)
    max_val = candidate_info.get("max_val", 255.0)
    line_h = candidate_info.get("line_h", 30)
    median_h = context.get("median_glyph_h", 35)

    is_dark = (mean_lum < 140) and (p75 < 180)
    is_oversized_sfx = bool(line_h > max(80, median_h * 2.8) and is_dark)

    line_w = candidate_info.get("line_w", 0)
    aspect = (line_w / max(1, line_h)) if line_w > 0 else 1.0
    is_graphic_banner = bool(aspect >= 4.0 and mean_sat >= 60)

    # General monochrome-tinted bubble check: low saturation variance and bright value channel
    has_tint_info = ("max_val" in candidate_info) or ("sat_std" in candidate_info)
    is_uniform_tint = (
        has_tint_info
        and not is_graphic_banner
        and (mean_lum >= 120 and p75 >= 155 and max_val >= 160)
        and (sat_std <= 22 or (mean_sat > 0 and sat_std / max(1.0, mean_sat) <= 0.40))
    )

    if is_uniform_tint:
        is_colored = False
    else:
        is_colored = bool(is_graphic_banner or (mean_sat > 85 and bright_ratio < 0.15))

    return bool(is_dark or is_colored or is_oversized_sfx or is_graphic_banner)


def _resolve_sfx_false_positive(candidate_info: Dict[str, Any], context: Dict[str, Any]) -> None:
    # Resolution: Reject candidate from speech lines by returning None
    return None


# ------------------------------------------------------------------------------
# 2. Diamond-Rule Line Split (Stage: line_pairing)
# ------------------------------------------------------------------------------
def _detect_diamond_rule_line_split(pair_info: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """
    Detects center-wrapped text lines in round/oval balloons where horizontal bboxes
    have little/no overlap (gap_x > 0), but vertical centers are diamond-aligned.
    """
    gap_x = pair_info.get("gap_x", 0)
    gap_y = pair_info.get("gap_y", 0)
    max_dy = context.get("max_dy", 30)
    c_dist = pair_info.get("c_dist", 0.0)
    wider_half = pair_info.get("wider_half", 50.0)
    diamond_tol = context.get("diamond_center_tolerance", 0.65)
    max_dx = context.get("max_dx", 25)

    center_aligned = c_dist <= max(wider_half * diamond_tol, max_dx)
    return bool(gap_y <= max_dy and gap_x > 0 and center_aligned)


def _resolve_diamond_rule_line_split(pair_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    # Resolution: Allow connection despite horizontal bbox gap
    pair_info["allow_connection"] = True
    return pair_info


# ------------------------------------------------------------------------------
# 3. Adjacent Bubble Merging Across Columns (Stage: line_clustering)
# ------------------------------------------------------------------------------
def _detect_adjacent_bubble_merging_cross_column(cluster: List[Dict[str, Any]], context: Dict[str, Any]) -> bool:
    """
    Detects if a cluster spans multiple adjacent speakers' speech balloons
    with a clear vertical dividing gap separating left and right lines.
    """
    if len(cluster) < 2:
        return False
    x_intervals = sorted([(l['x0'], l['x1']) for l in cluster], key=lambda iv: iv[0])
    min_split_gap = context.get("min_split_gap", 12)
    for k in range(len(x_intervals) - 1):
        cur_end = max(iv[1] for iv in x_intervals[:k + 1])
        next_start = min(iv[0] for iv in x_intervals[k + 1:])
        if next_start - cur_end >= min_split_gap:
            return True
    return False


def _resolve_adjacent_bubble_merging_cross_column(cluster: List[Dict[str, Any]], context: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """Splits multi-speaker cluster into separate Left and Right bubble clusters."""
    x_intervals = sorted([(l['x0'], l['x1']) for l in cluster], key=lambda iv: iv[0])
    min_split_gap = context.get("min_split_gap", 12)
    best_split_x = None
    max_split_gap = 0
    for k in range(len(x_intervals) - 1):
        cur_end = max(iv[1] for iv in x_intervals[:k + 1])
        next_start = min(iv[0] for iv in x_intervals[k + 1:])
        gap = next_start - cur_end
        if gap >= min_split_gap and gap > max_split_gap:
            max_split_gap = gap
            best_split_x = (cur_end + next_start) // 2

    if best_split_x is not None:
        left_col = [l for l in cluster if l['x1'] <= best_split_x + 5]
        right_col = [l for l in cluster if l['x0'] >= best_split_x - 5]
        if left_col and right_col and (len(left_col) + len(right_col) == len(cluster)):
            return [left_col, right_col]
    return [cluster]


# ------------------------------------------------------------------------------
# 4. Figure-8 Double Lobed Bubble (Stage: figure8_clustering)
# ------------------------------------------------------------------------------
def _detect_figure8_double_lobed(cluster: List[Dict[str, Any]], context: Dict[str, Any]) -> bool:
    """Detects vertically stacked double lobes connected with horizontal constriction or sentence break."""
    if not isinstance(cluster, list) or len(cluster) < 4:
        return False
    sorted_lines = sorted(cluster, key=lambda l: l['y0'])
    total_h = max(l['y1'] for l in sorted_lines) - min(l['y0'] for l in sorted_lines)
    total_w = max(l['x1'] for l in sorted_lines) - min(l['x0'] for l in sorted_lines)

    # Check tall vertical ratio and sentence transition
    if total_h >= 1.1 * total_w and total_h >= 130:
        for k in range(len(sorted_lines) - 1):
            t_prev = sorted_lines[k].get('text', '').strip()
            t_next = sorted_lines[k + 1].get('text', '').strip()
            gap_y = sorted_lines[k + 1]['y0'] - sorted_lines[k]['y1']
            if (t_prev and t_prev[-1] in '.!?:…' or gap_y >= 8) and t_next and t_next[0].isupper() and gap_y >= 6:
                return True
            if gap_y >= 25 and k >= 1 and (len(sorted_lines) - (k + 1)) >= 1:
                return True

    # Check clear vertical gap
    for i in range(len(sorted_lines) - 1):
        v_gap = sorted_lines[i + 1]['y0'] - sorted_lines[i]['y1']
        if v_gap >= 28 and i >= 1 and (len(sorted_lines) - (i + 1)) >= 1:
            return True

    return False


def _resolve_figure8_double_lobed(cluster: List[Dict[str, Any]], context: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """Splits vertically stacked double lobes into Top and Bottom lobe clusters."""
    if not isinstance(cluster, list) or len(cluster) < 4:
        return [cluster]
    sorted_lines = sorted(cluster, key=lambda l: l['y0'])
    split_k = None

    total_h = max(l['y1'] for l in sorted_lines) - min(l['y0'] for l in sorted_lines)
    total_w = max(l['x1'] for l in sorted_lines) - min(l['x0'] for l in sorted_lines)

    if total_h >= 1.1 * total_w and total_h >= 130:
        for k in range(len(sorted_lines) - 1):
            t_prev = sorted_lines[k].get('text', '').strip()
            t_next = sorted_lines[k + 1].get('text', '').strip()
            gap_y = sorted_lines[k + 1]['y0'] - sorted_lines[k]['y1']
            if (t_prev and t_prev[-1] in '.!?:…' or gap_y >= 8) and t_next and t_next[0].isupper() and gap_y >= 6:
                split_k = k + 1
                break
            if gap_y >= 25 and k >= 1 and (len(sorted_lines) - (k + 1)) >= 1:
                split_k = k + 1
                break

    if split_k is None:
        max_v_gap = 0
        for i in range(len(sorted_lines) - 1):
            v_gap = sorted_lines[i + 1]['y0'] - sorted_lines[i]['y1']
            if v_gap > max_v_gap and i >= 1 and (len(sorted_lines) - (i + 1)) >= 1:
                max_v_gap = v_gap
                split_k = i + 1
        if max_v_gap < 28:
            split_k = None

    if split_k is not None:
        top_lobe = sorted_lines[:split_k]
        bot_lobe = sorted_lines[split_k:]
        if top_lobe and bot_lobe:
            return [top_lobe, bot_lobe]
    return [cluster]


# ------------------------------------------------------------------------------
# 5. Background Art Leak (Stage: contour_extraction)
# ------------------------------------------------------------------------------
def _detect_background_art_leak(contour_state: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """
    Detects flood-fill leaking past broken/docked bubble borders into open panel art:
    - Low solidity (< 0.80 on filled area > bbox_area * 1.5)
    - Scale-invariant ratio blowout (filled_area > effective_ratio_cap * bbox_area)
    - Misplaced text coverage (< 0.20)
    - Border touch with large ROI fill ratio (> roi_leak_ratio)
    """
    misplaced = contour_state.get("misplaced", False)
    low_solidity = contour_state.get("low_solidity", False)
    ratio_to_text = contour_state.get("ratio_to_text", 1.0)
    effective_ratio_cap = contour_state.get("effective_ratio_cap", 90.0)
    border_touch = contour_state.get("border_touch", False)
    ratio_to_roi = contour_state.get("ratio_to_roi", 0.0)
    roi_leak_ratio = contour_state.get("roi_leak_ratio", 0.75)

    return bool(
        misplaced or
        low_solidity or
        (ratio_to_text > effective_ratio_cap) or
        (border_touch and ratio_to_roi > roi_leak_ratio)
    )


def _resolve_background_art_leak(contour_state: Dict[str, Any], context: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
    """
    Applies synthetic ellipse fallback to contain the bubble within plausible text bounds
    without spilling across comic art, maintaining smooth organic oval curvature.
    """
    cx = contour_state.get("cx", 0)
    cy = contour_state.get("cy", 0)
    bw = max(10, contour_state.get("bw", 50))
    bh = max(10, contour_state.get("bh", 30))
    W = context.get("image_w", 3000)
    H = context.get("image_h", 4000)
    x0 = contour_state.get("x0", cx - bw // 2)
    y0 = contour_state.get("y0", cy - bh // 2)
    x1 = contour_state.get("x1", cx + bw // 2)
    y1 = contour_state.get("y1", cy + bh // 2)

    aspect = bw / max(1.0, float(bh))
    if aspect >= 3.0 or bh / max(1.0, float(bw)) >= 3.0:
        # Wide caption / headline / chyron banner: must be rectangular, never a curved ellipse!
        pad_x = 18
        pad_y = 12
        rx0 = max(0, x0 - pad_x)
        ry0 = max(0, y0 - pad_y)
        rx1 = min(W - 1, x1 + pad_x)
        ry1 = min(H - 1, y1 + pad_y)
        rect_pts = np.array([
            [rx0, ry0],
            [rx1, ry0],
            [rx1, ry1],
            [rx0, ry1]
        ], dtype=np.int32).reshape(-1, 1, 2)
        return rect_pts, True

    rx = min(int(bw * 0.75 + 16), max(20, (x1 - x0) // 2 + 25))
    ry = min(int(bh * 0.85 + 16), max(15, (y1 - y0) // 2 + 20))
    pts = cv2.ellipse2Poly((int(cx), int(cy)), (rx, ry), 0, 0, 360, 6)
    synth = pts.reshape(-1, 1, 2).astype(np.int32)
    synth[:, :, 0] = np.clip(synth[:, :, 0], 0, W - 1)
    synth[:, :, 1] = np.clip(synth[:, :, 1], 0, H - 1)
    return synth, False


# ------------------------------------------------------------------------------
# 6. Earpiece Rectangle Misclassification (Stage: bubble_inpainting)
# ------------------------------------------------------------------------------
def _detect_earpiece_rect_misclassification(bubble_info: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """
    Detects if a speech bubble (z_order == 0 or non-caption) is mistakenly flagged
    as an earpiece rectangle, which would cause solid white box whiteout and brown borders.
    """
    z = bubble_info.get("z_order", 0)
    shape = bubble_info.get("shape_type", "oval")
    return bool(z == 0 and shape == "rectangle")


def _resolve_earpiece_rect_misclassification(bubble_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Demotes misclassified speech bubble to 'oval' so it receives contour-only text inpainting."""
    bubble_info["shape_type"] = "oval"
    return bubble_info


# ------------------------------------------------------------------------------
# 7. Empty Fragment / Bubble (Stage: bubble_validation)
# ------------------------------------------------------------------------------
def _detect_empty_fragment(bubble: Any, context: Dict[str, Any]) -> bool:
    """Detects a bubble or fragment that has empty or whitespace-only dialogue text."""
    if isinstance(bubble, (list, tuple)):
        return False
    text = getattr(bubble, "original_text", "") if hasattr(bubble, "original_text") else (bubble.get("original_text", "") if isinstance(bubble, dict) else "")
    return not text or not str(text).strip()


def _resolve_empty_fragment(bubble: Any, context: Dict[str, Any]) -> None:
    """Drops empty bubble/fragment from the pipeline."""
    return None


# ------------------------------------------------------------------------------
# 8. Duplicate Bubble ID (Stage: bubble_list_indexing)
# ------------------------------------------------------------------------------
def _detect_duplicate_bubble_id(bubbles: Any, context: Dict[str, Any]) -> bool:
    """Detects non-sequential or duplicate bubble IDs in final bubble list."""
    if not isinstance(bubbles, list) or len(bubbles) == 0:
        return False
    ids = [getattr(b, "bubble_id", 0) if hasattr(b, "bubble_id") else (b.get("bubble_id", 0) if isinstance(b, dict) else 0) for b in bubbles]
    return len(ids) != len(set(ids)) or ids != list(range(1, len(bubbles) + 1))


def _resolve_duplicate_bubble_id(bubbles: List[Any], context: Dict[str, Any]) -> List[Any]:
    """Re-indexes all bubbles sequentially 1..N in panel-aware reading order."""
    import reading_order
    panel_boxes = context.get("panel_boxes") if context else None
    sorted_b = reading_order.order_bubbles_reading_order(bubbles, panel_boxes=panel_boxes)
    for idx, b in enumerate(sorted_b, start=1):
        if hasattr(b, "bubble_id"):
            b.bubble_id = idx
        elif isinstance(b, dict):
            b["bubble_id"] = idx
    return sorted_b


# ------------------------------------------------------------------------------
# 9. Sliver Too Small for Text (Stage: fragment_processing)
# ------------------------------------------------------------------------------
def _detect_sliver_too_small(fragment_info: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Detects occluded fragments whose area or width cannot fit assigned text."""
    area = fragment_info.get("area", 0)
    text = fragment_info.get("text", "")
    w = fragment_info.get("w", 0)
    h = fragment_info.get("h", 0)
    char_count = len(text.strip())
    return bool(char_count >= 8 and (area < 1200 or h < 15 or w < 25))


def _resolve_sliver_too_small(fragment_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Flags warning on the fragment so it is visible in inspector overlay."""
    fragment_info["quality_warning"] = "sliver_too_small_for_text"
    return fragment_info


# ------------------------------------------------------------------------------
# 10. Residual Ink Under Lettering (Stage: post_clean)
# ------------------------------------------------------------------------------
def _detect_residual_ink(clean_check_info: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """
    Detects un-inpainted dark ink clusters remaining inside the interior of a bubble
    that should have been cleaned before lettering.
    """
    residual_ink_count = clean_check_info.get("residual_ink_count", 0)
    return bool(residual_ink_count > 30)


def _resolve_residual_ink(clean_check_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Applies a secondary inpaint cleanup pass with increased radius on residual ink."""
    img_cv = clean_check_info.get("image_bgr")
    residual_mask = clean_check_info.get("residual_mask")
    if img_cv is not None and residual_mask is not None and np.any(residual_mask > 0):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(residual_mask, k, iterations=2)
        clean_check_info["image_bgr"] = cv2.inpaint(img_cv, dilated, 3, cv2.INPAINT_TELEA)
    return clean_check_info


# ------------------------------------------------------------------------------
# 11. Conjoined Bubble Constriction Split (Stage: bubble_segmentation)
# ------------------------------------------------------------------------------
def _detect_conjoined_bubble_constriction(bubble_state: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """
    Detects conjoined speech bubbles connected by a narrow waist/isthmus constriction
    where vertical or diagonal neck dips and text lines exist on both lobes.
    """
    constriction_ratio = bubble_state.get("constriction_ratio", 1.0)
    has_split_lines = bubble_state.get("has_split_lines", False)
    is_multi_lobe = bubble_state.get("is_multi_lobe", False)
    return bool((constriction_ratio <= 0.88 or is_multi_lobe) and has_split_lines)


def _resolve_conjoined_bubble_constriction(bubble_state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Applies partition to split conjoined cluster into Left/Right or multi-lobe bubble clusters."""
    if "split_clusters" not in bubble_state or not bubble_state["split_clusters"]:
        lines_left = bubble_state.get("lines_left", [])
        lines_right = bubble_state.get("lines_right", [])
        bubble_state["split_clusters"] = [lines_left, lines_right]
    return bubble_state


# ------------------------------------------------------------------------------
# 12. Caption Box Shape Destruction / Flattening (Stage: bubble_inpainting)
# ------------------------------------------------------------------------------
def _detect_caption_shape_destruction(bubble_info: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """
    Detects if a caption / earpiece bubble has non-rectangular features (wings, antenna,
    spikes, tails) or a non-rectangular contour, but is being treated as a blunt axis-aligned
    rectangle or scheduled for solid rectangular fill whiteout.
    """
    shape = bubble_info.get("shape_type", "oval")
    contour = bubble_info.get("contour")
    if contour is None:
        contour = bubble_info.get("contour_points")
    if contour is not None:
        try:
            cnt_arr = np.array(contour, dtype=np.int32)
            if cnt_arr.ndim == 2 and cnt_arr.shape[1] == 2 and len(cnt_arr) >= 3:
                cnt_arr = cnt_arr.reshape(-1, 1, 2)
            elif cnt_arr.ndim == 3 and cnt_arr.shape[2] == 2 and len(cnt_arr) >= 3:
                pass
            else:
                return False
            # Check if contour is NOT a simple rectangle
            area = cv2.contourArea(cnt_arr)
            x, y, w, h = cv2.boundingRect(cnt_arr)
            fill_ratio = area / max(1.0, float(w * h))
            peri = cv2.arcLength(cnt_arr, True)
            approx = cv2.approxPolyDP(cnt_arr, 0.02 * peri, True)
            is_rect = fill_ratio >= 0.94 and len(approx) <= 5
            if not is_rect and (shape == "rectangle" or bubble_info.get("is_rectangular_fill", False)):
                return True
        except Exception:
            pass
    return False


def _resolve_caption_shape_destruction(bubble_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Protects the authentic non-rectangular shape and disables blunt rectangular fill."""
    bubble_info["shape_type"] = "shaped_caption"
    bubble_info["is_rectangular_fill"] = False
    return bubble_info


# ------------------------------------------------------------------------------
# 13. Touching Bubbles Merged (Stage: bubble_segmentation)
# ------------------------------------------------------------------------------
_DEFAULT_SCISSORS_THRESHOLDS = {
    "constriction_ratio": 0.65,
    "min_waist_gap": 15.0,
    "confidence_boost": 0.0
}
_LEARNED_SCISSORS_THRESHOLDS = dict(_DEFAULT_SCISSORS_THRESHOLDS)


def get_learned_scissors_thresholds() -> Dict[str, float]:
    """Returns the current learned thresholds for scissors / touching bubble split."""
    return dict(_LEARNED_SCISSORS_THRESHOLDS)


def _detect_touching_bubbles_merged(bubble_state: Any, context: Dict[str, Any]) -> bool:
    """
    Detects adjacent merged speech bubbles along waist constriction or sentence boundary:
    - Constriction ratio <= learned threshold
    - Touching bubble flag or manual scissors correction indicator
    """
    if not isinstance(bubble_state, dict):
        return False
    c_ratio = bubble_state.get("constriction_ratio", 1.0)
    has_split = bubble_state.get("has_split_lines", False)
    thresh = _LEARNED_SCISSORS_THRESHOLDS["constriction_ratio"]
    if has_split and c_ratio <= thresh:
        return True
    return bool(bubble_state.get("is_touching_merged", False))


def _resolve_touching_bubbles_merged(bubble_state: Any, context: Dict[str, Any]) -> Any:
    """Splits touching bubbles into two separate lobe clusters."""
    if not isinstance(bubble_state, dict):
        return bubble_state
    lines_l = bubble_state.get("lines_left", [])
    lines_r = bubble_state.get("lines_right", [])
    if lines_l and lines_r:
        bubble_state["split_clusters"] = [lines_l, lines_r]
    return bubble_state


# ------------------------------------------------------------------------------
# 14. Clipped Bubble Edge / Missed Tail (Stage: contour_extraction)
# ------------------------------------------------------------------------------
_DEFAULT_BRUSH_TOLERANCES = {
    "color_tolerance": 35.0,
    "min_lum": 110.0,
    "min_margin": 12.0
}
_LEARNED_BRUSH_TOLERANCES = dict(_DEFAULT_BRUSH_TOLERANCES)


def get_learned_brush_tolerances() -> Dict[str, float]:
    """Returns current learned tolerances for smart brush expansion."""
    return dict(_LEARNED_BRUSH_TOLERANCES)


def reset_learned_thresholds():
    """Resets learned thresholds to their baseline default values."""
    global _LEARNED_SCISSORS_THRESHOLDS, _LEARNED_BRUSH_TOLERANCES
    _LEARNED_SCISSORS_THRESHOLDS = dict(_DEFAULT_SCISSORS_THRESHOLDS)
    _LEARNED_BRUSH_TOLERANCES = dict(_DEFAULT_BRUSH_TOLERANCES)


def _detect_clipped_bubble_edge(contour_state: Any, context: Dict[str, Any]) -> bool:
    """
    Detects if extracted bubble contour clipped authentic bubble regions (tails/edges):
    - Text bounding box comes within min_margin of contour boundary on one or more sides
    - Contour solidity has an unnatural indentation or flat edge near text
    """
    if not isinstance(contour_state, dict):
        return False
    margin = contour_state.get("text_contour_margin", 99.0)
    thresh_margin = _LEARNED_BRUSH_TOLERANCES["min_margin"]
    return bool(margin < thresh_margin or contour_state.get("is_clipped_edge", False))


def _resolve_clipped_bubble_edge(contour_state: Any, context: Dict[str, Any]) -> Any:
    """Applies learned tolerance parameters to expand bubble contour."""
    if not isinstance(contour_state, dict):
        return contour_state
    contour_state["suggested_tolerance"] = _LEARNED_BRUSH_TOLERANCES["color_tolerance"]
    contour_state["is_expanded"] = True
    return contour_state


# ------------------------------------------------------------------------------
# 15. Tinted / Spectral Bubble Preservation (Stage: bubble_inpainting)
# ------------------------------------------------------------------------------
def _detect_tinted_spectral_bubble(bubble_info: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """
    Detects monochromatic tinted speech bubbles or spectral dialogue ink (e.g. ghost voices)
    requiring authentic tint preservation rather than destructive flat whiteout inpainting.
    """
    if not isinstance(bubble_info, dict):
        return False
    is_tinted = bubble_info.get("is_tinted", False)
    bg_is_tinted = bubble_info.get("bg_is_tinted", False)
    text_is_spectral = bubble_info.get("text_is_spectral", False)
    sampled_tint = bubble_info.get("tint_bgr") or bubble_info.get("tint_rgb")
    return bool(is_tinted or bg_is_tinted or text_is_spectral or (sampled_tint is not None and any(c != 255 for c in sampled_tint)))


def _resolve_tinted_spectral_bubble(bubble_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures bubble is flagged for adaptive ink cleaning with Zero Whiteout tint preservation."""
    if isinstance(bubble_info, dict):
        bubble_info["is_tinted"] = True
        if "c_bg_protected" not in bubble_info:
            bubble_info["c_bg_protected"] = True
    return bubble_info


# ------------------------------------------------------------------------------
# 16. Occluded Crescent Text Tuck (Stage: lettering)
# ------------------------------------------------------------------------------
def _detect_occluded_crescent_tuck(lettering_state: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """
    Detects occluded bubbles resting directly underneath an earpiece/caption rectangle
    where the top line must tuck snuggly under the occluding rectangle's lower border.
    """
    if not isinstance(lettering_state, dict):
        return False
    is_occluded = lettering_state.get("is_occluded", False)
    has_top_rect = lettering_state.get("top_rect") is not None
    return bool(is_occluded and has_top_rect)


def _resolve_occluded_crescent_tuck(lettering_state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Sets y1_pos tuck offset to ensure dialogue line 1 enters cleanly under rectangle border."""
    if isinstance(lettering_state, dict):
        top_rect = lettering_state.get("top_rect")
        if top_rect:
            rect_y1 = top_rect.get("y1", top_rect.get("y", 0) + top_rect.get("h", 0))
            lettering_state["y1_pos"] = rect_y1 - 3
            lettering_state["tuck_applied"] = True
    return lettering_state


# ------------------------------------------------------------------------------
# 17. Translation Sentence Drop (Stage: translation_validation)
# ------------------------------------------------------------------------------
def _detect_translation_sentence_drop(state: Any, context: Dict[str, Any]) -> bool:
    """
    Detects dropped dialogue sentences during translation:
    Compares sentence count (split on .!?) between source and translated text.
    If source has >= 2 sentences and translation has fewer sentences, returns True.
    """
    import re
    if isinstance(state, dict):
        src = state.get("source_text") or state.get("orig_text") or state.get("original_text") or context.get("source_text") or ""
        tgt = state.get("translated_text") or state.get("uzbek_translation") or state.get("uz_text") or ""
    elif hasattr(state, "original_text") and hasattr(state, "uzbek_translation"):
        src = getattr(state, "original_text", "") or ""
        tgt = getattr(state, "uzbek_translation", "") or ""
    else:
        src = context.get("source_text") or context.get("orig_text") or ""
        tgt = str(state) if state else ""

    src_s = [s.strip() for s in re.split(r"[\.!\?…]+", str(src)) if re.search(r"\w", s)]
    tgt_s = [s.strip() for s in re.split(r"[\.!\?…]+", str(tgt)) if re.search(r"\w", s)]

    if len(src_s) >= 2 and len(tgt_s) < len(src_s):
        return True
    return False


def _resolve_translation_sentence_drop(state: Any, context: Dict[str, Any]) -> Any:
    """Flags bubble/translation for review due to dropped sentences."""
    if isinstance(state, dict):
        state["needs_review"] = True
        state["quality_warning"] = "translation_sentence_drop"
        state["review_reason"] = "Tarjimada gap tushib qolgan (translation_sentence_drop)"
    elif hasattr(state, "needs_review"):
        try:
            state.needs_review = True
            setattr(state, "quality_warning", "translation_sentence_drop")
            setattr(state, "review_reason", "Tarjimada gap tushib qolgan (translation_sentence_drop)")
        except Exception:
            pass
    return state


def _detect_translation_cross_check_mismatch(state: Any, context: Dict[str, Any]) -> bool:
    """Detects whether translation cross-check flagged a significant mismatch."""
    if isinstance(state, dict):
        return bool(state.get("cross_check_mismatch", False))
    return bool(getattr(state, "cross_check_mismatch", False))


def _resolve_translation_cross_check_mismatch(state: Any, context: Dict[str, Any]) -> Any:
    """Flags bubble for review due to translation cross-check mismatch; leaves NLLB output intact."""
    msg = context.get("mismatch_reason") or "Tarjima nomuvofiqligi: NLLB va Google Translate farqlanadi (tekshiruv tavsiya etiladi)"
    if isinstance(state, dict):
        state["needs_review"] = True
        state["quality_warning"] = "translation_cross_check_mismatch"
        state["review_reason"] = msg
    elif hasattr(state, "needs_review"):
        try:
            state.needs_review = True
            setattr(state, "quality_warning", "translation_cross_check_mismatch")
            setattr(state, "review_reason", msg)
        except Exception:
            pass
    return state


def tune_pattern_from_correction(pattern_name: str, event_record: Dict[str, Any]):
    """
    Dynamically tunes IssuePattern thresholds from recorded manual corrections.
    """
    global _LEARNED_SCISSORS_THRESHOLDS, _LEARNED_BRUSH_TOLERANCES
    if pattern_name == "touching_bubbles_merged":
        cur = _LEARNED_SCISSORS_THRESHOLDS["constriction_ratio"]
        _LEARNED_SCISSORS_THRESHOLDS["constriction_ratio"] = min(0.75, round(cur + 0.02, 3))
        _LEARNED_SCISSORS_THRESHOLDS["confidence_boost"] = round(_LEARNED_SCISSORS_THRESHOLDS["confidence_boost"] + 0.1, 2)
    elif pattern_name == "clipped_bubble_edge":
        cur_tol = _LEARNED_BRUSH_TOLERANCES["color_tolerance"]
        _LEARNED_BRUSH_TOLERANCES["color_tolerance"] = min(50.0, round(cur_tol + 1.0, 1))
        _LEARNED_BRUSH_TOLERANCES["min_margin"] = max(8.0, round(_LEARNED_BRUSH_TOLERANCES["min_margin"] - 0.5, 1))


# ==============================================================================
# QUALITY GATES REGISTRY
# ==============================================================================

QUALITY_GATES_REGISTRY: List[IssuePattern] = [
    IssuePattern(
        name="sfx_false_positive",
        stage="ocr_filtering",
        description="Non-speech stylized sound effect or background text on dark/colored art",
        severity="auto_fix",
        detect=_detect_sfx_false_positive,
        resolve=_resolve_sfx_false_positive
    ),
    IssuePattern(
        name="diamond_rule_line_split",
        stage="line_pairing",
        description="Center-wrapped speech lines split due to lack of horizontal bbox overlap",
        severity="auto_fix",
        detect=_detect_diamond_rule_line_split,
        resolve=_resolve_diamond_rule_line_split
    ),
    IssuePattern(
        name="adjacent_bubble_merging_cross_column",
        stage="line_clustering",
        description="Distinct adjacent speakers' speech balloons merged across separate columns",
        severity="auto_fix",
        detect=_detect_adjacent_bubble_merging_cross_column,
        resolve=_resolve_adjacent_bubble_merging_cross_column
    ),
    IssuePattern(
        name="figure8_double_lobed_bubble",
        stage="figure8_clustering",
        description="Vertically stacked double lobes erroneously merged into single bubble",
        severity="auto_fix",
        detect=_detect_figure8_double_lobed,
        resolve=_resolve_figure8_double_lobed
    ),
    IssuePattern(
        name="background_art_leak",
        stage="contour_extraction",
        description="Flood-fill leak escaping into open background art past broken or docked border",
        severity="auto_fix",
        detect=_detect_background_art_leak,
        resolve=_resolve_background_art_leak
    ),
    IssuePattern(
        name="earpiece_rect_misclassification",
        stage="bubble_inpainting",
        description="Speech bubble misclassified as earpiece rectangle, risking whiteout corners",
        severity="auto_fix",
        detect=_detect_earpiece_rect_misclassification,
        resolve=_resolve_earpiece_rect_misclassification
    ),
    IssuePattern(
        name="empty_fragment",
        stage="bubble_validation",
        description="Bubble fragment with zero or whitespace-only dialogue",
        severity="auto_fix",
        detect=_detect_empty_fragment,
        resolve=_resolve_empty_fragment
    ),
    IssuePattern(
        name="duplicate_bubble_id",
        stage="bubble_list_indexing",
        description="Duplicate or non-sequential bubble IDs in assembled bubble list",
        severity="auto_fix",
        detect=_detect_duplicate_bubble_id,
        resolve=_resolve_duplicate_bubble_id
    ),
    IssuePattern(
        name="sliver_too_small_for_text",
        stage="fragment_processing",
        description="Occluded sliver too small to fit assigned dialogue glyphs",
        severity="flag_for_review",
        detect=_detect_sliver_too_small,
        resolve=_resolve_sliver_too_small
    ),
    IssuePattern(
        name="residual_ink_under_lettering",
        stage="post_clean",
        description="Residual dark ink clusters remaining inside bubble interior",
        severity="auto_fix",
        detect=_detect_residual_ink,
        resolve=_resolve_residual_ink
    ),
    IssuePattern(
        name="conjoined_bubble_constriction_split",
        stage="bubble_segmentation",
        description="Conjoined speech bubbles with narrow waist constriction between lobes split at the neck",
        severity="auto_fix",
        detect=_detect_conjoined_bubble_constriction,
        resolve=_resolve_conjoined_bubble_constriction
    ),
    IssuePattern(
        name="caption_shape_destruction",
        stage="bubble_inpainting",
        description="Non-rectangular caption or communicator bubble mistakenly flattened to rectangle, risking artwork destruction",
        severity="auto_fix",
        detect=_detect_caption_shape_destruction,
        resolve=_resolve_caption_shape_destruction
    ),
    IssuePattern(
        name="touching_bubbles_merged",
        stage="bubble_review_edit",
        description="Touching adjacent speech bubbles erroneously merged without dividing seam",
        severity="auto_fix",
        detect=_detect_touching_bubbles_merged,
        resolve=_resolve_touching_bubbles_merged
    ),
    IssuePattern(
        name="clipped_bubble_edge",
        stage="bubble_review_edit",
        description="Bubble contour artificially clipped tight, missing organic tail or shoulder curve",
        severity="auto_fix",
        detect=_detect_clipped_bubble_edge,
        resolve=_resolve_clipped_bubble_edge
    ),
    IssuePattern(
        name="tinted_spectral_bubble_preservation",
        stage="bubble_inpainting",
        description="Monochromatic tinted speech bubble or spectral dialogue ink requiring authentic tint preservation and Zero Whiteout inpainting",
        severity="auto_fix",
        detect=_detect_tinted_spectral_bubble,
        resolve=_resolve_tinted_spectral_bubble
    ),
    IssuePattern(
        name="occluded_crescent_text_tuck",
        stage="lettering",
        description="Occluded crescent bubble under caption/earpiece rectangle requires top line to tuck directly beneath rectangle bottom boundary",
        severity="auto_fix",
        detect=_detect_occluded_crescent_tuck,
        resolve=_resolve_occluded_crescent_tuck
    ),
    IssuePattern(
        name="translation_sentence_drop",
        stage="translation_validation",
        description="Dialogue translation dropped one or more complete sentences compared to source dialogue",
        severity="flag_for_review",
        detect=_detect_translation_sentence_drop,
        resolve=_resolve_translation_sentence_drop
    ),
    IssuePattern(
        name="translation_cross_check_mismatch",
        stage="translation_validation",
        description="Substantial semantic or phrasing mismatch between local NLLB-200 and advisory Google Cloud Translation",
        severity="flag_for_review",
        detect=_detect_translation_cross_check_mismatch,
        resolve=_resolve_translation_cross_check_mismatch
    ),
]


# ==============================================================================
# PIPELINE INVOCATION INTERFACE
# ==============================================================================

def apply_quality_gates(stage: str, state: Any, context: Optional[Dict[str, Any]] = None,
                        bubble_id: Optional[int] = None) -> Any:
    """
    Executes all registered IssuePatterns for `stage` against `state`.
    If detect() matches:
      - Records a QualityGateEvent in the logger.
      - If severity is 'auto_fix', applies resolve() and updates state.
      - If severity is 'flag_for_review', marks warning and retains state.
    Returns the resolved/updated state.
    """
    ctx = context or {}
    for pattern in QUALITY_GATES_REGISTRY:
        if pattern.stage != stage:
            continue

        try:
            if pattern.detect(state, ctx):
                action = "auto_fix_applied" if pattern.severity == "auto_fix" else "flagged_for_review"
                log_quality_gate_event(
                    pattern_name=pattern.name,
                    stage=stage,
                    severity=pattern.severity,
                    action_taken=action,
                    details={"description": pattern.description},
                    bubble_id=bubble_id,
                    page=ctx.get("page")
                )

                if pattern.severity == "auto_fix":
                    state = pattern.resolve(state, ctx)
                else:
                    state = pattern.resolve(state, ctx)
        except Exception as e:
            # Quality gate execution should never crash the pipeline
            log_quality_gate_event(
                pattern_name=pattern.name,
                stage=stage,
                severity="error",
                action_taken="exception_in_gate",
                details={"error": str(e)},
                bubble_id=bubble_id,
                page=ctx.get("page")
            )

    return state


# ------------------------------------------------------------------------------
# 5. UNTRANSLATED ENGLISH RESIDUE & HYBRID DIALOGUE GATE
# ------------------------------------------------------------------------------
PERMITTED_UZBEK_TERMS = {
    "SPIDEY", "SPIDER-MAN", "OTTO", "OKTAVIUS", "OCTAVIUS", "PITER", "PARKER", "PETER",
    "JONA", "JEYMSON", "JAMESON", "MARLA", "GOLDMAN", "RUTH", "DOKTOR", "OKTOPUS", "DOC", "OCK",
    "EMPIRE", "STATE", "UNIVERSITETI", "BUGLE", "DAILY", "MIDTOWN", "DEKAN", "MER", "MAYOR",
    "XOTIRA", "FONDI", "BINOSI"
}

UNTRANSLATED_ENGLISH_TOKENS = {
    "THE", "AND", "OR", "BUT", "NOT", "WITH", "FOR", "FROM", "HAVE", "HAS", "BEEN",
    "CAN", "CANNOT", "WILL", "SHALL", "WOULD", "COULD", "SHOULD", "ABOUT", "OVER",
    "UNDER", "AFTER", "BEFORE", "THESE", "THOSE", "THIS", "THAT", "MAKES", "NICE",
    "FINALLY", "BODY", "BRAIN", "TOWN", "JOB", "HERE", "SOME", "MORE", "ANYMORE",
    "WHY", "WHAT", "WHERE", "WHEN", "WHO", "HOW", "DEAD", "LIVE", "ALIVE", "FEEL",
    "TAKE", "FLAT", "FINK", "BANANA", "PANTS", "HECKUVA", "FLAAT", "TOPPED", "FINKA",
    "CRAZY", "TOWN", "BANANA-PANTS", "BELIEVE", "YEARS", "ALL", "DOES", "OVER"
}


def get_all_permitted_terms() -> set:
    """Returns combined set of permitted comic proper nouns from glossary and hardcoded list."""
    terms = set(PERMITTED_UZBEK_TERMS)
    try:
        import terminology_store
        store = terminology_store.get_terminology_store()
        for t in store._entries.values():
            for word in re.findall(r"[A-Za-z\-']+", f"{t.get('target_term', '')} {t.get('source_term', '')}"):
                terms.add(word.strip("'-").upper())
    except Exception:
        pass
    return terms


def check_untranslated_english_residue(uz_text: str, en_orig: str = "") -> Tuple[bool, List[str]]:
    """
    Quality gate: Detects untranslated English residue or hybrid English-Uzbek suffixes in translated text.
    Returns (has_residue: bool, flagged_tokens: List[str]).
    """
    import re
    if not uz_text:
        return False, []

    permitted = get_all_permitted_terms()
    flagged = []

    # 1. Check for unmasked token placeholders (e.g. XNAME_..._X)
    placeholder_matches = re.findall(r"XNAME_[A-Z0-9_]+_X", uz_text, flags=re.IGNORECASE)
    if placeholder_matches:
        flagged.extend(placeholder_matches)

    # 2. Check individual words against known English leakage words & hybrid suffixes
    words = re.findall(r"[A-Za-z\-']+", uz_text)
    for w in words:
        w_clean = w.strip("'-").upper()
        if not w_clean or len(w_clean) < 2:
            continue
        if w_clean in permitted:
            continue

        # Direct English leak word
        if w_clean in UNTRANSLATED_ENGLISH_TOKENS:
            flagged.append(w)
            continue

        # Check for English root with Uzbek suffix (e.g. BRAIN-I -> BRAIN, BANANA-PANTS-HI -> BANANA-PANTS, FINK-A -> FINK)
        for sfx in ["NING", "DAN", "GA", "DA", "NI", "HI", "SI", "I", "A"]:
            if len(w_clean) > len(sfx) + 2 and w_clean.endswith(sfx):
                stem = w_clean[:-len(sfx)]
                if stem in UNTRANSLATED_ENGLISH_TOKENS:
                    flagged.append(w)
                    break

    # De-duplicate while preserving order
    unique_flagged = list(dict.fromkeys(flagged))
    return bool(unique_flagged), unique_flagged


def post_translation_validation(
    uz_text: str,
    orig_text: str,
    is_error: bool = False
) -> Dict[str, Any]:
    """
    Validates translated dialogue against comic translation failure criteria:
    1. Translation exception / error flag -> needs_review = True
    2. Empty or punctuation-only translation when source had words -> needs_review = True
    3. Exact match with raw OCR (untranslated leak) -> needs_review = True
    4. Untranslated English residue or hybrid suffixes (exempting permitted terms) -> needs_review = True
    5. Dropped sentence / clause truncation:
       If original text had 2 or more distinct sentences (marked by ! ? . ;) containing
       substantive clauses, but translation collapsed into a single short fragment where
       an entire sentence was dropped.
    """
    import re
    result = {
        "needs_review": False,
        "reason": None,
        "flagged_tokens": []
    }

    # Criterion 1: Translation exception / error flag
    if is_error:
        result["needs_review"] = True
        result["reason"] = "Tarjima xatoligi / oflayn NMT istisnosi yuz berdi"
        return result

    u_strip = (uz_text or "").strip()
    o_strip = (orig_text or "").strip()

    # Criterion 2: Empty or punctuation-only translation when source had words
    has_words_orig = bool(re.search(r"[A-Za-z0-9]", o_strip))
    has_words_uz = bool(re.search(r"[A-Za-z0-9]", u_strip))
    if has_words_orig and not has_words_uz:
        result["needs_review"] = True
        result["reason"] = "Bo'sh yoki matnsiz tarjima"
        return result

    # Criterion 3: Exact match with raw OCR (untranslated leak)
    clean_uz = re.sub(r"[\W_]+", "", u_strip).upper()
    clean_orig = re.sub(r"[\W_]+", "", o_strip).upper()
    if clean_uz and clean_orig and clean_uz == clean_orig:
        result["needs_review"] = True
        result["reason"] = "Tarjima qilinmagan xom OCR (asl matn bilan bir xil)"
        return result

    # Criterion 4: Untranslated English residue / leakage
    has_residue, flagged_tokens = check_untranslated_english_residue(u_strip, en_orig=o_strip)
    if has_residue:
        result["needs_review"] = True
        result["reason"] = f"Inglizcha qoldiq so'zlar aniqlandi: {', '.join(flagged_tokens)}"
        result["flagged_tokens"] = flagged_tokens
        return result

    # Criterion 5: Truncated translation / dropped sentence
    # If source had 2 or more sentences, but translation collapsed into fewer sentences
    orig_sentences = [s.strip() for s in re.split(r"[\.!\?…]+", o_strip) if re.search(r"\w", s)]
    uz_sentences = [s.strip() for s in re.split(r"[\.!\?…]+", u_strip) if re.search(r"\w", s)]
    if len(orig_sentences) >= 2 and len(uz_sentences) < len(orig_sentences):
        result["needs_review"] = True
        result["reason"] = f"Qisman kesilgan tarjima / Tarjimada gap tushib qolgan (originalda {len(orig_sentences)} ta gap, tarjimada {len(uz_sentences)} ta)"
        return result

    # Criterion 6: Advisory Cloud Cross-Check (Google Translate)
    # Strictly inert if GOOGLE_TRANSLATE_API_KEY is not configured
    try:
        import cloud_enhancements
        if cloud_enhancements.is_translate_cross_check_enabled():
            check = cloud_enhancements.google_translate_cross_check(o_strip, u_strip)
            if check and check.get("is_mismatch"):
                result["needs_review"] = True
                result["reason"] = (
                    f"Tarjima nomuvofiqligi: NLLB va Google Translate farqlanadi "
                    f"({check.get('similarity', 0.0):.2f}): '{check.get('google_translation')}'"
                )
                result["cross_check"] = check
                log_quality_gate_event(
                    pattern_name="translation_cross_check_mismatch",
                    stage="translation_validation",
                    severity="flag_for_review",
                    action_taken="flagged_mismatch_for_human_review",
                    details=check
                )
                return result
    except Exception:
        pass

    return result


