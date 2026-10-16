import cv2
import numpy as np
import math
from typing import Optional, Dict, Any, Tuple, List

def detect_bubble_tail(
    contour: Any,
    box: Optional[List[int]] = None,
    shape_type: str = "oval"
) -> Dict[str, Any]:
    """
    Robust Comic Speech Bubble Tail Detector & Vectorizer.
    Analyzes bubble contour geometry to determine:
    - has_tail (bool)
    - tail_tip (x, y)
    - tail_base (x1, y1), (x2, y2)
    - tail_vector [dx, dy] normalized
    - tail_angle_deg (-180 to 180)
    - tail_length (pixels)
    - tail_direction_label ('bottom', 'bottom-left', 'left', etc., or 'none')
    """
    result = {
        "has_tail": False,
        "tail_tip": None,
        "tail_base": None,
        "tail_vector": None,
        "tail_angle_deg": None,
        "tail_length": 0.0,
        "tail_direction": "none"
    }

    if contour is None or len(contour) < 6:
        return result

    cnt = np.array(contour, dtype=np.int32)
    if cnt.ndim == 2:
        cnt = cnt.reshape((-1, 1, 2))

    # Caption boxes / rectangles generally have no tail
    if shape_type == "rectangle":
        # Check if rect_ratio is extremely high (> 0.92) - pure caption
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        if rw * rh > 0 and (cv2.contourArea(cnt) / (rw * rh)) > 0.92:
            return result

    # 1. Centroid and Bounding Metrics
    M = cv2.moments(cnt)
    if M["m00"] <= 0:
        return result
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    rx, ry, rw, rh = cv2.boundingRect(cnt)
    min_dim = min(rw, rh)
    min_tail_len = max(10.0, min_dim * 0.12)

    # 2. Convex Hull & Convexity Defects
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return result

    try:
        defects = cv2.convexityDefects(cnt, hull_idx)
    except Exception:
        defects = None

    candidates = []

    # Method A: Analyze convexity defect valleys that flank the tail
    if defects is not None and len(defects) >= 2:
        # Collect significant defects
        sig_defects = []
        for i in range(len(defects)):
            row = defects[i].flatten()
            s, e, f, d = int(row[0]), int(row[1]), int(row[2]), float(row[3])
            depth = d / 256.0
            if depth >= 5.0:  # at least 5px deep concavity
                sig_defects.append({
                    "start": s,
                    "end": e,
                    "far": f,
                    "depth": depth,
                    "pt": cnt[f, 0]
                })

        # Pair adjacent or nearby concavities that bound a tail protrusion
        N_pts = len(cnt)
        for i in range(len(sig_defects)):
            for j in range(i + 1, len(sig_defects)):
                d1 = sig_defects[i]
                d2 = sig_defects[j]
                
                # Contour index span between the two defects
                idx1 = d1["far"]
                idx2 = d2["far"]
                
                # Check shorter path along contour
                span_fwd = (idx2 - idx1) % N_pts
                span_bwd = (idx1 - idx2) % N_pts
                
                for span, start_idx in [(span_fwd, idx1), (span_bwd, idx2)]:
                    # A tail is narrow: span along contour should be between 3 and N_pts // 2
                    if 3 <= span <= N_pts * 0.40:
                        # Extract points on this contour arc
                        arc_indices = [(start_idx + k) % N_pts for k in range(span + 1)]
                        arc_pts = cnt[arc_indices, 0]
                        
                        base_pt1 = cnt[start_idx, 0]
                        base_pt2 = cnt[(start_idx + span) % N_pts, 0]
                        base_mid = (base_pt1 + base_pt2) / 2.0
                        base_width = float(np.linalg.norm(base_pt2 - base_pt1))
                        
                        if base_width > rw * 0.60:
                            continue  # Base too wide to be a tail
                            
                        # Find the point on arc furthest from base_mid
                        dists_to_base = np.linalg.norm(arc_pts - base_mid, axis=1)
                        max_k = np.argmax(dists_to_base)
                        tail_len = dists_to_base[max_k]
                        tip_pt = arc_pts[max_k]
                        
                        # Distance from centroid
                        dist_to_center = float(np.linalg.norm(tip_pt - np.array([cx, cy])))
                        
                        # Tail must protrude outward from base
                        if tail_len >= min_tail_len and (tail_len / max(1.0, base_width)) >= 0.35:
                            score = tail_len * 1.5 + (d1["depth"] + d2["depth"])
                            candidates.append({
                                "tip": tip_pt,
                                "base1": base_pt1,
                                "base2": base_pt2,
                                "base_mid": base_mid,
                                "length": float(tail_len),
                                "base_width": base_width,
                                "score": score
                            })

    # Method B: Radial outlier protrusion (Elliptical distance profile)
    # Fit ellipse to contour points excluding the outer 10% outliers
    pts_2d = cnt[:, 0].astype(np.float32)
    # Distances from centroid
    radial_dists = np.linalg.norm(pts_2d - np.array([cx, cy]), axis=1)
    q75 = np.percentile(radial_dists, 75)
    iqr = q75 - np.percentile(radial_dists, 25)
    outlier_thresh = q75 + 1.2 * iqr

    outlier_mask = radial_dists > outlier_thresh
    if np.any(outlier_mask):
        outlier_indices = np.where(outlier_mask)[0]
        # Cluster consecutive outlier indices
        clusters = []
        cur_clust = [outlier_indices[0]]
        for k in range(1, len(outlier_indices)):
            if outlier_indices[k] == (outlier_indices[k-1] + 1) % len(cnt):
                cur_clust.append(outlier_indices[k])
            else:
                clusters.append(cur_clust)
                cur_clust = [outlier_indices[k]]
        if cur_clust:
            clusters.append(cur_clust)

        for clust in clusters:
            if len(clust) >= 2:
                clust_pts = cnt[clust, 0]
                dists_c = np.linalg.norm(clust_pts - np.array([cx, cy]), axis=1)
                max_k = np.argmax(dists_c)
                tip_pt = clust_pts[max_k]
                tip_dist = dists_c[max_k]
                
                # Approximate base
                b1 = cnt[(clust[0] - 1) % len(cnt), 0]
                b2 = cnt[(clust[-1] + 1) % len(cnt), 0]
                base_mid = (b1 + b2) / 2.0
                tail_len = float(np.linalg.norm(tip_pt - base_mid))
                base_w = float(np.linalg.norm(b2 - b1))
                
                if tail_len >= min_tail_len:
                    candidates.append({
                        "tip": tip_pt,
                        "base1": b1,
                        "base2": b2,
                        "base_mid": base_mid,
                        "length": tail_len,
                        "base_width": base_w,
                        "score": tail_len * 2.0
                    })

    if not candidates:
        return result

    # Pick highest scoring candidate
    best = max(candidates, key=lambda c: c["score"])

    tip = best["tip"]
    base_mid = best["base_mid"]
    
    # Calculate directional vector from bubble centroid through tail tip
    # (Pointing outward towards speaker)
    v_cx = float(tip[0] - cx)
    v_cy = float(tip[1] - cy)
    v_len = math.hypot(v_cx, v_cy)
    
    if v_len <= 1e-4:
        return result

    ux = v_cx / v_len
    uy = v_cy / v_len
    angle_rad = math.atan2(uy, ux)
    angle_deg = math.degrees(angle_rad)  # -180 to 180

    # Direction categorization
    # 0 deg = right, 90 deg = bottom, 180/-180 = left, -90 = top
    if 67.5 <= angle_deg < 112.5:
        dir_label = "bottom"
    elif 112.5 <= angle_deg < 157.5:
        dir_label = "bottom-left"
    elif angle_deg >= 157.5 or angle_deg < -157.5:
        dir_label = "left"
    elif -157.5 <= angle_deg < -112.5:
        dir_label = "top-left"
    elif -112.5 <= angle_deg < -67.5:
        dir_label = "top"
    elif -67.5 <= angle_deg < -22.5:
        dir_label = "top-right"
    elif -22.5 <= angle_deg < 22.5:
        dir_label = "right"
    else:  # 22.5 <= angle_deg < 67.5
        dir_label = "bottom-right"

    result["has_tail"] = True
    result["tail_tip"] = [int(tip[0]), int(tip[1])]
    result["tail_base"] = [
        [int(best["base1"][0]), int(best["base1"][1])],
        [int(best["base2"][0]), int(best["base2"][1])]
    ]
    result["tail_vector"] = [round(ux, 4), round(uy, 4)]
    result["tail_angle_deg"] = round(angle_deg, 1)
    result["tail_length"] = round(best["length"], 1)
    result["tail_direction"] = dir_label
    result["is_off_panel"] = False

    return result


def is_tail_off_panel(
    tail_tip: Optional[List[int]],
    tail_vector: Optional[List[float]],
    panel_box: Optional[List[int]] = None,
    page_size: Optional[Tuple[int, int]] = None,
    proximity_thresh: int = 45
) -> bool:
    """
    Detects whether a tail represents an Off-Panel / Off-Screen speaker voice.
    In comics, when a character behind the camera or outside the frame speaks,
    their bubble's tail points directly toward or touches the panel border.
    """
    if not tail_tip or not tail_vector:
        return False

    tx, ty = tail_tip[0], tail_tip[1]
    ux, uy = tail_vector[0], tail_vector[1]

    if panel_box and len(panel_box) >= 4:
        px0, py0, px1, py1 = panel_box[:4]
        d_left = abs(tx - px0)
        d_right = abs(px1 - tx)
        d_top = abs(ty - py0)
        d_bottom = abs(py1 - ty)

        # Check proximity to each border combined with outward direction
        if (d_right <= proximity_thresh and ux > 0.35) or (tx >= px1 - 10 and ux >= 0):
            return True
        if (d_left <= proximity_thresh and ux < -0.35) or (tx <= px0 + 10 and ux <= 0):
            return True
        if (d_top <= proximity_thresh and uy < -0.35) or (ty <= py0 + 10 and uy <= 0):
            return True
        if (d_bottom <= proximity_thresh and uy > 0.35) or (ty >= py1 - 10 and uy >= 0):
            return True

    if page_size and len(page_size) >= 2:
        pw, ph = page_size[:2]
        if (tx >= pw - proximity_thresh and ux > 0.35) or (tx <= proximity_thresh and ux < -0.35):
            return True
        if (ty >= ph - proximity_thresh and uy > 0.35) or (ty <= proximity_thresh and uy < -0.35):
            return True

    return False


def project_tail_to_speaker_region(
    tail_tip: List[int],
    tail_vector: List[float],
    panel_box: Optional[List[int]] = None,
    page_size: Optional[Tuple[int, int]] = None,
    distance_px: int = 180
) -> Tuple[int, int]:
    """
    Projects the tail directional vector outward from the tail tip to locate the
    speaker's probable center of mass / face anchor within the comic panel.
    """
    tx, ty = tail_tip[0], tail_tip[1]
    ux, uy = tail_vector[0], tail_vector[1]

    proj_x = int(tx + ux * distance_px)
    proj_y = int(ty + uy * distance_px)

    if panel_box and len(panel_box) >= 4:
        px0, py0, px1, py1 = panel_box[:4]
        proj_x = max(px0 + 5, min(px1 - 5, proj_x))
        proj_y = max(py0 + 5, min(py1 - 5, proj_y))
    elif page_size and len(page_size) >= 2:
        pw, ph = page_size[:2]
        proj_x = max(5, min(pw - 5, proj_x))
        proj_y = max(5, min(ph - 5, proj_y))

    return (proj_x, proj_y)


def annotate_bubble_with_tail(
    b: Any,
    panel_box: Optional[List[int]] = None,
    page_size: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Computes tail metrics and assigns fields directly to a SpeechBubble instance or dict.
    """
    cnt = getattr(b, "contour", None) if not isinstance(b, dict) else b.get("contour")
    box = getattr(b, "box", None) if not isinstance(b, dict) else b.get("box")
    shape_type = getattr(b, "shape_type", "oval") if not isinstance(b, dict) else b.get("shape_type", "oval")

    tail = detect_bubble_tail(cnt, box, shape_type)

    pbox = panel_box or (getattr(b, "panel_box", None) if not isinstance(b, dict) else b.get("panel_box"))
    if tail.get("has_tail"):
        tail["is_off_panel"] = is_tail_off_panel(
            tail.get("tail_tip"), tail.get("tail_vector"), panel_box=pbox, page_size=page_size
        )
    else:
        tail["is_off_panel"] = False

    if isinstance(b, dict):
        b.update(tail)
    else:
        for k, v in tail.items():
            if hasattr(b, k):
                setattr(b, k, v)
    return tail


if __name__ == "__main__":
    import engine
    from PIL import Image

    img = Image.open("samples/sample_comic.png")
    bubbles = engine.scan_bubbles_ocr(img)
    print(f"Testing tail detection on {len(bubbles)} bubbles:")
    for b in bubbles:
        tail = annotate_bubble_with_tail(b)
        print(f"Bubble #{b.bubble_id} [{b.shape_type}]: has_tail={tail['has_tail']}, dir={tail['tail_direction']}, angle={tail['tail_angle_deg']} deg, tip={tail['tail_tip']}, len={tail['tail_length']}px")
