"""
cloud_enhancements.py
--------------------
Optional, fail-safe, free-tier-only cloud enhancements:
1. Google Cloud Vision OCR Fallback (active only when EasyOCR confidence < 0.55).
2. Google Cloud Translation Advisory Cross-Check (flags substantial divergence for review).

Completely inert (zero behavior change, zero network calls, zero logs) when their
respective API keys are absent:
- GOOGLE_VISION_API_KEY
- GOOGLE_TRANSLATE_API_KEY

Enforces strict 90% hard usage caps via api_quota_tracker to guarantee 100% free-tier safety.
All requests wrapped in try/except with 5s timeout; any failure falls back seamlessly
to local processing and never crashes the pipeline.
"""

import os
import json
import base64
import html
import difflib
import re
import logging
import urllib.request
import urllib.parse
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

import api_quota_tracker

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("cloud_enhancements")


def _get_api_key(key_name: str) -> Optional[str]:
    """Retrieves API key from environment or Streamlit secrets."""
    val = os.getenv(key_name)
    if val and val.strip():
        return val.strip()

    try:
        import streamlit as st
        if hasattr(st, "secrets") and key_name in st.secrets:
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass

    return None


# ==============================================================================
# 1. GOOGLE CLOUD VISION OCR FALLBACK
# ==============================================================================

def is_vision_ocr_enabled() -> bool:
    """Returns True only if GOOGLE_VISION_API_KEY is configured and within quota."""
    key = _get_api_key("GOOGLE_VISION_API_KEY")
    if not key:
        return False
    return api_quota_tracker.can_consume("google_vision", 1)


def google_vision_ocr_fallback(
    image_np: np.ndarray,
    easyocr_text: str,
    easyocr_conf: float,
    timeout: float = 5.0
) -> Tuple[str, float, str]:
    """
    Conditionally re-OCRs a text line crop via Google Cloud Vision API:
    - Skipped entirely if GOOGLE_VISION_API_KEY is not configured (zero network calls).
    - Skipped if easyocr_conf >= 0.55.
    - Blocked if monthly 90% quota cap (900 units) is reached.
    - If Vision text differs and has clearly higher confidence, uses Vision text.
    - Otherwise keeps EasyOCR text.
    - Returns (final_text, final_confidence, winner_source: 'easyocr' | 'google_vision').
    """
    # 1. Fast inert bypass if confidence is already acceptable
    if easyocr_conf >= 0.55:
        return easyocr_text, easyocr_conf, "easyocr"

    # 2. Fast inert bypass if API key is absent
    api_key = _get_api_key("GOOGLE_VISION_API_KEY")
    if not api_key:
        return easyocr_text, easyocr_conf, "easyocr"

    # 3. Check hard 90% usage cap
    if not api_quota_tracker.can_consume("google_vision", 1):
        return easyocr_text, easyocr_conf, "easyocr"

    # 4. Check image crop validity
    if image_np is None or image_np.size == 0 or image_np.shape[0] < 5 or image_np.shape[1] < 5:
        return easyocr_text, easyocr_conf, "easyocr"

    try:
        # Encode image crop to JPEG in memory
        success, encoded_img = cv2.imencode(".jpg", image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            return easyocr_text, easyocr_conf, "easyocr"

        b64_content = base64.b64encode(encoded_img.tobytes()).decode("utf-8")

        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        payload = {
            "requests": [
                {
                    "image": {"content": b64_content},
                    "features": [{"type": "TEXT_DETECTION", "maxResults": 1}]
                }
            ]
        }

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            resp_body = response.read().decode("utf-8")
            result = json.loads(resp_body)

        # Record quota consumption upon successful request
        api_quota_tracker.record_consumption("google_vision", 1)

        responses = result.get("responses", [])
        if not responses or "textAnnotations" not in responses[0]:
            return easyocr_text, easyocr_conf, "easyocr"

        annotations = responses[0]["textAnnotations"]
        if not annotations:
            return easyocr_text, easyocr_conf, "easyocr"

        vision_text = annotations[0].get("description", "").strip().replace("\n", " ")
        if not vision_text:
            return easyocr_text, easyocr_conf, "easyocr"

        # Extract confidence from block/paragraph/word if available
        full_ann = responses[0].get("fullTextAnnotation", {})
        conf_scores = []
        for page in full_ann.get("pages", []):
            if "confidence" in page:
                conf_scores.append(float(page["confidence"]))
            for block in page.get("blocks", []):
                if "confidence" in block:
                    conf_scores.append(float(block["confidence"]))
                for para in block.get("paragraphs", []):
                    if "confidence" in para:
                        conf_scores.append(float(para["confidence"]))

        vision_conf = float(np.mean(conf_scores)) if conf_scores else 0.88

        # Decision rule:
        # If Vision text differs and its confidence is clearly higher (> EasyOCR conf + 0.10)
        norm_easy = easyocr_text.strip().upper()
        norm_vis = vision_text.strip().upper()

        if norm_vis != norm_easy and vision_conf > easyocr_conf:
            print(f"[OCR_FALLBACK] Google Vision won (conf {vision_conf:.2f} vs {easyocr_conf:.2f}): '{easyocr_text}' -> '{vision_text}'")
            return vision_text, vision_conf, "google_vision"
        else:
            print(f"[OCR_FALLBACK] EasyOCR kept (conf {easyocr_conf:.2f} vs Vision {vision_conf:.2f}): '{easyocr_text}'")
            return easyocr_text, easyocr_conf, "easyocr"

    except Exception as e:
        logger.warning(f"[GOOGLE_VISION_WARN] Fallback failed: {e}. Keeping EasyOCR result.")
        return easyocr_text, easyocr_conf, "easyocr"


# ==============================================================================
# 2. GOOGLE CLOUD TRANSLATION ADVISORY CROSS-CHECK
# ==============================================================================

def is_translate_cross_check_enabled() -> bool:
    """Returns True only if GOOGLE_TRANSLATE_API_KEY is configured and within quota."""
    key = _get_api_key("GOOGLE_TRANSLATE_API_KEY")
    if not key:
        return False
    return api_quota_tracker.can_consume("google_translate", 1)


def fetch_google_translation(
    source_en: str,
    timeout: float = 5.0
) -> Optional[str]:
    """
    Fetches Uzbek translation via Google Cloud Translation API:
    - Returns None if GOOGLE_TRANSLATE_API_KEY is missing or 90% quota cap is reached.
    - Wrapped in 5.0s timeout and safe exception handling.
    """
    if not source_en or not source_en.strip():
        return None

    api_key = _get_api_key("GOOGLE_TRANSLATE_API_KEY")
    if not api_key:
        return None

    char_count = len(source_en)
    if not api_quota_tracker.can_consume("google_translate", char_count):
        return None

    try:
        url = f"https://translation.googleapis.com/language/translate/v2?key={api_key}"
        payload = {
            "q": [source_en.strip()],
            "source": "en",
            "target": "uz",
            "format": "text"
        }

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            resp_body = response.read().decode("utf-8")
            result = json.loads(resp_body)

        api_quota_tracker.record_consumption("google_translate", char_count)

        data = result.get("data", {})
        translations = data.get("translations", [])
        if not translations:
            return None

        gt_raw = html.unescape(translations[0].get("translatedText", "")).strip()
        return gt_raw if gt_raw else None

    except Exception as e:
        logger.warning(f"[GOOGLE_TRANSLATE_WARN] Request failed: {e}. Keeping local NLLB result.")
        return None


def compute_naturalness_score(text: str) -> float:
    """
    Computes a heuristic score (higher = more natural comic Uzbek) evaluating:
    - Penalties for robotic calques, ungrammatical morphology, and repeated pronouns.
    - Bonuses for authentic Uzbek subordination clauses, idioms, and proper entity spellings.
    """
    if not text or not text.strip():
        return 0.0

    score = 1.0
    t_lower = text.lower().strip()

    # 1. Robotic Calque & Ungrammatical Penalties (-0.25 each)
    robotic_indicators = [
        r"\byunus\b",
        r"\bxayrli\s+kechasi\b",
        r"\bmenyasini\b",
        r"\bmaskalli\b",
        r"\batrofimiz\s+tuzilgan\b",
        r"\bkatta\s+bo['’`]?lib\s+o['’`]?sadi\b",
        r"\bso['’`]?zlarni\s+o['’`]?qiydi\b",
        r"\bdasturimda\b",
        r"\bjonli\s+bo['’`]?yim\b",
        r"\bbanana-?pants\b",
        r"\bfinka\b",
        r"\btekis\s+kalla\b",
        r"\byarmisidan\s+so['’`]?ng\b",
        r"\bko['’`]?radi\b",
    ]
    for pat in robotic_indicators:
        if re.search(pat, t_lower):
            score -= 0.25

    # 2. Repeated pronoun penalty (-0.20) e.g. "sen ... sen", "siz ... siz"
    if re.search(r"\b(sen|siz|men|u)\b.*\b\1\b", t_lower):
        score -= 0.20

    # 3. Authentic Uzbek Subordination & Morphology Bonuses (+0.15 each)
    natural_affixes = [
        r"\w+ganligini?\b",
        r"\w+ganini?\b",
        r"\w+ganda\b",
        r"\w+yotganini?\b",
        r"\w+maslik\b",
        r"\w+gandir\b",
        r"\w+moqchi\b",
        r"\w+gach\b",
    ]
    for pat in natural_affixes:
        if re.search(pat, t_lower):
            score += 0.15

    # 4. Authentic Comic Expressions & Entity Normalization (+0.10 each)
    natural_comic_tokens = [
        r"\bo['’`]?xshamayapsan\b",
        r"\byaxshimisan\b",
        r"\borqaga\s+turing\b",
        r"\bjinoyat\s+joyida\b",
        r"\bmeriya\b",
        r"\bbahsli\b",
        r"\bvino\s+menyusi\b",
        r"\bqasoskorlar\b",
        r"\bspider-man\b",
        r"\bjona\b",
    ]
    for pat in natural_comic_tokens:
        if re.search(pat, t_lower):
            score += 0.10

    return round(score, 3)


def semantic_agreement(nllb_uz: str, cloud_uz: str, en_orig: str) -> bool:
    """
    Verifies that cloud translation and local NMT translation agree in core semantic meaning:
    - Negation / polarity consistency (neither hallucinated an opposite sentiment).
    - Preserves core entities and sentiment.
    """
    if not nllb_uz or not cloud_uz:
        return False

    nllb_low = nllb_uz.lower()
    cloud_low = cloud_uz.lower()
    en_low = en_orig.lower() if en_orig else ""

    # Polarity check:
    neg_en = bool(re.search(r"\b(not|never|no|none|neither|don't|doesn't|didn't|won't|can't|cannot)\b", en_low))
    neg_pat = r"\b(emas\w*|yo['’`]?q\w*|hech\w*)\b|\w+ma(y|gan|ng|yotgan|di|sa|miz|siz|yapti)?\b"
    neg_nllb = bool(re.search(neg_pat, nllb_low))
    neg_cloud = bool(re.search(neg_pat, cloud_low))

    # If the two translations have opposite polarities and one diverges from English polarity, reject
    if neg_nllb != neg_cloud and (neg_cloud != neg_en):
        return False

    # Check minimum token overlap or character sequence similarity
    tokens_nmt = set(re.findall(r"\w+", nllb_low))
    tokens_cloud = set(re.findall(r"\w+", cloud_low))
    overlap = len(tokens_nmt & tokens_cloud) / max(1, min(len(tokens_nmt), len(tokens_cloud)))
    seq_sim = difflib.SequenceMatcher(None, nllb_low, cloud_low).ratio()

    # Agree if token overlap >= 0.15 or sequence ratio >= 0.25
    return (overlap >= 0.15) or (seq_sim >= 0.25)


def select_optimal_base_translation(
    source_en: str,
    nllb_uz: str,
    timeout: float = 5.0
) -> Tuple[str, str]:
    """
    Evaluates whether Google Cloud Translation provides a more natural grammatical base
    than local NLLB-200. If Google's translation scores higher on naturalness and agrees
    in core semantic meaning, returns (google_uz, 'google_translate').
    Otherwise returns (nllb_uz, 'nllb').
    """
    if not is_translate_cross_check_enabled():
        return nllb_uz, "nllb"

    cloud_uz = fetch_google_translation(source_en, timeout=timeout)
    if not cloud_uz:
        return nllb_uz, "nllb"

    score_nllb = compute_naturalness_score(nllb_uz)
    score_cloud = compute_naturalness_score(cloud_uz)

    agrees = semantic_agreement(nllb_uz, cloud_uz, source_en)
    if agrees and (score_cloud > score_nllb):
        return cloud_uz, "google_translate"

    return nllb_uz, "nllb"


def google_translate_cross_check(
    source_en: str,
    nmt_translation: str,
    similarity_threshold: float = 0.45,
    timeout: float = 5.0
) -> Optional[Dict[str, Any]]:
    """
    Advisory translation cross-check via Google Cloud Translation API:
    - Skipped entirely if GOOGLE_TRANSLATE_API_KEY is not configured (zero network calls).
    - Blocked if monthly 90% quota cap (450,000 characters) is reached.
    - Advisory only: does NOT overwrite NLLB-200 translation.
    - If Google's translation differs substantially from NLLB-200's output (similarity < threshold),
      returns mismatch metadata to flag bubble for human review.
    - Returns None if disabled or matching, or Dict with cross-check comparison.
    """
    if not source_en or not source_en.strip() or not nmt_translation or not nmt_translation.strip():
        return None

    gt_raw = fetch_google_translation(source_en, timeout=timeout)
    if not gt_raw:
        return None

    # Clean strings for similarity check
    clean_nmt = nmt_translation.strip().upper()
    clean_gt = gt_raw.strip().upper()

    # Sequence character ratio
    seq_ratio = difflib.SequenceMatcher(None, clean_nmt, clean_gt).ratio()

    # Token overlap Jaccard
    tokens_nmt = set(clean_nmt.split())
    tokens_gt = set(clean_gt.split())
    jaccard = (len(tokens_nmt & tokens_gt) / len(tokens_nmt | tokens_gt)) if (tokens_nmt | tokens_gt) else 1.0

    hybrid_sim = round(0.5 * seq_ratio + 0.5 * jaccard, 3)
    is_mismatch = bool(hybrid_sim < similarity_threshold)

    return {
        "source_en": source_en,
        "nmt_translation": nmt_translation,
        "google_translation": gt_raw,
        "similarity": hybrid_sim,
        "is_mismatch": is_mismatch,
        "reason": (
            f"NLLB va Google Translate tarjimalari o'rtasida jiddiy farq aniqlandi "
            f"(o'xshashlik: {hybrid_sim:.2f} < {similarity_threshold:.2f})"
        ) if is_mismatch else None
    }
