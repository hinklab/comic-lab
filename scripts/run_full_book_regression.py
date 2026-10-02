"""
scripts/run_full_book_regression.py - Full-book regression runner across all 24 pages.
Executes Stage 1 (OCR + Telea clean), Stage 2 (Translation), Stage 3 (Typeset lettering).
Records per-page quality gates, OCR & translation path telemetry, review queue flags,
and novel failures.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Any, Optional
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import engine
import quality_gates

BOOK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "full_book"))
OUTPUT_LETTERED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "full_book", "lettered"))
OUTPUT_CLEANED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "full_book", "cleaned"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scratch", "full_book_results"))

os.makedirs(OUTPUT_LETTERED_DIR, exist_ok=True)
os.makedirs(OUTPUT_CLEANED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def process_single_page(page_filename: str) -> Dict[str, Any]:
    page_path = os.path.join(BOOK_DIR, page_filename)
    page_base = os.path.splitext(page_filename)[0]
    
    if not os.path.exists(page_path):
        return {
            "page_filename": page_filename,
            "status": "FAIL",
            "error": f"File not found: {page_path}"
        }

    im = Image.open(page_path)
    w, h = im.size
    t_start = time.time()

    record: Dict[str, Any] = {
        "page_filename": page_filename,
        "page_base": page_base,
        "image_size": [w, h],
        "bubble_count": 0,
        "ocr_path": "UNKNOWN",
        "ocr_reason": "",
        "ocr_avg_conf": 0.0,
        "ocr_duration_s": 0.0,
        "clean_duration_s": 0.0,
        "translation_path": "UNKNOWN",
        "translation_reason": "",
        "translation_duration_s": 0.0,
        "lettering_duration_s": 0.0,
        "total_duration_s": 0.0,
        "quality_gates_auto_fix": [],
        "quality_gates_flagged": [],
        "novel_issues": [],
        "lettered_output_path": "",
        "status": "PASS"
    }

    with quality_gates.quality_gate_context(page_base, clear_existing=True):
        # -------------------------------------------------------------
        # STAGE 1: OCR (Tesseract Primary -> EasyOCR Subprocess Fallback)
        # -------------------------------------------------------------
        t0 = time.time()
        tess_bubbles = []
        tess_err = None
        try:
            tess_bubbles = engine.scan_bubbles_ocr_tesseract(im)
        except Exception as e:
            tess_err = str(e)

        avg_conf = sum(b.confidence for b in tess_bubbles) / len(tess_bubbles) if tess_bubbles else 0.0
        record["ocr_avg_conf"] = round(avg_conf, 3)

        if tess_bubbles and avg_conf >= 0.35:
            record["ocr_path"] = "Tesseract"
            record["ocr_reason"] = f"Primary OCR success ({len(tess_bubbles)} bubbles, conf={avg_conf:.1%})"
            bubbles = tess_bubbles
        else:
            record["ocr_path"] = "EasyOCR (fallback)"
            if tess_err:
                record["ocr_reason"] = f"Tesseract error ({tess_err})"
            elif not tess_bubbles:
                record["ocr_reason"] = "Tesseract returned 0 bubbles"
            else:
                record["ocr_reason"] = f"Tesseract conf {avg_conf:.1%} < 35% threshold"

            try:
                bubbles = engine.scan_bubbles_ocr_subprocess(im)
            except Exception as fb_err:
                record["novel_issues"].append(f"EasyOCR fallback failure: {fb_err}")
                bubbles = tess_bubbles  # use whatever we got if subprocess fails

        record["ocr_duration_s"] = round(time.time() - t0, 2)
        record["bubble_count"] = len(bubbles)

        # -------------------------------------------------------------
        # STAGE 1: Telea Inpainting (Cleaning)
        # -------------------------------------------------------------
        t_clean = time.time()
        try:
            if bubbles:
                cleaned = engine.clean_page_ink_telea(im, bubbles)
            else:
                cleaned = im.copy()
            cleaned_path = os.path.join(OUTPUT_CLEANED_DIR, f"{page_base}_cleaned.png")
            cleaned.save(cleaned_path, format="PNG")
        except Exception as cl_err:
            record["novel_issues"].append(f"Inpainting clean exception: {cl_err}")
            cleaned = im.copy()
        record["clean_duration_s"] = round(time.time() - t_clean, 2)

        # -------------------------------------------------------------
        # STAGE 2: Translation (deep_translator -> NLLB-200 Fallback)
        # -------------------------------------------------------------
        t_trans = time.time()
        translated_bubbles = []
        if bubbles:
            # Check deep_translator availability
            deep_ok = False
            deep_reason = ""
            try:
                from deep_translator import GoogleTranslator
                test_tr = GoogleTranslator(source="en", target="uz").translate("TEST")
                if test_tr:
                    deep_ok = True
                    deep_reason = "GoogleTranslator active"
            except Exception as d_err:
                deep_ok = False
                deep_reason = f"{type(d_err).__name__}: {str(d_err)[:80]}"

            if deep_ok:
                record["translation_path"] = "deep_translator"
                record["translation_reason"] = deep_reason
            else:
                record["translation_path"] = "NLLB-200 (fallback)"
                record["translation_reason"] = f"deep_translator unavailable ({deep_reason})"

            try:
                # Use _translate_bubbles_list_core to translate all bubbles
                translated_bubbles = engine._translate_bubbles_list_core(bubbles, page_name=page_base)
            except Exception as tr_err:
                record["novel_issues"].append(f"Translation pipeline exception: {tr_err}")
                translated_bubbles = bubbles
        record["translation_duration_s"] = round(time.time() - t_trans, 2)

        # -------------------------------------------------------------
        # STAGE 3: Lettering & Typesetting
        # -------------------------------------------------------------
        t_let = time.time()
        lettered_img = None
        try:
            fonts = engine.get_available_fonts()
            f_path = fonts.get("CC Wild Words (Recommended)", next(iter(fonts.values())))
            
            lettered_img = engine.typeset_lettering_on_page(
                cleaned_image=cleaned,
                bubbles=translated_bubbles,
                font_path=f_path,
                max_font_size=26,
                min_font_size=10,
                line_gap=6,
                original_image=im
            )
            out_img_path = os.path.join(OUTPUT_LETTERED_DIR, f"{page_base}_lettered.png")
            lettered_img.save(out_img_path, format="PNG")
            record["lettered_output_path"] = out_img_path
        except Exception as let_err:
            record["novel_issues"].append(f"Lettering typeset exception: {let_err}")
            record["status"] = "FAIL"
        record["lettering_duration_s"] = round(time.time() - t_let, 2)

        # -------------------------------------------------------------
        # Collect Quality Gate Events
        # -------------------------------------------------------------
        events = quality_gates.get_quality_gate_log()
        for ev in events:
            ev_summary = {
                "pattern_name": ev.get("pattern_name"),
                "stage": ev.get("stage"),
                "action_taken": ev.get("action_taken"),
                "bubble_id": ev.get("bubble_id"),
                "details": ev.get("details", {})
            }
            if ev.get("severity") == "auto_fix":
                record["quality_gates_auto_fix"].append(ev_summary)
            else:
                record["quality_gates_flagged"].append(ev_summary)

        # Check for novel anomalies not matching registered patterns
        for b in translated_bubbles:
            # Check for NaN / negative coordinates
            if b.x0 < 0 or b.y0 < 0 or b.x1 > w + 50 or b.y1 > h + 50:
                record["novel_issues"].append(f"Bubble #{b.bubble_id} out of bounds: ({b.x0},{b.y0},{b.x1},{b.y1})")
            # Check for empty translation when original had text
            if b.original_text.strip() and not b.uzbek_translation.strip():
                record["novel_issues"].append(f"Bubble #{b.bubble_id} has empty translation for '{b.original_text[:30]}'")

    record["total_duration_s"] = round(time.time() - t_start, 2)
    return record


def run_batch(start_page: int, end_page: int) -> List[Dict[str, Any]]:
    print(f"\n================================================================================")
    print(f"STARTING REGRESSION BATCH: Pages {start_page:02d} to {end_page:02d}")
    print(f"================================================================================\n")

    batch_records = []
    for p_idx in range(start_page, end_page + 1):
        filename = f"page_{p_idx:02d}.png"
        print(f"--> Processing {filename} ({p_idx}/{end_page})...", flush=True)
        rec = process_single_page(filename)
        batch_records.append(rec)

        # Save individual page record
        page_json = os.path.join(RESULTS_DIR, f"{rec.get('page_base', filename)}_result.json")
        with open(page_json, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)

        # Quick console report
        print(
            f"    [DONE {filename}] Bubbles: {rec['bubble_count']} | "
            f"OCR: {rec['ocr_path']} | Trans: {rec['translation_path']} | "
            f"AutoFix: {len(rec['quality_gates_auto_fix'])} | "
            f"Flagged: {len(rec['quality_gates_flagged'])} | "
            f"Time: {rec['total_duration_s']}s | Status: {rec['status']}",
            flush=True
        )

    # Save batch summary
    batch_json = os.path.join(RESULTS_DIR, f"batch_{start_page:02d}_to_{end_page:02d}_results.json")
    with open(batch_json, "w", encoding="utf-8") as f:
        json.dump(batch_records, f, indent=2, ensure_ascii=False)

    return batch_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full book regression pass runner")
    parser.add_argument("--start-page", type=int, default=1, help="Starting page number (1-24)")
    parser.add_argument("--end-page", type=int, default=24, help="Ending page number (1-24)")
    args = parser.parse_args()

    results = run_batch(args.start_page, args.end_page)
    print(f"\nFinished batch pages {args.start_page} to {args.end_page}. Processed {len(results)} pages.")
