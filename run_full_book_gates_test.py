"""
run_full_book_gates_test.py
Executes the pipeline across all 5 sample pages with quality_gate_context:
1. C:/Users/Ozod/Desktop/3.png
2. samples/page_18.png
3. samples/page_19.png
4. samples/page_20.png
5. samples/restaurant_page.png

Collects and formats all QualityGateEvents into a structured report.
"""

import os
import sys
import json
import time
from PIL import Image
import engine
import quality_gates

TEST_PAGES = [
    ("3.png", "C:/Users/Ozod/Desktop/3.png"),
    ("page_18.png", "samples/page_18.png"),
    ("page_19.png", "samples/page_19.png"),
    ("page_20.png", "samples/page_20.png"),
    ("restaurant_page.png", "samples/restaurant_page.png"),
]

def run_multi_page_test():
    quality_gates.clear_quality_gate_log()
    print(f"Starting multi-page quality gate run across {len(TEST_PAGES)} pages...\n")

    page_results = {}

    for page_name, page_path in TEST_PAGES:
        if not os.path.exists(page_path):
            print(f"[-] Warning: {page_path} not found, skipping.")
            continue

        print(f"[+] Processing {page_name} ({page_path})...")
        t0 = time.time()
        start_event_idx = len(quality_gates.get_quality_gate_log())

        with quality_gates.quality_gate_context(page=page_name, clear_existing=False):
            try:
                img = Image.open(page_path).convert("RGB")
                # Step 1: Scan and segment bubbles (OCR filtering, line pairing, clustering, validation, indexing)
                bubbles = engine.scan_bubbles_ocr(img)
                num_bubbles = len(bubbles)

                # Step 2: Clean page ink (bubble inpainting demotion, contour extraction, post-clean)
                cleaned = engine.clean_page_ink_telea(img, bubbles)

                dur = time.time() - t0
                all_events = quality_gates.get_quality_gate_log()
                page_events = all_events[start_event_idx:]

                page_results[page_name] = {
                    "num_bubbles": num_bubbles,
                    "duration_sec": round(dur, 2),
                    "events_count": len(page_events),
                    "events": page_events
                }
                print(f"    Done in {dur:.2f}s: found {num_bubbles} bubbles, {len(page_events)} quality gate events fired.")
            except Exception as e:
                print(f"    [!] Error processing {page_name}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*80)
    print("QUALITY GATE FIRING SUMMARY REPORT")
    print("="*80)

    total_events = quality_gates.get_quality_gate_log()
    print(f"Total Quality Gate Events Recorded: {len(total_events)}\n")

    pattern_counts = {}
    for ev in total_events:
        pat = ev["pattern_name"]
        pattern_counts[pat] = pattern_counts.get(pat, 0) + 1

    print("Gate Firings by Pattern:")
    for pat, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  - {pat:40s}: {cnt} times")

    print("\nDetailed Gate Firings Table:")
    print("| Page | Bubble ID | Stage | Pattern Name | Severity | Action Taken | Details |")
    print("|---|---|---|---|---|---|---|")
    for ev in total_events:
        pg = ev.get("page") or "-"
        bid = ev.get("bubble_id")
        bid_str = str(bid) if bid is not None else "-"
        stg = ev.get("stage") or "-"
        pat = ev.get("pattern_name") or "-"
        sev = ev.get("severity") or "-"
        act = ev.get("action_taken") or "-"
        desc = ev.get("details", {}).get("description", "")
        print(f"| {pg} | {bid_str} | {stg} | {pat} | {sev} | {act} | {desc} |")

    # Save to JSON log for record
    os.makedirs("scratch", exist_ok=True)
    with open("scratch/quality_gate_report.json", "w") as f:
        json.dump(total_events, f, indent=2)
    print(f"\nSaved structured report to scratch/quality_gate_report.json")

if __name__ == "__main__":
    run_multi_page_test()
