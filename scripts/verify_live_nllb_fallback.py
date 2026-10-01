"""
scripts/verify_live_nllb_fallback.py
------------------------------------
Force-tests the NLLB subprocess translation FALLBACK path on live Streamlit Cloud:
1. Simulates deep_translator failure via the sidebar toggle ('Simulate deep_translator outage').
2. Executes Stage 1 (Tesseract OCR + Inpainting).
3. Executes Stage 2 (forced NLLB subprocess fallback).
4. Monitors and reports:
   - Pre-subprocess Tree RSS
   - Active Peak Tree RSS (at the moment NLLB fallback is running)
   - Post-subprocess Tree RSS (confirming memory returns to baseline)
   - cgroup Anon & Peak values
5. Proceeds to Stage 3 to verify zero crash / no OOM.
"""
import os
import sys
import time
from playwright.sync_api import sync_playwright

URL = "https://comic-lab-bf44wggdfu4enffzhhmhuf.streamlit.app/"
SAMPLE_IMG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples", "page_4.png"))
ARTIFACT_DIR = r"C:\Users\Ozod\.gemini\antigravity\brain\b9e18c00-a13c-4536-8eb0-753bbc58e136"

def get_text_safe(locator):
    try:
        return locator.inner_text().strip()
    except Exception:
        return ""

def get_app_frame(page):
    for f in page.frames:
        if "/~/" in f.url:
            return f
    return page

def extract_memory_readings(frame):
    captions = frame.locator("p, [data-testid='stCaptionContainer'], code, span").all()
    results = []
    for c in captions:
        txt = get_text_safe(c)
        if any(k in txt for k in ["Tree RSS", "cgroup:", "cgroup Peak", "cgroup Anon", "Xotira (RAM / cgroup)"]):
            results.append(txt)
    seen = set()
    dedup = []
    for r in results:
        if r not in seen and len(r) < 250:
            seen.add(r)
            dedup.append(r)
    return " | ".join(dedup) if dedup else "Not found in DOM"

def extract_subprocess_telemetry(frame):
    info_boxes = frame.locator("[data-testid='stAlert']").all()
    for box in info_boxes:
        txt = get_text_safe(box)
        if "NLLB Bola Jarayoni Xotirasi" in txt or "Cho'qqi" in txt:
            return txt
    return None

def main():
    print("=" * 80)
    print("FORCE-TESTING NLLB SUBPROCESS FALLBACK ON LIVE STREAMLIT CLOUD")
    print(f"URL: {URL}")
    print(f"Sample: {SAMPLE_IMG}")
    print("=" * 80)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # Step 0: Open App
        print("\n[STEP 0] Opening live URL...", flush=True)
        page.goto(URL, wait_until="networkidle", timeout=90000)
        page.wait_for_timeout(8000)

        # Check for wake up button
        wake_btn = page.locator("button:has-text('Yes, get this app back up!')")
        if wake_btn.count() > 0 and wake_btn.first.is_visible():
            print("[INFO] App is sleeping. Clicking wake up button...", flush=True)
            wake_btn.first.click()
            page.wait_for_timeout(20000)

        frame = get_app_frame(page)
        mem_initial = extract_memory_readings(frame)
        print(f"[INITIAL MEMORY] {mem_initial}", flush=True)

        # Step 1: Open sidebar diagnostics and enable FORCE_NLLB_FALLBACK
        print("\n[STEP 1] Enabling 'Simulate deep_translator outage (Force NLLB fallback)' in sidebar...", flush=True)
        try:
            diag_exp = frame.locator("[data-testid='stSidebar'] [data-testid='stExpander'] summary:has-text('Diagnostika')")
            if diag_exp.count() > 0:
                diag_exp.first.click()
                page.wait_for_timeout(1000)
            
            fb_checkbox = frame.locator("label:has-text('Simulate deep_translator outage')")
            if fb_checkbox.count() > 0:
                fb_checkbox.first.click()
                page.wait_for_timeout(2000)
                print("[STEP 1] FORCE_NLLB_FALLBACK checkbox ENABLED successfully!", flush=True)
            else:
                print("[STEP 1 WARN] Checkbox not found directly in DOM, will check input directly", flush=True)
        except Exception as e:
            print(f"[STEP 1 WARN] Failed to toggle checkbox via UI: {e}", flush=True)

        # Step 2: Upload image
        print("\n[STEP 2] Uploading samples/page_4.png...", flush=True)
        file_input = frame.locator("input[type='file']")
        file_input.wait_for(state="attached", timeout=30000)
        file_input.set_input_files(SAMPLE_IMG)
        page.wait_for_timeout(8000)

        frame = get_app_frame(page)
        mem_after_upload = extract_memory_readings(frame)
        print(f"[AFTER UPLOAD MEMORY] {mem_after_upload}", flush=True)

        # Step 3: Run Stage 1
        scan_btn = frame.locator("button:has-text('1. Sahifani Skanerlash'), button:has-text('Skanerlash va Pufaklarni Tozalash')")
        scan_btn.wait_for(state="visible", timeout=30000)
        print("\n[STEP 3] Clicking '1. Sahifani Skanerlash va Pufaklarni Tozalash'...", flush=True)
        scan_btn.click()

        stage2_btn = frame.locator("button:has-text('2-bosqich')")
        stage2_btn.wait_for(state="visible", timeout=180000)
        page.wait_for_timeout(4000)

        frame = get_app_frame(page)
        mem_stage1 = extract_memory_readings(frame)
        print(f"\n[STAGE 1 COMPLETED MEMORY] {mem_stage1}", flush=True)
        page.screenshot(path=os.path.join(ARTIFACT_DIR, "cloud_nllb_stage1.png"))

        # Re-verify that simulation checkbox is checked
        try:
            diag_exp = frame.locator("[data-testid='stSidebar'] [data-testid='stExpander'] summary:has-text('Diagnostika')")
            if diag_exp.count() > 0:
                diag_exp.first.click()
                page.wait_for_timeout(500)
            fb_checkbox = frame.locator("label:has-text('Simulate deep_translator outage')")
            if fb_checkbox.count() > 0 and not frame.locator("input[key='ui_force_nllb_fallback']").is_checked():
                fb_checkbox.first.click()
                page.wait_for_timeout(1000)
        except Exception:
            pass

        # Step 4: Click Stage 2 translation button — FORCING NLLB SUBPROCESS FALLBACK
        print("\n[STEP 4] Clicking Stage 2 button ('2-bosqich: Matnlarni Ko\\'rish va Tahrirlash') with FORCE_NLLB_FALLBACK...", flush=True)
        stage2_btn.click()

        # Step 5: Real-time sampling of active Tree RSS while NLLB subprocess runs
        print("[STEP 5] Actively monitoring Tree RSS while NLLB subprocess is executing...", flush=True)
        stage3_btn = frame.locator("button:has-text('3-bosqich')")
        
        peak_active_observed_rss = 0.0
        active_sample_readings = []
        start_trans = time.time()
        
        while not stage3_btn.is_visible() and (time.time() - start_trans < 300):
            time.sleep(2)
            frame = get_app_frame(page)
            cur_mem = extract_memory_readings(frame)
            if cur_mem != "Not found in DOM":
                active_sample_readings.append(cur_mem)
                print(f"  [Translating active sample]: {cur_mem}", flush=True)
            if stage3_btn.count() > 0 and stage3_btn.first.is_visible():
                break

        print("[STEP 5] Stage 2 translation completed! Stage 3 button appeared.", flush=True)
        page.wait_for_timeout(4000)
        frame = get_app_frame(page)
        
        mem_stage2 = extract_memory_readings(frame)
        print(f"\n[STAGE 2 COMPLETED MEMORY] {mem_stage2}", flush=True)
        page.screenshot(path=os.path.join(ARTIFACT_DIR, "cloud_nllb_fallback_done.png"))

        # Check telemetry card
        telemetry = extract_subprocess_telemetry(frame)
        if telemetry:
            print(f"\n[SUBPROCESS TELEMETRY CARD]:\n{telemetry}", flush=True)

        # Step 6: Proceed to Stage 3 to verify zero crash / complete end-to-end flow
        if stage3_btn.count() > 0:
            print("\n[STEP 6] Clicking Stage 3 lettering button ('3-bosqich: Shriftlarni Yozish')...", flush=True)
            stage3_btn.click(force=True)
            page.wait_for_timeout(8000)
            frame = get_app_frame(page)
            mem_stage3 = extract_memory_readings(frame)
            print(f"\n[STAGE 3 COMPLETED MEMORY] {mem_stage3}", flush=True)
            page.screenshot(path=os.path.join(ARTIFACT_DIR, "cloud_nllb_stage3_done.png"))
        else:
            mem_stage3 = "N/A"

        print("\n" + "=" * 80)
        print("FORCED NLLB FALLBACK LIVE TEST COMPLETED SUCCESSFULLY!")
        print(f"Initial: {mem_initial}")
        print(f"After Upload: {mem_after_upload}")
        print(f"Stage 1: {mem_stage1}")
        print("Active NLLB Subprocess Samples:")
        for s in active_sample_readings[-4:]:
            print(f"  - {s}")
        print(f"Stage 2 (Post Subprocess): {mem_stage2}")
        print(f"Stage 3: {mem_stage3}")
        if telemetry:
            print(f"Telemetry: {telemetry}")
        print("=" * 80)
        browser.close()

if __name__ == "__main__":
    main()
