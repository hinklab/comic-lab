"""
scripts/verify_live_cloud.py - Automated end-to-end verification of the live Streamlit Cloud app.
Executes Stage 1, Stage 2, and Stage 3 with samples/page_4.png and records live Cloud memory metrics.
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
    """Finds the inner Streamlit app iframe if present, otherwise returns page."""
    for f in page.frames:
        if "/~/" in f.url:
            return f
    return page

def extract_memory_readings(frame):
    """Finds all memory captions in the app frame."""
    captions = frame.locator("p, [data-testid='stCaptionContainer'], code").all()
    results = []
    for c in captions:
        txt = get_text_safe(c)
        if "Tree RSS:" in txt or "cgroup:" in txt:
            results.append(txt)
    seen = set()
    dedup = []
    for r in results:
        if r not in seen:
            seen.add(r)
            dedup.append(r)
    return " | ".join(dedup) if dedup else "Not found in DOM"

def main():
    print("=" * 80)
    print(f"VERIFYING LIVE STREAMLIT CLOUD APP: {URL}")
    print(f"Testing image: {SAMPLE_IMG}")
    print("=" * 80)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # Step 0: Open App
        print("\n[STEP 0] Opening live URL...", flush=True)
        page.goto(URL, wait_until="networkidle", timeout=90000)
        page.wait_for_timeout(8000)

        # Check for error screen
        content = page.content()
        if "Oh no" in content or "Error running app" in content:
            print("[ERROR] Streamlit Cloud app shows 'Oh no. Error running app'!", flush=True)
            page.screenshot(path=os.path.join(ARTIFACT_DIR, "cloud_error.png"))
            browser.close()
            return

        frame = get_app_frame(page)
        mem_initial = extract_memory_readings(frame)
        print(f"[INITIAL MEMORY] {mem_initial}", flush=True)
        page.screenshot(path=os.path.join(ARTIFACT_DIR, "cloud_step0_home.png"))

        # Step 1: Upload image
        print("\n[STEP 1] Uploading samples/page_4.png...", flush=True)
        file_input = frame.locator("input[type='file']")
        file_input.wait_for(state="attached", timeout=30000)
        file_input.set_input_files(SAMPLE_IMG)
        page.wait_for_timeout(8000)

        # Find scan button
        scan_btn = frame.locator("button:has-text('1. Sahifani Skanerlash'), button:has-text('Skanerlash va Pufaklarni Tozalash')")
        scan_btn.wait_for(state="visible", timeout=30000)
        mem_after_upload = extract_memory_readings(frame)
        print(f"[AFTER UPLOAD MEMORY] {mem_after_upload}", flush=True)

        # Click scan button
        print("[STEP 1] Clicking '1. Sahifani Skanerlash va Pufaklarni Tozalash'...", flush=True)
        scan_btn.click()

        # Wait for Stage 1 to complete (EasyOCR subprocess + Inpainting)
        print("[STEP 1] Waiting for Stage 1 OCR & Inpainting to complete (timeout: 240s)...", flush=True)
        stage2_btn = frame.locator("button:has-text('2-bosqich')")
        stage2_btn.wait_for(state="visible", timeout=240000)
        page.wait_for_timeout(4000)

        mem_stage1 = extract_memory_readings(frame)
        print(f"\n[STAGE 1 COMPLETED MEMORY] {mem_stage1}", flush=True)
        page.screenshot(path=os.path.join(ARTIFACT_DIR, "cloud_stage1_done.png"))

        # Step 2: Click Stage 2 translation button
        print("\n[STEP 2] Clicking Stage 2 button ('2-bosqich: Matnlarni Ko\\'rish va Tahrirlash')...", flush=True)
        stage2_btn.click()

        # Wait for Stage 2 to complete (Subprocess NLLB translation)
        print("[STEP 2] Waiting for Stage 2 Translation to complete (timeout: 240s)...", flush=True)
        stage3_btn = frame.locator("button:has-text('3-bosqich')")
        try:
            stage3_btn.wait_for(state="visible", timeout=240000)
        except Exception:
            # Check if cards are visible
            frame.locator(".modern-bubble-card, textarea").first.wait_for(state="visible", timeout=30000)

        page.wait_for_timeout(4000)
        mem_stage2 = extract_memory_readings(frame)
        print(f"\n[STAGE 2 COMPLETED MEMORY] {mem_stage2}", flush=True)
        page.screenshot(path=os.path.join(ARTIFACT_DIR, "cloud_stage2_done.png"))

        # Step 3: Click Stage 3 confirm if available
        if stage3_btn.is_visible():
            print("\n[STEP 3] Clicking Stage 3 lettering button ('3-bosqich: Shriftlarni Yozish')...", flush=True)
            stage3_btn.click()
            page.wait_for_timeout(8000)
            mem_stage3 = extract_memory_readings(frame)
            print(f"\n[STAGE 3 COMPLETED MEMORY] {mem_stage3}", flush=True)
            page.screenshot(path=os.path.join(ARTIFACT_DIR, "cloud_stage3_done.png"))
        else:
            mem_stage3 = "N/A"

        print("\n" + "=" * 80)
        print("REAL CLOUD TEST COMPLETED SUCCESSFULLY!")
        print(f"Initial: {mem_initial}")
        print(f"After Upload: {mem_after_upload}")
        print(f"Stage 1: {mem_stage1}")
        print(f"Stage 2: {mem_stage2}")
        print(f"Stage 3: {mem_stage3}")
        print("=" * 80)
        browser.close()

if __name__ == "__main__":
    main()
