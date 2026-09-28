"""
scripts/test_subprocess_integration.py
Full integration verification for scan_bubbles_ocr(use_subprocess=True):
1. Measures execution time and bubble count on samples/page_4.png (target: 14 bubbles).
2. Verifies zero orphan Python processes remain after execution.
3. Tests timeout enforcement and forceful process tree killing.
"""
import os
import sys
import time
import subprocess
import psutil
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import engine

def get_current_python_pids():
    pids = set()
    for p in psutil.process_iter(['pid', 'name']):
        try:
            if 'python' in p.info['name'].lower():
                pids.add(p.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return pids

def main():
    print("=== 1. SUBPROCESS INTEGRATSIYASINI SINASH ===", flush=True)
    sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples", "page_4.png"))
    im = Image.open(sample_path)

    pids_before = get_current_python_pids()
    print(f"Testdan oldingi Python jarayonlari soni: {len(pids_before)} ta", flush=True)

    t0 = time.perf_counter()
    bubbles = engine.scan_bubbles_ocr(im, use_subprocess=True, timeout_seconds=120)
    elapsed = time.perf_counter() - t0

    pids_after = get_current_python_pids()
    leftover_pids = pids_after - pids_before

    print(f"Topilgan pufaklar soni: {len(bubbles)} ta (Kutilgan: 14 ta)", flush=True)
    print(f"Ketgan vaqt: {elapsed:.2f} soniya", flush=True)
    print(f"Testdan keyingi Python jarayonlari soni: {len(pids_after)} ta", flush=True)
    print(f"Qolib ketgan yetim jarayonlar (Orphan PIDs): {leftover_pids if leftover_pids else 'HECH QANDAY (0 ta)'}", flush=True)

    assert len(bubbles) == 14, f"Xato: 14 ta pufak kutilgan edi, lekin {len(bubbles)} ta topildi!"
    assert len(leftover_pids) == 0, f"Xato: Yetim jarayonlar qolib ketdi: {leftover_pids}"

    print("\n=== 2. TIMEOUT VA FORCE-KILL HIMOYaSINI SINASH ===", flush=True)
    print("Sun'iy 2 soniyalik timeout bilan osilib qolish holati tekshirilmoqda...", flush=True)

    timeout_caught = False
    pids_before_to = get_current_python_pids()

    try:
        # Run with a 1-second timeout to trigger timeout protection
        engine.scan_bubbles_ocr(im, use_subprocess=True, timeout_seconds=1)
    except TimeoutError as ex:
        timeout_caught = True
        print(f"Muvaffaqiyatli: TimeoutError ushlandi! Xabar: '{ex}'", flush=True)
    except Exception as ex:
        print(f"Kutilmagan xatolik turi: {type(ex)}: {ex}", flush=True)

    time.sleep(1.0)
    pids_after_to = get_current_python_pids()
    leaked_to_pids = pids_after_to - pids_before_to

    print(f"Timeout'dan keyin qolib ketgan yetim jarayonlar: {leaked_to_pids if leaked_to_pids else 'HECH QANDAY (0 ta)'}", flush=True)

    assert timeout_caught, "Xato: TimeoutError chaqirilmadi!"
    assert len(leaked_to_pids) == 0, f"Xato: Timeout bo'lgandan keyin jarayon o'lmadi: {leaked_to_pids}"

    print("\nBARCHA INTEGRATSION VA TIMEOUT TESTLAR 100% MUVAFFAQIShI O'TDI!", flush=True)

if __name__ == "__main__":
    main()
