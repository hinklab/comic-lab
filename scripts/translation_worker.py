"""
scripts/translation_worker.py - Isolated subprocess worker for NLLB translation.
Runs CTranslate2 NLLB in an isolated OS child process so that the ~700 MB model
memory is 100% reclaimed by the OS upon process exit.
"""
import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force stdout to UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import engine


def main():
    if len(sys.argv) < 3:
        print("Usage: python translation_worker.py <input_json_path> <output_json_path> [default_speaker] [page_name]", file=sys.stderr)
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    default_speaker = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "None" else None
    page_name = sys.argv[4] if len(sys.argv) > 4 else "auto"

    with open(in_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    bubbles = [engine.SpeechBubble(**d) for d in raw_data]

    import psutil
    proc = psutil.Process()
    print(f"[TRANSLATION_WORKER] Child PID {proc.pid} started. Initial RSS: {proc.memory_info().rss / (1024*1024):.1f} MB (FORCE_NLLB_FALLBACK={os.getenv('FORCE_NLLB_FALLBACK', '0')})", flush=True)

    # Execute core in-process translation inside this isolated child process
    translated = engine._translate_bubbles_list_core(
        bubbles,
        default_speaker=default_speaker,
        page_name=page_name
    )

    peak_rss = proc.memory_info().rss / (1024 * 1024)
    out_data = [b.model_dump() for b in translated]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)

    print(f"[TRANSLATION_WORKER] Successfully translated {len(translated)} bubbles to {out_path}. Active Peak Child RSS: {peak_rss:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
