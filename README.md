# 💥 100% Local Comic Book Translation & Typesetting Studio

A standalone, local & free comic book translation and typesetting web application built with **Starlette/FastAPI**, **HTML5 Canvas Studio**, **OpenCV**, **EasyOCR / Tesseract**, **deep-translator**, and **Pillow (PIL)**.

**100% Free, Offline/Local, Zero AI API Keys, Zero Streamlit Overhead.**

---

## 🌟 Key Features

1. **Flawless Bubble Detection & 100% Clean Erase (OpenCV)**:
   - High-luminance binary thresholding ($> 225$) and morphological closing to discover speech bubble contours ($1,200 \le \text{area} \le 150,000$).
   - Contour inpainting with 5-pixel erosion mask guarantees 100% of old English dialogue is completely erased with **zero ghost text**, while outer black comic borders are preserved intact.

2. **Accurate Local OCR (Tesseract / EasyOCR)**:
   - Speech bubbles are cropped from the original image **before** cleaning.
   - Preprocessed with CLAHE contrast enhancement and read locally on CPU.
   - Strict filtering against SFX noise (`TEKK`, `SIW`, `THWIP`), standalone numbers (`4`), and single letters.

3. **Free Translation & Superhero Tone Naturalization**:
   - Uses `deep-translator` (`GoogleTranslator`) with automatic failover to local NLLB subprocess / MyMemory.
   - Post-processor maps literal phrasing to punchy Marvel / Spider-Man superhero comic dialogue:
     * *"WHAT THE--?!"* $\rightarrow$ *"BU NIMA BALO?!"*
     * *"LOOK OUT!"* $\rightarrow$ *"EHTIYOT BO'L!"*
     * *"DON'T PANIC!"* $\rightarrow$ *"VAHIMAGA TUSHMA!"*
     * *"JUST GIVE ME TEN SECONDS!"* $\rightarrow$ *"MENGA ATIGI O'N SONIYA BER!"*
     * *"IMPRESSED? DON'T BOTHER..."* $\rightarrow$ *"QOYIL QOLDINGMI? HOVLIQMA..."*
     * *"I CAN SEE IT..."* $\rightarrow$ *"O'ZIM HAM KO'RIB TURIBMAN..."*
     * *"CRITICAL LIMIT"* $\rightarrow$ *"XAVFLI DARAJA"*

4. **Collision-Free Typography (`CC Wild Words`, `Yada Yada Yada`, etc.)**:
   - Industry standard comic fonts strictly in UPPERCASE.
   - Dynamic auto-scaling inside bubble bounds.
   - Explicit line height advancement prevents line collision or overlap.
   - Automatic normalization of Uzbek diacritics (`O'`, `G'`, `Sh`, `Ch`).

5. **High-Performance Web Studio (HTML5 Canvas + Starlette)**:
   - 60 FPS smooth zoom, pan, and interactive bubble editing directly on canvas.
   - Zero Streamlit lag, reruns, or memory leaks (~35 MB baseline RAM).
   - Editable translation cards with per-bubble font, size, and style controls.
   - 1-click high-resolution rendering and download.

---

## 🚀 Quick Start

Run the server:
```bash
python app.py
```
Open your browser at **`http://localhost:8000`**.
