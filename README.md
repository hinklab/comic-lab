---
title: Comic Lab
emoji: ⚡
colorFrom: red
colorTo: purple
sdk: streamlit
sdk_version: "1.30.0"
app_file: streamlit_app.py
pinned: false
---

# 💥 100% Local Comic Book Translation & Typesetting Studio

A standalone, local & free comic book translation and typesetting web application built with **Streamlit**, **OpenCV**, **EasyOCR**, **deep-translator**, and **Pillow (PIL)**.

**100% Free, Offline/Local, Zero AI API Keys.**

---

## 🌟 Key Features

1. **Flawless Bubble Detection & 100% Clean Erase (OpenCV)**:
   - High-luminance binary thresholding ($> 225$) and morphological closing to discover speech bubble contours ($1,200 \le \text{area} \le 150,000$).
   - Contour inpainting with 5-pixel erosion mask guarantees 100% of old English dialogue is completely erased with **zero ghost text**, while outer black comic borders are preserved intact.

2. **Accurate Local OCR (EasyOCR)**:
   - Speech bubbles are cropped from the original image **before** cleaning.
   - Preprocessed with CLAHE contrast enhancement and read locally on CPU.
   - Strict filtering against SFX noise (`TEKK`, `SIW`, `THWIP`), standalone numbers (`4`), and single letters.

3. **Free Translation & Spider-Man Superhero Tone Mapper**:
   - Uses `deep-translator` (`GoogleTranslator`) with automatic failover to `MyMemoryTranslator`.
   - Post-processor maps literal phrasing to punchy Marvel / Spider-Man superhero comic dialogue:
     * *"WHAT THE--?!"* $\rightarrow$ *"BU NIMA BALO?!"*
     * *"LOOK OUT!"* $\rightarrow$ *"EHTIYOT BO'L!"*
     * *"DON'T PANIC!"* $\rightarrow$ *"VAHIMAGA TUSHMA!"*
     * *"JUST GIVE ME TEN SECONDS!"* $\rightarrow$ *"MENGA ATIGI O'N SONIYA BER!"*
     * *"IMPRESSED? DON'T BOTHER..."* $\rightarrow$ *"QOYIL QOLDINGMI? HOVLIQMA..."*
     * *"I CAN SEE IT..."* $\rightarrow$ *"O'ZIM HAM KO'RIB TURIBMAN..."*
     * *"CRITICAL LIMIT"* $\rightarrow$ *"XAVFLI DARAJA"*

4. **Collision-Free Typography (`CC Wild Words`)**:
   - Industry standard font: `fonts/CCWildWords.ttf` strictly in UPPERCASE.
   - Dynamic auto-scaling from 24px down to 9px inside 70% oval bounds.
   - Explicit line height advancement: $(\text{bottom} - \text{top}) + 6\text{px}$ gap prevents line collision or overlap.
   - Automatic normalization of Uzbek diacritics (`O'`, `G'`, `Sh`, `Ch`).

5. **Streamlit UI Workspace**:
   - Zero API key fields.
   - **"🧹 Cleaned Preview"** tab to visually verify that all dialogue was erased cleanly before applying lettering.
   - Editable translation cards with per-bubble font size nudge sliders.
   - 1-click high-resolution PNG export to `output/`.

---

## 🚀 Quick Start

Run the Streamlit server:
```bash
py -m streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.
