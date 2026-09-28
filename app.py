"""
Comic Book Translation and Typesetting Studio
Zero AI API Keys (free/local translation model); online services for data storage are fine.
Powered by OpenCV, EasyOCR, deep-translator (Spider-Man tone), and CC Wild Words typography.
"""

import os
import warnings
warnings.filterwarnings("ignore")
import io
import re
import html
import base64
import time
from typing import Dict, Tuple, List, Any, Optional
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import streamlit as st

import importlib
import local_translator
import character_profiles
import naturalization_rules
import quality_gates
import bubble_lettering
import bubble_mask_editor
import history_manager
import engine
import sfx_engine

# Dynamic cache busting: Ensure updated modules are always loaded on deploy
try:
    importlib.reload(engine)
    importlib.reload(sfx_engine)
    importlib.reload(bubble_lettering)
    importlib.reload(quality_gates)
except Exception:
    pass

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Page config MUST be the very first Streamlit call -----------------
_favicon_path = os.path.join(_BASE_DIR, "assets", "favicon.png")
_page_icon = Image.open(_favicon_path) if os.path.exists(_favicon_path) else None
st.set_page_config(
    page_title="COMIC-LAB",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)
# ------------------------------------------------------------------------

# Invalidate stale session state from older deployments automatically
APP_BUILD_VERSION = "2026.09.28.v4_precision"
if st.session_state.get("_app_build_version") != APP_BUILD_VERSION:
    st.session_state.clear()
    st.session_state["_app_build_version"] = APP_BUILD_VERSION


class BubbleDict(dict):

    """Dictionary supporting both attribute (dot) and dict-item (bracket) access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def pil_to_png_bytes(img: Image.Image) -> bytes:
    """Converts PIL Image to raw PNG bytes using fast compression level."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


@st.cache_data
def get_comic_star_b64() -> str:
    """Returns Base64 string of the comic explosion sticker (cached in memory)."""
    candidates = [
        os.path.join(_BASE_DIR, "assets", "comic-lab-logo.png"),
        os.path.join(_BASE_DIR, "assets", "comic_explosion_transparent.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    return ""



def reset_page_state(image: Optional[Image.Image], image_name: str):
    """Resets all per-page analysis, cleaning, and rendering state for a new image."""
    keys_to_clear = [k for k in list(st.session_state.keys()) if k.startswith(("trans_", "font_size_", "line_spacing_", "pad_", "overlay_"))]
    for k in keys_to_clear:
        del st.session_state[k]

    # Memory guard: clamp extreme poster-sized images to max 4096px to prevent Out-Of-Memory (OOM)
    if image is not None:
        max_dim = max(image.size)
        if max_dim > 4096:
            scale = 4096.0 / max_dim
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

    st.session_state.image = image
    st.session_state.image_name = image_name
    st.session_state.raw_bubbles = []
    st.session_state.bubbles = []
    st.session_state.cleaned_page = None
    st.session_state.cleaned_page_bytes = None
    st.session_state.cleaned_preview = None
    st.session_state.rendered_image = None
    st.session_state.rendered_image_bytes = None
    st.session_state.saved_file_path = ""
    st.session_state.last_render_hash = ""
    st.session_state.current_stage = 1
    st.session_state.analysis_version = st.session_state.get("analysis_version", 0) + 1


def capture_bubble_edits_into_memory(page_name: str = "auto"):
    """
    Captures user-edited translations into character memory as concrete sample_lines.
    Called on review and confirm actions in Stage 2 and Stage 3.
    Only captures meaningful edits (skips trivial whitespace/typos) and prevents double-counting.
    """
    if not st.session_state.get("bubbles"):
        return
    if page_name == "auto":
        page_name = st.session_state.get("image_name", "unknown")

    for b in st.session_state.bubbles:
        b_id = b["bubble_id"] if isinstance(b, dict) else b.bubble_id
        spk = b.get("speaker") if isinstance(b, dict) else getattr(b, "speaker", None)
        if not spk:
            spk = "Superior Spider-Man"

        en_text = b.get("original_text", "") if isinstance(b, dict) else getattr(b, "original_text", "")
        cur_uz = b.get("uzbek_translation", "") if isinstance(b, dict) else getattr(b, "uzbek_translation", "")
        pipe_uz = b.get("pipeline_uzbek_translation", "") if isinstance(b, dict) else getattr(b, "pipeline_uzbek_translation", "")
        last_captured = b.get("last_captured_uzbek", "") if isinstance(b, dict) else getattr(b, "last_captured_uzbek", "")

        if cur_uz and cur_uz != last_captured and en_text:
            recorded, reason = character_profiles.record_user_edited_sample_line(
                character_name=spk,
                en_text=en_text,
                edited_uzbek=cur_uz,
                pipeline_uzbek=pipe_uz,
                page=page_name
            )
            # Mark this text state as recorded/evaluated so we don't repeat on identical runs
            if recorded or reason in ("duplicate", "trivial_or_identical"):
                if isinstance(b, dict):
                    b["last_captured_uzbek"] = cur_uz
                else:
                    b.last_captured_uzbek = cur_uz

        # Implicit confirmation: grow character's visual signature if detected with confidence
        colors = b.get("tail_region_colors") if isinstance(b, dict) else getattr(b, "tail_region_colors", None)
        weights = b.get("tail_region_weights") if isinstance(b, dict) else getattr(b, "tail_region_weights", None)
        conf = b.get("speaker_confidence") if isinstance(b, dict) else getattr(b, "speaker_confidence", 0.0)
        sig_done = b.get("visual_sig_confirmed", False) if isinstance(b, dict) else getattr(b, "visual_sig_confirmed", False)
        actual_spk = b.get("speaker") if isinstance(b, dict) else getattr(b, "speaker", None)
        if actual_spk and conf and conf >= 0.55 and colors and weights and not sig_done:
            character_profiles.update_visual_signature(
                character_name=actual_spk,
                new_colors=colors,
                new_weights=weights,
                page=page_name
            )
            if isinstance(b, dict):
                b["visual_sig_confirmed"] = True
            else:
                b.visual_sig_confirmed = True



def sync_bubble_widgets():
    """
    Explicitly updates all bubbles in st.session_state.bubbles from widget inputs.
    Must be called right before calling engine.render_page(...) (both in live re-render
    and inside the 'Sahifaga Shriftlarni Qayta Yozish' button click handler).
    Also triggers user-edit capture into character memory.
    """
    if not st.session_state.get("bubbles"):
        return
    for b in st.session_state.bubbles:
        b_id = b["bubble_id"] if isinstance(b, dict) else b.bubble_id

        # Update translation text
        trans_key = f"trans_{b_id}"
        if trans_key in st.session_state:
            val = st.session_state[trans_key]
            if isinstance(b, dict):
                b["uzbek_translation"] = val
            else:
                b.uzbek_translation = val

        # Update active/deleted status
        active_key = f"active_{b_id}"
        if active_key in st.session_state:
            is_active = bool(st.session_state[active_key])
            if isinstance(b, dict):
                b["is_active"] = is_active
                b["enabled"] = is_active
            else:
                b.is_active = is_active
                b.enabled = is_active

        # Update font nudge
        nudge_key = f"nudge_{b_id}"
        if nudge_key in st.session_state:
            nudge = st.session_state[nudge_key]
            if isinstance(b, dict):
                b["font_size_offset"] = nudge
                b["size_offset"] = nudge
            else:
                b.font_size_offset = nudge
                b.size_offset = nudge

    # Capture any confirmed text edits into persistent character memory
    capture_bubble_edits_into_memory()


def trigger_render():
    """
    Real-time reactive render callback:
    Renders directly on cleaned_page canvas using typeset_lettering_on_page.
    """
    if st.session_state.get("image") is None or not st.session_state.get("bubbles"):
        return

    sync_bubble_widgets()

    fonts = engine.get_available_fonts()
    f_choice = st.session_state.get("font_choice_select")
    f_path = fonts.get(f_choice) if f_choice else next(iter(fonts.values()), "")

    max_fs = st.session_state.get("max_font_size_slider", 26)
    min_fs = st.session_state.get("min_font_size_slider", 10)
    lg = st.session_state.get("line_gap_slider", 6)

    active_bubbles = [
        b for b in st.session_state.bubbles
        if (b.get("is_active", b.get("enabled", True)) if isinstance(b, dict) else getattr(b, "is_active", getattr(b, "enabled", True)))
    ]

    base_canvas = st.session_state.get("cleaned_page")
    if base_canvas is None:
        base_canvas = engine.clean_page_ink_telea(st.session_state.image, active_bubbles)
        st.session_state.cleaned_page = base_canvas

    had_rendered = st.session_state.get("rendered_image") is not None

    st.session_state.rendered_image = engine.typeset_lettering_on_page(
        cleaned_image=base_canvas,
        bubbles=active_bubbles,
        font_path=f_path,
        max_font_size=max_fs,
        min_font_size=min_fs,
        line_gap=lg,
        original_image=st.session_state.get("image")
    )
    st.session_state.rendered_image_bytes = pil_to_png_bytes(st.session_state.rendered_image)
    st.session_state.render_timestamp = time.time()
    if had_rendered:
        st.toast("Jonli natija yangilandi!", icon=":material/auto_awesome:")


def get_stepper_component(current_stage: int) -> str:
    """
    Renders the modern capsule pill stepper exactly matching the user's reference image:
    - Dark capsule pill container (#181A20)
    - Continuous connecting track with active progress fill (#FF3366)
    - 3 circular nodes with minimal task-specific icons:
      1. Skanerlash: Minimal scan aperture frame
      2. Tarjima: Minimal speech bubble with dialogue lines
      3. Shriftlar: Minimal calligraphy pen nib / typography
    """
    c_bg = "#000000"       # Black Void
    c_active = "#b62b1a"   # Crimson Heat (brand accent)
    c_inactive = "#4d4d4d" # Smoke

    x1, y1 = 75, 45
    x2, y2 = 230, 45
    x3, y3 = 385, 45
    r = 28
    track_h = 14

    s1 = current_stage >= 1
    s2 = current_stage >= 2
    s3 = current_stage >= 3

    ic1 = "#FFFFFF" if s1 else "#757B8A"
    ic2 = "#FFFFFF" if s2 else "#757B8A"
    ic3 = "#FFFFFF" if s3 else "#757B8A"

    active_track_w = 0
    if current_stage == 2:
        active_track_w = x2 - x1
    elif current_stage >= 3:
        active_track_w = x3 - x1

    glow1 = 'filter="url(#stepperGlow)"' if current_stage == 1 else ''
    glow2 = 'filter="url(#stepperGlow)"' if current_stage == 2 else ''
    glow3 = 'filter="url(#stepperGlow)"' if current_stage == 3 else ''

    raw_html = f'''
    <div style="display: flex; justify-content: center; margin: 4px auto 12px auto; max-width: 480px; width: 100%;">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 90" width="100%" height="90" style="display: block; filter: drop-shadow(0 6px 18px rgba(0,0,0,0.4));">
        <defs>
          <filter id="stepperGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="#b62b1a" flood-opacity="0.6"/>
          </filter>
        </defs>

        <rect x="2" y="2" width="456" height="86" rx="43" ry="43" fill="{c_bg}" stroke="#4d4d4d" stroke-width="2" />
        <rect x="{x1}" y="{y1 - track_h//2}" width="{x3 - x1}" height="{track_h}" rx="{track_h//2}" ry="{track_h//2}" fill="{c_inactive}" />
        {f'<rect x="{x1}" y="{y1 - track_h//2}" width="{active_track_w}" height="{track_h}" rx="{track_h//2 if active_track_w == (x3-x1) else 0}" fill="{c_active}" />' if active_track_w > 0 else ''}

        <circle cx="{x1}" cy="{y1}" r="{r}" fill="{c_active if s1 else c_inactive}" {glow1} />
        <circle cx="{x2}" cy="{y2}" r="{r}" fill="{c_active if s2 else c_inactive}" {glow2} />
        <circle cx="{x3}" cy="{y3}" r="{r}" fill="{c_active if s3 else c_inactive}" {glow3} />

        <g transform="translate({x1 - 12}, {y1 - 12})" stroke="{ic1}" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" fill="none">
          <path d="M3 7V4a1 1 0 0 1 1-1h3" />
          <path d="M21 7V4a1 1 0 0 0-1-1h-3" />
          <path d="M3 17v3a1 1 0 0 0 1 1h3" />
          <path d="M21 17v3a1 1 0 0 1-1 1h-3" />
          <circle cx="12" cy="12" r="3.5" />
          <line x1="7" y1="12" x2="8.5" y2="12" />
          <line x1="15.5" y1="12" x2="17" y2="12" />
        </g>

        <g transform="translate({x2 - 12}, {y2 - 12})" stroke="{ic2}" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" fill="none">
          <path d="M4 4h16a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H8l-5 4V6a2 2 0 0 1 2-2z" />
          <line x1="8" y1="9" x2="16" y2="9" />
          <line x1="8" y1="13" x2="13" y2="13" />
        </g>

        <g transform="translate({x3 - 12}, {y3 - 12})" stroke="{ic3}" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" fill="none">
          <path d="M12 19l7-7 3 3-7 7-3-3z"/>
          <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/>
          <path d="M2 2l7.5 7.5"/>
          <circle cx="11" cy="11" r="1.5" fill="{ic3}"/>
        </g>
      </svg>
    </div>
    '''
    return "".join(line.strip() for line in raw_html.splitlines() if line.strip())


def get_progress_bar_html(
    value: Optional[float] = None,
    max_val: float = 100.0,
    label: str = "Jarayon",
    pending_label: str = "Bajarilmoqda...",
    complete_label: str = "Tugatildi"
) -> str:
    """
    Renders the exact sleek ProgressBar design (ported from shadcn/motion React component).
    - If value is None: animated indeterminate sweeping gleam bar
    - If value is numeric: smooth determinate progress bar with tabular-nums percentage
    """
    indeterminate = value is None
    if indeterminate:
        status_text = html.escape(pending_label)
        inner_html = '<span class="comic-progress-indeterminate"></span>'
    else:
        frac = 0.0 if max_val <= 0 else max(0.0, min(1.0, float(value) / float(max_val)))
        pct = int(round(frac * 100))
        status_text = html.escape(complete_label) if frac >= 1.0 else f"{pct}%"
        inner_html = f'<span class="comic-progress-fill" style="width: {pct}%;"></span>'

    return (
        f'<div class="comic-progress-wrap">'
        f'<div class="comic-progress-header">'
        f'<span class="comic-progress-label">{html.escape(label)}</span>'
        f'<span class="comic-progress-status">{status_text}</span>'
        f'</div>'
        f'<div class="comic-progress-track">'
        f'<div class="comic-progress-inner">{inner_html}</div>'
        f'</div>'
        f'</div>'
    )


def split_bubble_action(b, bubble_id: int, live_render: bool = False):
    """Splits a speech bubble horizontally into two separate bubbles."""
    sync_bubble_widgets()
    if isinstance(b, dict):
        x0, y0, x1, y1 = int(b["x0"]), int(b["y0"]), int(b["x1"]), int(b["y1"])
        orig_t = b.get("original_text", "")
        uz_t = b.get("uzbek_translation", "")
        cur_offset = b.get("size_offset", 0)
    else:
        x0, y0, x1, y1 = int(b.x0), int(b.y0), int(b.x1), int(b.y1)
        orig_t = getattr(b, "original_text", "")
        uz_t = getattr(b, "uzbek_translation", "")
        cur_offset = getattr(b, "size_offset", 0)

    mid_x = (x0 + x1) // 2
    words_orig = orig_t.split()
    mid_orig = max(1, len(words_orig) // 2)
    orig_left = " ".join(words_orig[:mid_orig]) if words_orig else orig_t
    orig_right = " ".join(words_orig[mid_orig:]) if len(words_orig) > 1 else orig_t

    # Translate complete clustered sentence first if not yet translated
    if not uz_t:
        cur_spk = getattr(b, "speaker", None) if not isinstance(b, dict) else b.get("speaker")
        uz_t = engine.translate_spiderman_uzbek(orig_t, speaker=cur_spk)

    # Partition translated text across the two split halves using natural syntax/clause boundaries
    weights = [max(1, len(orig_left.split())), max(1, len(orig_right.split()))]
    sub_uz = engine.partition_translated_text_for_lobes(uz_t, weights)
    uz_left = sub_uz[0] if len(sub_uz) > 0 else uz_t
    uz_right = sub_uz[1] if len(sub_uz) > 1 else ""

    b_left = engine.SpeechBubble(
        bubble_id=bubble_id,
        x0=x0, y0=y0, x1=mid_x - 5, y1=y1,
        original_text=orig_left,
        uzbek_translation=uz_left,
        confidence=1.0,
        enabled=True,
        is_active=True,
        size_offset=cur_offset,
        font_size_offset=cur_offset
    )
    b_right = engine.SpeechBubble(
        bubble_id=bubble_id + 1,
        x0=mid_x + 5, y0=y0, x1=x1, y1=y1,
        original_text=orig_right,
        uzbek_translation=uz_right,
        confidence=1.0,
        enabled=True,
        is_active=True,
        size_offset=cur_offset,
        font_size_offset=cur_offset
    )

    new_bubbles = []
    for old_b in st.session_state.bubbles:
        old_id = old_b["bubble_id"] if isinstance(old_b, dict) else old_b.bubble_id
        if old_id == bubble_id:
            new_bubbles.append(BubbleDict(b_left.model_dump()))
            new_bubbles.append(BubbleDict(b_right.model_dump()))
        else:
            new_bubbles.append(old_b)

    for new_idx, nb in enumerate(new_bubbles, 1):
        if isinstance(nb, dict):
            nb["bubble_id"] = new_idx
        else:
            nb.bubble_id = new_idx

    # Push undo snapshot before applying split
    history_manager.push_undo_snapshot(f"✂️ Pufak #{bubble_id} ni ikkiga ajratish")

    st.session_state.bubbles = new_bubbles
    for nb in new_bubbles:
        nb_id = nb["bubble_id"] if isinstance(nb, dict) else nb.bubble_id
        st.session_state[f"trans_{nb_id}"] = nb["uzbek_translation"] if isinstance(nb, dict) else nb.uzbek_translation
        st.session_state[f"active_{nb_id}"] = nb["is_active"] if isinstance(nb, dict) else nb.is_active
        st.session_state[f"nudge_{nb_id}"] = nb.get("font_size_offset", 0) if isinstance(nb, dict) else getattr(nb, "font_size_offset", 0)

    if live_render:
        trigger_render()
    st.toast(f"Bubble #{bubble_id} ikkiga muvaffaqiyatli ajratildi!", icon=":material/content_cut:")
    st.rerun()


DRAWER_W = 440                       # px
ANIM = "0.35s cubic-bezier(.4,0,.2,1)"

def set_drawer(value: bool):
    """Callback triggered before rerun to update drawer state."""
    st.session_state.show_editor_panel = value

def inject_drawer_css():
    """Dynamically injects CSS for drawer push/shrink behavior on stAppViewContainer."""
    is_open = bool(st.session_state.get("show_editor_panel", True) and st.session_state.get("image") is not None)
    css = """
<style>
:root {
  --drawer-w: __W__px;
  --drawer-pad: __PAD__px;
  --panel-anim: __ANIM__;
  --panel-bg: var(--secondary-background-color, #111114);
  --panel-border: rgba(250, 250, 250, 0.12);
  --panel-text: var(--text-color, #fafafa);
}

/* 1) PUSH/SHRINK: [sidebar | main] qatorining o'ng tomoniga joy ochamiz */
[data-testid="stAppViewContainer"] {
  box-sizing: border-box !important;
  padding-right: var(--drawer-pad) !important;
  transition: padding-right var(--panel-anim) !important;
}

/* 2) Header boshqaruvi: stHeader stAppViewContainer ichida joylashgani sababli
      padding-right hisobiga uning kengligi o'zi to'g'ri chegaralanadi.
      Ikki marta qisqarib taskbar markazga surilib ketmasligi uchun width: 100% o'rnatiladi. */
[data-testid="stHeader"] {
  width: 100% !important;
}
[data-testid="stToolbar"] {
  margin-right: __TB__ !important;
  transition: margin-right var(--panel-anim) !important;
}

/* 3) CHAP PANEL: slide animatsiyani majburlash (o'ng panel bilan bir xil rang va tezlik) */
section[data-testid="stSidebar"],
div[data-testid="stSidebar"] {
  background-color: var(--panel-bg) !important;
  border-right: 1px solid var(--panel-border) !important;
  box-shadow: 4px 0 24px rgba(0, 0, 0, 0.45) !important;
  transition: transform var(--panel-anim), min-width var(--panel-anim),
              max-width var(--panel-anim), margin-left var(--panel-anim) !important;
}

/* 4) O'NG PANEL: st.container(key="right_drawer") -> .st-key-right_drawer */
.st-key-right_drawer {
  position: fixed !important;
  top: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
  width: var(--drawer-w) !important;
  max-width: 100vw !important;
  box-sizing: border-box !important;
  overflow-y: auto !important;
  padding: 0.85rem 1.25rem 2rem 1.25rem !important;
  background: var(--panel-bg) !important;
  border-left: 1px solid var(--panel-border) !important;
  border-right: none !important;
  border-top: none !important;
  border-bottom: none !important;
  border-radius: 0 !important;
  color: var(--panel-text) !important;
  z-index: 1000001 !important;
  box-shadow: -4px 0 28px rgba(0, 0, 0, 0.55) !important;
  transform: translateX(__TX__) !important;
  visibility: __VIS__ !important;
  transition: transform var(--panel-anim), visibility 0s linear __VISDELAY__ !important;
}

.st-key-right_drawer::-webkit-scrollbar {
  width: 5px;
}
.st-key-right_drawer::-webkit-scrollbar-track {
  background: transparent;
}
.st-key-right_drawer::-webkit-scrollbar-thumb {
  background: #27272f;
  border-radius: 9999px;
}
.st-key-right_drawer::-webkit-scrollbar-thumb:hover {
  background: #b62b1a;
}

.panel-title {
  margin: 0 !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.8px !important;
  text-transform: uppercase !important;
  color: #a1a1aa !important;
}

/* 5) CHAP VA O'NG TUGMALAR BIR XIL USLUBDA (Shaffof, hoshiyasiz, hover yorug') */
.st-key-drawer_open button,
[class*="st-key-drawer_close"] button,
button[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapsedControl"] button {
  width: 2rem !important;
  height: 2rem !important;
  min-height: 2rem !important;
  padding: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: var(--panel-text) !important;
  opacity: .65 !important;
  border-radius: .5rem !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  transition: opacity .2s, background-color .2s !important;
}
.st-key-drawer_open button:hover,
[class*="st-key-drawer_close"] button:hover,
button[data-testid="stSidebarCollapseButton"]:hover,
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="collapsedControl"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover {
  opacity: 1 !important;
  background: rgba(250, 250, 250, .08) !important;
}
.st-key-drawer_open button p,
[class*="st-key-drawer_close"] button p,
.st-key-drawer_open button span,
[class*="st-key-drawer_close"] button span {
  font-size: 1.25rem !important;
  line-height: 1 !important;
  margin: 0 !important;
}

[class*="st-key-drawer_close"] {
  display: flex !important;
  justify-content: flex-end !important;
}

/* Floating top-right Open Button when drawer is closed */
.st-key-drawer_open {
  position: fixed !important;
  top: .6rem !important;
  right: .75rem !important;
  z-index: 1000002 !important;
  width: auto !important;
  animation: panelFade var(--panel-anim) !important;
}
@keyframes panelFade { from {opacity: 0} to {opacity: 1} }

/* rerun paytida Streamlit elementlarni xiralashtirmasin (flicker) */
[data-stale="true"] { opacity: 1 !important; }

/* tor ekranlarda push yo'q, panel ustidan chiqadi */
@media (max-width: 900px) {
  [data-testid="stAppViewContainer"] { padding-right: 0 !important; }
  [data-testid="stHeader"] { width: 100% !important; }
  .st-key-right_drawer { width: 100vw !important; }
}
</style>
"""
    css = (css.replace("__W__", str(DRAWER_W))
              .replace("__PAD__", str(DRAWER_W if is_open else 0))
              .replace("__ANIM__", ANIM)
              .replace("__TX__", "0" if is_open else "105%")
              .replace("__VIS__", "visible" if is_open else "hidden")
              .replace("__VISDELAY__", "0s" if is_open else "0.35s")
              .replace("__TB__", "0" if is_open else "2.75rem"))
    st.markdown(css, unsafe_allow_html=True)


def render_right_panel_header(title: str, badge_text: str = "", close_key: str = "close_panel"):
    """Renders the sleek modern IDE-style right panel header matching left sidebar symmetry."""
    badge_html = f' <span class="modern-badge-pill">{html.escape(badge_text)}</span>' if badge_text else ''
    
    col_rh_title, col_rh_btn = st.columns([0.86, 0.14], vertical_alignment="center")
    with col_rh_title:
        st.markdown(
            f'<div style="display: flex; align-items: center; gap: 8px;">'
            f'<span style="color: #b62b1a; font-size: 14px; font-weight: 700;">◨</span>'
            f'<p class="panel-title">{html.escape(title)}</p>{badge_html}'
            f'</div>',
            unsafe_allow_html=True
        )
    with col_rh_btn:
        st.button(
            ":material/keyboard_double_arrow_right:",
            key=f"drawer_close_{close_key}",
            on_click=set_drawer,
            args=(False,),
            help="Yopish",
            use_container_width=True
        )

    st.markdown('<hr style="border: none; border-top: 1px solid var(--panel-border, rgba(250,250,250,.12)); margin: 6px 0 14px 0;" />', unsafe_allow_html=True)


def render_bubble_editor_panel(live_render: bool = False, key_suffix: str = ""):
    """
    Renders the rich bubble text cards in a sleek modern IDE/Cursor-like inspector panel:
    - In Stage 2: normal edits
    - In Stage 3: live edits with on_change=trigger_render for instant re-rendering!
    """
    if not st.session_state.get("bubbles"):
        st.info("Pufaklar ro'yxati bo'sh.")
        return

    col_p_title, col_p_undo, col_p_redo = st.columns([0.64, 0.18, 0.18])
    with col_p_title:
        st.markdown(
            f'<div class="modern-panel-section-header"><span>PUFAKLAR ({len(st.session_state.bubbles)})</span></div>',
            unsafe_allow_html=True
        )
    with col_p_undo:
        u_dis = not history_manager.can_undo()
        u_desc = history_manager.get_undo_description() or ""
        u_tip = f"Bekor qilish: {u_desc} (Ctrl+Z)" if u_desc else "Bekor qilish (Ctrl+Z)"
        if st.button("↩️", key=f"panel_undo_{key_suffix}", disabled=u_dis, help=u_tip, use_container_width=True):
            succ, desc = history_manager.undo_action()
            if succ:
                if live_render:
                    trigger_render()
                st.toast(f"↩️ Bekor qilindi: {desc}", icon=":material/undo:")
                st.rerun()
    with col_p_redo:
        r_dis = not history_manager.can_redo()
        if st.button("↪️", key=f"panel_redo_{key_suffix}", disabled=r_dis, help="Qaytarish (Ctrl+Y)", use_container_width=True):
            succ, desc = history_manager.redo_action()
            if succ:
                if live_render:
                    trigger_render()
                st.toast(f"↪️ Qaytarildi: {desc}", icon=":material/redo:")
                st.rerun()

    cb_on_change = trigger_render if live_render else None

    for i, b in enumerate(st.session_state.bubbles):
        bubble_id = i + 1
        is_active = b.get("is_active", b.get("enabled", True)) if isinstance(b, dict) else getattr(b, "is_active", getattr(b, "enabled", True))
        cur_text = b["uzbek_translation"] if isinstance(b, dict) else b.uzbek_translation
        orig_dialogue = (b.get("clean_text") or b.get("original_text", "")) if isinstance(b, dict) else (getattr(b, "clean_text", None) or getattr(b, "original_text", ""))
        has_residue, residue_words = quality_gates.check_untranslated_english_residue(cur_text, en_orig=orig_dialogue)
        val = quality_gates.post_translation_validation(cur_text, orig_dialogue)

        needs_review = (
            b.get("needs_review", False) if isinstance(b, dict) else getattr(b, "needs_review", False)
        ) or has_residue or val.get("needs_review", False)

        review_reason = (
            b.get("review_reason") if isinstance(b, dict) else getattr(b, "review_reason", None)
        ) or val.get("reason")

        card_classes = ["modern-bubble-card"]
        if needs_review:
            card_classes.append("needs-review")
        if not is_active:
            card_classes.append("disabled")
        class_str = " ".join(card_classes)

        with st.container():
            orig_text = html.escape(b.get("original_text", "") if isinstance(b, dict) else getattr(b, "original_text", ""))
            
            # Speaker badge
            cur_spk = b.get("speaker") if isinstance(b, dict) else getattr(b, "speaker", None)
            cur_conf = b.get("speaker_confidence") if isinstance(b, dict) else getattr(b, "speaker_confidence", None)
            if cur_conf is None:
                cur_conf = 0.0
            if cur_spk and cur_conf >= 0.55:
                spk_html = f'<span class="modern-speaker-pill">{html.escape(cur_spk)} ({cur_conf:.0%})</span>'
            else:
                spk_html = '<span class="modern-speaker-pill-muted">Umumiy ovoz</span>'

            # Review badge
            review_badge = '<span class="modern-review-pill">⚠️ KO\'RIB CHIQISH</span>' if needs_review else ''

            card_html = (
                f'<div class="{class_str}">'
                f'<div class="mbc-header">'
                f'<div class="mbc-tags"><span class="mbc-id">#{bubble_id}</span>{spk_html}</div>'
                f'{review_badge}'
                f'</div>'
                f'<div class="mbc-orig-dialogue"><span class="mbc-orig-label">EN</span> {orig_text}</div>'
            )
            if needs_review:
                card_html += f'<div class="mbc-error-notice">⚠️ {html.escape(review_reason or "Tarjima sifati tekshiruvi talab etiladi")}</div>'
            card_html += '</div>'
            st.markdown(card_html, unsafe_allow_html=True)

            col_active, col_text = st.columns([1.2, 3.8])
            with col_active:
                active_val = st.checkbox(
                    "Faol / Active",
                    value=is_active,
                    key=f"active_{bubble_id}",
                    on_change=cb_on_change,
                    help="Belgini olib tashlasangiz, bu pufak tozalanmaydi va tarjima yozilmaydi."
                )
                if active_val != is_active:
                    act_str = "faollashtirildi" if active_val else "o'chirildi"
                    history_manager.push_undo_snapshot(f"👁️ Pufak #{bubble_id} {act_str}")
                if isinstance(b, dict):
                    b["is_active"] = active_val
                    b["enabled"] = active_val
                else:
                    b.is_active = active_val
                    b.enabled = active_val

            with col_text:
                new_uzbek = st.text_area(
                    f"Uzbek Translation #{bubble_id}",
                    value=cur_text,
                    key=f"trans_{bubble_id}",
                    label_visibility="collapsed",
                    on_change=cb_on_change,
                    help=f"Bubble #{bubble_id} tarjimasini tahrirlang"
                )
                if new_uzbek != cur_text:
                    history_manager.push_undo_snapshot(f"✏️ Pufak #{bubble_id} tarjimasi tahrirlandi")
                if isinstance(b, dict):
                    b["uzbek_translation"] = new_uzbek
                    if new_uzbek != cur_text:
                        new_val = quality_gates.post_translation_validation(new_uzbek, orig_dialogue)
                        if not new_val.get("needs_review"):
                            b["needs_review"] = False
                            b["review_reason"] = None
                else:
                    b.uzbek_translation = new_uzbek
                    if new_uzbek != cur_text:
                        new_val = quality_gates.post_translation_validation(new_uzbek, orig_dialogue)
                        if not new_val.get("needs_review"):
                            b.needs_review = False
                            b.review_reason = None

            if active_val:
                cur_nudge = int(b.get("font_size_offset", b.get("size_offset", 0))) if isinstance(b, dict) else int(getattr(b, "font_size_offset", getattr(b, "size_offset", 0)))
                size_offset = st.slider(
                    f"Shrift o'lchami tuzatmasi #{bubble_id}",
                    min_value=-8, max_value=12, value=cur_nudge,
                    key=f"nudge_{bubble_id}",
                    on_change=cb_on_change,
                    help="Ushbu pufak shriftini kattalashtirish yoki kichraytirish."
                )
                if size_offset != cur_nudge:
                    history_manager.push_undo_snapshot(f"📏 Pufak #{bubble_id} shrift o'lchami o'zgartirildi")
                if isinstance(b, dict):
                    b["font_size_offset"] = size_offset
                    b["size_offset"] = size_offset
                else:
                    b.font_size_offset = size_offset
                    b.size_offset = size_offset

                col_split, _ = st.columns([1.5, 2.5])
                with col_split:
                    if st.button(f"✂️ Ikkiga ajratish", key=f"split_btn_{key_suffix}_{bubble_id}", help="Pufakni gorizontal ikkiga ajratish"):
                        split_bubble_action(b, bubble_id, live_render=live_render)





SYSTEM_FONTS_CSS_MAP = {
    "Comic Sans MS": "'Comic Sans MS', cursive, sans-serif",
    "Segoe UI Bold": "'Segoe UI', sans-serif",
    "Arial Bold": "Arial, sans-serif",
}


@st.cache_data
def get_active_font_css(font_path: str) -> str:
    """
    Returns @font-face CSS for 'ActiveComicFont'.
    System fonts use native font-family without heavy base64 strings.
    """
    if not os.path.exists(font_path):
        return ""

    norm_path = font_path.lower()
    for s_name, s_fam in SYSTEM_FONTS_CSS_MAP.items():
        if s_name.lower().split()[0] in norm_path or "windows\\fonts" in norm_path:
            return f"""
            <style>
            .comic-preview-text {{
                font-family: {s_fam} !important;
            }}
            div[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] div,
            div[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
            div[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p {{
                font-family: {s_fam} !important;
                font-size: 1.1rem !important;
                letter-spacing: 0.5px !important;
            }}
            </style>
            """

    with open(font_path, "rb") as f:
        font_bytes = f.read()

    b64_str = base64.b64encode(font_bytes).decode("utf-8")
    is_otf = font_bytes.startswith(b"OTTO")
    font_mime = "font/otf" if is_otf else "font/ttf"
    font_fmt = "opentype" if is_otf else "truetype"

    return f"""
    <style>
    @font-face {{
        font-family: 'ActiveComicFont';
        src: url('data:{font_mime};charset=utf-8;base64,{b64_str}') format('{font_fmt}');
        font-weight: bold;
        font-style: normal;
        font-display: swap;
    }}
    .comic-preview-text {{
        font-family: 'ActiveComicFont', cursive, sans-serif !important;
    }}
    div[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] div,
    div[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
    div[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p {{
        font-family: 'ActiveComicFont', cursive, sans-serif !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px !important;
    }}
    </style>
    """


@st.cache_data
def get_dropdown_fonts_css() -> str:
    """
    Loads custom fonts with correct format (OTF/TTF) for selectbox options.
    System fonts (Arial, Segoe UI, Comic Sans) use direct OS font-family to avoid 3MB base64 DOM payload.
    """
    fonts = engine.get_available_fonts()
    rules = []
    dropdown_rules = []

    for idx, (name, path) in enumerate(fonts.items(), start=1):
        norm_path = path.lower()
        sys_family = None
        for s_name, s_fam in SYSTEM_FONTS_CSS_MAP.items():
            if s_name.lower().split()[0] in norm_path or "windows\\fonts" in norm_path:
                sys_family = s_fam
                break

        if sys_family:
            dropdown_rules.append(f"""
div[data-baseweb="popover"] ul[role="listbox"] li:nth-child({idx}),
div[data-baseweb="popover"] ul[role="listbox"] li:nth-child({idx}) *,
div[data-baseweb="popover"] li[role="option"]:nth-child({idx}),
div[data-baseweb="popover"] li[role="option"]:nth-child({idx}) *,
ul[role="listbox"] li:nth-child({idx}),
ul[role="listbox"] li:nth-child({idx}) * {{
    font-family: {sys_family} !important;
    font-size: 1.12rem !important;
}}""")
        elif os.path.exists(path):
            with open(path, "rb") as f:
                fb = f.read()
            is_otf = fb.startswith(b"OTTO")
            mime = "font/otf" if is_otf else "font/ttf"
            fmt = "opentype" if is_otf else "truetype"
            b64 = base64.b64encode(fb).decode("utf-8")
            alias = f"MenuFont_{idx}"
            rules.append(f"""
@font-face {{
    font-family: '{alias}';
    src: url('data:{mime};charset=utf-8;base64,{b64}') format('{fmt}');
    font-weight: bold;
    font-style: normal;
    font-display: swap;
}}""")
            dropdown_rules.append(f"""
div[data-baseweb="popover"] ul[role="listbox"] li:nth-child({idx}),
div[data-baseweb="popover"] ul[role="listbox"] li:nth-child({idx}) *,
div[data-baseweb="popover"] li[role="option"]:nth-child({idx}),
div[data-baseweb="popover"] li[role="option"]:nth-child({idx}) *,
ul[role="listbox"] li:nth-child({idx}),
ul[role="listbox"] li:nth-child({idx}) * {{
    font-family: '{alias}', cursive, sans-serif !important;
    font-size: 1.12rem !important;
}}""")

    return "<style>\n" + "\n".join(rules) + "\n" + "\n".join(dropdown_rules) + "\n</style>"

# Custom Styling

st.markdown("""
<style>
    /* --- Design System Tokens ---------------------------------------------------
       Color palette  (ONLY these four -- no other chromatic colors allowed)
         Brand accent : #b62b1a  -- UI chrome only: buttons, active borders, progress
         Black Void   : #000000  -- page canvas, deepest surface, dark borders
         Smoke        : #4d4d4d  -- muted dividers, secondary borders, muted text
         Paper White  : #ffffff  -- primary text, strokes, ghost-button borders

       Type scale  (Perfect Fourth 1.333, base 14px)
         display-xl  : 245px / 380 / lh 1
         display-lg  : 136px / 300 / lh 1.3
         display     : 130px / 300 / lh 1   (uppercase)
         heading-lg  : 106px / 300 / lh 1.3
         heading     :  79px / 300 / lh 1.3
         heading-sm  :  47px / 300 / lh 1
         subheading  :  20px / 300 / lh 1.15
         body/base   :  18px / 380 / lh 1.11
    --------------------------------------------------------------------------- */

    /* --- Google Fonts ------------------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Inter:wght@300;400;600;700&display=swap');

    /* --- Legacy .comic-header (kept for backward-compat, maps to heading-sm) */
    .comic-header {
        font-family: 'Bangers', cursive, sans-serif;
        font-size: 47px;           /* heading-sm */
        font-weight: 300;
        line-height: 1;
        color: #b62b1a;            /* brand accent */
        letter-spacing: 2px;
        text-shadow: 2px 2px 0px #000000;
        margin-bottom: 0px;
    }

    /* --- Site Logo (sidebar): heading-sm scale --------------------------- */
    .comic-site-logo {
        display: inline-flex !important;
        align-items: center !important;
        gap: 10px !important;
        text-decoration: none !important;
        cursor: pointer !important;
        padding: 4px 0 !important;
        margin: 2px 0 14px 0 !important;
        border: none !important;
        background: transparent !important;
        transition: transform 0.15s ease, opacity 0.15s ease !important;
    }
    .comic-site-logo:hover {
        text-decoration: none !important;
        opacity: 0.88 !important;
        transform: scale(1.02) !important;
    }
    .comic-logo-icon {
        width: 54px !important;
        height: 54px !important;
        object-fit: contain !important;
        display: inline-block !important;
        vertical-align: middle !important;
        flex-shrink: 0 !important;
    }
    .comic-logo-text {
        font-family: 'Bangers', cursive, sans-serif !important;
        font-style: italic !important;
        font-size: 2.3rem !important;          /* original size */
        font-weight: 400 !important;
        line-height: 1 !important;
        color: #FF3366 !important;             /* original logo color */
        letter-spacing: 2px !important;
        text-shadow: 2.5px 2.5px 0px #000000 !important;
        display: inline-block !important;
        vertical-align: middle !important;
    }

    /* --- Subheading helper text ------------------------------------------- */
    .comic-sub {
        color: #4d4d4d;                    /* Smoke */
        font-size: 20px;                   /* subheading */
        font-weight: 300;
        line-height: 1.15;
        margin-bottom: 1.2rem;
    }

    /* --- Offline badge --------------------------------------------------- */
    .local-badge {
        background-color: #000000;        /* Black Void surface */
        border: 1px solid #ffffff;        /* Paper White border */
        color: #ffffff;
        font-size: 18px;                  /* body/base */
        font-weight: 380;
        line-height: 1.11;
        padding: 4px 12px;
        border-radius: 9999px;
        display: inline-block;
        margin-bottom: 12px;
    }

    /* --- Bubble Cards ---------------------------------------------------- */
    .bubble-card {
        background-color: #0d0d0d !important;  /* Sleek Dark */
        border: 1px solid #333333 !important;  /* Subtle Outline */
        border-radius: 8px !important;
        padding: 8px 10px !important;
        margin-bottom: 6px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
        color: #ffffff !important;
    }
    .bubble-card-ignored {
        background-color: #0a0a0a !important;
        border: 1px dashed #333333 !important;
        border-radius: 8px !important;
        padding: 6px 8px !important;
        margin-bottom: 6px !important;
        opacity: 0.45 !important;
        color: #666666 !important;
    }
    .bubble-badge {
        background-color: #b62b1a !important;  /* brand accent */
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 12px !important;            /* compact badge */
        line-height: 1.2 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        display: inline-block !important;
        margin-bottom: 4px !important;
        letter-spacing: 0.3px !important;
    }
    .dialogue-en {
        background-color: #141414 !important;
        border: 1px solid #2a2a2a !important;
        padding: 5px 8px !important;
        border-radius: 4px !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-size: 12.5px !important;          /* compact secondary font size */
        font-weight: 400 !important;
        line-height: 1.35 !important;
        color: #9e9e9e !important;             /* subtle secondary text color */
        margin-bottom: 4px !important;
        word-break: break-word !important;
    }

    /* --- Home Upload Zone ------------------------------------------------ */
    .home-uploader-zone {
        max-width: 840px;
        margin: 0 auto 16px auto;
        width: 100%;
    }
    div[data-testid="stFileUploader"] {
        width: 100% !important;
    }
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
        min-height: 200px !important;
        height: auto !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        border: 2px dashed #b62b1a !important;  /* brand accent */
        border-radius: 16px !important;
        background: rgba(182, 43, 26, 0.03) !important;
        padding: 30px 24px !important;
        transition: all 0.2s ease-in-out !important;
        cursor: pointer !important;
    }
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #b62b1a !important;
        background: rgba(182, 43, 26, 0.07) !important;
        box-shadow: 0 0 25px rgba(182, 43, 26, 0.25) !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        margin-top: 14px !important;
        text-align: center !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"]::before {
        content: "Komiks sahifasini bu yerga sudrab tashlang yoki yuklang" !important;
        display: block !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 18px !important;            /* body/base */
        font-weight: 380 !important;
        line-height: 1.11 !important;
        color: #ffffff !important;             /* Paper White */
        margin-bottom: 5px !important;
        letter-spacing: 0.2px !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
        font-size: 18px !important;            /* body/base */
        font-weight: 380 !important;
        color: #ffffff !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
        font-size: 18px !important;
        color: #4d4d4d !important;             /* Smoke */
    }

    /* --- Upload / Browse button ------------------------------------------ */
    .home-uploader-zone button,
    section[data-testid="stFileUploaderDropzone"] button,
    div[data-testid="stFileUploaderDropzone"] button {
        font-family: 'Bangers', cursive, sans-serif !important;
        font-size: 20px !important;            /* subheading */
        font-weight: 300 !important;
        line-height: 1.15 !important;
        letter-spacing: 1.2px !important;
        padding: 10px 32px !important;
        background: #b62b1a !important;        /* brand accent -- solid, no gradient */
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 2px 2px 0px #000000 !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
    }
    .home-uploader-zone button:hover,
    section[data-testid="stFileUploaderDropzone"] button:hover,
    div[data-testid="stFileUploaderDropzone"] button:hover {
        transform: scale(1.04) translateY(-1px) !important;
        background: #9a2416 !important;        /* darkened brand accent */
        box-shadow: 3px 3px 0px #000000 !important;
    }

    /* --- Shiny Gleam Button Effect (Subtle Low-Contrast Muted Gleam) -------- */
    @property --gradient-angle {
        syntax: "<angle>";
        initial-value: 0deg;
        inherits: false;
    }
    @property --gradient-angle-offset {
        syntax: "<angle>";
        initial-value: 0deg;
        inherits: false;
    }
    @property --gradient-percent {
        syntax: "<percentage>";
        initial-value: 4%;
        inherits: false;
    }
    @property --gradient-shine {
        syntax: "<color>";
        initial-value: rgba(210, 75, 60, 0.4);
        inherits: false;
    }

    /* Target all primary continue/stage-advance buttons and .comic-shiny-btn */
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    .comic-shiny-btn {
        --gleam-base: #000000;
        --gleam-inset: #161515;
        --gleam-label: #ffffff;
        --gleam-accent: rgba(182, 43, 26, 0.35);
        --gleam-accent-soft: rgba(215, 65, 50, 0.55);
        --animation: gradient-angle linear infinite;
        --duration: 3.5s;
        --shadow-size: 2px;
        --transition: 0.8s cubic-bezier(0.25, 1, 0.5, 1);

        isolation: isolate;
        position: relative !important;
        overflow: hidden !important;
        cursor: pointer !important;
        outline-offset: 4px;
        padding: 0.9rem 2rem !important;
        font-size: 1.125rem !important;
        line-height: 1.2 !important;
        font-weight: 500 !important;
        border: 1px solid transparent !important;
        border-radius: 12px !important;
        color: var(--gleam-label) !important;
        background:
            linear-gradient(var(--gleam-base), var(--gleam-base)) padding-box,
            conic-gradient(
                from calc(var(--gradient-angle) - var(--gradient-angle-offset)),
                transparent,
                var(--gleam-accent) var(--gradient-percent),
                var(--gradient-shine) calc(var(--gradient-percent) * 2),
                var(--gleam-accent) calc(var(--gradient-percent) * 3),
                transparent calc(var(--gradient-percent) * 4)
            ) border-box !important;
        box-shadow: inset 0 0 0 1px var(--gleam-inset) !important;
        transition: var(--transition) !important;
        transition-property:
            --gradient-angle-offset,
            --gradient-percent,
            --gradient-shine !important;
    }

    button[kind="primary"]::before,
    button[kind="primary"]::after,
    button[data-testid="baseButton-primary"]::before,
    button[data-testid="baseButton-primary"]::after,
    .comic-shiny-btn::before,
    .comic-shiny-btn::after {
        content: "" !important;
        pointer-events: none !important;
        position: absolute !important;
        inset-inline-start: 50% !important;
        inset-block-start: 50% !important;
        translate: -50% -50% !important;
        z-index: -1 !important;
    }

    button[kind="primary"]:active,
    button[data-testid="baseButton-primary"]:active,
    .comic-shiny-btn:active {
        translate: 0 1px !important;
    }

    button[kind="primary"]::before,
    button[data-testid="baseButton-primary"]::before,
    .comic-shiny-btn::before {
        --size: calc(100% - var(--shadow-size) * 3);
        --position: 2px;
        --space: calc(var(--position) * 2);
        width: var(--size) !important;
        height: var(--size) !important;
        background: radial-gradient(
            circle at var(--position) var(--position),
            rgba(255, 255, 255, 0.12) calc(var(--position) / 4),
            transparent 0
        ) padding-box !important;
        background-size: var(--space) var(--space) !important;
        background-repeat: space !important;
        -webkit-mask-image: conic-gradient(
            from calc(var(--gradient-angle) + 45deg),
            black,
            transparent 10% 90%,
            black
        ) !important;
        mask-image: conic-gradient(
            from calc(var(--gradient-angle) + 45deg),
            black,
            transparent 10% 90%,
            black
        ) !important;
        border-radius: inherit !important;
        opacity: 0.12 !important;
        z-index: -1 !important;
    }

    button[kind="primary"]::after,
    button[data-testid="baseButton-primary"]::after,
    .comic-shiny-btn::after {
        --animation: shimmer linear infinite;
        width: 100% !important;
        aspect-ratio: 1 !important;
        background: linear-gradient(
            -50deg,
            transparent,
            rgba(182, 43, 26, 0.4),
            transparent
        ) !important;
        -webkit-mask-image: radial-gradient(circle at bottom, transparent 40%, black) !important;
        mask-image: radial-gradient(circle at bottom, transparent 40%, black) !important;
        opacity: 0.18 !important;
        z-index: -1 !important;
    }

    button[kind="primary"] div[data-testid="stMarkdownContainer"],
    button[kind="primary"] p,
    button[data-testid="baseButton-primary"] div[data-testid="stMarkdownContainer"],
    button[data-testid="baseButton-primary"] p,
    .comic-shiny-btn span {
        z-index: 1 !important;
        position: relative !important;
        color: var(--gleam-label) !important;
    }

    button[kind="primary"],
    button[kind="primary"]::before,
    button[kind="primary"]::after,
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-primary"]::before,
    button[data-testid="baseButton-primary"]::after,
    .comic-shiny-btn,
    .comic-shiny-btn::before,
    .comic-shiny-btn::after {
        animation:
            var(--animation) var(--duration),
            var(--animation) calc(var(--duration) / 0.4) reverse paused !important;
        animation-composition: add !important;
    }

    button[kind="primary"]:is(:hover, :focus-visible),
    button[data-testid="baseButton-primary"]:is(:hover, :focus-visible),
    .comic-shiny-btn:is(:hover, :focus-visible) {
        --gradient-percent: 7% !important;
        --gradient-angle-offset: 95deg !important;
        --gradient-shine: rgba(225, 95, 80, 0.6) !important;
        box-shadow: 0 0 10px rgba(182, 43, 26, 0.22), inset 0 0 0 1px var(--gleam-inset) !important;
    }

    button[kind="primary"]:is(:hover, :focus-visible),
    button[kind="primary"]:is(:hover, :focus-visible)::before,
    button[kind="primary"]:is(:hover, :focus-visible)::after,
    button[data-testid="baseButton-primary"]:is(:hover, :focus-visible),
    button[data-testid="baseButton-primary"]:is(:hover, :focus-visible)::before,
    button[data-testid="baseButton-primary"]:is(:hover, :focus-visible)::after,
    .comic-shiny-btn:is(:hover, :focus-visible),
    .comic-shiny-btn:is(:hover, :focus-visible)::before,
    .comic-shiny-btn:is(:hover, :focus-visible)::after {
        animation-play-state: running !important;
    }

    @keyframes gradient-angle {
        to {
            --gradient-angle: 360deg;
        }
    }

    @keyframes shimmer {
        to {
            rotate: 360deg;
        }
    }

    /* Disabled State */
    button[kind="primary"]:disabled,
    button[data-testid="baseButton-primary"]:disabled,
    .comic-shiny-btn:disabled {
        opacity: 0.45 !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
        translate: 0 0 !important;
        border-color: #4d4d4d !important;
        background: #111111 !important;
    }
    button[kind="primary"]:disabled::before,
    button[kind="primary"]:disabled::after,
    button[data-testid="baseButton-primary"]:disabled::before,
    button[data-testid="baseButton-primary"]:disabled::after,
    .comic-shiny-btn:disabled::before,
    .comic-shiny-btn:disabled::after {
        animation: none !important;
        display: none !important;
    }

    /* Accessibility: respect prefers-reduced-motion */
    @media (prefers-reduced-motion: reduce) {
        button[kind="primary"],
        button[kind="primary"]::before,
        button[kind="primary"]::after,
        button[kind="primary"] div[data-testid="stMarkdownContainer"]::before,
        button[kind="primary"] p::before,
        button[data-testid="baseButton-primary"],
        button[data-testid="baseButton-primary"]::before,
        button[data-testid="baseButton-primary"]::after,
        .comic-shiny-btn,
        .comic-shiny-btn::before,
        .comic-shiny-btn::after,
        .comic-shiny-btn span::before {
            animation: none !important;
            transition: none !important;
        }

        button[kind="primary"]:is(:hover, :focus-visible),
        button[kind="primary"]:is(:hover, :focus-visible)::before,
        button[kind="primary"]:is(:hover, :focus-visible)::after,
        button[data-testid="baseButton-primary"]:is(:hover, :focus-visible),
        button[data-testid="baseButton-primary"]:is(:hover, :focus-visible)::before,
        button[data-testid="baseButton-primary"]:is(:hover, :focus-visible)::after,
        .comic-shiny-btn:is(:hover, :focus-visible),
        .comic-shiny-btn:is(:hover, :focus-visible)::before,
        .comic-shiny-btn:is(:hover, :focus-visible)::after {
            animation-play-state: paused !important;
        }

        button[kind="primary"]:is(:hover, :focus-visible) div[data-testid="stMarkdownContainer"]::before,
        button[kind="primary"]:is(:hover, :focus-visible) p::before,
        .comic-shiny-btn:is(:hover, :focus-visible) span::before {
            opacity: 0 !important;
        }
    }


    /* --- Home Meta Pills ------------------------------------------------- */
    .uploader-badges-wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        margin: 0 0 16px 0;
    }
    .uploader-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #000000;               /* Black Void */
        border: 1px solid #4d4d4d;        /* Smoke */
        border-radius: 9999px;
        padding: 5px 14px;
        font-size: 18px;                   /* body/base */
        font-weight: 380;
        line-height: 1.11;
        color: #4d4d4d;                   /* Smoke */
        letter-spacing: 0.5px;
    }
    .uploader-pill-accent {
        background: #000000;
        border-color: #ffffff;            /* Paper White -- highlights the "secure" badge */
        color: #ffffff;
    }
    .uploader-pill-comic {
        background: #000000;
        border-color: #b62b1a;           /* brand accent border */
        color: #b62b1a;
    }

    /* --- Sample launch ghost button ------------------------------------- */
    .sample-launch-btn button {
        background: #000000 !important;      /* Black Void */
        border: 1.5px solid #4d4d4d !important; /* Smoke */
        color: #ffffff !important;
        border-radius: 10px !important;
        font-size: 18px !important;          /* body/base */
        font-weight: 380 !important;
        line-height: 1.11 !important;
        padding: 8px 12px !important;
        transition: all 0.15s ease !important;
    }
    .sample-launch-btn button:hover {
        border-color: #b62b1a !important;    /* brand accent on hover */
        color: #b62b1a !important;
        background: rgba(182, 43, 26, 0.06) !important;
        transform: translateY(-1px) !important;
    }

    /* --- Modern Left Sidebar System -------------------------------------- */
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"] {
        background-color: var(--panel-bg, #111114) !important;
        border-right: 1px solid var(--panel-border, rgba(250, 250, 250, 0.12)) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.45) !important;
        transition: transform var(--panel-anim, 0.35s cubic-bezier(.4,0,.2,1)),
                    min-width var(--panel-anim, 0.35s cubic-bezier(.4,0,.2,1)),
                    max-width var(--panel-anim, 0.35s cubic-bezier(.4,0,.2,1)),
                    margin-left var(--panel-anim, 0.35s cubic-bezier(.4,0,.2,1)) !important;
    }
    section[data-testid="stSidebar"] hr {
        border: none !important;
        border-top: 1px solid var(--panel-border, rgba(250, 250, 250, 0.12)) !important;
        margin: 12px 0 16px 0 !important;
    }
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.8px !important;
        text-transform: uppercase !important;
        color: #71717a !important;
        margin: 14px 0 8px 0 !important;
    }

    /* Modern Expander matching reference sections */
    div[data-testid="stExpander"] {
        background-color: #141418 !important;
        border: 1px solid #1f1f25 !important;
        border-radius: 8px !important;
        margin-bottom: 8px !important;
        overflow: hidden !important;
    }
    div[data-testid="stExpander"]:hover {
        border-color: #2b2b36 !important;
    }
    div[data-testid="stExpander"] > details > summary {
        padding: 8px 12px !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #d4d4d8 !important;
    }
    div[data-testid="stExpander"] > details > summary:hover {
        color: #ffffff !important;
    }
    div[data-testid="stExpander"] > details > div {
        padding: 10px 12px !important;
        border-top: 1px solid #1f1f25 !important;
    }



    /* Modern Panel Header */
    .modern-panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 8px;
        margin-bottom: 8px;
    }
    .modern-panel-title-wrap {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .modern-panel-icon {
        color: #b62b1a;
        font-size: 15px;
        font-weight: 700;
    }
    .modern-panel-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        letter-spacing: 0.6px !important;
        text-transform: uppercase !important;
        color: #f4f4f5 !important;
    }
    .modern-badge-pill {
        background: #1f1f25;
        border: 1px solid #2b2b34;
        color: #a1a1aa;
        font-size: 11px;
        font-weight: 500;
        padding: 2px 7px;
        border-radius: 9999px;
    }

    /* Modern Section Header inside Right Panel */
    .modern-panel-section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: #71717a;
        margin: 10px 0 6px 0;
        padding: 4px 0;
    }


    /* Modern Bubble Cards in Right Panel */
    .modern-bubble-card {
        background: #16161a;
        border: 1px solid #23232a;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
        transition: border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
    }
    .modern-bubble-card:hover {
        border-color: #33333d;
        background: #18181d;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .modern-bubble-card.needs-review {
        border-color: rgba(239, 68, 68, 0.45);
        background: rgba(239, 68, 68, 0.04);
        box-shadow: 0 0 16px rgba(239, 68, 68, 0.12);
    }
    .modern-bubble-card.disabled {
        opacity: 0.45;
        border-style: dashed;
    }
    .mbc-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .mbc-tags {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .mbc-id {
        background: #23232c;
        border: 1px solid #2f2f3a;
        color: #f4f4f5;
        font-family: ui-monospace, monospace;
        font-weight: 700;
        font-size: 11px;
        padding: 2px 7px;
        border-radius: 4px;
        letter-spacing: 0.3px;
    }
    .modern-speaker-pill {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.22);
        color: #93c5fd;
        font-size: 11px;
        font-weight: 500;
        padding: 1px 7px;
        border-radius: 4px;
    }
    .modern-speaker-pill-muted {
        background: #1c1c22;
        border: 1px solid #272730;
        color: #71717a;
        font-size: 11px;
        font-weight: 400;
        padding: 1px 6px;
        border-radius: 4px;
    }
    .modern-review-pill {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fca5a5;
        font-size: 10px;
        font-weight: 700;
        padding: 1px 6px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    }
    .mbc-orig-dialogue {
        background: #0d0d10;
        border: 1px solid #1c1c23;
        border-radius: 6px;
        padding: 6px 9px;
        font-size: 12px;
        line-height: 1.35;
        color: #a1a1aa;
        margin-bottom: 6px;
        word-break: break-word;
    }
    .mbc-orig-label {
        font-family: ui-monospace, monospace;
        font-size: 10px;
        font-weight: 700;
        color: #52525b;
        background: #18181f;
        padding: 1px 4px;
        border-radius: 3px;
        margin-right: 4px;
    }
    .mbc-error-notice {
        margin-top: 6px;
        padding: 5px 8px;
        border-radius: 5px;
        background: rgba(239, 68, 68, 0.08);
        border-left: 3px solid #ef4444;
        color: #fca5a5;
        font-size: 12px;
        font-weight: 500;
    }

    /* Workspace breadcrumb matching reference header */
    .workspace-breadcrumb-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 0 12px 0;
        border-bottom: 1px solid #1f1f25;
        margin-bottom: 12px;
    }
    .workspace-breadcrumb {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 13px;
        color: #71717a;
    }
    .workspace-breadcrumb .wb-root {
        color: #e4e4e7;
        font-weight: 600;
    }
    .workspace-breadcrumb .wb-sep {
        color: #52525b;
    }
    .workspace-breadcrumb .wb-leaf {
        color: #a1a1aa;
    }
    .workspace-stage-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #71717a;
        background: #141418;
        border: 1px solid #23232a;
        padding: 2px 8px;
        border-radius: 9999px;
    }
    .wsi-dot {
        width: 6px;
        height: 6px;
        background: #b62b1a;
        border-radius: 50%;
        display: inline-block;
    }

    /* --- Hide Streamlit footer ------------------------------------------- */
    footer {
        display: none !important;
        visibility: hidden !important;
    }

    /* --- Suppress Streamlit element-toolbar green outline on markdown blocks */
    [data-testid="element-toolbar"],
    [data-testid="StyledFullScreenButton"],
    .stElementToolbar,
    div[data-testid="stElementToolbarButton"] {
        display: none !important;
    }
    /* Remove browser default focus outline from logo anchor */
    .comic-site-logo,
    .comic-site-logo:focus,
    .comic-site-logo:focus-visible {
        outline: none !important;
        box-shadow: none !important;
    }

    /* --- Speaker Detection Badge & Status --------------------------------- */
    .speaker-tag-wrap {
        display: flex;
        align-items: center;
        gap: 6px;
        margin: 2px 0 6px 0;
        flex-wrap: wrap;
    }
    .speaker-tag-detected {
        background: #000000;
        border: 1px solid #b62b1a;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.3px;
    }
    .speaker-tag-generic {
        background: #000000;
        border: 1px solid #4d4d4d;
        color: #888888;
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.3px;
    }
    .speaker-conf-pill {
        background: rgba(182, 43, 26, 0.15);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        padding: 2px 8px;
        border-radius: 9999px;
        border: 1px solid #b62b1a;
    }
    .speaker-conf-pill-muted {
        background: rgba(77, 77, 77, 0.2);
        color: #4d4d4d;
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        padding: 2px 8px;
        border-radius: 9999px;
        border: 1px solid #4d4d4d;
    }

    /* --- Sleek ProgressBar & Loading Indicator (Exact Port from shadcn React component) --- */
    div[data-testid="stProgress"] {
        margin: 10px 0 !important;
    }
    div[data-testid="stProgress"] > div {
        background: transparent !important;
    }
    div[data-testid="stProgress"] > div > div {
        background: #1D1D1A !important;
        border-radius: 4px !important;
        padding: 2px !important;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.45), inset 0 0 0 1px rgba(255, 255, 255, 0.06) !important;
        height: 12px !important;
        overflow: hidden !important;
    }
    div[data-testid="stProgress"] > div > div > div {
        border-radius: 2px !important;
        background: #b62b1a !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4), inset 0 -1px 0 rgba(0, 0, 0, 0.25) !important;
        height: 8px !important;
        transition: width 0.4s cubic-bezier(0.25, 1, 0.5, 1) !important;
    }

    div[data-testid="stFileUploader"] [role="progressbar"],
    div[data-testid="stFileUploaderProgressBar"] {
        background: #1D1D1A !important;
        border-radius: 4px !important;
        padding: 2px !important;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.45), inset 0 0 0 1px rgba(255, 255, 255, 0.06) !important;
        height: 12px !important;
    }
    div[data-testid="stFileUploader"] [role="progressbar"] > div,
    div[data-testid="stFileUploaderProgressBar"] > div {
        border-radius: 2px !important;
        background: #b62b1a !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4), inset 0 -1px 0 rgba(0, 0, 0, 0.25) !important;
        height: 8px !important;
    }

    div[data-testid="stSpinner"] > div {
        border-top-color: #b62b1a !important;
    }

    .comic-progress-wrap {
        width: 100%;
        margin: 10px 0;
        font-family: 'Inter', sans-serif;
    }
    .comic-progress-header {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 6px;
    }
    .comic-progress-label {
        font-size: 13px;
        font-weight: 500;
        color: #e7e5e4;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .comic-progress-status {
        font-size: 12px;
        font-weight: 500;
        color: #a8a29e;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-variant-numeric: tabular-nums;
        flex-shrink: 0;
    }
    .comic-progress-track {
        border-radius: 4px;
        background: #1D1D1A;
        padding: 2px;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.45), inset 0 0 0 1px rgba(255, 255, 255, 0.06);
    }
    .comic-progress-inner {
        position: relative;
        height: 8px;
        overflow: hidden;
        border-radius: 2px;
    }
    .comic-progress-fill {
        position: absolute;
        inset: 0;
        display: block;
        transform-origin: left;
        border-radius: 2px;
        background: #b62b1a;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4), inset 0 -1px 0 rgba(0, 0, 0, 0.25);
        transition: width 0.4s cubic-bezier(0.25, 1, 0.5, 1);
    }
    .comic-progress-indeterminate {
        position: absolute;
        top: 0;
        bottom: 0;
        left: 0;
        width: 40%;
        border-radius: 2px;
        background: #b62b1a;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4), inset 0 -1px 0 rgba(0, 0, 0, 0.25);
        animation: progress-indeterminate 1.25s ease-in-out infinite;
    }
    @keyframes progress-indeterminate {
        0% {
            transform: translateX(-100%);
            opacity: 0;
        }
        15% {
            opacity: 1;
        }
        85% {
            opacity: 1;
        }
        100% {
            transform: translateX(250%);
            opacity: 0;
        }
    }
</style>
""", unsafe_allow_html=True)


# Inject native browser Base64 @font-face and dropdown rules
st.markdown(get_dropdown_fonts_css(), unsafe_allow_html=True)

# Initialize Session State
sample_page_19 = os.path.join(_BASE_DIR, "samples", "page_19.png")
sample_page_18 = os.path.join(_BASE_DIR, "samples", "page_18.png")
default_comic_path = sample_page_19 if os.path.exists(sample_page_19) else sample_page_18

# Handle logo click / reset via query param
if "reset" in st.query_params:
    st.query_params.clear()
    reset_page_state(None, "")
    st.session_state.image = None
    st.session_state.image_name = ""

if "current_stage" not in st.session_state:
    st.session_state.current_stage = 1
if "image" not in st.session_state:
    st.session_state.image = None
if "image_name" not in st.session_state:
    st.session_state.image_name = ""
if "raw_bubbles" not in st.session_state:
    st.session_state.raw_bubbles = []
if "cleaned_page" not in st.session_state:
    st.session_state.cleaned_page = None
if "cleaned_page_bytes" not in st.session_state:
    st.session_state.cleaned_page_bytes = None
if "bubbles" not in st.session_state:
    st.session_state.bubbles = []
if "cleaned_preview" not in st.session_state:
    st.session_state.cleaned_preview = None
if "rendered_image" not in st.session_state:
    st.session_state.rendered_image = None
if "rendered_image_bytes" not in st.session_state:
    st.session_state.rendered_image_bytes = None
if "render_timestamp" not in st.session_state:
    st.session_state.render_timestamp = 0.0
if "saved_file_path" not in st.session_state:
    st.session_state.saved_file_path = ""
if "analysis_version" not in st.session_state:
    st.session_state.analysis_version = 0
if "last_render_hash" not in st.session_state:
    st.session_state.last_render_hash = ""
if "show_editor_panel" not in st.session_state:
    st.session_state.show_editor_panel = True

# Inject modern Right Drawer CSS on stAppViewContainer
inject_drawer_css()



# Sidebar Controls
with st.sidebar:
    star_b64 = get_comic_star_b64()
    st.markdown(
        f'<a href="/?reset=1" target="_self" class="comic-site-logo" title="Bosh sahifaga qaytish">'
        f'<img src="data:image/png;base64,{star_b64}" class="comic-logo-icon" alt="Logo" />'
        f'<span class="comic-logo-text">COMIC-LAB</span>'
        f'</a>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### Tipografiya va Lettering")

    available_fonts = engine.get_available_fonts()
    font_choice = st.selectbox(
        "Comic Lettering Font",
        list(available_fonts.keys()),
        index=0,
        key="font_choice_select",
        on_change=trigger_render,
        help="CC Wild Words is the industry standard comic font for Marvel and manga lettering."
    )
    selected_font_path = available_fonts[font_choice]

    # Inject ActiveComicFont CSS matching the exact format of selected_font_path (OTTO -> font/otf & format('opentype'))
    st.markdown(get_active_font_css(selected_font_path), unsafe_allow_html=True)

    # Editable Preview Text Box
    preview_text = st.text_input("Prevyu matni (Preview Text)", value="QOYIL QOLDINGMI?!")

    # Simplified Crisp Vector HTML/CSS Preview Block (100% native vector rendering, zero blur)
    cur_font_size = st.session_state.get("max_font_size_slider", 26)
    cur_line_gap = st.session_state.get("line_gap_slider", 6)
    clean_font_name = font_choice.replace(" (Recommended)", "")
    line_gap_mult = f"{1.15 + (cur_line_gap / max(1, cur_font_size)):.2f}"
    escaped_text = html.escape(preview_text or "QOYIL QOLDINGMI?!").upper()

    st.markdown(
        f'<div style="background: #141418; border: 1px solid #23232a; border-radius: 8px; padding: 14px; text-align: center; margin: 10px 0 14px 0;">'
        f'<div style="display: inline-flex; align-items: center; gap: 6px; font-size: 10px; color: #a1a1aa; background: #1c1c22; border: 1px solid #272730; padding: 2px 8px; border-radius: 9999px; margin-bottom: 8px; font-family: ui-monospace, monospace; letter-spacing: 0.5px;">'
        f'{clean_font_name.upper()} · {cur_font_size}PX · GAP {cur_line_gap}PX'
        f'</div>'
        f'<div class="comic-preview-text" style="font-size: {cur_font_size}px; line-height: {line_gap_mult}; color: #ffffff; text-shadow: 2px 2px 0px #000000, -1px -1px 0px #000000, 1px -1px 0px #000000, -1px 1px 0px #000000; word-break: break-word; text-transform: uppercase;">'
        f'{escaped_text}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    with st.expander("Qahramon Shaxsiyati (Voice)", expanded=False):
        st.caption("Avtomatik xarakter aniqlash (Classical CV) faol.")
        known_profiles = character_profiles.list_available_profiles()
        st.markdown(f"**Xotiradagi profillar:**<br>`{', '.join(known_profiles)}`", unsafe_allow_html=True)


    with st.expander("Tipografiya va Shrift Sozlamalari", expanded=True):
        max_font_size = st.slider(
            "Max Font Size (px)", min_value=14, max_value=46, value=26,
            key="max_font_size_slider", on_change=trigger_render,
            help="Maximum dialogue font size."
        )
        min_font_size = st.slider(
            "Min Font Size (px)", min_value=6, max_value=18, value=10,
            key="min_font_size_slider", on_change=trigger_render,
            help="Minimum font size threshold."
        )
        line_gap = st.slider(
            "Line Gap Safety (px)", min_value=0, max_value=20, value=6,
            key="line_gap_slider", on_change=trigger_render,
            help="Explicit spacing added between lines to completely prevent line collision."
        )

    # Dedicated Re-render Action (only shown when comic is loaded and bubbles exist)
    if st.session_state.image is not None and st.session_state.get("bubbles"):
        apply_settings_btn = st.button(
            "Sahifaga Shriftlarni Qayta Yozish",
            type="primary",
            use_container_width=True,
            help="Yangi shrift, o'lcham va oraliqlarni sahifadagi barcha pufakchalarga darhol qayta yozish."
        )

        if apply_settings_btn:
            trigger_render()
            st.toast("Shrift va sozlamalar sahifaga muvaffaqiyatli qo'llandi", icon=":material/check_circle:")
            st.rerun()

if st.session_state.image is None:
    star_b64 = get_comic_star_b64()
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align: center; margin-bottom: 24px;">'
        f'<div style="display: inline-flex; align-items: center; justify-content: center; gap: 16px;">'
        f'<img src="data:image/png;base64,{star_b64}" style="height: 80px; width: 80px; object-fit: contain;" />'
        f'<span style="font-family: \'Bangers\', cursive, sans-serif; font-style: italic; font-size: 3.8rem; color: #FF3366; letter-spacing: 3px; text-shadow: 3px 3px 0px #000000; line-height: 1;">COMIC-LAB</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="home-uploader-zone">'
        f'<div class="uploader-badges-wrap">'
        f'<span class="uploader-pill">PNG, JPG, WEBP</span>'
        f'<span class="uploader-pill">200MB gacha</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="home-uploader-zone">', unsafe_allow_html=True)
    home_file = st.file_uploader(
        "Komiks sahifasini yuklang",
        type=["png", "jpg", "jpeg", "webp"],
        key="home_file_uploader",
        label_visibility="collapsed",
        help="Komiks sahifasini bu yerga tashlang (drag & drop) yoki fayl tanlash orqali yuklang."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if home_file is not None:
        reset_page_state(Image.open(home_file).convert("RGB"), home_file.name)
        st.rerun()
else:
    # 3-Stage Visual Stepper Header matching reference pill design
    stage = st.session_state.get("current_stage", 1)
    has_raw = bool(st.session_state.get("raw_bubbles"))
    has_trans = bool(st.session_state.get("bubbles"))

    # Render Visual Capsule Stepper
    st.markdown(get_stepper_component(stage), unsafe_allow_html=True)

    # Modern Workspace Breadcrumb + Floating Symmetrical Toggle Button
    cur_doc_name = st.session_state.image_name or "comic_page"
    is_panel_open = st.session_state.get("show_editor_panel", True)

    # If right inspector drawer is closed, render floating symmetrical open button at top-right
    if not is_panel_open:
        st.button(
            ":material/keyboard_double_arrow_left:",
            key="drawer_open",
            on_click=set_drawer,
            args=(True,),
            help="Inspektorni ochish (Slide Open)",
        )

    # Modern Workspace Breadcrumb (full width)
    st.markdown(
        f'<div class="workspace-breadcrumb-bar">'
        f'<div class="workspace-breadcrumb">'
        f'<span class="wb-root">comic-lab</span>'
        f'<span class="wb-sep">/</span>'
        f'<span class="wb-leaf">{html.escape(cur_doc_name)}</span>'
        f'</div>'
        f'<div class="workspace-stage-indicator">'
        f'<span class="wsi-dot"></span> Bosqich {stage}/3'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    col_nav1, col_nav2, col_nav3 = st.columns(3)
    with col_nav1:
        if st.button("1. Skanerlash & Tozalash", type="primary" if stage == 1 else "secondary", use_container_width=True):
            st.session_state.current_stage = 1
            st.rerun()
    with col_nav2:
        if st.button("2. Tarjima & Tahrir", type="primary" if stage == 2 else "secondary", disabled=not has_raw, use_container_width=True):
            need_translate = not has_trans or (len(st.session_state.get("bubbles", [])) != len(st.session_state.raw_bubbles))
            if need_translate and has_raw:
                p_slot = st.empty()
                p_slot.markdown(
                    get_progress_bar_html(None, label="NLLB-200 AI Model", pending_label="O'zbek tiliga tarjima qilinmoqda..."),
                    unsafe_allow_html=True
                )
                translated = engine.translate_bubbles_list(st.session_state.raw_bubbles)
                st.session_state.bubbles = [BubbleDict(b.model_dump()) for b in translated]
                for b in st.session_state.bubbles:
                    b["pipeline_uzbek_translation"] = b.get("pipeline_uzbek_translation") or b.get("uzbek_translation", "")
                    b["last_captured_uzbek"] = b["pipeline_uzbek_translation"]
                    st.session_state[f"trans_{b.bubble_id}"] = b.uzbek_translation
                    st.session_state[f"active_{b.bubble_id}"] = b.is_active
                    st.session_state[f"nudge_{b.bubble_id}"] = b.font_size_offset
                p_slot.markdown(
                    get_progress_bar_html(100, label="NLLB-200 AI Model", complete_label="Tarjima yakunlandi"),
                    unsafe_allow_html=True
                )
                time.sleep(0.3)
                p_slot.empty()

            st.session_state.current_stage = 2
            st.rerun()
    with col_nav3:
        if st.button("3. Shriftlarni Yozish", type="primary" if stage == 3 else "secondary", disabled=not has_trans, use_container_width=True):
            sync_bubble_widgets()
            trigger_render()
            st.session_state.current_stage = 3
            st.rerun()

    st.markdown("---")

    # ----------------------------------------------------
    # STAGE 1: SCAN & CLEAN (Skanerlash va Tozalash)
    # ----------------------------------------------------
    if stage == 1:
        # --- Main Workspace: Full Width Canvas ---
        st.markdown("### Sahifa Ko'rinishi")
        if st.session_state.cleaned_page is not None:
            tab_clean, tab_orig = st.tabs([":material/cleaning_services: Tozalangan Sahifa", ":material/image: Asl Sahifa"])
            with tab_clean:
                show_markers = st.checkbox("Pufak chegaralarini ko'rsatish (Green Markers)", value=True, key="markers_st1")
                if show_markers and st.session_state.raw_bubbles:
                    overlay_key = f"overlay_st1_{len(st.session_state.raw_bubbles)}_{st.session_state.get('analysis_version', 0)}"
                    if overlay_key not in st.session_state:
                        st.session_state[overlay_key] = engine.draw_bounding_box_overlay(st.session_state.cleaned_page, st.session_state.raw_bubbles)
                    annotated = st.session_state[overlay_key]
                    st.image(annotated, use_container_width=True, caption=f"Tozalangan sahifa ({len(st.session_state.raw_bubbles)} ta pufak)")
                else:
                    st.image(st.session_state.cleaned_page, use_container_width=True, caption="Tozalangan sahifa (Siyoh Telea orqali tozalangan, to'rtburchak oq dog'siz)")
            with tab_orig:
                st.image(st.session_state.image, use_container_width=True, caption=st.session_state.image_name)
        else:
            st.image(st.session_state.image, use_container_width=True, caption="Asl sahifa (Hali skanerlanmagan)")

        # --- Right Drawer: Push/Slide Container ---
        with st.container(key="right_drawer"):
            p_cnt_str = f"{len(st.session_state.raw_bubbles)} ta pufak" if st.session_state.raw_bubbles else ""
            render_right_panel_header("Skanerlash & Tozalash", p_cnt_str, close_key="btn_close_panel_1")

            st.caption("EasyOCR pufaklarni aniqlaydi va qog'oz teksturasini buzmasdan faqat qora siyohni tozalaydi (Pure Ink Inpainting).")

            scan_label = "Qayta Skanerlash va Tozalash" if st.session_state.cleaned_page is not None else "1. Sahifani Skanerlash va Pufaklarni Tozalash"
            scan_type = "secondary" if st.session_state.cleaned_page is not None else "primary"
            scan_btn = st.button(scan_label, type=scan_type, use_container_width=True)

            if scan_btn:
                p_slot = st.empty()
                p_slot.markdown(
                    get_progress_bar_html(None, label=st.session_state.image_name or "Komiks sahifasi", pending_label="EasyOCR matnlarni tahlil qilmoqda..."),
                    unsafe_allow_html=True
                )
                try:
                    raw_bubbles = engine.scan_bubbles_ocr(st.session_state.image)
                    if not raw_bubbles:
                        p_slot.empty()
                        st.error("Hech qanday pufak aniqlanmadi!")
                    else:
                        p_slot.markdown(
                            get_progress_bar_html(65, label=st.session_state.image_name or "Komiks sahifasi", pending_label="Siyoh Telea orqali tozalanmoqda..."),
                            unsafe_allow_html=True
                        )
                        keys_to_clear = [k for k in list(st.session_state.keys()) if k.startswith("overlay_")]
                        for k in keys_to_clear:
                            del st.session_state[k]
                        st.session_state.analysis_version = st.session_state.get("analysis_version", 0) + 1
                        st.session_state.raw_bubbles = raw_bubbles
                        cleaned = engine.clean_page_ink_telea(st.session_state.image, raw_bubbles)
                        st.session_state.cleaned_page = cleaned
                        st.session_state.bubbles = []
                        st.session_state.rendered_image = None
                        st.session_state.rendered_image_bytes = None
                        import gc
                        gc.collect()
                        p_slot.markdown(
                            get_progress_bar_html(100, label=st.session_state.image_name or "Komiks sahifasi", complete_label=f"{len(raw_bubbles)} ta pufak tozalandi"),
                            unsafe_allow_html=True
                        )
                        time.sleep(0.3)
                        p_slot.empty()
                        st.toast(f"{len(raw_bubbles)} ta pufak topildi va sahifa tozalandi!", icon=":material/check_circle:")
                        st.rerun()
                except Exception as ex:
                    p_slot.empty()
                    st.error(f"Skanerlashda xatolik yuz berdi: {ex}")

            if st.session_state.cleaned_page is not None and st.session_state.raw_bubbles:
                if st.button("2-bosqich: Matnlarni Ko'rish va Tahrirlash", type="primary", use_container_width=True):
                    p_slot = st.empty()
                    p_slot.markdown(
                        get_progress_bar_html(None, label="NLLB-200 AI Model", pending_label="O'zbek tiliga tarjima qilinmoqda..."),
                        unsafe_allow_html=True
                    )
                    translated = engine.translate_bubbles_list(st.session_state.raw_bubbles)
                    st.session_state.bubbles = [BubbleDict(b.model_dump()) for b in translated]
                    for b in st.session_state.bubbles:
                        b["pipeline_uzbek_translation"] = b.get("pipeline_uzbek_translation") or b.get("uzbek_translation", "")
                        b["last_captured_uzbek"] = b["pipeline_uzbek_translation"]
                        st.session_state[f"trans_{b.bubble_id}"] = b.uzbek_translation
                        st.session_state[f"active_{b.bubble_id}"] = b.is_active
                        st.session_state[f"nudge_{b.bubble_id}"] = b.font_size_offset
                    p_slot.markdown(
                        get_progress_bar_html(100, label="NLLB-200 AI Model", complete_label="Tarjima yakunlandi"),
                        unsafe_allow_html=True
                    )
                    time.sleep(0.3)
                    p_slot.empty()
                    st.session_state.current_stage = 2
                    st.rerun()

                st.success(f"Tozalash muvaffaqiyatli! {len(st.session_state.raw_bubbles)} ta pufak topildi va matnlar to'liq o'chirildi.", icon=":material/check_circle:")
                
                st.markdown(
                    f'<div class="modern-panel-section-header"><span>ANIQLANGAN MATNLAR ({len(st.session_state.raw_bubbles)})</span></div>',
                    unsafe_allow_html=True
                )
                for b in st.session_state.raw_bubbles:
                    esc_orig = html.escape(b.original_text)
                    card_html = (
                        f'<div class="modern-bubble-card">'
                        f'<div class="mbc-header"><div class="mbc-tags"><span class="mbc-id">#{b.bubble_id}</span></div></div>'
                        f'<div class="mbc-orig-dialogue"><span class="mbc-orig-label">EN</span>{esc_orig}</div>'
                        f'</div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

    # ----------------------------------------------------
    # STAGE 2: TRANSLATE & REVIEW (Tarjima va Tahrir)
    # ----------------------------------------------------
    elif stage == 2:
        sync_bubble_widgets()

        # --- Main Workspace: Full Width Mask Editor ---
        canvas = st.session_state.cleaned_page if st.session_state.cleaned_page is not None else st.session_state.image
        active_sig = tuple((b.get("is_active", True) if isinstance(b, dict) else getattr(b, "is_active", True)) for b in st.session_state.bubbles)
        overlay_st2_key = f"overlay_st2_{len(st.session_state.bubbles)}_{hash(active_sig)}_{st.session_state.get('analysis_version', 0)}"
        if overlay_st2_key not in st.session_state:
            st.session_state[overlay_st2_key] = engine.draw_bounding_box_overlay(canvas, st.session_state.bubbles)
        annotated = st.session_state[overlay_st2_key]

        page_bgr = cv2.cvtColor(np.array(st.session_state.image), cv2.COLOR_RGB2BGR)
        cur_page_name = st.session_state.get("uploaded_file_name", "page.png")
        bubble_mask_editor.render_stage2_mask_editor(
            annotated_page_image=annotated,
            page_bgr=page_bgr,
            page_name=cur_page_name
        )

        # --- Right Drawer: Push/Slide Container ---
        with st.container(key="right_drawer"):
            p_cnt_str = f"{len(st.session_state.bubbles)} ta pufak" if st.session_state.bubbles else ""
            render_right_panel_header("Tarjima & Tahrirlash", p_cnt_str, close_key="btn_close_panel_2")

            st.caption("Tarjimalarni ko'rib chiqing va tahrirlang. Bu bosqichda og'ir grafik qayta ishlanmaydi.")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("1-bosqichga qaytish", use_container_width=True):
                    st.session_state.current_stage = 1
                    st.rerun()
            with col_btn2:
                unresolved = [
                    b for b in st.session_state.bubbles
                    if (b.get("needs_review", False) if isinstance(b, dict) else getattr(b, "needs_review", False))
                    and (b.get("is_active", True) if isinstance(b, dict) else getattr(b, "is_active", True))
                ]
                if unresolved:
                    st.markdown(
                        f'<div style="color: #ef4444; font-size: 12px; font-weight: 700; margin-bottom: 4px;">'
                        f'⚠️ {len(unresolved)} ta pufak ko\'rib chiqilishi kerak'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                if st.button("3-bosqich: Shriftlarni Yozish", type="primary", use_container_width=True):
                    sync_bubble_widgets()
                    trigger_render()
                    st.session_state.current_stage = 3
                    st.rerun()

            st.markdown("---")
            render_bubble_editor_panel(live_render=False, key_suffix="st2")

    # ----------------------------------------------------
    # STAGE 3: LETTERING & EXPORT (Yakuniy Lettering)
    # ----------------------------------------------------
    elif stage == 3:
        sync_bubble_widgets()
        if st.session_state.rendered_image is None:
            trigger_render()

        # --- Main Workspace: Full Width Lettering Result ---
        st.markdown("### Yakuniy Lettering Natijasi")
        tab_res, tab_clean, tab_orig = st.tabs([":material/auto_awesome: Jonli Natija", ":material/cleaning_services: Tozalangan Sahifa", ":material/image: Asl Sahifa"])
        with tab_res:
            img_data = st.session_state.get("rendered_image_bytes") or st.session_state.rendered_image
            st.image(img_data, use_container_width=True, caption="Jonli Lettering (CC Wild Words, Solid Black, Zero Stroke)")
        with tab_clean:
            st.image(st.session_state.cleaned_page, use_container_width=True, caption="1-bosqichda tozalangan sahifa")
        with tab_orig:
            st.image(st.session_state.image, use_container_width=True, caption="Asl sahifa")

        # --- Right Drawer: Push/Slide Container ---
        with st.container(key="right_drawer"):
            p_cnt_str = f"{len(st.session_state.bubbles)} ta pufak" if st.session_state.bubbles else ""
            render_right_panel_header("Shriftlar & Jonli Tahrir", p_cnt_str, close_key="btn_close_panel_3")

            col_back, col_re = st.columns(2)
            with col_back:
                if st.button("Matnlarni Qayta Tahrirlash (2-bosqich)", use_container_width=True):
                    st.session_state.current_stage = 2
                    st.rerun()
            with col_re:
                if st.button("Jonli Qayta Chizish", type="primary", use_container_width=True):
                    trigger_render()
                    st.rerun()

            st.markdown("---")
            st.markdown("#### Yuklab olish va Saqlash")

            if st.session_state.rendered_image is not None:
                unresolved = [
                    b for b in st.session_state.bubbles
                    if (b.get("needs_review", False) if isinstance(b, dict) else getattr(b, "needs_review", False))
                    and (b.get("is_active", True) if isinstance(b, dict) else getattr(b, "is_active", True))
                ]
                if unresolved:
                    st.markdown(
                        f'<div style="border: 1.5px solid #ef4444; background: rgba(239, 68, 68, 0.12); border-radius: 8px; padding: 10px 12px; margin-bottom: 12px; color: #fca5a5; font-size: 13px; line-height: 1.4;">'
                        f'⚠️ <strong>Eksport Ogohlantirishi:</strong> {len(unresolved)} ta pufakda tarjima xatosi/qoldiq aniqlangan. Ushbu pufaklar komiksda asl inglizcha tasvir holatida tegilmasdan saqlanadi (xom matn ustiga yozilmaydi).'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                png_bytes = st.session_state.get("rendered_image_bytes") or pil_to_png_bytes(st.session_state.rendered_image)
                base_name = os.path.splitext(st.session_state.image_name)[0] or "comic"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dl_filename = f"{base_name}_uzbek_{timestamp}.png"

                st.download_button(
                    label="Yuklab Olish (Download HD PNG)",
                    data=png_bytes,
                    file_name=dl_filename,
                    mime="image/png",
                    type="primary",
                    use_container_width=True
                )

                if st.button("Server 'output/' papkasiga saqlash", use_container_width=True):
                    os.makedirs("output", exist_ok=True)
                    out_path = os.path.abspath(os.path.join("output", dl_filename))
                    st.session_state.rendered_image.save(out_path, format="PNG")
                    st.session_state.saved_file_path = out_path
                    st.success(f"Komiks saqlandi: output/{dl_filename}", icon=":material/check_circle:")

            st.markdown(
                '<div style="background: #16161a; border: 1px solid #23232a; border-left: 3px solid #b62b1a; border-radius: 6px; padding: 8px 12px; margin: 12px 0 10px 0; font-size: 12px; line-height: 1.4; color: #a1a1aa;">'
                '<strong style="color: #ffffff;">Jonli Tahrirlash:</strong> Matnni yoki shrift o\'lchamini o\'zgartirsangiz, komiks sahifasi avtomatik qayta chiziladi.'
                '</div>',
                unsafe_allow_html=True
            )

            render_bubble_editor_panel(live_render=True, key_suffix="st3")


