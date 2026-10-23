"""
gemini_enhancements.py
----------------------
High-performance, free-tier Google Gemini integration for Comic-Lab:
1. Ultra-fast Multimodal Vision extraction (Speech bubbles, bounding boxes, speaker detection).
2. Context-aware, natural Uzbek dialogue translation with character voice preservation.
3. Resilient networking with forced IPv4 (bypasses Windows IPv6 latency/hangs) and automatic retries.
4. Seamless fallback: if network is unavailable or quota is exceeded, pipeline falls back to local engine.
"""

import os
import sys
import io
import json
import time
import base64
import socket
import re
import hashlib
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image

# Gemini API Disk & RAM Caching
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "gemini")
try:
    os.makedirs(_CACHE_DIR, exist_ok=True)
except Exception:
    pass

_MEMORY_CACHE: Dict[str, str] = {}

def get_cache_key(prompt: str, image: Optional[Image.Image] = None) -> str:
    """Generates a collision-resistant SHA256 cache key for prompt + image."""
    h = hashlib.sha256()
    h.update(prompt.strip().encode("utf-8"))
    try:
        import glossary_manager
        h.update(glossary_manager.get_glossary_hash().encode("utf-8"))
    except Exception:
        pass
    if image is not None:
        try:
            w, h_dim = image.size
            h.update(f"{w}x{h_dim}".encode("utf-8"))
            # High-resolution 256x256 bilinear thumbnail for robust uniqueness across comic pages
            thumb = image.resize((256, 256), Image.Resampling.BILINEAR).convert("RGB")
            h.update(thumb.tobytes())
        except Exception:
            try:
                raw = image.convert("RGB").tobytes()
                h.update(raw[:32768])
                h.update(raw[-32768:])
            except Exception:
                pass
    return h.hexdigest()

def clear_gemini_cache():
    """Flushes RAM cache and purges disk cache files."""
    global _MEMORY_CACHE
    _MEMORY_CACHE.clear()
    if os.path.exists(_CACHE_DIR):
        for f in os.listdir(_CACHE_DIR):
            if f.endswith(".json"):
                try:
                    os.remove(os.path.join(_CACHE_DIR, f))
                except Exception:
                    pass

def get_gemini_cache_stats() -> Dict[str, Any]:
    """Returns number of cached entries and storage footprint."""
    count = 0
    size_bytes = 0
    if os.path.exists(_CACHE_DIR):
        for f in os.listdir(_CACHE_DIR):
            if f.endswith(".json"):
                count += 1
                try:
                    size_bytes += os.path.getsize(os.path.join(_CACHE_DIR, f))
                except Exception:
                    pass
    return {
        "ram_entries": len(_MEMORY_CACHE),
        "disk_entries": count,
        "size_kb": round(size_bytes / 1024, 1)
    }

# Enforce IPv4 socket resolution to prevent Windows IPv6 connection timeouts to Google APIs
_ORIG_GETADDRINFO = socket.getaddrinfo

def _getaddrinfo_v4(host, port, family=0, type=0, proto=0, flags=0):
    return _ORIG_GETADDRINFO(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = _getaddrinfo_v4

# Default Models (ranked by active quota, speed, and multimodal capabilities)
GEMINI_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
]

def get_config_file_path() -> str:
    """Returns persistent JSON config file path in %APPDATA%/ComicLab."""
    appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
    target_dir = os.path.join(appdata, "ComicLab")
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, "config.json")

def load_user_config() -> Dict[str, Any]:
    """Loads persistent user configuration safely."""
    cfg_path = get_config_file_path()
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                return content if isinstance(content, dict) else {}
        except Exception:
            return {}
    return {}

def save_user_api_key(api_key: str) -> bool:
    """Saves user API key to persistent AppData config and activates in os.environ."""
    api_key = api_key.strip().strip('"').strip("'")
    if not api_key:
        return False
    cfg_path = get_config_file_path()
    try:
        cfg = load_user_config()
        cfg["gemini_api_key"] = api_key
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        os.environ["GEMINI_API_KEY"] = api_key
        return True
    except Exception:
        return False

def delete_user_api_key() -> bool:
    """Removes saved API key from persistent AppData config and clears os.environ."""
    cfg_path = get_config_file_path()
    try:
        cfg = load_user_config()
        if "gemini_api_key" in cfg:
            del cfg["gemini_api_key"]
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ.pop("GEMINI_API_KEYS", None)
        return True
    except Exception:
        return False

def test_gemini_key(api_key: str) -> Tuple[bool, str]:
    """Tests if a Gemini API key is valid by making a fast lightweight query to Google AI API."""
    api_key = api_key.strip().strip('"').strip("'")
    if not api_key or len(api_key) < 15:
        return False, "Kalit formati noto'g'ri yoki juda qisqa."
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        import urllib.request
        import urllib.error
        req = urllib.request.Request(url, headers={"User-Agent": "ComicLab/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            if resp.status == 200:
                return True, "API kalit muvaffaqiyatli tekshirildi va faol."
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return False, "Google API xatosi: API kalit noto'g'ri (Invalid API key)."
        elif e.code == 403:
            return False, "Google API xatosi: Kalitga ruxsat cheklangan yoki bloklangan."
        return False, f"Google API xatosi: HTTP {e.code}"
    except Exception as e:
        return False, f"Ulanish xatosi: {str(e)}"
    return False, "Kalitni tekshirib bo'lmadi."

def get_gemini_api_keys() -> List[str]:
    """Retrieves all configured Gemini API keys (supports multiple keys comma-separated or multi-variable)."""
    keys = []
    # 1. Environment variables
    for var in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEYS"]:
        raw = os.getenv(var)
        if raw:
            for k in raw.split(","):
                k = k.strip().strip('"').strip("'")
                if k and len(k) >= 15 and k not in keys:
                    keys.append(k)

    # 2. User AppData config JSON
    cfg = load_user_config()
    cfg_key = cfg.get("gemini_api_key")
    if cfg_key and isinstance(cfg_key, str):
        for k in cfg_key.split(","):
            k = k.strip().strip('"').strip("'")
            if k and len(k) >= 15 and k not in keys:
                keys.append(k)

    # 2. .env files from multiple candidate locations
    candidate_paths = [
        os.path.join(os.path.dirname(sys.executable), ".env"),
        os.path.join(os.environ.get("APPDATA") or "", "ComicLab", ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.getcwd(), ".env")
    ]
    seen_paths = set()
    for env_path in candidate_paths:
        if not env_path or env_path in seen_paths or not os.path.exists(env_path):
            continue
        seen_paths.add(env_path)
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#"):
                        continue
                    if any(line.startswith(prefix) for prefix in ["GEMINI_API_KEY=", "GOOGLE_API_KEY=", "GEMINI_API_KEYS="]):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        for k in val.split(","):
                            k = k.strip().strip('"').strip("'")
                            if k and len(k) >= 15 and k not in keys:
                                keys.append(k)
        except Exception:
            pass

    return keys


def get_gemini_api_key() -> Optional[str]:
    """Retrieves primary Gemini API key."""
    keys = get_gemini_api_keys()
    return keys[0] if keys else None


def is_gemini_enabled() -> bool:
    """Returns True if a valid Gemini API key is configured and not explicitly disabled."""
    keys = get_gemini_api_keys()
    return bool(len(keys) > 0)


def call_gemini_api(
    prompt: str,
    image: Optional[Image.Image] = None,
    api_key: Optional[str] = None,
    timeout: int = 45,
    max_retries: int = 3,
    response_json: bool = False,
    use_cache: bool = True
) -> str:
    """
    Executes a resilient call to Google Gemini API with automatic model fallback, retries,
    and instantaneous Disk + RAM caching.
    """
    cache_key = get_cache_key(prompt, image)
    if use_cache:
        # 1. Immediate RAM cache lookup (0.0001s)
        if cache_key in _MEMORY_CACHE:
            print(f"[GEMINI_CACHE] RAM keshdan olindi (key: {cache_key[:8]})", flush=True)
            return _MEMORY_CACHE[cache_key]

        # 2. Local Disk cache lookup (0.001s)
        cache_path = os.path.join(_CACHE_DIR, f"{cache_key}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    txt = cached.get("text")
                    if txt:
                        _MEMORY_CACHE[cache_key] = txt
                        print(f"[GEMINI_CACHE] Disk keshdan olindi (key: {cache_key[:8]})", flush=True)
                        return txt
            except Exception:
                pass

    active_keys = [api_key] if api_key else get_gemini_api_keys()
    if not active_keys:
        raise ValueError("Gemini API key is not configured.")

    parts: List[Dict[str, Any]] = [{"text": prompt}]

    if image is not None:
        # Resize image for optimal bandwidth & Gemini Vision inference (max dimension 1280px)
        img_rgb = image.convert("RGB")
        w, h = img_rgb.size
        scale = min(1.0, 1280.0 / max(w, h))
        if scale < 1.0:
            img_to_send = img_rgb.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        else:
            img_to_send = img_rgb

        buf = io.BytesIO()
        img_to_send.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img_b64
            }
        })

    payload: Dict[str, Any] = {
        "contents": [{"parts": parts}]
    }
    if response_json:
        payload["generationConfig"] = {
            "response_mime_type": "application/json"
        }
    data_bytes = json.dumps(payload).encode("utf-8")

    last_error = None
    for key in active_keys:
        for model_name in GEMINI_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
            for attempt in range(1, max_retries + 1):
                try:
                    req = urllib.request.Request(
                        url,
                        data=data_bytes,
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        res = json.loads(resp.read().decode("utf-8"))
                        candidates = res.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            c_parts = candidates[0]["content"].get("parts", [])
                            if c_parts and "text" in c_parts[0]:
                                res_text = str(c_parts[0]["text"]).strip()
                                # Save to memory and disk cache
                                if cache_key:
                                    _MEMORY_CACHE[cache_key] = res_text
                                    try:
                                        cache_path = os.path.join(_CACHE_DIR, f"{cache_key}.json")
                                        with open(cache_path, "w", encoding="utf-8") as cf:
                                            json.dump({"key": cache_key, "text": res_text, "created_at": time.time()}, cf, ensure_ascii=False)
                                    except Exception:
                                        pass
                                return res_text
                        raise ValueError(f"No text candidate returned by Gemini ({model_name})")
                except urllib.error.HTTPError as he:
                    last_error = he
                    if he.code == 404:
                        print(f"[GEMINI_WARN] Model {model_name} HTTP 404 (Not Found). Trying next model...", flush=True)
                        break
                    if he.code in [429, 500, 503] and attempt < max_retries:
                        sleep_time = 0.8 * (2 ** (attempt - 1))
                        print(f"[GEMINI_RETRY] Model {model_name} HTTP {he.code}, retrying in {sleep_time:.1f}s (attempt {attempt}/{max_retries})...", flush=True)
                        time.sleep(sleep_time)
                        continue
                    # If model is exhausted or fails, fall through to next model in GEMINI_MODELS
                    print(f"[GEMINI_WARN] Model {model_name} failed with HTTP {he.code}. Trying next model...", flush=True)
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        sleep_time = 0.8 * attempt
                        print(f"[GEMINI_RETRY] Model {model_name} {type(e).__name__}, retrying in {sleep_time:.1f}s (attempt {attempt}/{max_retries})...", flush=True)
                        time.sleep(sleep_time)
                        continue
                    print(f"[GEMINI_WARN] Model {model_name} failed with {type(e).__name__}. Trying next model...", flush=True)
                    break

    raise RuntimeError(f"Gemini API request failed across all models: {last_error}")


UZBEK_QUALITY_PROMPT_INSTRUCTIONS = (
    "O'ZBEK TILI IMILOVIY VA GRAMMATIK QOIDALARI (MUTLAQO AMAL QILINSIN):\n"
    "1. 'H' VA 'X' HARFLARINI ANIQLIK BILAN ISHLATING:\n"
    "   - 'H' bilan: hozir, har doim, har kim, har bir, har qanday, har narsa, hech qachon, hech kim, hech narsa, hamma, ham, hatto, harakat, hayot, haqiqat, haqiqatan, haq, rahmat, mehnat, mashhur, sahna, muhim, himoya, hujum, his qilmoq, hurmat, shubhasiz, nahotki, shahar, bahor.\n"
    "   - 'X' bilan: xato, xavf, xavfli, xabar, xohlamoq (xohlayman, xohlaysan), xursand, xizmat, xayol, xayr, xalos, xarid, xasis, xiyonat, xos, xotira, xotirjam, xush kelibsiz, xona.\n"
    "2. TO'G'RI TUTUQ BELGISI VA APOSTROFLAR (Faqat standart ASCII ' belgisidan foydalaning):\n"
    "   - to'g'ri, og'ir, bo'ldi, bo'ladi, bo'lgan, bo'lsa, ko'rdi, ko'raman, o'zim, o'zing, o'zi, o'zimiz, bog'liq, yolg'on, o'tmish, o'g'il, o'g'irlamoq, o'ldi, o'yin, o'rganmoq, o'rin, o'chir, o'zgarish, e'tibor, mo'jiza, ma'lum, ma'no.\n"
    "3. IKKILANGAN UNDOSHLAR:\n"
    "   - 'albatta' (HECH QACHON 'albatda' yoki 'albata' EMAS).\n"
    "4. SO'ZMA-SO'Z KALKALARDAN QOCHING:\n"
    "   - 'mening uchun', 'sening uchun' EMAS -> 'men uchun', 'sen uchun' deb yozing.\n"
    "   - Keraksiz kishilik olmoshlarini ('Men buni ko'rdim' o'rniga 'Buni ko'rdim') tushirib qoldiring.\n"
    "5. QISQARTIRILGAN JONLI FE'LLAR (KOMIKSLARGA XOS JONLI TILDAGI SHAKLLARI):\n"
    "   - 'qila olmayman' -> 'qilolmayman', 'qila olmaysan' -> 'qilolmaysan', 'qila olmaydi' -> 'qilolmaydi'\n"
    "   - 'ko'ra olmayman' -> 'ko'rolmayman', 'bila olmayman' -> 'bilolmayman', 'topa olmayman' -> 'topolmayman'\n"
    "   - 'ayta olmayman' -> 'aytolmayman', 'chidolmayman', 'ishonolmayman', 'to'xtatolmayman'\n"
    "   - 'bilar edim' -> 'bilardim', 'aytgan edim' -> 'aytgandim', 'kelgan edim' -> 'kelgandim'.\n"
    "6. QAHRAMON OHANGI VA HISSISI:\n"
    "   - Peter Parker / O'rgimchak-odam: chaqqon, hazilkash, kattalarga hurmat bilan 'SIZ'.\n"
    "   - J. Jonah Jameson: jahldor, shovqin-suronli, Nyu-York gazetchisi ohangi.\n"
    "   - Urg'u va hissiy belgilarni (!, ?, --, ...) asl nusxadek to'liq saqlang.\n"
    "7. KOMIKS FRAZEOLOGIZMLARI VA IDIOMALARINI TABIIYLASHTIRING:\n"
    "   - 'Put him on ice' -> 'Uni muzlatib qo'ying' ('Unga muzlab turish kerak' deb xato qilmang).\n"
    "   - 'Hook him up' -> 'Uni kishanlang' / 'Uni ulang'.\n"
    "   - 'One and the same' -> 'Bitta narsadir' / 'Aynan bir narsadir' ('Bir jondir' deb xato qilmang).\n"
    "   - 'Schedule an infusion and wake him up' -> 'Infuziya tayinlab, keyin uni uyg'otamiz'.\n"
    "8. KOMIKSLARGA XOS EMOTSIONAL URG'U (SEMANTIC EMPHASIS):\n"
    "   - Har bir gapda dialog ritmini va his-tuyg'uni belgilovchi 1-2 ta eng muhim so'zni (inkor, kuchaytiruvchi yuklama, fe'l yoki qahramon ismini) markdown formatida **so'z** ko'rinishida belgilang (masalan: 'Biz buni **albatta** bajaramiz!', 'Men **hech qachon** taslim bo'lmayman!', '**Piter**, orqangga qara!').\n"
    "   - QAT'IY QOIDA: Butun gapni HECH QACHON to'liq bold qilmang! Faqat 1-2 ta markaziy so'z urg'u bilan ajratilsin.\n"
)


def gemini_translate_dialogue(text: str, speaker: Optional[str] = None, context: Optional[str] = None, use_cache: bool = True) -> str:
    """
    Translates comic dialogue into natural, colloquial Uzbek using Gemini.
    Preserves speaker character voice, respectful 'SIZ' address when appropriate, and comic slang.
    Enforces team-wide persistent glossary rules.
    """
    try:
        import engine
        if engine.is_pure_sfx_text(text):
            return text
    except Exception:
        pass

    import glossary_manager
    glossary_instr = glossary_manager.format_glossary_prompt()
    spk_info = f"Speaker: {speaker}. " if speaker else ""
    ctx_info = f"Context: {context}. " if context else ""
    prompt = (
        "Translate comic dialogue to natural Uzbek.\n"
        f"{spk_info}{ctx_info}"
        f"{glossary_instr}\n"
        f"{UZBEK_QUALITY_PROMPT_INSTRUCTIONS}\n"
        "Rules: Colloquial punchy Uzbek, preserve character voice (Jameson fiery, Peter Parker respectful 'SIZ'), keep (!, ?, --).\n"
        "Avoid word-by-word literal translations (e.g. 'That\\'s me.' -> 'Aynan o\\'zim.' / 'O\\'sha menman.', never 'Bu men'; 'Do you?' -> 'Shundaymi?' / 'Rostdanmi?'; 'You\\'re the rocket lady.' -> 'Siz o\\'sha raketachi xonimsiz.'). Return ONLY translation.\n\n"
        f"Text:\n{text}"
    )
    result = call_gemini_api(prompt, use_cache=use_cache)
    # Clean up quotes or markdown wrapper if returned
    clean = result.strip().strip('"').strip("'").strip("`")
    if clean.startswith("Tarjima:") or clean.startswith("O'zbekcha:"):
        clean = clean.split(":", 1)[1].strip()
    try:
        import naturalization_rules
        clean = naturalization_rules.apply_naturalization(clean, en_orig=text, character_profile=speaker)
    except Exception:
        pass
    clean = glossary_manager.apply_glossary_enforcement(clean)
    try:
        import auto_emphasis
        clean = auto_emphasis.apply_auto_emphasis(clean, en_orig=text)
    except Exception:
        pass
    return clean


def gemini_translate_dialogues_batch(items: List[Dict[str, Any]]) -> List[str]:
    """
    Translates multiple comic dialogue bubbles in a SINGLE unified Gemini call.
    Drastically reduces API calls and eliminates 429 rate limit errors.
    items: list of dicts with {"text": str, "speaker": Optional[str]}
    Returns list of Uzbek translations corresponding 1-to-1 with input items.
    """
    if not items:
        return []
    import glossary_manager
    glossary_instr = glossary_manager.format_glossary_prompt()
    lines = []
    for idx, it in enumerate(items, 1):
        txt = it.get("text", "").strip()
        spk = it.get("speaker") or "Unknown"
        context_hint = ""
        if idx > 1:
            prev_txt = items[idx-2].get("text", "").strip()
            if (prev_txt.endswith("...") or prev_txt.endswith("…") or prev_txt.endswith("--") or
                txt.startswith("...") or txt.startswith("…") or txt.startswith("--") or
                (not prev_txt.endswith((".", "!", "?")) and txt and (txt[0].islower() or txt.split()[0].lower() in ("of", "the", "and", "about", "for", "to", "in", "on", "with")))):
                context_hint = f" [NOTE: Sentence continuation of bubble {idx-1}]"
        lines.append(f'{idx}. [{spk}]: "{txt}"{context_hint}')
    dialogues_text = "\n".join(lines)

    prompt = (
        "Translate each of these numbered comic dialogue bubbles into authentic, natural Uzbek.\n"
        f"{glossary_instr}\n"
        f"{UZBEK_QUALITY_PROMPT_INSTRUCTIONS}\n"
        "MULTI-BUBBLE SENTENCE CONTINUATION RULE:\n"
        "- When dialogue spans multiple bubbles (e.g. ends with '...' or marked [NOTE: Sentence continuation]), maintain fluent sentence coherence across the bubbles!\n"
        "- NEVER translate a continuation fragment in isolation as a dangling literal calque (e.g. '...of the beast' must NEVER be translated as 'HAYVONNING', but as '...maxluq tabiati haqida' or '...maxluq haqida').\n"
        'Return strictly JSON in format: {"translations": ["uzbek 1", "uzbek 2", ...]}\n'
        "Rules: The 'translations' array MUST have exactly the same length and order as input.\n\n"
        f"Dialogues:\n{dialogues_text}"
    )
    try:
        res = call_gemini_api(prompt, timeout=35, response_json=True)
        clean = extract_clean_json_string(res)
        data = json.loads(clean)
        translations = data.get("translations", [])
        if isinstance(translations, list) and len(translations) == len(items):
            results = []
            for tr, it in zip(translations, items):
                t_str = str(tr).strip().strip('"').strip("'")
                try:
                    import naturalization_rules
                    t_str = naturalization_rules.apply_naturalization(t_str, en_orig=it.get("text", ""), character_profile=it.get("speaker"))
                except Exception:
                    pass
                t_str = glossary_manager.apply_glossary_enforcement(t_str)
                try:
                    import auto_emphasis
                    t_str = auto_emphasis.apply_auto_emphasis(t_str, en_orig=it.get("text", ""))
                except Exception:
                    pass
                results.append(t_str)
            return results
    except Exception as e:
        print(f"[GEMINI_BATCH_WARN] Batch translation error: {e}", flush=True)

    # Fallback: translate individually
    results = []
    for it in items:
        try:
            tr = gemini_translate_dialogue(it.get("text", ""), speaker=it.get("speaker"))
            results.append(tr)
        except Exception:
            results.append("")
    return results


def extract_clean_json_string(raw_text: str) -> str:
    """
    Robustly extracts the primary JSON string from LLM response.
    Handles:
    - Enclosed ```json ... ``` or ``` ... ``` code blocks
    - Bare JSON with only trailing ```
    - Leading or trailing commentary text
    - Trailing commas before closing braces/brackets
    """
    text = (raw_text or "").strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if match:
        inner = match.group(1).strip()
        if inner.startswith("{") or inner.startswith("["):
            return re.sub(r',\s*([\]\}])', r'\1', inner)

    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    first_brace = text.find('{')
    first_bracket = text.find('[')

    start_idx = -1
    end_idx = -1
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start_idx = first_brace
        end_idx = text.rfind('}')
    elif first_bracket != -1:
        start_idx = first_bracket
        end_idx = text.rfind(']')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx + 1]

    return re.sub(r',\s*([\]\}])', r'\1', text.strip())


def gemini_extract_comic_bubbles(image: Image.Image, use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Analyzes a full comic page with Gemini Vision:
    Identifies all authentic speech bubbles, their bounding boxes (scaled to image resolution),
    speakers, original text, and natural Uzbek translations in one unified call.
    Uses Gemini JSON output mode for 100% structured reliability.
    Enforces team-wide persistent glossary rules.
    """
    import glossary_manager
    glossary_instr = glossary_manager.format_glossary_prompt()
    prompt = (
        "Extract all dialogue speech bubbles and narrative monologue/story captions from this comic page.\n"
        f"{glossary_instr}\n"
        f"{UZBEK_QUALITY_PROMPT_INSTRUCTIONS}\n"
        'Output JSON strictly as: {"bubbles": [{"box_2d": [ymin, xmin, ymax, xmax], "speaker": "str", "original_text": "str", "uzbek_translation": "str", "bubble_type": "speech"|"caption"}]}\n'
        "CRITICAL: Each bubble must be an independent JSON object with its OWN keys: 'box_2d', 'speaker', 'original_text', 'uzbek_translation', 'bubble_type'. NEVER omit the keys and NEVER merge multiple bubbles into a single array.\n"
        "Rules:\n"
        "- box_2d: normalized 0-1000 integers [ymin, xmin, ymax, xmax] tightly enclosing EACH bubble or caption box's full outer border (including all text lines). CRITICAL: Do NOT shift box coordinates into adjacent walls, counters, or characters!\n"
        "- FULL-WIDTH CAPTION RULE: For wide rectangular narrative monologue/story caption boxes spanning across panels, ensure box_2d encloses the ENTIRE rectangle horizontally from its left outer border all the way across to its right outer border. NEVER truncate wide boxes or cut off dialogue/monologue text on the right side.\n"
        "- MANDATORY: Extract EVERY authentic spoken dialogue bubble AND narrative monologue/caption box (whether rounded bubbles or rectangular caption/monologue boxes, including character inner thoughts, questions, or declarations like 'WHO AM I?', 'I AM DOCTOR DOOM...'), including short 1-2 word calls (e.g. 'No...', 'Yes!', 'Look!'), across ALL panels strictly from top to bottom, left to right.\n"
        "- SEQUENTIAL PANEL ORDER: Read panels strictly in comic reading order. Never hallucinate or copy dialogues from earlier panels into later panels unless the character genuinely repeats that exact phrase.\n"
        "- MANDATORY CONJOINED BUBBLE RULE: If two or more bubble lobes touch, chain, or connect in a figure-8 / double-lobed / peanut shape (stacked vertically or placed horizontally), you MUST extract EACH LOBE AS ITS OWN SEPARATE OBJECT in the 'bubbles' list with its own box_2d, original_text, and uzbek_translation! NEVER merge conjoined lobes into a single box! Each lobe holds its own separate sentence/thought and must be an independent bubble.\n"
        "- MANDATORY MULTI-BUBBLE RULE: When a character speaks in multiple distinct bubbles within the same panel, NEVER merge them into one box. Output each distinct bubble as its own separate entry.\n"
        "- NATURAL COLLOQUIAL UZBEK: Translate dialogues in relation to the conversational back-and-forth between characters. Avoid robotic word-for-word calques (e.g. 'That\\'s me.' -> 'Aynan o\\'zim.' or 'O\\'sha menman.', never 'Bu men'; 'Do you?' -> 'Shundaymi?' or 'Rostdanmi?'; 'You\\'re the rocket lady.' -> 'Siz o\\'sha raketachi xonimsiz.'). Keep (!, ?, --).\n"
        "- ONLY extract spoken character dialogue bubbles and narrator/monologue story captions.\n"
        "- DO NOT extract raw un-boxed environmental text or standalone background scene text (like 'YEARS AGO.' floating on background art without an enclosing speech bubble or narrative box). Only extract text that is enclosed inside a clear speech bubble or narrative caption/monologue box.\n"
        "- IGNORE in-scene environmental text, architectural signs, establishment names, building headers, street plaques, room numbers, storefronts, and logos.\n"
        "- STRICT SFX RULE: Do NOT extract or translate sound effects (SFX) or non-dialogue onomatopoeia (e.g. 'Koff Koff', 'COUGH', 'GRRGLL', 'GAAAASP', 'GASP', 'BAM', 'THWIP', 'CLICK'). SFX are part of the original comic art and must NEVER be extracted as speech bubbles and must NEVER be translated. Only extract authentic spoken conversational dialogue and narrator story captions."
    )
    res_text = call_gemini_api(prompt, image=image, timeout=30, response_json=True, use_cache=use_cache)
    
    clean_json = extract_clean_json_string(res_text)

    data = []
    try:
        norm_json = re.sub(r',\s*([\]\}])', r'\1', clean_json)
        parsed = json.loads(norm_json)
        if isinstance(parsed, dict) and "bubbles" in parsed and isinstance(parsed["bubbles"], list):
            valid_items = []
            for b in parsed["bubbles"]:
                if isinstance(b, dict) and "box_2d" in b:
                    raw_b = b["box_2d"]
                    if isinstance(raw_b, list) and len(raw_b) >= 4 and all(isinstance(x, (int, float)) for x in raw_b[:4]):
                        valid_items.append(b)
                    elif isinstance(raw_b, list) and len(raw_b) > 0 and all(isinstance(sb, list) and len(sb) >= 4 for sb in raw_b):
                        # Handles multi-box grouping inside a single object
                        for sb in raw_b:
                            valid_items.append({
                                "box_2d": sb[:4],
                                "speaker": b.get("speaker") or "Unknown",
                                "original_text": b.get("original_text") or "",
                                "uzbek_translation": b.get("uzbek_translation") or "",
                                "bubble_type": b.get("bubble_type") or "speech"
                            })
                    elif isinstance(raw_b, list) and len(raw_b) > 0 and isinstance(raw_b[0], list) and len(raw_b[0]) >= 4:
                        # Handles positional [ [ymin, xmin, ymax, xmax], "speaker", "orig", "uz", "type" ]
                        coords = raw_b[0][:4]
                        strings = [s for s in raw_b[1:] if isinstance(s, str)]
                        spk = strings[0] if len(strings) >= 1 else (b.get("speaker") or "Unknown")
                        orig = strings[1] if len(strings) >= 2 else (b.get("original_text") or "")
                        uz = strings[2] if len(strings) >= 3 else (b.get("uzbek_translation") or "")
                        b_type = strings[3] if len(strings) >= 4 else (b.get("bubble_type") or "speech")
                        if orig:
                            valid_items.append({
                                "box_2d": coords,
                                "speaker": spk,
                                "original_text": orig,
                                "uzbek_translation": uz,
                                "bubble_type": b_type
                            })
            if valid_items:
                data = valid_items
        elif isinstance(parsed, list):
            data = [b for b in parsed if isinstance(b, dict) and "box_2d" in b]
    except Exception as e:
        pass

    # Fallback: if data is empty or fewer items than occurrences of box_2d in raw text
    box2d_count = len(re.findall(r'"box_2d"\s*:', res_text))
    if len(data) < box2d_count:
        # Resilient chunk extractor for malformed / nested / positional Gemini JSON
        box_iter = list(re.finditer(r'\"box_2d\"\s*:\s*(?:\[\s*)?\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', res_text))
        if box_iter:
            extracted = []
            for i, m in enumerate(box_iter):
                coords = [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))]
                start_pos = m.end()
                end_pos = box_iter[i+1].start() if i + 1 < len(box_iter) else len(res_text)
                chunk = res_text[start_pos:end_pos]

                spk_match = re.search(r'\"speaker\"\s*:\s*\"([^\"]*)\"', chunk)
                orig_match = re.search(r'\"original_text\"\s*:\s*\"(.*?)(?<!\\)\"', chunk, re.DOTALL)
                uz_match = re.search(r'\"uzbek_translation\"\s*:\s*\"(.*?)(?<!\\)\"', chunk, re.DOTALL)
                type_match = re.search(r'\"bubble_type\"\s*:\s*\"([^\"]*)\"', chunk)

                spk = spk_match.group(1) if spk_match else None
                orig = orig_match.group(1) if orig_match else None
                uz = uz_match.group(1) if uz_match else None
                b_type = type_match.group(1) if type_match else "speech"

                # Positional string literals fallback if JSON keys were omitted in chunk
                if not orig or not uz:
                    str_literals = re.findall(r'\"(.*?)(?<!\\)\"', chunk, re.DOTALL)
                    filtered_strs = [s for s in str_literals if s not in ("box_2d", "speaker", "original_text", "uzbek_translation", "bubble_type")]
                    if len(filtered_strs) >= 4:
                        spk = spk or filtered_strs[0]
                        orig = orig or filtered_strs[1]
                        uz = uz or filtered_strs[2]
                        b_type = filtered_strs[3]
                    elif len(filtered_strs) == 3:
                        spk = spk or "Unknown"
                        orig = orig or filtered_strs[0]
                        uz = uz or filtered_strs[1]
                        b_type = filtered_strs[2]
                    elif len(filtered_strs) == 2:
                        spk = spk or "Unknown"
                        orig = orig or filtered_strs[0]
                        uz = uz or filtered_strs[1]

                orig = (orig or "").replace('\\"', '"').replace('\\n', '\n').strip()
                uz = (uz or "").replace('\\"', '"').replace('\\n', '\n').strip()
                spk = (spk or "Unknown").replace('\\"', '"').strip()

                if orig:
                    extracted.append({
                        "box_2d": coords,
                        "speaker": spk,
                        "original_text": orig,
                        "uzbek_translation": uz,
                        "bubble_type": b_type
                    })
            if len(extracted) > len(data):
                data = extracted

    if not data:
        print(f"[GEMINI_WARN] Failed to parse bubbles from Gemini response. Raw: {res_text[:150]}...", flush=True)
        return []

    expanded_data = []
    for item in data:
        raw_box = item.get("box_2d") or [0, 0, 100, 100]
        if isinstance(raw_box, list) and len(raw_box) > 0 and isinstance(raw_box[0], list):
            box_lists = [b for b in raw_box if isinstance(b, list) and len(b) >= 4]
            if len(box_lists) > 1:
                for b_sub in box_lists:
                    new_item = dict(item)
                    new_item["box_2d"] = b_sub
                    expanded_data.append(new_item)
                continue
        expanded_data.append(item)
    data = expanded_data

    W, H = image.size
    formatted_bubbles = []
    for item in data:
        raw_box = item.get("box_2d")
        if raw_box is None and "box_ymin" in item:
            raw_box = [item.get("box_ymin", 0), item.get("box_xmin", 0), item.get("box_ymax", 100), item.get("box_xmax", 100)]
        if raw_box is None:
            raw_box = [0, 0, 100, 100]

        box_coords = [0, 0, 100, 100]
        orig = str(item.get("original_text") or item.get("text") or "").strip()
        uz = str(
            item.get("uzbek_translation") or
            item.get("translation") or
            item.get("uzbek_text") or
            item.get("translated_text") or
            item.get("uzbek") or
            item.get("uz") or
            ""
        ).strip()
        spk = str(item.get("speaker") or "Unknown").strip()
        b_type = str(item.get("bubble_type") or "speech").strip()

        if isinstance(raw_box, list) and len(raw_box) > 0:
            if isinstance(raw_box[0], list):
                box_coords = raw_box[0]
                strings = [s for s in raw_box[1:] if isinstance(s, str)]
                if len(strings) >= 2 and not orig:
                    if len(strings) >= 3:
                        spk = strings[0]
                        orig = strings[1]
                        uz = strings[2]
                        if len(strings) >= 4:
                            b_type = strings[3]
                    else:
                        orig = strings[0]
                        uz = strings[1]
            else:
                box_coords = raw_box[:4]
                strings = [s for s in raw_box[4:] if isinstance(s, str)]
                if len(strings) >= 2 and not orig:
                    orig = strings[0]
                    uz = strings[1]

        ymin, xmin, ymax, xmax = [float(c) for c in (box_coords + [0, 0, 100, 100])[:4]]
        # Denormalize 0-1000 to pixel coordinates
        x0 = int((xmin / 1000.0) * W)
        y0 = int((ymin / 1000.0) * H)
        x1 = int((xmax / 1000.0) * W)
        y1 = int((ymax / 1000.0) * H)

        bw = x1 - x0
        bh = y1 - y0
        if bw <= 10 or bh <= 10:
            continue

        has_letters = bool(re.search(r'[a-zA-Z0-9\u0400-\u04FF]', orig))
        if not has_letters and len(orig.strip()) < 1:
            continue

        aspect = bw / float(bh) if bh > 0 else 1.0
        # Discard extreme horizontal noise strips with less than 2 letters
        letters = re.findall(r'[a-zA-Z0-9\u0400-\u04FF]', orig)
        if len(letters) < 2 and (aspect >= 6.0 or bh <= 12 or bw <= 12):
            continue

        if orig:
            try:
                import engine
                if engine.is_pure_sfx_text(orig):
                    print(f"[GEMINI_FILTER] Ignored pure SFX sound effect '{orig}' to preserve comic art.", flush=True)
                    continue
            except Exception:
                pass

            # Automatic fallback: if Uzbek translation is missing, translate original text
            if not uz:
                try:
                    uz = gemini_translate_dialogue(orig, speaker=spk)
                except Exception as ex:
                    print(f"[GEMINI_WARN] On-the-fly bubble translation fallback failed: {ex}", flush=True)

            final_uz = ""
            if uz:
                try:
                    import naturalization_rules
                    uz = naturalization_rules.apply_naturalization(uz, en_orig=orig, character_profile=spk)
                except Exception:
                    pass
                final_uz = glossary_manager.apply_glossary_enforcement(uz)
                try:
                    import auto_emphasis
                    final_uz = auto_emphasis.apply_auto_emphasis(final_uz, en_orig=orig)
                except Exception:
                    pass
            formatted_bubbles.append({
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "box": [x0, y0, x1, y1],
                "original_text": orig,
                "uzbek_translation": final_uz,
                "speaker": spk,
                "bubble_type": b_type
            })

    return formatted_bubbles
