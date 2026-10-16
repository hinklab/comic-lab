"""
glossary_manager.py
-------------------
Centralized, persistent team glossary and terminology enforcer for Comic-Lab:
1. Shared server-side store (glossary.json) accessible to all team members.
2. Injects mandatory terminology rules directly into Gemini vision & translation prompts.
3. Post-processes translations with regex enforcement to guarantee 100% consistent naming.
4. Auto-seeds from existing comic terminology store if available.
"""

import os
import json
import re
import threading
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GLOSSARY_FILE = os.path.join(BASE_DIR, "glossary.json")
TERMINOLOGY_STORE_FILE = os.path.join(BASE_DIR, "terminology_store.json")

_LOCK = threading.Lock()
_CACHE: Optional[Dict[str, str]] = None
_CACHE_VERSION: float = 0.0


def _seed_initial_glossary() -> Dict[str, str]:
    """Builds initial dictionary, pulling from terminology_store.json if present."""
    initial: Dict[str, str] = {
        "Peter Parker": "Piter Parker",
        "Spider-Man": "O'rgimchak-odam",
        "Spidey": "O'rgimchak",
        "Batman": "Betmen",
        "Bruce Wayne": "Bryus Ueyn",
        "Gotham": "Gotem",
        "Aunt May": "Mey xola",
        "Uncle Ben": "Ben amaki",
        "J. Jonah Jameson": "J. Jona Jeymson",
        "Daily Bugle": "Deyli Byugl",
        "Web-Shooter": "To'r otgich",
        "Green Goblin": "Yashil Goblin",
        "Doctor Octopus": "Doktor Oktopus",
        "Doc Ock": "Dok Ock"
    }
    if os.path.exists(TERMINOLOGY_STORE_FILE):
        try:
            with open(TERMINOLOGY_STORE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                terms_list = data.get("terms", [])
                for t in terms_list:
                    src = t.get("source_term", "").strip()
                    uz = t.get("canonical_uzbek", "").strip()
                    if src and uz:
                        title_src = src.title() if src.islower() else src
                        initial[title_src] = uz
        except Exception:
            pass
    return initial


def load_glossary() -> Dict[str, str]:
    """Loads glossary terms from disk with memory caching."""
    global _CACHE, _CACHE_VERSION
    with _LOCK:
        if _CACHE is not None:
            return dict(_CACHE)

        if os.path.exists(GLOSSARY_FILE):
            try:
                with open(GLOSSARY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        terms = data.get("terms", data)
                        _CACHE = {str(k).strip(): str(v).strip() for k, v in terms.items() if str(k).strip()}
                        _CACHE_VERSION = data.get("updated_at", os.path.getmtime(GLOSSARY_FILE))
                        return dict(_CACHE)
            except Exception as e:
                print(f"[Glossary] Error loading glossary.json: {e}", flush=True)

        # Seed and save initial
        _CACHE = _seed_initial_glossary()
        _CACHE_VERSION = 1.0
        try:
            with open(GLOSSARY_FILE, "w", encoding="utf-8") as f:
                json.dump({"terms": _CACHE, "updated_at": _CACHE_VERSION}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return dict(_CACHE)


def save_glossary(terms: Dict[str, str]) -> bool:
    """Persists updated glossary terms to glossary.json atomically."""
    global _CACHE, _CACHE_VERSION
    import time
    with _LOCK:
        clean_terms = {str(k).strip(): str(v).strip() for k, v in terms.items() if str(k).strip() and str(v).strip()}
        _CACHE = clean_terms
        _CACHE_VERSION = time.time()
        try:
            tmp_file = GLOSSARY_FILE + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump({"terms": clean_terms, "updated_at": _CACHE_VERSION}, f, ensure_ascii=False, indent=2)
            if os.path.exists(GLOSSARY_FILE):
                os.remove(GLOSSARY_FILE)
            os.rename(tmp_file, GLOSSARY_FILE)
            return True
        except Exception as e:
            print(f"[Glossary] Failed to write glossary.json: {e}", flush=True)
            return False


def set_term(source: str, target: str) -> bool:
    """Adds or updates a single term."""
    source = source.strip()
    target = target.strip()
    if not source or not target:
        return False
    current = load_glossary()
    current[source] = target
    return save_glossary(current)


def remove_term(source: str) -> bool:
    """Removes a term from glossary."""
    source = source.strip()
    current = load_glossary()
    matched_key = None
    for k in current.keys():
        if k.lower() == source.lower():
            matched_key = k
            break
    if matched_key:
        del current[matched_key]
        return save_glossary(current)
    return False


def get_glossary_hash() -> str:
    """Returns a short hash for cache key invalidation when glossary changes."""
    import hashlib
    terms = load_glossary()
    raw = "|".join(f"{k}:{v}" for k, v in sorted(terms.items()))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def format_glossary_prompt() -> str:
    """Formats glossary terms into strict instruction for Gemini system prompt."""
    terms = load_glossary()
    if not terms:
        return ""
    lines = [
        "MAJBURIY JAMOA LUG'ATI (Ushbu maxsus ismlar va atamalarni FAQAT quyidagicha tarjima qil):"
    ]
    # Sort by length descending so multi-word phrases appear first
    sorted_terms = sorted(terms.items(), key=lambda x: len(x[0]), reverse=True)
    for src, tgt in sorted_terms[:40]:  # include top terms
        lines.append(f"- {src} -> {tgt}")
    return "\n" + "\n".join(lines) + "\n"


def _match_case(orig: str, repl: str) -> str:
    """Matches the casing style of repl to orig: ALL CAPS, Capitalized/Title, or lowercase."""
    if orig.isupper():
        return repl.upper()
    if orig.islower():
        return repl.lower()
    if orig and orig[0].isupper():
        if len(repl) <= 1:
            return repl.upper()
        return repl[0].upper() + repl[1:].lower()
    return repl


def _make_sub(pattern_str: str, repl_str: str):
    regex = re.compile(pattern_str, flags=re.IGNORECASE)
    def replacer(match):
        return _match_case(match.group(0), repl_str)
    return lambda s: regex.sub(replacer, s)


# Exhaustive compiled table of Uzbek orthography, H vs X, apostrophes, double consonants, and verb contractions
UZBEK_TYPO_RULES = [
    # 1. Double consonants and common morphological typos
    (r'\balbatda\b', 'albatta'),
    (r'\balbata\b', 'albatta'),
    (r'\balbatdami\b', 'albattami'),
    (r'\btasurot\b', 'taassurot'),
    (r'\btasurotlar', 'taassurotlar'),
    (r'\bmukamal\b', 'mukammal'),
    (r'\bmukamallik', 'mukammallik'),

    # 2. Clumsy Russian/English calques with "uchun"
    (r'\bmening uchun\b', 'men uchun'),
    (r'\bsening uchun\b', 'sen uchun'),
    (r'\buning uchun\b', 'u uchun'),
    (r'\bbizning uchun\b', 'biz uchun'),
    (r'\bsizning uchun\b', 'siz uchun'),
    (r'\bularning uchun\b', 'ular uchun'),

    # 3. Words that strictly require 'H' (frequently mistyped as 'X')
    (r'\bxozir\b', 'hozir'),
    (r'\bxozircha\b', 'hozircha'),
    (r'\bxoziroq\b', 'hoziroq'),
    (r'\bxozirgacha\b', 'hozirgacha'),
    (r'\bxamm', 'hamm'), # covers hamma, hammamiz, hammangiz, hammasi, hammaga, etc.
    (r'\bxam\b', 'ham'),
    (r'\bxatto', 'hatto'), # covers hatto, hattoki
    (r'\bxar doim\b', 'har doim'),
    (r'\bxar kim\b', 'har kim'),
    (r'\bxar bir\b', 'har bir'),
    (r'\bxar narsa\b', 'har narsa'),
    (r'\bxar qanday\b', 'har qanday'),
    (r'\bxar xil\b', 'har xil'),
    (r'\bxar holda\b', 'har holda'),
    (r'\bxar qalay\b', 'har qalay'),
    (r'\bxech qachon\b', 'hech qachon'),
    (r'\bxech kim', 'hech kim'),
    (r'\bxech narsa', 'hech narsa'),
    (r'\bxech qanday\b', 'hech qanday'),
    (r'\bxech qayer', 'hech qayer'),
    (r'\bxech qisi yo\'q\b', "hech qisi yo'q"),
    (r'\bxech bo\'lmasa\b', "hech bo'lmasa"),
    (r'\bxech\b', 'hech'),
    (r'\bxarakat', 'harakat'), # covers harakat, harakatlar, harakatda, harakatlanmoq
    (r'\bxayot', 'hayot'), # covers hayot, hayotim, hayotda, hayotiy
    (r'\bxaqiqat', 'haqiqat'), # covers haqiqat, haqiqatan, haqiqatdan, haqiqiy
    (r'\bxaq\b', 'haq'),
    (r'\bxaqqi', 'haqqi'),
    (r'\bxaqli\b', 'haqli'),
    (r'\braxmat', 'rahmat'),
    (r'\bmexnat', 'mehnat'),
    (r'\bmashxur', 'mashhur'),
    (r'\bsaxna', 'sahna'),
    (r'\bmuxim', 'muhim'),
    (r'\bximoya', 'himoya'), # covers himoya, himoyachi, himoyalash
    (r'\bxujum', 'hujum'), # covers hujum, hujumchi
    (r'\bxurmat', 'hurmat'),
    (r'\bshubxasiz\b', 'shubhasiz'),
    (r'\bshubxa', 'shubha'),
    (r'\bnaxotki\b', 'nahotki'),
    (r'\bshaxar', 'shahar'),
    (r'\bbaxor', 'bahor'),
    (r'\bbaxo\b', 'baho'),
    (r'\bguvox\b', 'guvoh'),
    (r'\bxamkor', 'hamkor'),
    (r'\bxissa', 'hissa'),
    (r'\bxushyor', 'hushyor'),
    (r'\bxushidan ket', 'hushidan ket'),
    (r'\bxushiga kel', 'hushiga kel'),
    (r'\bxis qil', 'his qil'),
    (r'\bxissiyot', 'hissiyot'),
    (r'\bxal qil', 'hal qil'),

    # 4. Words that strictly require 'X' (frequently mistyped as 'H')
    (r'\bhato', 'xato'), # covers xato, xatolar, xatolik
    (r'\bhavf', 'xavf'), # covers xavf, xavfli, xavfsiz, xavfsizlik
    (r'\bhabar', 'xabar'), # covers xabar, xabarlar, xabardor
    (r'\bhohla', 'xohla'), # covers xohlamoq, xohlayman, xohlaysan, xohlaydi, xohlardim
    (r'\bhohish', 'xohish'),
    (r'\bhursand', 'xursand'), # covers xursand, xursandman, xursandchilik
    (r'\bhizmat', 'xizmat'), # covers xizmat, xizmatchi
    (r'\bhayol', 'xayol'), # covers xayol, xayollar, xayolan
    (r'\bhayr\b', 'xayr'),
    (r'\bhayrlash', 'xayrlash'),
    (r'\bhayrli\b', 'xayrli'),
    (r'\bhalos', 'xalos'),
    (r'\bharid', 'xarid'),
    (r'\bhasis', 'xasis'),
    (r'\bhiyonat', 'xiyonat'),
    (r'\bhosdir\b', 'xosdir'),
    (r'\bhos\b', 'xos'),
    (r'\bhotir', 'xotir'), # covers xotira, xotirjam
    (r'\bhona', 'xona'), # covers xona, xonadon
    (r'\bhush kelibsiz\b', 'xush kelibsiz'),
    (r'\bhushfe\'l\b', "xushfe'l"),
    (r'\bhushbroy\b', "xushro'y"),
    (r'\bhushroy\b', "xushro'y"),
    (r'\bhushchaqchaq\b', 'xushchaqchaq'),

    # 5. Missing root apostrophes (O' and G' and Tutuq belgisi)
    (r'\btogri\b', "to'g'ri"),
    (r'\bnotogri\b', "noto'g'ri"),
    (r'\bogir\b', "og'ir"),
    (r'\bboldim\b', "bo'ldim"),
    (r'\bbolding\b', "bo'lding"),
    (r'\bboldi\b', "bo'ldi"),
    (r'\bboldik\b', "bo'ldik"),
    (r'\bboldingiz\b', "bo'ldingiz"),
    (r'\bboldilar\b', "bo'ldilar"),
    (r'\bbolaman\b', "bo'laman"),
    (r'\bbolasan\b', "bo'lasan"),
    (r'\bboladi\b', "bo'ladi"),
    (r'\bbolamiz\b', "bo'lamiz"),
    (r'\bbolasiz\b', "bo'lasiz"),
    (r'\bboladilar\b', "bo'ladilar"),
    (r'\bbolgan\b', "bo'lgan"),
    (r'\bbolsa\b', "bo'lsa"),
    (r'\bbolmasin\b', "bo'lmasin"),
    (r'\bbolmaydi\b', "bo'lmaydi"),
    (r'\bbolmoq\b', "bo'lmoq"),
    (r'\bkordim\b', "ko'rdim"),
    (r'\bkording\b', "ko'rding"),
    (r'\bkordi\b', "ko'rdi"),
    (r'\bkordik\b', "ko'rdik"),
    (r'\bkordingiz\b', "ko'rdingiz"),
    (r'\bkordilar\b', "ko'rdilar"),
    (r'\bkoraman\b', "ko'raman"),
    (r'\bkorasan\b', "ko'rasan"),
    (r'\bkoradi\b', "ko'radi"),
    (r'\bkoramiz\b', "ko'ramiz"),
    (r'\bkorasiz\b', "ko'rasiz"),
    (r'\bkoradilar\b', "ko'radilar"),
    (r'\bkormadim\b', "ko'rmadim"),
    (r'\bkormading\b', "ko'rmading"),
    (r'\bkormadi\b', "ko'rmadi"),
    (r'\bkorinyapti\b', "ko'rinyapti"),
    (r'\bkorindi\b', "ko'rindi"),
    (r'\bkorinadi\b', "ko'rinadi"),
    (r'\bkoringan\b', "ko'ringan"),
    (r'\bozim\b', "o'zim"),
    (r'\bozing\b', "o'zing"),
    (r'\bozi\b', "o'zi"),
    (r'\bozimiz\b', "o'zimiz"),
    (r'\bozingiz\b', "o'zingiz"),
    (r'\bozlari\b', "o'zlari"),
    (r'\bozimning\b', "o'zimning"),
    (r'\bozingning\b', "o'zingning"),
    (r'\bozining\b', "o'zining"),
    (r'\bbogliq', "bog'liq"), # covers bog'liq, bog'liqlik
    (r'\byolgon', "yolg'on"), # covers yolg'on, yolg'onchi, yolg'onlar
    (r'\botmish', "o'tmish"), # covers o'tmish, o'tmishda
    (r'\botkir\b', "o'tkir"),
    (r'\botir\b', "o'tir"),
    (r'\botiring\b', "o'tiring"),
    (r'\botirdi\b', "o'tirdi"),
    (r'\botirish\b', "o'tirish"),
    (r'\botdi\b', "o'tdi"),
    (r'\botgan\b', "o'tgan"),
    (r'\botib\b', "o'tib"),
    (r'\botmoq\b', "o'tmoq"),
    (r'\bogil\b', "o'g'il"),
    (r'\boglim\b', "o'g'lim"),
    (r'\bogling\b', "o'g'ling"),
    (r'\bogli\b', "o'g'li"),
    (r'\bogirlan', "o'g'irlan"),
    (r'\bogirlamoq\b', "o'g'irlamoq"),
    (r'\bogri', "o'g'ri"),
    (r'\boldirmoq\b', "o'ldirmoq"),
    (r'\boldirdi\b', "o'ldirdi"),
    (r'\boldirilgan\b', "o'ldirilgan"),
    (r'\boyin', "o'yin"), # covers o'yin, o'yinchoq, o'yinlar
    (r'\borgan', "o'rgan"), # covers o'rganmoq, o'rgandim, o'rgangan
    (r'\borgat', "o'rgat"), # covers o'rgatmoq, o'rgatdi
    (r'\bozgar', "o'zgar"), # covers o'zgarish, o'zgardi, o'zgaradi
    (r'\betibor', "e'tibor"), # covers e'tibor, e'tiborsiz, e'tiborli
    (r'\bmojiza', "mo'jiza"), # covers mo'jiza, mo'jizakor
    (r'\bmalumot', "ma'lumot"), # covers ma'lumot, ma'lumotlar
    (r'\bmanoli\b', "ma'noli"),
    (r'\bmanosiz\b', "ma'nosiz"),
    (r'\bmanosi\b', "ma'nosi"),

    # 6. Authentic colloquial verb contractions for superhero comic dialogue
    (r'\bqila olmayman\b', 'qilolmayman'),
    (r'\bqila olmaysan\b', 'qilolmaysan'),
    (r'\bqila olmaydi\b', 'qilolmaydi'),
    (r'\bqila olmaymiz\b', 'qilolmaymiz'),
    (r'\bqila olmaysiz\b', 'qilolmaysiz'),
    (r'\bqila olmaydilar\b', 'qilolmaydilar'),
    (r'\bqila olmadim\b', 'qilolmadim'),
    (r'\bqila olmading\b', 'qilolmading'),
    (r'\bqila olmadi\b', 'qilolmadi'),
    (r'\bqila olmadik\b', 'qilolmadik'),
    (r'\bqila olmadingiz\b', 'qilolmadingiz'),
    (r'\bqila olmadilar\b', 'qilolmadilar'),
    (r'\bbila olmayman\b', 'bilolmayman'),
    (r'\bbila olmaysan\b', 'bilolmaysan'),
    (r'\bbila olmaydi\b', 'bilolmaydi'),
    (r'\bbila olmaymiz\b', 'bilolmaymiz'),
    (r'\bbila olmaysiz\b', 'bilolmaysiz'),
    (r'\bbila olmadim\b', 'bilolmadim'),
    (r'\bbila olmadi\b', 'bilolmadi'),
    (r'\bko[\'ʻ`’]?ra olmayman\b', "ko'rolmayman"),
    (r'\bko[\'ʻ`’]?ra olmaysan\b', "ko'rolmaysan"),
    (r'\bko[\'ʻ`’]?ra olmaydi\b', "ko'rolmaydi"),
    (r'\bko[\'ʻ`’]?ra olmaymiz\b', "ko'rolmaymiz"),
    (r'\bko[\'ʻ`’]?ra olmaysiz\b', "ko'rolmaysiz"),
    (r'\bko[\'ʻ`’]?ra olmadim\b', "ko'rolmadim"),
    (r'\bko[\'ʻ`’]?ra olmadi\b', "ko'rolmadi"),
    (r'\btopa olmayman\b', 'topolmayman'),
    (r'\btopa olmaysan\b', 'topolmaysan'),
    (r'\btopa olmaydi\b', 'topolmaydi'),
    (r'\btopa olmadim\b', 'topolmadim'),
    (r'\btopa olmadi\b', 'topolmadi'),
    (r'\bayta olmayman\b', 'aytolmayman'),
    (r'\bayta olmaysan\b', 'aytolmaysan'),
    (r'\bayta olmaydi\b', 'aytolmaydi'),
    (r'\bayta olmadim\b', 'aytolmadim'),
    (r'\bayta olmadi\b', 'aytolmadi'),
    (r'\byeta olmayman\b', 'yetolmayman'),
    (r'\byeta olmaysan\b', 'yetolmaysan'),
    (r'\byeta olmaydi\b', 'yetolmaydi'),
    (r'\byeta olmadim\b', 'yetolmadim'),
    (r'\byeta olmadi\b', 'yetolmadi'),
    (r'\bbora olmayman\b', 'borolmayman'),
    (r'\bbora olmaysan\b', 'borolmaysan'),
    (r'\bbora olmaydi\b', 'borolmaydi'),
    (r'\bbora olmadim\b', 'borolmadim'),
    (r'\bbora olmadi\b', 'borolmadi'),
    (r'\bkela olmayman\b', 'kelolmayman'),
    (r'\bkela olmaysan\b', 'kelolmaysan'),
    (r'\bkela olmaydi\b', 'kelolmaydi'),
    (r'\bkela olmadim\b', 'kelolmadim'),
    (r'\bkela olmadi\b', 'kelolmadi'),
    (r'\bchida(?:y)? olmayman\b', 'chidolmayman'),
    (r'\bchida(?:y)? olmaysan\b', 'chidolmaysan'),
    (r'\bchida(?:y)? olmaydi\b', 'chidolmaydi'),
    (r'\bchida(?:y)? olmadim\b', 'chidolmadim'),
    (r'\bchida(?:y)? olmadi\b', 'chidolmadi'),
    (r'\bishona olmayman\b', 'ishonolmayman'),
    (r'\bishona olmaysan\b', 'ishonolmaysan'),
    (r'\bishona olmaydi\b', 'ishonolmaydi'),
    (r'\bto[\'ʻ`’]?xtata olmayman\b', "to'xtatolmayman"),
    (r'\bto[\'ʻ`’]?xtata olmaysan\b', "to'xtatolmaysan"),
    (r'\bto[\'ʻ`’]?xtata olmaydi\b', "to'xtatolmaydi"),
    (r'\bqaolmasligingni\b', 'qololmasligingni'),
    (r'\bqaolmasligini\b', 'qololmasligini'),
    (r'\bkirolmasligingizni\b', 'qololmasligingizni'),
    (r'\bketa olmasligingni\b', 'ketib qololmasligingni'),
    (r'\bketa olmasligini\b', 'ketib qololmasligini'),
    (r'\bketib qola olmasligingni\b', 'ketib qololmasligingni'),
    (r'\bketib qola olmasligini\b', 'ketib qololmasligini'),
    (r'\bqola olmasligingni\b', 'qololmasligingni'),
    (r'\bqola olmasligini\b', 'qololmasligini'),
    (r'\bbilar edim\b', 'bilardim'),
    (r'\bbilar eding\b', 'bilarding'),
    (r'\bbilar edi\b', 'bilardi'),
    (r'\bbilar edik\b', 'bilardik'),
    (r'\bbilar edingiz\b', 'bilardingiz'),
    (r'\bkelar edim\b', 'kelardim'),
    (r'\bkelar eding\b', 'kelarding'),
    (r'\bkelar edi\b', 'kelardi'),
    (r'\bborar edim\b', 'borardim'),
    (r'\bborar eding\b', 'borarding'),
    (r'\bborar edi\b', 'borardi'),
    (r'\baytar edim\b', 'aytardim'),
    (r'\baytar eding\b', 'aytarding'),
    (r'\baytar edi\b', 'aytardi'),
    (r'\bqilar edim\b', 'qilardim'),
    (r'\bqilar eding\b', 'qilarding'),
    (r'\bqilar edi\b', 'qilardi'),
    (r'\baytgan edim\b', 'aytgandim'),
    (r'\baytgan eding\b', 'aytganding'),
    (r'\baytgan edi\b', 'aytgandi'),
    (r'\bkelgan edim\b', 'kelgandim'),
    (r'\bkelgan eding\b', 'kelganding'),
    (r'\bkelgan edi\b', 'kelgandi'),
    (r'\bko[\'ʻ`’]?rgan edim\b', "ko'rgandim"),
    (r'\bko[\'ʻ`’]?rgan eding\b', "ko'rganding"),
    (r'\bko[\'ʻ`’]?rgan edi\b', "ko'rgandi"),
    (r'\bborgan edim\b', 'borgandim'),
    (r'\bborgan eding\b', 'borganding'),
    (r'\bborgan edi\b', 'borgandi'),
]

# Pre-compile the substitutions for microsecond execution speed
_COMPILED_TYPO_SUBS = [_make_sub(p, r) for p, r in UZBEK_TYPO_RULES]


def clean_uzbek_dialogue_typos(text: str) -> str:
    """
    Cleans common AI translation typographical, orthographical (H vs X, apostrophes),
    and morphological quirks in Uzbek comic dialogue with strict case-preservation.
    """
    if not text:
        return ""
    res = text

    # 1. Normalize modifier and curly apostrophes to standard ASCII single quote
    for ap in ['\u02bb', '\u02bc', '\u2018', '\u2019', '`', '´', 'ʻ', 'ʼ', '’', '‘']:
        res = res.replace(ap, "'")

    # 2. Apply all case-preserving orthographical rules
    for sub_fn in _COMPILED_TYPO_SUBS:
        res = sub_fn(res)

    # 3. Deduplicate accidental immediately-repeated words (e.g. "Shunchaki shunchaki" -> "Shunchaki")
    res = re.sub(r'\b([a-zA-Z\u0400-\u04FF\']+)\s+\1\b', r'\1', res, flags=re.I)

    # 4. Collapse accidental double spaces
    res = re.sub(r'[ \t]{2,}', ' ', res)

    return res.strip()


def apply_glossary_enforcement(text: str) -> str:
    """
    Applies case-insensitive regex substitution to enforce glossary terms
    in translated Uzbek text (handles model misspellings or English bleed).
    Preserves original casing style (ALL CAPS, Title Case, lowercase).
    """
    if not text:
        return ""
    terms = load_glossary()
    result = text
    if terms:
        # Sort longer phrases first to prevent sub-term collisions
        sorted_items = sorted(terms.items(), key=lambda x: len(x[0]), reverse=True)
        for src, tgt in sorted_items:
            if not src or not tgt:
                continue
            pattern = re.compile(r'\b' + re.escape(src) + r'\b', re.IGNORECASE)
            def _repl(match, target=tgt):
                return _match_case(match.group(0), target)
            result = pattern.sub(_repl, result)

    return clean_uzbek_dialogue_typos(result)


