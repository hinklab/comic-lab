"""
auto_emphasis.py
----------------
Step 3.3: Automatic Semantic Emphasis Prediction (Avtomatik Semantik Urg'u)
for Comic Book Dialogue Lettering.

Comic Lettering Standards (Marvel / DC / Manga):
1. Bubbles are lettered in standard 'normal' weight by default (never universally bold).
2. The letterer applies bold emphasis (**word**) to 1-2 key rhythmic words per sentence
   to convey emotional cadence, dramatic intensity, and spoken stress.
3. Common clitics and grammatical glue (va, ham, bu, bilan, uchun) are NEVER emphasized.
4. Punctuation is kept outside the bold tags (**SO'Z**!).
"""

from __future__ import annotations

import re
from typing import List, Tuple, Optional, Set, Dict


# Stop words that should NEVER be emphasized alone
CLITICS_AND_STOP_WORDS: Set[str] = {
    "va", "ham", "bu", "u", "bilan", "uchun", "deb", "esa", "bir", "oq", "yoq",
    "shu", "o'sha", "mana", "ana", "men", "sen", "biz", "siz", "ular"
}

# Tier 1: Strong Negations and Absolutes (Highest comic emotional stress)
TIER_1_NEGATIONS: List[str] = [
    "hech qachon", "hech kim", "hech narsa", "hech qayerda", "hech qanday", "hech qanaqa",
    "aslo", "mutlaqo", "emas", "aslida"
]

# Tier 2: High Intensifiers & Dramatic Modifiers
TIER_2_INTENSIFIERS: List[str] = [
    "albatta", "juda", "eng", "faqat", "nahotki", "darhol", "tezroq", "aniq",
    "haqiqatan", "shunchaki", "qattiq", "butunlay", "mutlaq", "aynan"
]

# Tier 3: Action Exclamations & Imperatives
TIER_3_IMPERATIVES: List[str] = [
    "to'xta", "to'xtang", "qara", "qarang", "qoch", "qoching", "yordam",
    "tingla", "tinglang", "eshit", "eshiting", "ketdik", "tashla", "tashlang",
    "ehtiyot", "qutqar", "berkit"
]

# Tier 4: Core Modals & Critical Verbs
TIER_4_MODALS: List[str] = [
    "shart", "kerak", "lozim", "mumkin emas", "bilmayman", "bilolmayman",
    "qilolmayman", "qilolmaysan", "chidolmayman", "bo'lmaydi", "yetadi",
    "yetarli", "tushundim", "tushunmaysan", "to'xtatolmayman", "ishonolmayman"
]

# Tier 5: Direct Vocatives / Hero Personas
TIER_5_HEROES: List[str] = [
    "piter", "parker", "jonni", "doom", "dum", "mey", "ueyn", "oktavius",
    "spidey", "o'rgimchak", "betmen", "xalk", "hulk", "stark", "jona", "jeymson",
    "doktor", "kapitan"
]


def clean_token_for_matching(token: str) -> str:
    """Strips markdown and punctuation, converting to lowercase for rule matching."""
    s = re.sub(r"\*{1,3}", "", token)
    s = re.sub(r"[^\w'-]", "", s).strip().lower()
    return s


def has_existing_emphasis(text: str) -> bool:
    """Checks whether the text already contains markdown bold formatting."""
    return bool(re.search(r"\*\*[^\*\s].*?\*\*", text))


def is_fully_bolded(text: str) -> bool:
    """
    Returns True if the entire text (or >80% of words) was erroneously bolded.
    In comic lettering, a bubble must never be 100% bold by default.
    """
    clean = text.strip()
    if clean.startswith("**") and clean.endswith("**"):
        inner = clean[2:-2].strip()
        if "**" not in inner:
            return True

    words = clean.split()
    if len(words) <= 1:
        return False

    bold_words = 0
    for w in words:
        if w.startswith("**") or w.endswith("**") or ("**" in w):
            bold_words += 1

    return (bold_words / len(words)) >= 0.80


def strip_all_bold_tags(text: str) -> str:
    """Removes all markdown bold markers from the text."""
    return re.sub(r"\*\*", "", text).strip()


def extract_sentences(text: str) -> List[str]:
    """Splits dialogue into sentences while preserving trailing punctuation and newlines."""
    # Split by newlines first
    lines = text.split("\n")
    all_chunks = []
    for line in lines:
        if not line.strip():
            all_chunks.append("")
            continue
        # Split line by sentence terminators (. ! ? … --)
        parts = re.split(r"(?<=[.!?…])\s+", line)
        for p in parts:
            if p.strip():
                all_chunks.append(p.strip())
    return all_chunks


def project_english_emphasis(en_orig: str, uz_text: str) -> Optional[str]:
    """
    If the English source text has clear focal emphasis (markdown bold or ALL-CAPS word
    in mixed-case dialogue), maps that emphasis directly to the corresponding Uzbek word.
    """
    if not en_orig or not uz_text:
        return None

    # 1. Check for markdown bold in English: "I **never** said that"
    en_bold_matches = re.findall(r"\*\*([^\*]+)\*\*", en_orig)
    targets: List[str] = []
    for bm in en_bold_matches:
        c = clean_token_for_matching(bm)
        if c and c not in CLITICS_AND_STOP_WORDS:
            targets.append(c)

    # 2. Check for single/double all-caps emphasis in mixed English: "I WILL find you"
    en_words = en_orig.split()
    if any(any(c.islower() for c in w) for w in en_words):
        for w in en_words:
            c = clean_token_for_matching(w)
            if len(c) >= 2 and w.isupper() and c not in ("i", "a", "an", "the", "in", "on", "at"):
                targets.append(c)

    if not targets:
        return None

    uz_words = uz_text.split()
    uz_lower = [clean_token_for_matching(w) for w in uz_words]

    # Semantic cross-lingual keyword bridges
    BRIDGES = {
        "never": ["hech qachon", "aslo"],
        "no": ["yo'q", "emas"],
        "not": ["emas", "bo'lmaydi"],
        "must": ["shart", "kerak"],
        "will": ["albatta"],
        "really": ["haqiqatan", "nahotki", "juda"],
        "stop": ["to'xta", "to'xtang"],
        "look": ["qara", "qarang"],
        "always": ["har doim"],
        "why": ["nega", "nimaga"],
        "who": ["kim"],
        "what": ["nima"],
        "how": ["qanday"],
        "help": ["yordam"],
        "peter": ["piter"],
        "parker": ["parker"],
        "doom": ["dum", "doom"],
        "danger": ["xavf", "xavfli"],
    }

    for t in targets:
        candidates = BRIDGES.get(t, [t])
        for cand in candidates:
            cand_parts = cand.split()
            if len(cand_parts) == 1:
                for idx, uw in enumerate(uz_lower):
                    if uw == cand:
                        return wrap_word_index_with_bold(uz_words, idx)
            elif len(cand_parts) == 2:
                for idx in range(len(uz_lower) - 1):
                    if uz_lower[idx] == cand_parts[0] and uz_lower[idx + 1] == cand_parts[1]:
                        return wrap_word_range_with_bold(uz_words, idx, idx + 1)

    return None


def wrap_word_index_with_bold(words: List[str], target_idx: int) -> str:
    """Wraps the word at target_idx in **...**, preserving leading/trailing punctuation."""
    result = []
    for idx, raw_w in enumerate(words):
        if idx == target_idx:
            result.append(format_word_with_bold(raw_w))
        else:
            result.append(raw_w)
    return " ".join(result)


def wrap_word_range_with_bold(words: List[str], start_idx: int, end_idx: int) -> str:
    """Wraps words from start_idx to end_idx in a unified **...** block."""
    result = []
    for idx, raw_w in enumerate(words):
        if idx == start_idx:
            # Strip trailing punctuation from end word if any
            w_start = raw_w.lstrip("¿¡\"'«(„")
            lead = raw_w[:len(raw_w) - len(w_start)]
            result.append(f"{lead}**{w_start}")
        elif idx == end_idx:
            w_end = raw_w.rstrip(".,!?:;…\")»”")
            trail = raw_w[len(w_end):]
            result.append(f"{w_end}**{trail}")
        else:
            result.append(raw_w)
    return " ".join(result)


def format_word_with_bold(word: str) -> str:
    """Wraps a single word with **...**, keeping outer punctuation outside."""
    if not word:
        return ""
    if word.startswith("**") and word.endswith("**"):
        return word

    match = re.match(r"^([^\w'’ʼʻ`]*)([\w'’ʼʻ`-]+)([^\w'’ʼʻ`]*)$", word)
    if not match:
        return f"**{word}**"

    lead_punct, core, trail_punct = match.groups()
    return f"{lead_punct}**{core}**{trail_punct}"


def predict_semantic_emphasis_sentence(sentence: str) -> str:
    """
    Analyzes an unemphasized Uzbek sentence and injects bold emphasis (**word**)
    onto 1-2 words that carry the primary emotional and dramatic rhythm.
    """
    clean_s = sentence.strip()
    if not clean_s:
        return sentence

    words = clean_s.split()
    total_words = len(words)

    # Short 1-2 word utterance: e.g. "Rahmat.", "Salom!"
    # Only emphasize if it's an urgent imperative exclamation (e.g. "Qoch!", "Yordam!")
    if total_words <= 2:
        for idx, w in enumerate(words):
            cw = clean_token_for_matching(w)
            if cw in ("qoch", "yordam", "to'xta", "tashla") and ("!" in w):
                return wrap_word_index_with_bold(words, idx)
        return sentence

    clean_words = [clean_token_for_matching(w) for w in words]

    # Search for multi-word phrases first (e.g. "hech qachon", "mumkin emas")
    multi_phrases = [
        "hech qachon", "hech kim", "hech narsa", "hech qayerda", "hech qanday",
        "mumkin emas", "bo'lishi mumkin emas", "ehtiyot bo'l", "ehtiyot bo'ling"
    ]
    for phrase in multi_phrases:
        p_tokens = phrase.split()
        p_len = len(p_tokens)
        for i in range(len(clean_words) - p_len + 1):
            if clean_words[i:i + p_len] == p_tokens:
                return wrap_word_range_with_bold(words, i, i + p_len - 1)

    # Score each single word
    scores: List[Tuple[int, int]] = []  # (priority_tier, word_index)

    for idx, cw in enumerate(clean_words):
        if not cw or cw in CLITICS_AND_STOP_WORDS:
            continue

        raw_w = words[idx]

        # Tier 1: Strong Negations (score = 50)
        if cw in TIER_1_NEGATIONS:
            scores.append((50, idx))
            continue

        # Tier 2: High Intensifiers (score = 40)
        if cw in TIER_2_INTENSIFIERS:
            scores.append((40, idx))
            continue

        # Tier 3: Action Imperatives (score = 30)
        if cw in TIER_3_IMPERATIVES:
            scores.append((30, idx))
            continue

        # Tier 4: Core Modals & Strong Verbs (score = 25)
        if cw in TIER_4_MODALS or cw.endswith("olmayman") or cw.endswith("olmaysan") or cw.endswith("olmadi"):
            scores.append((25, idx))
            continue

        # Tier 5: Hero Vocatives (score = 20)
        if cw in TIER_5_HEROES:
            # Often at start or end of sentence with comma: "Piter, qara!"
            scores.append((20, idx))
            continue

        # Exclamatory word marked with exclamation inside sentence
        if "!" in raw_w and len(cw) >= 3:
            scores.append((15, idx))

    if not scores:
        # Fallback for dramatic sentences: if sentence ends with '!', emphasize the main verb / penultimate word
        if ("!" in clean_s) and total_words >= 3:
            # Pick the last non-stop word
            for idx in range(total_words - 1, -1, -1):
                cw = clean_words[idx]
                if cw and cw not in CLITICS_AND_STOP_WORDS and len(cw) >= 3:
                    return wrap_word_index_with_bold(words, idx)
        return sentence

    # Sort candidates by highest score
    scores.sort(key=lambda item: item[0], reverse=True)

    # Determine number of words to emphasize:
    # 3 to 7 words: exactly 1 word
    # 8+ words: up to 2 words if both have high tier scores (>= 25) and are separated
    best_tier, best_idx = scores[0]
    emphasized_indices = [best_idx]

    if total_words >= 8 and len(scores) > 1:
        second_tier, second_idx = scores[1]
        # Only add second emphasis if it's high tier and at least 2 words away
        if second_tier >= 25 and abs(second_idx - best_idx) >= 2:
            emphasized_indices.append(second_idx)

    # Build final sentence with **...**
    res = []
    for idx, w in enumerate(words):
        if idx in emphasized_indices:
            res.append(format_word_with_bold(w))
        else:
            res.append(w)

    return " ".join(res)


def apply_auto_emphasis(uz_text: str, en_orig: str = "") -> str:
    """
    Main entry point for Step 3.3: Avtomatik Semantik Urg'u.
    
    1. If the entire text is wrapped in **bold**, strips the full bolding and predicts
       key focal emphasis so the bubble is never universally bold.
    2. If the text already contains 1-3 well-formed bold tags, validates and keeps them.
    3. If en_orig has explicit uppercase or bold emphasis, projects it to Uzbek.
    4. Otherwise, processes each sentence and predicts comic semantic emphasis.
    """
    if not uz_text or not uz_text.strip():
        return ""

    text = uz_text.strip()

    # Rule 1: Check for erroneous full bolding
    if is_fully_bolded(text):
        text = strip_all_bold_tags(text)
        # Proceed to re-predict emphasis correctly

    # Rule 2: If existing balanced emphasis is already present and reasonable (1 to 3 bold words)
    if has_existing_emphasis(text):
        bold_matches = re.findall(r"\*\*([^\*]+)\*\*", text)
        total_bold_words = sum(len(m.split()) for m in bold_matches)
        all_words = len(text.split())
        # If between 1 and 3 words bolded and <= 40% of total words, it's valid comic emphasis!
        if 1 <= total_bold_words <= 3 and (total_bold_words / max(1, all_words)) <= 0.40:
            return text

    # Rule 3: Try projecting English source emphasis if available
    if en_orig:
        projected = project_english_emphasis(en_orig, text)
        if projected:
            return projected

    # Rule 4: Algorithmic sentence-by-sentence prediction
    lines = text.split("\n")
    processed_lines = []

    for line in lines:
        if not line.strip():
            processed_lines.append("")
            continue

        # Split into sentence clauses (. ! ? …)
        # Using lookbehind so delimiter remains with preceding sentence
        raw_sentences = re.split(r"(?<=[.!?…])\s+", line)
        emphasized_sentences = []

        for s in raw_sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            if has_existing_emphasis(s_clean):
                emphasized_sentences.append(s_clean)
            else:
                emp = predict_semantic_emphasis_sentence(s_clean)
                emphasized_sentences.append(emp)

        processed_lines.append(" ".join(emphasized_sentences))

    return "\n".join(processed_lines)
