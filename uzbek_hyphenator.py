"""
uzbek_hyphenator.py
-------------------
Algorithmic Uzbek Syllabification and Hyphenation Engine for Comic Lettering.
Adheres to the official rules of Uzbek orthography (O'zbek adabiy tili imlo qoidalari):
1. Vowels and syllable counts: Exactly one vowel per syllable.
2. V-C-V: Single consonant moves to the next syllable (bo-la, o-na, ki-tob, qah-ra-mon).
3. V-CC-V: Two consonants split (mak-tab, ish-chi, kat-ta, par-da), except digraphs (sh, ch, ng).
4. V-CCC-V: Three consonants split after the second consonant or digraph (dast-lab, hukm-dor).
5. Digraph protection: 'sh', 'ch', 'ng' and 'g\'' are atomic and never split.
6. Vowel 'o\'' is atomic and never split.
7. Apostrophe/tutuq (', ’, ʼ) stays with the preceding syllable (ta'-lim, a'-lo).
8. 2-Letter Boundary Constraint: Never leave a single letter at line end (a-ka -> forbidden).
9. Entity & Proper Noun Protection: Character names (Spider-Man, Jonni, Doom) are never split.
"""

from __future__ import annotations

import os
import json
import re
from typing import List, Tuple, Optional, Set, Callable, Any

# Core Uzbek vowel representations (Latin & common variations)
VOWEL_PATTERNS = [
    "o'", "o‘", "oʻ", "o’", "o`",
    "a", "e", "i", "o", "u"
]

# Core Uzbek consonant digraphs (treated as atomic single consonants)
CONSONANT_DIGRAPHS = [
    "g'", "g‘", "gʻ", "g’", "g`",
    "sh", "ch", "ng"
]

# Base protected proper nouns and hero entities that must NEVER be hyphenated internally
DEFAULT_PROTECTED_ENTITIES: Set[str] = {
    # Marvel / DC Universe & Comic characters
    "spider-man", "spiderman", "o'rgimchak-odam", "o'rgimchak", "spidey",
    "peter", "parker", "piter", "jonni", "johnny", "storm",
    "doctor doom", "doom", "dum", "victor", "von doom", "viktor",
    "otto", "octavius", "oktavius", "doc ock",
    "batman", "betmen", "bruce", "wayne", "bryus", "ueyn", "gotham", "gotem",
    "hulk", "xalk", "thor", "tor", "loki", "iron man", "tony", "stark",
    "captain america", "kapitan amerika", "steve", "rogers",
    "superman", "supermen", "clark", "kent",
    "goblin", "norman", "osborn", "jameson", "jeymson", "jona", "jonah",
    "mary jane", "meri jeyn", "watson", "uotson", "gwen", "stacy",
    "aunt may", "mey xola", "uncle ben", "ben amaki",
    "toshkent", "samarqand", "buxoro", "nyu-york", "amerika", "o'zbekiston"
}


def load_protected_entities() -> Set[str]:
    """Loads all known protected character and glossary names."""
    names = set(DEFAULT_PROTECTED_ENTITIES)

    # 1. Load from glossary.json
    try:
        glossary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glossary.json")
        if os.path.exists(glossary_path):
            with open(glossary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                terms = data.get("terms", {})
                for en_k, uz_v in terms.items():
                    for token in en_k.split():
                        clean = re.sub(r"[^\w'-]", "", token.lower())
                        if clean and len(clean) >= 3:
                            names.add(clean)
                    for token in uz_v.split():
                        clean = re.sub(r"[^\w'-]", "", token.lower())
                        if clean and len(clean) >= 3:
                            names.add(clean)
    except Exception:
        pass

    # 2. Load from character_memory.json
    try:
        mem_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "character_memory.json")
        if os.path.exists(mem_path):
            with open(mem_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                profiles = c_data.get("profiles", {})
                for c_name, p_info in profiles.items():
                    names.add(c_name.lower())
                    for token in c_name.split():
                        names.add(token.lower())
                    for alias in p_info.get("aliases", []):
                        names.add(alias.lower())
    except Exception:
        pass

    return names


_PROTECTED_CACHE: Set[str] = load_protected_entities()


def is_protected_proper_noun(word: str) -> bool:
    """
    Returns True if the word is a protected character persona or proper noun
    that must not be broken across syllables with a hyphen.
    """
    clean = re.sub(r"[^\w'-]", "", word).strip().lower()
    if not clean:
        return False

    if clean in _PROTECTED_CACHE:
        return True

    # Check capitalized proper nouns (e.g. 'Jonni', 'Parker', 'Oktavius')
    # If the word starts with capital letter, is not at index 0 of sentence, or is title-cased:
    raw_clean = re.sub(r"[^\w'-]", "", word).strip()
    if raw_clean and raw_clean[0].isupper() and len(raw_clean) >= 3:
        # Title case names with no common Uzbek suffix
        if raw_clean.istitle() and raw_clean.lower() in _PROTECTED_CACHE:
            return True

    return False


def tokenize_phonemes(word: str) -> List[Tuple[str, str]]:
    """
    Tokenizes a single clean Uzbek word into phonetic units:
    - ('V', vowel_str): 'a', 'e', 'i', 'o', 'u', or 'o\'', 'o‘', etc.
    - ('C', consonant_str): single consonant or digraph 'g\'', 'sh', 'ch', 'ng'
    - ('A', apostrophe_str): glottal stop / tutuq mark
    """
    w_lower = word.lower()
    n = len(w_lower)
    tokens: List[Tuple[str, str]] = []
    i = 0

    while i < n:
        matched = False

        # 1. Two-character vowel ('o\'', 'o‘', 'oʻ', 'o’', 'o`')
        if i + 1 < n:
            pair = w_lower[i:i+2]
            if pair in ("o'", "o‘", "oʻ", "o’", "o`"):
                tokens.append(("V", word[i:i+2]))
                i += 2
                continue

            # 2. Two-character consonant ('g\'', 'g‘', 'gʻ', 'g’', 'g`', 'sh', 'ch', 'ng')
            if pair in ("g'", "g‘", "gʻ", "g’", "g`", "sh", "ch", "ng"):
                tokens.append(("C", word[i:i+2]))
                i += 2
                continue

        # 3. Single-character vowel
        ch = w_lower[i]
        if ch in "aeiouаеёиоуўэюя":
            tokens.append(("V", word[i]))
            i += 1
            continue

        # 4. Apostrophe / Tutuq mark
        if ch in ("'", "’", "ʼ", "`"):
            tokens.append(("A", word[i]))
            i += 1
            continue

        # 5. Consonant (or symbol)
        if ch.isalpha():
            tokens.append(("C", word[i]))
        else:
            tokens.append(("S", word[i]))
        i += 1

    return tokens


def syllabify_uzbek(word: str) -> List[str]:
    """
    Decomposes an Uzbek word into its linguistic syllables (bo'g'inlar).
    Example:
        'qahramon' -> ['qah', 'ra', 'mon']
        'maktab'   -> ['mak', 'tab']
        'bolalar'  -> ['bo', 'la', 'lar']
        'katta'    -> ['kat', 'ta']
        'pichoq'   -> ['pi', 'choq']
        'dastlab'  -> ['dast', 'lab']
        'singil'   -> ['sin', 'gil']
    """
    # Clean surrounding punctuation
    match = re.match(r"^([^\w]*)([\w'’ʼʻ`-]+)([^\w]*)$", word)
    if not match:
        return [word]

    prefix_punct, core_word, suffix_punct = match.groups()
    if len(core_word) < 3:
        return [word]

    tokens = tokenize_phonemes(core_word)
    vowel_indices = [idx for idx, (kind, _) in enumerate(tokens) if kind == "V"]

    # Monosyllabic word: cannot be split
    if len(vowel_indices) <= 1:
        return [word]

    cut_indices: List[int] = []

    for v_idx in range(len(vowel_indices) - 1):
        v_curr = vowel_indices[v_idx]
        v_next = vowel_indices[v_idx + 1]

        # Consonants between current vowel and next vowel
        intervening = tokens[v_curr + 1 : v_next]
        num_c = len(intervening)

        if num_c == 0:
            # V - V: split directly between vowels (e.g. do-ir-a, te-atr)
            cut_indices.append(v_curr + 1)
        elif num_c == 1:
            # V - C - V: consonant attaches to next syllable (e.g. bo-la, ki-tob, qah-ra-mon)
            # If intervening is apostrophe, it stays with the current syllable
            if intervening[0][0] == "A":
                cut_indices.append(v_curr + 2)
            else:
                cut_indices.append(v_curr + 1)
        elif num_c == 2:
            # V - C1 C2 - V: split between consonants (e.g. mak-tab, ish-chi, kat-ta)
            # If first is apostrophe: ta'-lim -> split after apostrophe
            if intervening[0][0] == "A":
                cut_indices.append(v_curr + 2)
            else:
                cut_indices.append(v_curr + 2)
        else:
            # V - C1 C2 C3... - V (3 or more consonants, e.g. dast-lab, past-lik, hukm-dor)
            # Standard rule: split after second consonant
            cut_indices.append(v_curr + 3)

    # Reconstruct syllables
    syllables: List[str] = []
    start = 0
    for cut in cut_indices:
        syllable_tokens = tokens[start:cut]
        syllables.append("".join(t[1] for t in syllable_tokens))
        start = cut

    if start < len(tokens):
        syllables.append("".join(t[1] for t in tokens[start:]))

    # Reattach outer punctuation if any
    if prefix_punct and syllables:
        syllables[0] = prefix_punct + syllables[0]
    if suffix_punct and syllables:
        syllables[-1] = syllables[-1] + suffix_punct

    return syllables


def get_hyphenation_points(word: str) -> List[Tuple[str, str]]:
    """
    Returns a list of valid (prefix_with_hyphen, suffix) pairs where this word
    can be legally hyphenated at line wrap boundaries.
    
    Guarantees:
    - Never breaks protected character names (Spider-Man, Jonni, Doom).
    - Minimum word length: 5 characters.
    - Minimum 2 characters on line 1 before the hyphen (no 'a-ka').
    - Minimum 2 characters on line 2 after the hyphen (no 'eshik-i').
    """
    clean_word = re.sub(r"[^\w'-]", "", word).strip()
    # Foydalanuvchi qoidasi: 9 ta va undan kam harfli so'zlarda bo'g'in ko'chirilmaydi.
    # Faqat 10 ta va undan ortiq harfli (kamida 10 ta harf) uzun so'zlargina bo'g'inlab ko'chiriladi.
    if len(clean_word) < 10 or is_protected_proper_noun(word):
        # If word has an existing internal hyphen (like 'O'rgimchak-Odam'), allow break ONLY at that hyphen
        if "-" in clean_word and not is_protected_proper_noun(clean_word):
            parts = clean_word.split("-", 1)
            if len(parts[0]) >= 2 and len(parts[1]) >= 2:
                return [(parts[0] + "-", parts[1])]
        return []

    syllables = syllabify_uzbek(word)
    if len(syllables) <= 1:
        return []

    points: List[Tuple[str, str]] = []
    total_s = len(syllables)

    for i in range(1, total_s):
        prefix_part = "".join(syllables[:i])
        suffix_part = "".join(syllables[i:])

        clean_pref = re.sub(r"[^\w'-]", "", prefix_part)
        clean_suff = re.sub(r"[^\w'-]", "", suffix_part)

        # Enforce 2-letter minimum rule for both parts
        if len(clean_pref) >= 2 and len(clean_suff) >= 2:
            points.append((prefix_part + "-", suffix_part))

    return points


def fit_word_with_hyphen(
    word: str,
    max_w: float,
    measure_fn: Callable[[str], float]
) -> Optional[Tuple[str, str]]:
    """
    Finds the longest valid hyphenated prefix of `word` that fits within `max_w`.
    Returns (prefix_with_hyphen, suffix) or None if no valid hyphenation fits.
    """
    points = get_hyphenation_points(word)
    if not points:
        return None

    best: Optional[Tuple[str, str]] = None
    # Reverse search: prefer longer prefixes on current line for tighter comic text packing
    for pref_hyphen, suff in reversed(points):
        w = measure_fn(pref_hyphen)
        if w <= max_w:
            best = (pref_hyphen, suff)
            break

    return best


def wrap_words_with_hyphenation(
    words: List[str],
    row_widths: List[float],
    measure_fn: Callable[[str], float]
) -> Optional[List[List[str]]]:
    """
    Wraps words across comic bubble rows. When a long word does not fit on
    the remaining row space, it attempts to hyphenate the word so the prefix
    fills the current row and the suffix starts the next row.
    """
    if not words or not row_widths:
        return None

    lines: List[List[str]] = []
    cur_line: List[str] = []
    row_idx = 0
    w_idx = 0
    num_rows = len(row_widths)

    space_w = measure_fn(" ")
    cur_line_w = 0.0

    while w_idx < len(words) and row_idx < num_rows:
        max_w = row_widths[row_idx]
        word = words[w_idx]
        word_w = measure_fn(word)
        needed_w = word_w if not cur_line else (space_w + word_w)

        if cur_line_w + needed_w <= max_w:
            cur_line.append(word)
            cur_line_w += needed_w
            w_idx += 1
        else:
            # Word does not fit in remaining space: can we hyphenate it?
            rem_w = max_w - cur_line_w - (space_w if cur_line else 0.0)
            hyphen_fit = fit_word_with_hyphen(word, rem_w, measure_fn) if rem_w >= 30.0 else None

            if hyphen_fit is not None:
                pref, suff = hyphen_fit
                cur_line.append(pref)
                lines.append(cur_line)
                cur_line = []
                cur_line_w = 0.0
                row_idx += 1
                # Replace current word with remainder suffix for next line
                words = words[:w_idx] + [suff] + words[w_idx + 1:]
            else:
                # Word cannot be hyphenated into remaining space: move to next line
                if cur_line:
                    lines.append(cur_line)
                    cur_line = []
                    cur_line_w = 0.0
                    row_idx += 1
                else:
                    # Single word overflows even an entire empty line: try to hyphenate into full line width
                    full_hyphen = fit_word_with_hyphen(word, max_w, measure_fn)
                    if full_hyphen is not None:
                        pref, suff = full_hyphen
                        lines.append([pref])
                        row_idx += 1
                        words = words[:w_idx] + [suff] + words[w_idx + 1:]
                    else:
                        # Cannot hyphenate: place as-is
                        lines.append([word])
                        row_idx += 1
                        w_idx += 1

    if cur_line:
        lines.append(cur_line)

    if w_idx < len(words):
        return None  # Could not fit all words within the allotted rows

    return lines
