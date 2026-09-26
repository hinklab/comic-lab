"""
names_glossary.py - Mandatory Proper Noun & Name Transliteration Glossary.

Backed by persistent TerminologyStore (terminology_store.json) to ensure permanent
terminology locking across runs, characters, and pages.
Uses NLLB-safe alphanumeric tokens (e.g. XNAME_SPIDER_MAN_X) that survive NMT untouched.
"""

import re
from typing import Dict, List, Tuple
import terminology_store

# Access the persistent terminology store singleton
_store = terminology_store.get_terminology_store()

# Canonical Name Mappings (maintained for backwards compatibility with tests)
GLOSSARY_ENTRIES: List[Tuple[str, str, str]] = _store.get_all_entries()


def mask_proper_nouns(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replaces recognized proper nouns in English text with safe token placeholders
    before translation so the MT engine cannot corrupt or transliterate them erratically.
    Delegates to persistent TerminologyStore.
    """
    return _store.mask_terms(text)


def unmask_proper_nouns(text: str, unmask_map: Dict[str, str] = None) -> str:
    """
    Restores token placeholders in translated text to their standardized Uzbek spelling.
    Also acts as a safety pass to catch any unmasked variations (e.g. 'Jamison' -> 'Jeymson').
    Delegates to persistent TerminologyStore.
    """
    return _store.unmask_terms(text, unmask_map)

