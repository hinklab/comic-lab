"""
terminology_store.py - Persistent, Self-Growing Canonical Terminology & Name Lock.

Guarantees that once a character name or proper noun is translated in this project,
it is permanently locked in its canonical Uzbek form across all characters, pages,
and runs. Survives across restarts via disk persistence in terminology_store.json.
"""

import json
import os
import re
import threading
from typing import Dict, List, Optional, Tuple

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terminology_store.json")


VALID_ENTRY_TYPES = {"title", "character_name", "real_person", "term"}
VALID_METHODS = {"meaning", "transliteration", "transliteration_override", "fixed"}


def validate_entry(entry: dict) -> Tuple[bool, str]:
    """
    Enforces project translation rulebook policies:
    - Rule 1: 'title' must use method 'meaning'.
    - Rule 2: 'character_name' must use method 'meaning', or 'transliteration_override' with a mandatory 'reason'.
    - Rule 3: 'real_person' must use method 'transliteration' (never meaning).
    - Rule 4: 'transliteration_override' must include non-empty 'reason'.
    - Rule 5: 'term' must use method 'fixed' or 'meaning'.
    """
    entry_type = entry.get("entry_type")
    method = entry.get("method")

    if not entry_type or entry_type not in VALID_ENTRY_TYPES:
        return False, f"Invalid or missing entry_type '{entry_type}'. Must be one of {sorted(VALID_ENTRY_TYPES)}"

    if not method or method not in VALID_METHODS:
        return False, f"Invalid or missing method '{method}'. Must be one of {sorted(VALID_METHODS)}"

    if entry_type == "title" and method != "meaning":
        return False, f"Rule 1 violation: Series/comic title '{entry.get('source_term')}' must use method 'meaning', got '{method}'"

    if entry_type == "character_name":
        if method not in ("meaning", "transliteration_override"):
            return False, f"Rule 2 violation: Character name '{entry.get('source_term')}' must use method 'meaning' or 'transliteration_override', got '{method}'"
        if method == "transliteration_override":
            reason = (entry.get("reason") or "").strip()
            if not reason:
                return False, f"Rule 4 violation: Character name override for '{entry.get('source_term')}' requires a non-empty 'reason'"

    if entry_type == "real_person" and method != "transliteration":
        return False, f"Rule 3 violation: Real person '{entry.get('source_term')}' must use method 'transliteration' (never meaning), got '{method}'"

    if entry_type == "term" and method not in ("fixed", "meaning"):
        return False, f"Rule 5 violation: Term '{entry.get('source_term')}' must use method 'fixed' or 'meaning', got '{method}'"

    return True, "valid"


class TerminologyStore:
    """Thread-safe persistent store for canonical names and terms."""
    _instance: Optional["TerminologyStore"] = None
    _lock = threading.Lock()

    def __init__(self, store_path: str = STORE_PATH):
        self.store_path = store_path
        self._entries: Dict[str, dict] = {}
        self._mtime: float = 0.0
        self.load()

    @classmethod
    def get_instance(cls, store_path: str = STORE_PATH) -> "TerminologyStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(store_path)
            else:
                cls._instance._reload_if_modified()
            return cls._instance

    def _reload_if_modified(self) -> None:
        """Reloads from disk if another process or user edited the JSON file."""
        if os.path.exists(self.store_path):
            current_mtime = os.path.getmtime(self.store_path)
            if current_mtime > self._mtime:
                self.load()

    def load(self) -> None:
        """Loads terminology entries from Supabase or disk JSON and validates policies."""
        # Try loading from Supabase terminology_store table if standard store path
        if self.store_path == STORE_PATH:
            try:
                import supabase_db
                db_terms = supabase_db.fetch_all_terms()
                if db_terms is not None and len(db_terms) > 0:
                    self._entries = {}
                    for row in db_terms:
                        norm_key = row["source_term"].strip().lower()
                        entry = {
                            "source_term": norm_key,
                            "canonical_uzbek": row["canonical_uzbek"],
                            "first_seen_page": row.get("first_seen_page", "auto"),
                            "locked": row.get("locked", True),
                            "aliases": row.get("aliases", []),
                            "entry_type": row.get("entry_type", "term"),
                            "method": row.get("method", "fixed"),
                        }
                        if row.get("reason"):
                            entry["reason"] = row["reason"]
                        self._entries[norm_key] = entry
                    if os.path.exists(self.store_path):
                        self._mtime = os.path.getmtime(self.store_path)
                    return
            except Exception as e:
                print(f"[TERMINOLOGY_WARN] Supabase load failed: {e}. Falling back to local file.")

        if not os.path.exists(self.store_path):
            self._entries = {}
            return

        try:
            with open(self.store_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
                terms_list = data.get("terms", [])
                self._entries = {}
                for t in terms_list:
                    norm_key = t["source_term"].strip().lower()
                    is_valid, err_msg = validate_entry(t)
                    if not is_valid:
                        print(f"[TERMINOLOGY_POLICY_VIOLATION] Entry '{norm_key}' failed validation: {err_msg}")
                    self._entries[norm_key] = t
            self._mtime = os.path.getmtime(self.store_path)
        except Exception as e:
            print(f"[TERMINOLOGY_WARN] Failed to load {self.store_path}: {e}")

    def save(self) -> None:
        """Atomically saves terminology entries to Supabase and disk JSON."""
        # Sync to Supabase terminology_store table if standard store path
        if self.store_path == STORE_PATH:
            try:
                import supabase_db
                for t in self._entries.values():
                    supabase_db.upsert_term(t)
            except Exception as e:
                print(f"[TERMINOLOGY_WARN] Supabase save failed: {e}. Falling back to local file.")

        data = {
            "version": 2,
            "terms": list(self._entries.values())
        }
        tmp_path = self.store_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.store_path)
            self._mtime = os.path.getmtime(self.store_path)
        except Exception as e:
            print(f"[TERMINOLOGY_WARN] Failed to save {self.store_path}: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def register_term(
        self,
        source_term: str,
        canonical_uzbek: str,
        first_seen_page: str = "auto",
        locked: bool = True,
        aliases: Optional[List[str]] = None,
        entry_type: str = "term",
        method: str = "fixed",
        reason: Optional[str] = None
    ) -> bool:
        """
        Registers a term in the store after policy validation.
        If already present and locked, never overwrites (prevents drift).
        If new, validates and saves immediately to disk.
        """
        norm_key = source_term.strip().lower()
        if not norm_key:
            return False

        candidate_entry = {
            "source_term": norm_key,
            "canonical_uzbek": canonical_uzbek,
            "first_seen_page": first_seen_page,
            "locked": locked,
            "aliases": aliases or [],
            "entry_type": entry_type,
            "method": method,
        }
        if reason:
            candidate_entry["reason"] = reason

        is_valid, err_msg = validate_entry(candidate_entry)
        if not is_valid:
            print(f"[TERMINOLOGY_POLICY_ERROR] Rejected term '{norm_key}': {err_msg}")
            raise ValueError(f"Rulebook Policy Violation: {err_msg}")

        with self._lock:
            self._reload_if_modified()
            if norm_key in self._entries:
                existing = self._entries[norm_key]
                if existing.get("locked", True):
                    # Never overwrite locked canonical entries
                    return False
                # Non-locked entry can be updated
                existing["canonical_uzbek"] = canonical_uzbek
                existing["locked"] = locked
                existing["entry_type"] = entry_type
                existing["method"] = method
                if reason:
                    existing["reason"] = reason
                if aliases:
                    existing["aliases"] = list(set(existing.get("aliases", []) + aliases))
                self.save()
                return True

            # Register brand new entry
            self._entries[norm_key] = candidate_entry
            self.save()
            print(f"[TERMINOLOGY_STORE] Registered new locked term ({entry_type}/{method}): '{norm_key}' -> '{canonical_uzbek}' (page: {first_seen_page})")
            return True

    def get_all_entries(self) -> List[Tuple[str, str, str]]:
        """
        Returns all pattern -> placeholder -> canonical_uzbek mappings,
        sorted longest pattern first to prevent substring collision.
        """
        self._reload_if_modified()
        items = []
        for norm_key, data in self._entries.items():
            canonical_uz = data["canonical_uzbek"]
            token_slug = re.sub(r"[^A-Za-z0-9]", "_", norm_key).upper().strip("_")
            placeholder = f"XNAME_{token_slug}_X"

            # Primary term pattern with word boundaries
            escaped_term = re.escape(norm_key).replace(r"\-", r"[\-\s]?")
            pattern = rf"\b{escaped_term}\b"
            items.append((pattern, placeholder, canonical_uz, len(norm_key)))

            # Alias patterns
            for alias in data.get("aliases", []):
                norm_alias = alias.strip().lower()
                if norm_alias and norm_alias != norm_key:
                    escaped_alias = re.escape(norm_alias).replace(r"\-", r"[\-\s]?")
                    alias_pat = rf"\b{escaped_alias}\b"
                    items.append((alias_pat, placeholder, canonical_uz, len(norm_alias)))

        # Sort longest match first
        items.sort(key=lambda x: x[3], reverse=True)
        return [(pat, tok, uz) for pat, tok, uz, _ in items]

    def mask_terms(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Replaces recognized terms in source English text with safe NLLB-safe tokens
        before MT runs, preventing MT from mangling or re-translating locked names.
        """
        if not text:
            return text, {}

        masked = text
        unmask_map: Dict[str, str] = {}
        entries = self.get_all_entries()

        for pattern, placeholder, canonical_uz in entries:
            if re.search(pattern, masked, flags=re.IGNORECASE):
                masked = re.sub(pattern, placeholder, masked, flags=re.IGNORECASE)
                unmask_map[placeholder] = canonical_uz

        return masked, unmask_map

    def unmask_terms(self, text: str, unmask_map: Optional[Dict[str, str]] = None) -> str:
        """
        Restores token placeholders in translated text to canonical Uzbek spelling.
        Also runs safety corrections against known phonetically mangled variants.
        """
        if not text:
            return text

        result = text

        # Step 1: Explicit unmask map
        if unmask_map:
            for placeholder, canonical_uz in unmask_map.items():
                result = result.replace(placeholder, canonical_uz)

        # Step 2: Global unmask from all stored entries
        for _, placeholder, canonical_uz in self.get_all_entries():
            result = result.replace(placeholder, canonical_uz)

        # Step 2b: Catch NLLB transliterated/mutated token placeholders
        result = re.sub(r"XNAME_M[AJ]Y?OR[A-Z0-9_]*_X", "Mer J. Jona Jeymson", result, flags=re.IGNORECASE)
        result = re.sub(r"XNAME_SPIDER[A-Z0-9_]*_X", "O'rgimchak-Odam", result, flags=re.IGNORECASE)
        result = re.sub(r"XNAME_OCTAVIUS[A-Z0-9_]*_X", "Otto Oktavius", result, flags=re.IGNORECASE)
        result = re.sub(r"XNAME_DOCTOR_OCTOPUS[A-Z0-9_]*_X", "Doktor Oktopus", result, flags=re.IGNORECASE)

        # Step 3: Global safety corrections for common MT/phonetic corruptions
        safety_corrections = [
            (r"\bJamison\b", "Jeymson"),
            (r"\bJeymison\b", "Jeymson"),
            (r"\bGoldamon\b", "Goldman"),
            (r"\bGoldmanmanman\b", "Goldman"),
            (r"\bRut\s+Goldamon\b", "Rut Goldman"),
            (r"\bRuta\s+Goldman\b", "Rut Goldman"),
            (r"\bRuta\s+Goldamon\b", "Rut Goldman"),
            (r"\bJonasi\b", "Jona"),
            (r"\bJonay\b", "Jona"),
            (r"\bXotiriq\b", "Xotira"),
            (r"\bQo['’]?rg['’]?oqchi\b", "O'rgimchak-Odam"),
            (r"\bQo['’]?riqchi\s+Minorasi\s+ning\s+o['’]?zi\s+emas\b", "O'rgimchak-Odam emas"),
            (r"\bQo['’]?riqchi\s+Minorasi\s+Jamiyati\b", "O'rgimchak-Odam"),
            (r"\bQo['’]?riqchi\s+Minorasi\b", "O'rgimchak-Odam"),
            (r"\bo['’]?z-o['’]?zidan\s+ko['’]?p\s+narsa\s+emas\b", "O'rgimchak-Odam emas"),
            (r"\bDoktor\s+Oktopus\s+mening\s+dasturimda\b", "Doktor Oktopus mening tanamda"),
            (r"\bDoctor\s+Octopus\s+mening\s+dasturimda\b", "Doktor Oktopus mening tanamda"),
            (r"\bDoktor\s+Oktopusning\s+braini\b", "Doktor Oktopusning miyasi"),
            (r"\bDoctor\s+Octopusning\s+braini\b", "Doktor Oktopusning miyasi"),
            (r"\byou\s+tekis\s+kalla\s+ablah\b", "ey tekis kalla ablah"),
            (r"\bflaat-topped\s+finka?\b", "tekis kalla ablah"),
            (r"\bflat-topped\s+finka?\b", "tekis kalla ablah"),
            (r"\bmening\s+o['’]?lim\s+tanam\s+ustidan\b", "faqat mening murdam ustidan"),
            (r"\bo['’]?lim\s+tanam\s+ustidan\b", "murdam ustidan"),
            (r"\bO['’]?simlik\s+odam\b", "O'rgimchak-Odam"),
            (r"\bo['’]?simlikdagi\s+o['’]?simlik\s+odam\b", "O'rgimchak-Odam"),
            (r"\bO['’]?rgimchak[\-\s]yigit\b", "O'rgimchak-Odam"),
            (r"\bO['’]?rgimchak[\-\s]bola\b", "O'rgimchak-Odam"),
            (r"\bO['’]?rgimchak[\-\s]odam\b", "O'rgimchak-Odam"),
        ]
        for pat, rep in safety_corrections:
            result = re.sub(pat, rep, result, flags=re.IGNORECASE)

        return result

    def learn_from_page(self, dialogue_pairs: List[Tuple[str, str]], page_source: str = "auto") -> None:
        """
        Scans translated dialogue pairs from a page for new proper noun candidates.
        If a new proper noun appears consistently, locks it into the store.
        """
        known_terms = set(self._entries.keys())
        for en, uz in dialogue_pairs:
            if not en or not uz:
                continue
            # Look for Title Case multi-word named entities in English
            candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", en)
            for cand in candidates:
                norm_cand = cand.strip().lower()
                if norm_cand not in known_terms:
                    # If this candidate term is not in store, we register it
                    # Attempt to find corresponding capitalized phrase in Uzbek
                    uz_matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", uz)
                    if uz_matches:
                        canonical_candidate = uz_matches[0]
                        self.register_term(
                            source_term=norm_cand,
                            canonical_uzbek=canonical_candidate,
                            first_seen_page=page_source,
                            locked=True,
                            entry_type="term",
                            method="fixed"
                        )
                        known_terms.add(norm_cand)


# Convenience singleton accessor
def get_terminology_store() -> TerminologyStore:
    return TerminologyStore.get_instance()
