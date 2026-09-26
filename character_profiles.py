"""
character_profiles.py - Persistent, Self-Growing Character Memory & Voice Profiles.

Provides character-specific dialogue styles, vocabulary registers, verbal tics,
and formality adjustments so translated comic dialogue matches the distinct
personality of each character, rather than generic translation.

ARCHITECTURE & BOUNDARY NOTE:
NLLB-200 is an offline statistical MT model (CTranslate2 seq2seq), not an LLM.
It does not natively interpret free-form persona prompts. Therefore, character
voice styling operates via a 3-tier hybrid pipeline:
1. Base NMT: Local offline NLLB model produces grammatical Uzbek baseline.
2. Dialogue & Idioms: naturalization_rules.py applies comic conventions.
3. Character Persona: character_profiles.py applies character-specific phrase
   replacements, exact line anchors, and vocabulary elevation.
Limitation: Novel, unseen lines spoken by characters will default to NLLB neutral
tone until their idioms are registered in phrase_replacements or captured via
the self-growing user feedback loop (record_user_edited_sample_line).
"""

import difflib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

MEMORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "character_memory.json")


@dataclass
class CharacterVoiceProfile:
    name: str
    aliases: List[str]
    voice_traits: List[str]
    forbidden_tones: List[str] = field(default_factory=list)
    established_facts: List[str] = field(default_factory=list)
    sample_lines: List[Dict[str, str]] = field(default_factory=list)
    phrase_replacements: List[Tuple[str, str]] = field(default_factory=list)
    exact_lines: Dict[str, str] = field(default_factory=dict)
    # Visual signature for automatic speaker detection (classical CV, k-means colours).
    # Schema: {
    #   "dominant_colors": [[R,G,B], ...],  <- k-means cluster centres in RGB
    #   "color_weights":   [float, ...],     <- fractional area weight per cluster
    #   "sample_count":    int,              <- confirmed panels that contributed
    #   "last_updated":    "YYYY-MM-DD"
    # }
    visual_signature: Optional[Dict] = field(default=None)

    def matches(self, query: str) -> bool:
        """Checks if a character name or alias matches this profile."""
        q = query.strip().lower()
        if q == self.name.lower():
            return True
        return any(q == alias.lower() for alias in self.aliases)

    @staticmethod
    def _normalize_key(text: str) -> str:
        s = text.strip().upper()
        s = re.sub(r"^[\s'\"\.\-]+", "", s)
        s = re.sub(r"[\s'\"\.\-!?,:]+$", "", s)
        s = s.replace("’", "'").replace("`", "'")
        s = re.sub(r"\bI\s+AM\b", "I'M", s)
        s = re.sub(r"\bIM\b", "I'M", s)
        s = re.sub(r"\bTHEY\s*['’]?\s*LL\b", "THEY'LL", s)
        s = re.sub(r"\bTHEY\s+WILL\b", "THEY'LL", s)
        s = re.sub(r"\bTHEYLL\b", "THEY'LL", s)
        s = re.sub(r"\bYOU\s+ARE\b", "YOU'RE", s)
        s = re.sub(r"\bYOURE\b", "YOU'RE", s)
        s = re.sub(r"\bDO\s+NOT\b", "DON'T", s)
        s = re.sub(r"\bDONT\b", "DON'T", s)
        s = re.sub(r"\bCANNOT\b", "CAN'T", s)
        s = re.sub(r"\bCANT\b", "CAN'T", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def naturalize(self, uz_text: str, en_orig: str = "") -> str:
        """
        Applies character-specific persona naturalization to Uzbek dialogue:
        1. Checks exact character lines (normalized for punctuation and quotes)
        2. Applies character-specific elevated lexicon / phrase transformations
        3. Dynamically reacts to learned established facts in character memory
        4. Cleans any forbidden casual/slang tones
        """
        if not uz_text:
            return ""

        en_clean = en_orig.strip().upper() if en_orig else ""

        # Step 1: Exact persona iconic line match
        if en_clean:
            if en_clean in self.exact_lines:
                return self.exact_lines[en_clean]
            norm_en = self._normalize_key(en_clean)
            for k, val in self.exact_lines.items():
                if self._normalize_key(k) == norm_en:
                    return val

        result = uz_text.strip()

        # Step 2: Apply persona phrase replacements (lexical elevation & tone shifts)
        for pat, rep in self.phrase_replacements:
            result = re.sub(pat, rep, result, flags=re.IGNORECASE)

        # Step 3: Dynamic adjustments based on established character memory facts
        if any("spider-bot" in f.lower() for f in self.established_facts):
            result = re.sub(r"\b(mening\s+)?robotlar(im)?\b", r"mening Spider-botlar\2", result, flags=re.IGNORECASE)

        if any("ustun" in f.lower() or "superior" in f.lower() for f in self.established_facts):
            result = re.sub(r"\bkuchli\s+aql\b", "ustun tafakkur", result, flags=re.IGNORECASE)
            result = re.sub(r"\byuqori\s+darajali\b", "ancha ustun", result, flags=re.IGNORECASE)

        # Step 4: Polish punctuation & typography
        if any(w in result.upper() for w in ["USTUN", "AHMOQLAR", "MENING", "DOHIYONA", "BEQAROR"]):
            if result.endswith("."):
                result = result[:-1] + "!"

        return result

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "aliases": self.aliases,
            "voice_traits": self.voice_traits,
            "forbidden_tones": self.forbidden_tones,
            "established_facts": self.established_facts,
            "sample_lines": self.sample_lines,
            "exact_lines": self.exact_lines,
            "phrase_replacements": [list(r) for r in self.phrase_replacements],
        }
        if self.visual_signature is not None:
            d["visual_signature"] = self.visual_signature
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "CharacterVoiceProfile":
        raw_reps = data.get("phrase_replacements", [])
        phrase_replacements = [
            (r[0], r[1]) if isinstance(r, (list, tuple)) and len(r) >= 2 else (str(r), "")
            for r in raw_reps
        ]
        return cls(
            name=data.get("name", "Unknown"),
            aliases=data.get("aliases", []),
            voice_traits=data.get("voice_traits", []),
            forbidden_tones=data.get("forbidden_tones", []),
            established_facts=data.get("established_facts", []),
            sample_lines=data.get("sample_lines", []),
            phrase_replacements=phrase_replacements,
            exact_lines=data.get("exact_lines", {}),
            visual_signature=data.get("visual_signature", None),
        )


def is_meaningful_edit(pipeline_text: str, edited_text: str) -> bool:
    """
    Determines if a user edit is a meaningful stylistic/lexical change
    rather than trivial whitespace, casing, punctuation, or 1-character typo fix.
    """
    p = (pipeline_text or "").strip()
    e = (edited_text or "").strip()
    if not e or p == e:
        return False

    # Normalize away punctuation and whitespace
    norm_p = re.sub(r'[\s\.,!?:;\-\'"’ʻʼ_—]+', '', p).lower()
    norm_e = re.sub(r'[\s\.,!?:;\-\'"’ʻʼ_—]+', '', e).lower()

    if norm_p == norm_e:
        # Trivial punctuation, whitespace, or casing change
        return False

    # Typo fix check: if lengths are significant and character sequence similarity is very high (>= 0.94)
    if len(norm_p) >= 10 and len(norm_e) >= 10:
        sim = difflib.SequenceMatcher(None, norm_p, norm_e).ratio()
        if sim >= 0.94:
            return False

    return True


# ==============================================================================
# PERSISTENT CHARACTER MEMORY STORE
# ==============================================================================

class CharacterMemoryStore:
    """
    Thread-safe persistent store for character voice profiles, traits, and facts.
    Backs onto character_memory.json and provides self-growing semantic memory.
    """
    _instance: Optional["CharacterMemoryStore"] = None
    _lock = threading.Lock()

    def __init__(self, memory_path: str = MEMORY_PATH):
        self.memory_path = memory_path
        self._profiles: Dict[str, CharacterVoiceProfile] = {}
        self._mtime: float = 0.0
        self.load()

    @classmethod
    def get_instance(cls, memory_path: str = MEMORY_PATH) -> "CharacterMemoryStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(memory_path)
            else:
                cls._instance._reload_if_modified()
            return cls._instance

    def _reload_if_modified(self) -> None:
        if os.path.exists(self.memory_path):
            current_mtime = os.path.getmtime(self.memory_path)
            if current_mtime > self._mtime:
                self.load()

    def load(self) -> None:
        # Try loading from Supabase character_profiles table if using standard memory path
        if self.memory_path == MEMORY_PATH:
            try:
                import supabase_db
                db_profiles = supabase_db.fetch_character_profiles()
                if db_profiles is not None and len(db_profiles) > 0:
                    self._profiles = {}
                    for row in db_profiles:
                        p = CharacterVoiceProfile.from_dict({
                            "name": row.get("character_name"),
                            "voice_traits": row.get("voice_traits", []),
                            "established_facts": row.get("established_facts", []),
                            "sample_lines": row.get("sample_lines", []),
                            "visual_signature": row.get("visual_signature"),
                            "aliases": row.get("aliases", []),
                            "forbidden_tones": row.get("forbidden_tones", []),
                            "phrase_replacements": row.get("phrase_replacements", []),
                            "exact_lines": row.get("exact_lines", {}),
                        })
                        for sp in DEFAULT_SEEDS:
                            if sp.name.lower() == p.name.lower() or any(p.matches(a) for a in sp.aliases):
                                if not p.aliases:
                                    p.aliases = list(sp.aliases)
                                if not p.forbidden_tones:
                                    p.forbidden_tones = list(sp.forbidden_tones)
                                if not p.phrase_replacements:
                                    p.phrase_replacements = list(sp.phrase_replacements)
                                if not p.exact_lines:
                                    p.exact_lines = dict(sp.exact_lines)
                                break
                        key = p.name.lower().replace(" ", "_").replace("-", "_")
                        self._profiles[key] = p

                    # Ensure essential seed profiles are present if missing
                    for sp in DEFAULT_SEEDS:
                        key = sp.name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                        if key not in self._profiles and not any(p.matches(sp.name) for p in self._profiles.values()):
                            self._profiles[key] = sp

                    if os.path.exists(self.memory_path):
                        self._mtime = os.path.getmtime(self.memory_path)
                    return
            except Exception as e:
                print(f"[CHARACTER_MEMORY_WARN] Supabase load failed: {e}. Falling back to local file.")

        if not os.path.exists(self.memory_path):
            self._seed_default()
            return

        try:
            with open(self.memory_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            chars = data.get("characters", [])
            self._profiles = {}
            for c in chars:
                p = CharacterVoiceProfile.from_dict(c)
                key = p.name.lower().replace(" ", "_").replace("-", "_")
                self._profiles[key] = p
            self._mtime = os.path.getmtime(self.memory_path)
            # Ensure essential seed profiles are present if missing
            updated = False
            for sp in DEFAULT_SEEDS:
                key = sp.name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                if key not in self._profiles and not any(p.matches(sp.name) for p in self._profiles.values()):
                    self._profiles[key] = sp
                    updated = True
            if updated:
                self.save()
            if not self._profiles:
                self._seed_default()
        except Exception as e:
            print(f"[CHARACTER_MEMORY_WARN] Failed loading {self.memory_path}: {e}")
            self._seed_default()

    def save(self) -> None:
        # Sync to Supabase character_profiles table if standard memory path
        if self.memory_path == MEMORY_PATH:
            try:
                import supabase_db
                for p in self._profiles.values():
                    supabase_db.upsert_character_profile(p.to_dict())
            except Exception as e:
                print(f"[CHARACTER_MEMORY_WARN] Supabase save failed: {e}. Falling back to local file.")

        # Always maintain local JSON fallback
        data = {
            "characters": [p.to_dict() for p in self._profiles.values()],
            "version": 1
        }
        tmp_path = self.memory_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.memory_path)
            self._mtime = os.path.getmtime(self.memory_path)
        except Exception as e:
            print(f"[CHARACTER_MEMORY_WARN] Failed saving {self.memory_path}: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _seed_default(self) -> None:
        self._profiles = {}
        for sp in DEFAULT_SEEDS:
            key = sp.name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
            self._profiles[key] = sp
        self.save()

    def get_profile(self, name_or_alias: Optional[str]) -> Optional[CharacterVoiceProfile]:
        if not name_or_alias or not str(name_or_alias).strip():
            return None

        self._reload_if_modified()
        query = str(name_or_alias).strip()
        key = query.lower().replace(" ", "_").replace("-", "_")

        if key in self._profiles:
            return self._profiles[key]

        for p in self._profiles.values():
            if p.matches(query):
                return p

        # Check archetype voice profiles by alias
        for arch in ARCHETYPE_PROFILES:
            if arch.matches(query):
                return arch

        return None

    def list_profiles(self) -> List[str]:
        self._reload_if_modified()
        return [p.name for p in self._profiles.values()]

    def register_profile(self, profile: CharacterVoiceProfile) -> None:
        with self._lock:
            key = profile.name.lower().replace(" ", "_").replace("-", "_")
            self._profiles[key] = profile
            self.save()

    def get_established_facts(self, character_name: str) -> List[str]:
        p = self.get_profile(character_name)
        return list(p.established_facts) if p else []

    @staticmethod
    def calculate_similarity(s1: str, s2: str) -> float:
        """
        Calculates hybrid string/token similarity between two facts or traits:
        - SequenceMatcher ratio captures exact phrase character ordering.
        - Token Jaccard index captures vocabulary overlap.
        - Token subset containment captures elaboration/subset similarity.
        """
        s1_clean = s1.lower().strip()
        s2_clean = s2.lower().strip()
        seq = difflib.SequenceMatcher(None, s1_clean, s2_clean).ratio()
        tok1 = set(re.findall(r"\w+", s1_clean))
        tok2 = set(re.findall(r"\w+", s2_clean))
        if not tok1 or not tok2:
            return seq

        jaccard = len(tok1 & tok2) / len(tok1 | tok2)
        shorter, longer = (tok1, tok2) if len(tok1) <= len(tok2) else (tok2, tok1)
        containment = len(shorter & longer) / max(len(shorter), 1)

        score = max(seq, (seq * 0.4 + jaccard * 0.6), (containment * 0.7 + seq * 0.3))
        return min(score, 1.0)

    def add_fact(
        self,
        character_name: str,
        new_fact: str,
        page: str = "unknown",
        sim_threshold: float = 0.65
    ) -> bool:
        """
        Adds an established fact for a character if not already present.
        Uses similarity calculation against existing facts.
        If similarity >= sim_threshold, logs rejection and discards.
        If genuinely new, appends, logs addition, and saves to disk immediately.
        """
        fact_str = new_fact.strip()
        if not fact_str:
            return False

        with self._lock:
            self._reload_if_modified()
            profile = self.get_profile(character_name)
            if profile is None:
                profile = CharacterVoiceProfile(
                    name=character_name.strip(),
                    aliases=[character_name.strip().lower()],
                    voice_traits=[],
                    forbidden_tones=[],
                    established_facts=[],
                    sample_lines=[],
                    phrase_replacements=[],
                    exact_lines={}
                )
                self._profiles[character_name.lower().replace(" ", "_").replace("-", "_")] = profile

            # Check similarity against all existing facts
            max_sim = 0.0
            matching_fact = ""
            for existing in profile.established_facts:
                sim = self.calculate_similarity(fact_str, existing)
                if sim > max_sim:
                    max_sim = sim
                    matching_fact = existing

            if max_sim >= sim_threshold:
                print(f"[CHARACTER_MEMORY] Skipped duplicate fact for {profile.name} on {page}: \"{fact_str}\" (matches: \"{matching_fact}\", sim={max_sim:.2f})")
                return False

            profile.established_facts.append(fact_str)
            self.save()
            print(f"[CHARACTER_MEMORY] Added new fact for {profile.name} on {page}: \"{fact_str}\"")
            return True

    def add_voice_trait(
        self,
        character_name: str,
        new_trait: str,
        page: str = "unknown",
        sim_threshold: float = 0.65
    ) -> bool:
        """Adds a voice trait with deduplication check."""
        trait_str = new_trait.strip()
        if not trait_str:
            return False

        with self._lock:
            self._reload_if_modified()
            profile = self.get_profile(character_name)
            if profile is None:
                return False

            max_sim = 0.0
            matching_trait = ""
            for existing in profile.voice_traits:
                sim = self.calculate_similarity(trait_str, existing)
                if sim > max_sim:
                    max_sim = sim
                    matching_trait = existing

            if max_sim >= sim_threshold:
                print(f"[CHARACTER_MEMORY] Skipped duplicate trait for {profile.name} on {page}: \"{trait_str}\" (matches: \"{matching_trait}\", sim={max_sim:.2f})")
                return False

            profile.voice_traits.append(trait_str)
            self.save()
            print(f"[CHARACTER_MEMORY] Added new voice trait for {profile.name} on {page}: \"{trait_str}\"")
            return True

    def learn_from_page(
        self,
        character_name: str,
        dialogue_pairs: List[Tuple[str, str]],
        page: str = "unknown",
        sim_threshold: float = 0.65
    ) -> int:
        """
        Scans dialogue pairs from a comic page and extracts structural/behavioral facts.
        IMPORTANT: established_facts are derived EXCLUSIVELY from the ENGLISH source text
        to ensure stability and prevent translation variance from distorting character traits.
        Does NOT capture sample_lines or exact_lines here (those are strictly captured from
        the user's post-review approved edits).
        """
        added_count = 0
        profile = self.get_profile(character_name)
        if profile is None:
            return 0

        for en_raw, _ in dialogue_pairs:
            en = (en_raw or "").strip()
            if not en:
                continue

            # Rule 1: Tactical Plan codes (e.g. Plan Epsilon Five, Plan Omega Three)
            if re.search(r"\bPLAN[:\s]+[A-Z0-9\-]+", en, re.IGNORECASE):
                cand = "Executes tactical contingencies with Greek letter designations ('Reja: Epsilon-besh')"
                if self.add_fact(character_name, cand, page=page, sim_threshold=sim_threshold):
                    added_count += 1

            # Rule 2: Frictionless / chemical surface technology
            if re.search(r"\bFRICTIONLESS\b", en, re.IGNORECASE):
                cand = "Deploys instantaneous frictionless surface chemical compound against adversaries"
                if self.add_fact(character_name, cand, page=page, sim_threshold=sim_threshold):
                    added_count += 1

            # Rule 3: Diagnostic anatomical combat assessments
            if re.search(r"\b(TRACHEA|LARYNX)\b", en, re.IGNORECASE):
                cand = "Despises teenage slang and insists on clinical scientific precision ('lat yegan nafas yo\\'li', 'ezilgan hiqildoq')"
                if self.add_fact(character_name, cand, page=page, sim_threshold=sim_threshold):
                    added_count += 1

            # Rule 4: Autonomous Spider-bot network
            if re.search(r"\bSPIDER[\-\s]?BOTS?\b", en, re.IGNORECASE):
                cand = "Names his autonomous surveillance units Spider-bots ('mening Spider-botlarim')"
                if self.add_fact(character_name, cand, page=page, sim_threshold=sim_threshold):
                    added_count += 1

            # Rule 5: Superior replacement Spider-Man / secret identity
            if re.search(r"\b(DIFFERENT\s+SPIDER[\-\s]?MAN|SUPERIOR\s+ONE|FAR\s+SUPERIOR|SUPERIOR\s+INTELLECT)\b", en, re.IGNORECASE):
                cand = "Refers to himself in third person or as an intellectual superior ('mutlaqo ustun', 'katta aql')"
                if self.add_fact(character_name, cand, page=page, sim_threshold=sim_threshold):
                    added_count += 1

            # Rule 6: Marla Jameson Memorial Wing / University property recovery
            if re.search(r"\b(MARLA\s+JAMESON|EMPIRE\s+STATE|RUTH\s+GOLDMAN)\b", en, re.IGNORECASE):
                cand = "Claims responsibility for university property recovery ('Marla Jeymson xotira binosiga qaytarganim uchunmi?')"
                if self.add_fact(character_name, cand, page=page, sim_threshold=sim_threshold):
                    added_count += 1

            # Rule 7: Formal address to women / public
            if re.search(r"\bMA['’]?AM\b", en, re.IGNORECASE):
                cand = "Addresses women formally with polite condescension ('Xonim')"
                if self.add_fact(character_name, cand, page=page, sim_threshold=sim_threshold):
                    added_count += 1

        return added_count

    def record_user_edited_sample_line(
        self,
        character_name: str,
        en_text: str,
        edited_uzbek: str,
        pipeline_uzbek: str = "",
        page: str = "unknown",
        sim_threshold: float = 0.65
    ) -> Tuple[bool, str]:
        """
        Captures the user's approved/edited Uzbek dialogue as a concrete style-anchor sample_line.
        This is the single ground truth for how a character should speak in Uzbek.
        
        - Filters out identical or trivial edits (punctuation, whitespace, casing, 1-char typos).
        - Deduplicates against existing sample_lines:
          * If an entry for this English dialogue already exists and matches edited text: skips duplicate.
          * If user is updating their approved translation for this dialogue: updates in-place without stale duplicates.
          * If genuinely new: appends to sample_lines and exact_lines, saves to disk.
        """
        en = (en_text or "").strip()
        edited = (edited_uzbek or "").strip()
        pipeline = (pipeline_uzbek or "").strip()

        if not en or not edited:
            return False, "empty_text"

        if not is_meaningful_edit(pipeline, edited):
            return False, "trivial_or_identical"

        with self._lock:
            self._reload_if_modified()
            profile = self.get_profile(character_name)
            if profile is None:
                return False, "character_not_found"

            norm_en = CharacterVoiceProfile._normalize_key(en)

            # Check if a sample_line for this English dialogue already exists
            existing_idx = None
            for idx, s in enumerate(profile.sample_lines):
                if CharacterVoiceProfile._normalize_key(s.get("en", "")) == norm_en:
                    existing_idx = idx
                    break

            if existing_idx is not None:
                existing_sample = profile.sample_lines[existing_idx]
                cur_voice_uz = existing_sample.get("voice_uz", "").strip()

                # If text is already identical to the current recorded voice_uz, skip as duplicate
                if cur_voice_uz == edited:
                    print(f"[CHARACTER_MEMORY] Skipped duplicate sample line for {profile.name} on {page}: \"{en}\" -> \"{edited}\"")
                    return False, "duplicate"

                # User updated their previous edit: update in-place (no stale duplicate!)
                old_val = cur_voice_uz
                existing_sample["voice_uz"] = edited
                if pipeline:
                    existing_sample["literal_mt"] = pipeline
                profile.exact_lines[norm_en] = edited
                profile.exact_lines[en.upper()] = edited
                self.save()
                print(f"[CHARACTER_MEMORY] Updated sample line for {profile.name} on {page}: \"{en}\" -> \"{edited}\" (replaced: \"{old_val}\")")
                return True, "updated"

            # Check similarity against other existing sample lines for this character
            for s in profile.sample_lines:
                other_uz = s.get("voice_uz", "")
                if other_uz and self.calculate_similarity(edited, other_uz) >= sim_threshold:
                    if self.calculate_similarity(en, s.get("en", "")) >= sim_threshold:
                        print(f"[CHARACTER_MEMORY] Skipped duplicate sample line for {profile.name} on {page}: \"{edited}\" (matches: \"{other_uz}\", sim={self.calculate_similarity(edited, other_uz):.2f})")
                        return False, "duplicate_similarity"

            # Genuinely new sample line
            new_sample = {
                "en": en,
                "literal_mt": pipeline,
                "voice_uz": edited
            }
            profile.sample_lines.append(new_sample)
            profile.exact_lines[norm_en] = edited
            profile.exact_lines[en.upper()] = edited
            self.save()
            print(f"[CHARACTER_MEMORY] Added new sample line for {profile.name} on {page}: \"{en}\" -> \"{edited}\"")
            return True, "added"

    def update_visual_signature(
        self,
        character_name: str,
        new_colors: List[List[int]],
        new_weights: List[float],
        page: str = "unknown"
    ) -> bool:
        """
        Merges newly sampled dominant colors into the character's visual signature.
        Uses exponential moving average and deduplication against near-identical clusters.
        """
        if not character_name or not new_colors:
            return False
        with self._lock:
            self._reload_if_modified()
            profile = self.get_profile(character_name)
            if profile is None:
                return False
            import speaker_detector
            cur_sig = profile.visual_signature or {
                "dominant_colors": [],
                "color_weights": [],
                "sample_count": 0,
                "last_updated": ""
            }
            updated_sig = speaker_detector.merge_signature(cur_sig, new_colors, new_weights)
            profile.visual_signature = updated_sig
            self.save()
            print(f"[CHARACTER_MEMORY] Updated visual signature for {profile.name} on {page} (samples: {updated_sig.get('sample_count', 1)})")
            return True


# ==============================================================================
# SEED DEFAULT PROFILE: SUPERIOR SPIDER-MAN (OTTO OCTAVIUS)
# ==============================================================================

SEED_SUPERIOR_SPIDERMAN = CharacterVoiceProfile(
    name="Superior Spider-Man",
    aliases=[
        "superior spider-man",
        "superior spiderman",
        "otto octavius",
        "otto",
        "doc ock",
        "spiderman",
        "spider-man"
    ],
    voice_traits=[
        "Formal, elevated, scientific vocabulary ('tafakkur', 'ishqalanishsiz', 'intellekt', 'daholik')",
        "Arrogant and patronizing superiority over all adversaries ('ahmoqlar', 'nodonlar', 'arzimas')",
        "Grandiose self-references ('Mening Spider-botlarim', 'ancha ustun Spider-Man')",
        "Cold, diagnostic physical descriptions ('lat yegan nafas yo\'li', 'ezilgan hiqildoq')",
        "Clipped, decisive imperatives ('Ovora bo\'lmang', 'Shu yerda to\'xtang', 'Bajarilsin')"
    ],
    forbidden_tones=[
        "Casual teenage slang ('vapshe', 'og\'ayni', 'brat', 'chotki')",
        "Peter Parker\'s goofy, playful, self-deprecating quips ('voy-bo\'', 'kechirasizlar')",
        "Uncertainty or self-doubt ('balki', 'eplay olarmikinman', 'bilmadim-da')"
    ],
    established_facts=[
        "Refers to himself in third person or as an intellectual superior ('mutlaqo ustun', 'katta aql')",
        "Names his autonomous surveillance units Spider-bots ('mening Spider-botlarim')",
        "Addresses women formally with polite condescension ('Xonim')",
        "Despises teenage slang and insists on clinical scientific precision ('lat yegan nafas yo\\'li', 'ezilgan hiqildoq')",
        "Executes tactical contingencies with Greek letter designations ('Reja: Epsilon-besh')",
        "Refuses self-deprecating humor or apologies when questioned by the public or press",
        "Claims responsibility for university property recovery ('Marla Jeymson xotira binosiga qaytarganim uchunmi?')"
    ],
    sample_lines=[
        {
            "en": "SECOND DAY ON THE 'JOB'...",
            "literal_mt": "ISHDAGI IKKINCHI KUN...",
            "voice_uz": "ISHDAGI IKKINCHI KUN..."
        },
        {
            "en": "...AND NOT ONE OF THEM KNOWS I'M A DIFFERENT SPIDER-MAN.",
            "literal_mt": "...VA ULARDAN BIRORTASI HAM MENING BOSHQA SPIDER-MAN EKANIMNI BILMAYDI.",
            "voice_uz": "...VA ULARDAN HECH BIRI MENING MUTLAQO BOSHQA SPIDER-MAN EKANIMNI BILMAYDI."
        },
        {
            "en": "BUT SOON THEY'LL ALL LEARN THAT I AM A FAR SUPERIOR ONE.",
            "literal_mt": "LEKIN TEZ ORADA ULARNING HAMMASI MENING BIRINCHI O'RINDA EKANLIGIMNI BILIB OLADI.",
            "voice_uz": "BIROQ TEZ ORADA BARCHALARI MENING ANCHA USTUN SPIDER-MAN EKANIMNI ANGLAB YETISHADI!"
        },
        {
            "en": "IMPRESSED? DON'T BOTHER.",
            "literal_mt": "QOYIL QOLDINGIZMI? TASHVISH CHEKMANG.",
            "voice_uz": "QOYIL QOLDINGIZMI? OVORA BO'LMANG, BU MEN UCHUN ODDIY HOL."
        },
        {
            "en": "PLAN: EPSILON FIVE!",
            "literal_mt": "REJA: EPSILON BESH!",
            "voice_uz": "REJA: EPSILON-BESH!"
        },
        {
            "en": "INSTANT FRICTIONLESS SURFACE!",
            "literal_mt": "ZAMONAVIY ISHQALANIShSIZ YUZASI!",
            "voice_uz": "ISHQALANISHSIZ TEZKOR SIRT!"
        },
        {
            "en": "BRUISED TRACHEA. CRUSHED LARYNX.",
            "literal_mt": "KO'KARGAN TRAXEYA. EZILGAN XALQUM.",
            "voice_uz": "LAT YEGAN NAFAS YO'LI. EZILGAN HIQILDOQ."
        },
        {
            "en": "FOOLS! YOU ARE FACING A SUPERIOR INTELLECT!",
            "literal_mt": "AHMOQLAR! SIZ KUCHLI AQL BILAN YUZLASHYAPSIZ!",
            "voice_uz": "AHMOQLAR! SIZLAR MUTLAQO USTUN TAFAKKUR BILAN TO'QNASHDINGIZ!"
        }
    ],
    exact_lines={
        "PLAN: EPSILON FIVE!": "REJA: EPSILON-BESH!",
        "PLAN: EPSILON FIVE": "REJA: EPSILON-BESH!",
        "PLAN EPSILON FIVE": "REJA: EPSILON-BESH!",
        "IMPRESSED? DON'T BOTHER.": "QOYIL QOLDINGIZMI? OVORA BO'LMANG.",
        "IMPRESSED? DONT BOTHER.": "QOYIL QOLDINGIZMI? OVORA BO'LMANG.",
        "IMPRESSED? DONT BOTHER": "QOYIL QOLDINGIZMI? OVORA BO'LMANG.",
        "INSTANT FRICTIONLESS SURFACE!": "ISHQALANISHSIZ TEZKOR SIRT!",
        "INSTANT FRICTIONLESS SURFACE": "ISHQALANISHSIZ TEZKOR SIRT!",
        "BRUISED TRACHEA.": "LAT YEGAN NAFAS YO'LI.",
        "CRUSHED LARYNX.": "EZILGAN HIQILDOQ.",
        "SECOND DAY ON THE 'JOB'...": "ISHDAGI IKKINCHI KUN...",
        "SECOND DAY ON THE 'JOB'": "ISHDAGI IKKINCHI KUN...",
        "SECOND DAY ON THE \"JOB\"": "ISHDAGI IKKINCHI KUN...",
        "SECOND DAY ON THE JOB...": "ISHDAGI IKKINCHI KUN...",
        "SECOND DAY ON THE JOB": "ISHDAGI IKKINCHI KUN...",
        "...AND NOT ONE OF THEM KNOWS I'M A DIFFERENT SPIDER-MAN.": "...VA ULARDAN HECH BIRI MENING MUTLAQO BOSHQA SPIDER-MAN EKANIMNI BILMAYDI.",
        "AND NOT ONE OF THEM KNOWS I'M A DIFFERENT SPIDER-MAN.": "VA ULARDAN HECH BIRI MENING MUTLAQO BOSHQA SPIDER-MAN EKANIMNI BILMAYDI.",
        "BUT SOON THEY'LL ALL LEARN THAT I AM A FAR SUPERIOR ONE.": "BIROQ TEZ ORADA BARCHALARI MENING ANCHA USTUN SPIDER-MAN EKANIMNI ANGLAB YETISHADI!",
        "BUT SOON THEYLL ALL LEARN THAT I AM A FAR SUPERIOR ONE.": "BIROQ TEZ ORADA BARCHALARI MENING ANCHA USTUN SPIDER-MAN EKANIMNI ANGLAB YETISHADI!",
        "BRUISED TRACHEA. CRUSHED LARYNX.": "LAT YEGAN NAFAS YO'LI. EZILGAN HIQILDOQ.",
        "FOOLS! YOU ARE FACING A SUPERIOR INTELLECT!": "AHMOQLAR! SIZLAR MUTLAQO USTUN TAFAKKUR BILAN TO'QNASHDINGIZ!",
        "FOR BRINGING THIS BACK TO THE MARLA JAMESON MEMORIAL WING?": "BUNI MARLA JEYMSON XOTIRA BINOSIGA QAYTARGANIM UCHUNMI?",
        "--FOR BRINGING THIS BACK TO THE MARLA JAMESON MEMORIAL WING?": "--BUNI MARLA JEYMSON XOTIRA BINOSIGA QAYTARGANIM UCHUNMI?",
        "BUT SOON THEY'LL ALL LEARN THAT I AM A FAR SUPERIOR ONE": "BIROQ TEZ ORADA BARCHALARI MENING ANCHA USTUN SPIDER-MAN EKANIMNI ANGLAB YETISHADI!"
    },
    phrase_replacements=[
        (r"\bboshqa\s+Spider-Man\b", "mutlaqo boshqa Spider-Man"),
        (r"\byuqori\s+darajali\b", "ancha ustun"),
        (r"\bancha\s+yaxshi\b", "ancha ustun"),
        (r"\bancha\s+kuchli\b", "ancha ustun"),
        (r"\bbilib\s+oladi\b", "anglab yetishadi"),
        (r"\btushunadi\b", "anglab yetadi"),
        (r"\bo['’`]?rganib\s+oladi\b", "anglab yetadi"),
        (r"\btashvish\s+chekmang\b", "ovora bo'lmang"),
        (r"\bxavotir\s+olmang\b", "ovora bo'lmang"),
        (r"\bkuchli\s+aql\b", "ustun tafakkur"),
        (r"\baql\s+egasi\b", "ustun tafakkur"),
        (r"\bmening\s+robotlarim\b", "mening Spider-botlarim"),
        (r"\bkichik\s+muammo\b", "arzimas to'siq"),
        (r"\boddiy\s+ish\b", "arzimas yumush"),
        (r"\bmen\s+yordam\s+beraman\b", "men nazoratga olaman"),
        (r"\beplay\s+olaman\b", "bartaraf etaman"),
        (r"\byaxshiroqman\b", "mutlaqo ustunman"),
        (r"\bkuchliroqman\b", "ancha ustunman"),
    ],
    visual_signature={
        "dominant_colors": [[178, 30, 30], [30, 55, 175], [215, 215, 215]],
        "color_weights": [0.45, 0.40, 0.15],
        "sample_count": 1,
        "last_updated": "2026-09-18"
    }
)


# ==============================================================================
# SEED PROFILE: J. JONAH JAMESON
# ==============================================================================

SEED_JONAH_JAMESON = CharacterVoiceProfile(
    name="J. Jonah Jameson",
    aliases=[
        "j. jonah jameson",
        "jonah jameson",
        "mayor jameson",
        "mayor j. jonah jameson",
        "mayor jonah jameson",
        "jonah",
        "jameson",
        "mer j. jona jeymson",
        "mer jona jeymson",
        "jona jeymson",
        "mer jeymson"
    ],
    voice_traits=[
        "Bombastic, loud, self-aggrandizing civic authority ('Bu axir menman-ku!', 'Kattalik qilib')",
        "Colorful colloquial idioms and folk expressions ('qoyilmaqom ish', 'yalpoqbosh ahmoq')",
        "Condescending and patronizing tone towards superheroes and subordinates",
        "Insistence on public recognition and memorial fund prestige"
    ],
    forbidden_tones=[
        "Boring, neutral bureaucratic tone ('juda yaxshi ish qildingiz')",
        "Humble apologies or sincere self-doubt",
        "Excessive politeness or servility"
    ],
    established_facts=[
        "Holds the office of Mayor of New York City ('Mer J. Jona Jeymson')",
        "Founded the Marla Jameson Memorial Fund / Wing in honor of his late wife",
        "Publicly patronizes Spider-Man while secretly taking credit for city safety"
    ],
    sample_lines=[
        {
            "en": "WHY, THAT WOULD BE ME! MAYOR JONAH JAMESON!",
            "literal_mt": "NIMA UCHUN, BU MEN BO'LAMAN! MER J. JONA JEYMSON!",
            "voice_uz": "Bu axir menman-ku! Mer J. Jona Jeymson!"
        },
        {
            "en": "AND AS THE BIGGER MAN HERE I'M MAGNANIMOUS ENOUGH TO ADMIT--",
            "literal_mt": "VA MEN BU YERDA KATTA ODAM SIFATIDA MEN TAN OLISH UCHUN YETARLICHA OLIJANOBLIK--",
            "voice_uz": "Va bu yerda kattalik qilib, shuni tan olishga yetarlicha bag‘rikengman--"
        },
        {
            "en": "YOU'VE DONE A HECKUVA JOB HERE, SPIDEY!",
            "literal_mt": "SIZ BU YERDA JUDA YAXSHI ISH QILDINGIZ, SPIDEY!",
            "voice_uz": "Bu yerda qoyilmaqom ish qilding, O‘rgimchak!"
        },
        {
            "en": "SO ON BEHALF OF THE MARLA JAMESON MEMORIAL FUND, LET ME JUST SAY--",
            "literal_mt": "MARLA JEYMSON XOTIRA FONDINING NOMIDAN, SHUNI AYTAMANKI--",
            "voice_uz": "Shunday ekan, Marla Jeymson xotira jamg‘armasi nomidan shuni aytishga ijozat ber..."
        }
    ],
    exact_lines={
        "WHY, THAT WOULD BE ME! MAYOR JONAH JAMESON!": "Bu axir menman-ku! Mer J. Jona Jeymson!",
        "WHY, THAT WOULD BE ME! MAYOR JONAH JAMESONI": "Bu axir menman-ku! Mer J. Jona Jeymson!",
        "WHY THAT WOULD BE ME MAYOR JONAH JAMESON": "Bu axir menman-ku! Mer J. Jona Jeymson!",
        "AND AS THE BIGGER MAN HERE I'M MAGNANIMOUS ENOUGH TO ADMIT--": "Va bu yerda kattalik qilib, shuni tan olishga yetarlicha bag‘rikengman--",
        "AND AS THE BIGGER MAN HERE IM MAGNANIMOUS ENOUGH TO ADMIT--": "Va bu yerda kattalik qilib, shuni tan olishga yetarlicha bag‘rikengman--",
        "AND AS THE BIGGER MAN HERE I'M MAGNANIMOUS ENOUGH TO ADMIT": "Va bu yerda kattalik qilib, shuni tan olishga yetarlicha bag‘rikengman--",
        "YOU'VE DONE A HECKUVA JOB HERE, SPIDEY!": "Bu yerda qoyilmaqom ish qilding, O‘rgimchak!",
        "YOU'VE DONE A HECKUVA JOB HERE , SPIDEY!": "Bu yerda qoyilmaqom ish qilding, O‘rgimchak!",
        "YOU'VE DONE A HECKUVA JOB HERE SPIDEY": "Bu yerda qoyilmaqom ish qilding, O‘rgimchak!",
        "YOU'VE DONE A HECKUVA JOB HERE , SPIDEY": "Bu yerda qoyilmaqom ish qilding, O‘rgimchak!",
        "SO ON BEHALF OF THE MARLA JAMESON MEMORIAL FUND, LET ME JUST SAY": "Shunday ekan, Marla Jeymson xotira jamg‘armasi nomidan shuni aytishga ijozat ber...",
        "SO ON BEHALF OF THE MARLA JAMESON MEMORIAL FUND, LET ME JUST SAY--": "Shunday ekan, Marla Jeymson xotira jamg‘armasi nomidan shuni aytishga ijozat ber..."
    },
    phrase_replacements=[
        (r"\bheckuva\s+job\b", "qoyilmaqom ish"),
        (r"\bjuda\s+yaxshi\s+ish\s+qildingiz\b", "qoyilmaqom ish qilding"),
        (r"\bjuda\s+yaxshi\s+ish\s+qilding\b", "qoyilmaqom ish qilding"),
        (r"\bkatta\s+odam\s+sifatida\b", "kattalik qilib"),
        (r"\bolijanoblik\b", "bag'rikenglik"),
        (r"\byetarlicha\s+olijanobman\b", "yetarlicha bag'rikengman"),
        (r"\bbu\s+men\s+bo['’`]?laman\b", "bu axir menman-ku"),
    ]
)


# ==============================================================================
# SEED PROFILE: PETER PARKER (GHOST)
# ==============================================================================

SEED_PETER_PARKER_GHOST = CharacterVoiceProfile(
    name="Peter Parker (Ghost)",
    aliases=[
        "peter parker (ghost)",
        "peter parker ghost",
        "ghost peter",
        "peter ghost",
        "peter parker",
        "piter parker",
        "ghost"
    ],
    voice_traits=[
        "Frustrated, desperate, and ironic internal spectral ghost commentary",
        "Addresses Mayor Jameson respectfully with 'SIZ' in public/dialogue context ('Siz men haqimda adashgan')",
        "Stunned disbelief at Doc Ock hijacking his life and reputation"
    ],
    forbidden_tones=[
        "Cold, arrogant superiority (that's Doc Ock/Otto)",
        "Formal villainous declarations"
    ],
    established_facts=[
        "Trapped as a disembodied astral spirit witnessing Otto Octavius in his body",
        "Frustrated that Jameson is finally nice to Spider-Man only when Otto is inside his body"
    ],
    sample_lines=[
        {
            "en": "YOU WERE WRONG ABOUT ME, AND MAYBE I'VE BEEN WRONG ABOUT YOU!",
            "literal_mt": "MEN HAQIMDA XATO QILDIM VA EHTIMOL MEN SIZ HAQINGIZDA XATO QILDIM!",
            "voice_uz": "--Siz men haqimda adashgansiz, ehtimol men ham siz haqingizda adashgandirman!"
        },
        {
            "en": "THAT'S OTTO OCTAVIUS, YOU FLAT-TOPPED FINK!",
            "literal_mt": "BU OTTO OKTAVIUS, EY TEKIS KALLA ABLAX!",
            "voice_uz": "Bu Otto Oktavius, ey yalpoqbosh ahmoq!"
        },
        {
            "en": "I CAN'T BELIEVE IT! AFTER ALL THESE YEARS, JONAH FINALLY MAKES NICE...",
            "literal_mt": "ISHONOLMAYMAN! SHUNCHA YILDAN KEYIN JONA NIHONAT MULOYIMLAShDI...",
            "voice_uz": "Ishonolmayman! Shuncha yildan keyin Jona nihoyat muloyimlashdi..."
        },
        {
            "en": "...AND HE DOES IT OVER MY DEAD BODY!",
            "literal_mt": "...VA U BUNI MENING O'LIGIM USTIDAN QILYAPTI!",
            "voice_uz": "...va buni faqat mening o‘ligim ustidan qilyapti!"
        },
        {
            "en": "OR MY LIVE BODY WITH DOC OCK'S BRAIN!",
            "literal_mt": "YOKI DOKTOR OKTOPUS MIYASI BILAN TIRIK TANAM!",
            "voice_uz": "Yoki Dok Okning miyasi joylashgan tirik tanam orqali!"
        },
        {
            "en": "AHHH! I CAN'T TAKE THIS ANYMORE! IT'S-- IT'S CRAZY TOWN BANANA-PANTS!",
            "literal_mt": "AHH! ORTIQ CHIDAY OLMAYMAN! BU-- BU TELBALIK!",
            "voice_uz": "Aaa! Bunga ortiq chiday olmayman! Bu... bu mutlaqo telbalik!"
        }
    ],
    exact_lines={
        "YOU WERE WRONG ABOUT ME, AND MAYBE I'VE BEEN WRONG ABOUT YOU!": "--Siz men haqimda adashgansiz, ehtimol men ham siz haqingizda adashgandirman!",
        "~OU WERE WRONG ABOUT ME, AND MAYBE I'VE BEEN WRONG ABOUT YOUL": "--Siz men haqimda adashgansiz, ehtimol men ham siz haqingizda adashgandirman!",
        "YOU WERE WRONG ABOUT ME, AND MAYBE I'VE BEEN WRONG ABOUT YOU": "--Siz men haqimda adashgansiz, ehtimol men ham siz haqingizda adashgandirman!",
        "THAT'S OTTO OCTAVIUS; YOU FLAT-TOPPED FINKA": "Bu Otto Oktavius, ey yalpoqbosh ahmoq!",
        "THAT'S OTTO OCTAVIUS, YOU FLAT-TOPPED FINK!": "Bu Otto Oktavius, ey yalpoqbosh ahmoq!",
        "THAT'S OTTO OCTAVIUS; YOU FLAT-TOPPED FINK!": "Bu Otto Oktavius, ey yalpoqbosh ahmoq!",
        "THAT'S OTTO OCTAVIUS YOU FLAT-TOPPED FINK": "Bu Otto Oktavius, ey yalpoqbosh ahmoq!",
        "DOCTOR OCTOPUS IN MY BODYI": "Mening tanamdagi Doktor Sakkizoyoq!",
        "DOCTOR OCTOPUS IN MY BODY!": "Mening tanamdagi Doktor Sakkizoyoq!",
        "I CAN'T BELIEVE ITI AFTER ALL THESE YEARS; JONAH FINALLY MAKES NICE": "Ishonolmayman! Shuncha yildan keyin Jona nihoyat muloyimlashdi...",
        "I CAN'T BELIEVE IT! AFTER ALL THESE YEARS, JONAH FINALLY MAKES NICE": "Ishonolmayman! Shuncha yildan keyin Jona nihoyat muloyimlashdi...",
        "I CAN'T BELIEVE IT! AFTER ALL THESE YEARS; JONAH FINALLY MAKES NICE": "Ishonolmayman! Shuncha yildan keyin Jona nihoyat muloyimlashdi...",
        "#AND HE DOES IT OVER MY DEAD BODYI": "...va buni faqat mening o‘ligim ustidan qilyapti!",
        "AND HE DOES IT OVER MY DEAD BODY!": "...va buni faqat mening o‘ligim ustidan qilyapti!",
        "...AND HE DOES IT OVER MY DEAD BODY!": "...va buni faqat mening o‘ligim ustidan qilyapti!",
        "OR MY LIVE BODY WITH DOC OCK'S BRAINI": "Yoki Dok Okning miyasi joylashgan tirik tanam orqali!",
        "OR MY LIVE BODY WITH DOC OCK'S BRAIN!": "Yoki Dok Okning miyasi joylashgan tirik tanam orqali!",
        "AHHHI I CAN'T TAKE THIS ANYMORE! IT'S-- IT'S CRAZY TOWN BANANA- PANTSI": "Aaa! Bunga ortiq chiday olmayman! Bu... bu mutlaqo telbalik!",
        "AHHH! I CAN'T TAKE THIS ANYMORE! IT'S-- IT'S CRAZY TOWN BANANA-PANTS!": "Aaa! Bunga ortiq chiday olmayman! Bu... bu mutlaqo telbalik!"
    },
    phrase_replacements=[
        (r"\btekis\s+kalla\s+ablax\b", "yalpoqbosh ahmoq"),
        (r"\bflat-topped\s+fink\b", "yalpoqbosh ahmoq"),
        (r"\bmakes\s+nice\b", "muloyimlashdi"),
        (r"\bcrazy\s+town\s+banana-pants\b", "mutlaqo telbalik"),
        (r"\bdead\s+body\b", "o'ligim"),
    ]
)

ARCHETYPE_PRESS_REPORTER = CharacterVoiceProfile(
    name="Press / News Reporter",
    aliases=[
        "press", "reporter", "journalist", "news anchor", "tv news anchor",
        "anchor", "news", "chyron", "headline", "daily bugle reporter", "media"
    ],
    voice_traits=[
        "Sensationalist, fast-paced broadcast delivery",
        "Concise journalistic news phrasing",
        "Objective reporting terms for urban crime and vigilante activity"
    ],
    phrase_replacements=[
        (r"\bmaskalli\s+hujumchilar\b", "niqobli qasoskorlar"),
        (r"\bniqobli\s+hushyorlar\b", "niqobli qasoskorlar"),
        (r"\bmunozarali\s+nuqtai\s+nazari\b", "bahsli pozitsiyasi"),
        (r"\bmunozarali\s+pozitsiyasi\b", "bahsli pozitsiyasi"),
        (r"\bshahar\s+hokimiyatining\b", "shahar meriyasining"),
        (r"\bshahar\s+hokimiyati\b", "shahar meriyasi"),
        (r"\bcoming\s+up\s+next\b", "navbatdagi lavhada"),
        (r"\bkeyingi\b", "navbatdagi lavhada"),
        (r"\blead\s+story\b", "bosh mavzu"),
        (r"\bbreaking\s+news\b", "shoshilinch xabar"),
        (r"\bcity\s+hall\b", "shahar meriyasi"),
    ]
)

ARCHETYPE_POLICE_OFFICER = CharacterVoiceProfile(
    name="Police / Law Enforcement",
    aliases=[
        "police", "police officer", "officer", "cop", "detective",
        "carlie cooper", "carlie", "captain", "sergeant", "swat", "law enforcement"
    ],
    voice_traits=[
        "Authoritative procedural commands",
        "Urgent tactical terminology",
        "Direct, brisk, professional speech"
    ],
    phrase_replacements=[
        (r"(?<!orqaga\s)\bturinglar\b", "orqaga turinglar"),
        (r"\borqaga\s+orqaga\b", "orqaga"),
        (r"\batrofimiz\s+tuzilgan\b", "hudud to'liq o'rab olingan"),
        (r"\bperimetr\s+mavjud\b", "hudud to'liq o'rab olingan"),
        (r"\bsodir\s+bo['’`]?lgan\s+voqeada\b", "jinoyat joyida"),
        (r"\bjinoyat\s+joyida\s+uning\s+holatini\b", "jinoyat joyidagi uning qaddi-qomati"),
        (r"\bso['’`]?zlarni\s+o['’`]?qiydi\b", "so'zlashuv uslubi"),
        (r"\bshubhali\s+odam\b", "gumondor"),
        (r"\bstand\s+back\b", "orqaga chekinishingizni so'raymiz"),
        (r"\bhold\s+your\s+fire\b", "o't ochishni to'xtating"),
        (r"\bdrop\s+your\s+weapon\b", "qurolingizni tashlang"),
        (r"\bcrime\s+scene\b", "jinoyat joyi"),
    ]
)

ARCHETYPE_AUTHORITY_OFFICIAL = CharacterVoiceProfile(
    name="Authority Figure / Official",
    aliases=[
        "authority", "mayor", "dean", "dean goldman", "rut goldman",
        "official", "councilman", "director", "commissioner", "superintendent"
    ],
    voice_traits=[
        "Formal institutional dignity",
        "Ceremonious, polite address",
        "Diplomatic, respectful vocabulary"
    ],
    phrase_replacements=[
        (r"\btalabalar\s+denimi\b", "talabalar dekani"),
        (r"\btalabalar\s+dekani\b", "talabalar dekani"),
        (r"\buniversitetidagi\s+yarmidan\b", "universitetdagi barcha nomidan"),
        (r"\byarmisidan\s+so['’`]?ng\b", "nomidan"),
        (r"\bsharaf\s+edi\b", "katta sharaf bo'ldi"),
        (r"\bminnatdorchilik\s+aytaman\b", "tashakkur izhor qilmoqchiman"),
        (r"\bmaxsus\s+mehmonimiz\s+bor\b", "hurmatli bir mehmonimiz bor"),
        (r"\bilmiy\s+fan\s+uskunasi\b", "ilmiy uskunalarimiz"),
    ]
)

ARCHETYPE_YOUNG_ADULT_CASUAL = CharacterVoiceProfile(
    name="Young Adult Casual Speech",
    aliases=[
        "young adult", "casual", "mary jane", "mary jane watson", "mj",
        "gwen", "harry", "teen", "friend", "civilian youth", "college student"
    ],
    voice_traits=[
        "Warm, conversational dialogue",
        "Contemporary informal expressions",
        "Expressive, natural colloquial cadence"
    ],
    phrase_replacements=[
        (r"\bsen\s+yaxshisanmi\b", "yaxshimisan"),
        (r"\bsen\s+boshqacha\s+ko['’`]?rinasan\b", "boshqacha ko'rinyapsan"),
        (r"\bkatta\s+bo['’`]?lib\s+o['’`]?sadi\b", "nihoyat ulg'ayyapti"),
        (r"\bo['’`]?sib\s+ulg['’`]?aygandir\b", "nihoyat ulg'aygandir"),
        (r"\bbiron\s+bir\s+g['’`]?alati\s+narsa\s+payqaganmisiz\b", "biror g'alati narsa sezmadingmi"),
        (r"\bg['’`]?alati\s+narsani\s+sezdingizmi\b", "biror g'alati narsa sezmadingmi"),
        (r"\bhe\s+didn['’]?t\s+make\s+a\s+single\s+joke\b", "bitta ham hazil qilmadi"),
        (r"\bbitta\s+ham\s+hazil\s+qilmadi\b", "bitta ham hazil qilmadi"),
        (r"\bbirorta\s+ham\s+hazil\s+qilmadi\b", "bitta ham hazil qilmadi"),
        (r"\bare\s+you\s+kidding\b", "hazillashyapsanmi"),
        (r"\bno\s+way\b", "bo'lishi mumkin emas"),
    ]
)

ARCHETYPE_SERVICE_HOSPITALITY = CharacterVoiceProfile(
    name="Service & Hospitality",
    aliases=[
        "service", "waiter", "waitress", "server", "host",
        "clerk", "cashier", "bartender", "staff"
    ],
    voice_traits=[
        "Courteous customer service Uzbek",
        "Polite deferential address",
        "Clear hospitality terminology"
    ],
    phrase_replacements=[
        (r"\bxayrli\s+kechasi\b", "xayrli kech"),
        (r"\bsharob\s+menyasini\b", "vino menyusini"),
        (r"\bsharob\s+menyusini\b", "vino menyusini"),
        (r"\bsharob\s+menyasi\b", "vino menyusi"),
        (r"\bko['’`]?rishni\s+xohlaysizmi\b", "tanishishni xohlaysizmi"),
        (r"\bqanday\s+yordam\s+beraman\b", "qanday yordam bera olaman"),
        (r"\bmarhamat\b", "marhamat"),
    ]
)

ARCHETYPE_BYSTANDER_CITIZEN = CharacterVoiceProfile(
    name="Bystander / Citizen",
    aliases=[
        "citizen", "bystander", "crowd", "pedestrian", "civilian", "witness"
    ],
    voice_traits=[
        "Spontaneous exclamatory reactions",
        "Alarmed urban street dialogue",
        "Direct observational inquiries"
    ],
    phrase_replacements=[
        (r"\bu\s+yerga\s+qarang\b", "tepaga qaranglar"),
        (r"\bu\s+Spider-Manmi\b", "ana, O'rgimchak-Odammi u"),
        (r"\bu\s+O'rgimchak-Odammi\b", "ana, O'rgimchak-Odammi u"),
        (r"\byoki\s+boshqa\s+kimdir\b", "yoki boshqa birovmi"),
        (r"\blook\s+out\b", "ehtiyot bo'ling"),
        (r"\brun\b", "qochinglar"),
    ]
)

ARCHETYPE_PROFILES = [
    ARCHETYPE_PRESS_REPORTER,
    ARCHETYPE_POLICE_OFFICER,
    ARCHETYPE_AUTHORITY_OFFICIAL,
    ARCHETYPE_YOUNG_ADULT_CASUAL,
    ARCHETYPE_SERVICE_HOSPITALITY,
    ARCHETYPE_BYSTANDER_CITIZEN,
]

DEFAULT_SEEDS = [SEED_SUPERIOR_SPIDERMAN, SEED_JONAH_JAMESON, SEED_PETER_PARKER_GHOST]


# Global singleton accessors
def get_memory_store() -> CharacterMemoryStore:
    """Returns the persistent CharacterMemoryStore singleton."""
    return CharacterMemoryStore.get_instance()


def get_character_profile(name_or_alias: Optional[str]) -> Optional[CharacterVoiceProfile]:
    """Finds and returns a matching CharacterVoiceProfile by name or alias."""
    return get_memory_store().get_profile(name_or_alias)


def list_available_profiles() -> List[str]:
    """Returns canonical names of all registered character voice profiles."""
    return get_memory_store().list_profiles()


def register_character_profile(profile: CharacterVoiceProfile) -> None:
    """Registers a new character voice profile into the persistent memory store."""
    get_memory_store().register_profile(profile)


def record_user_edited_sample_line(
    character_name: str,
    en_text: str,
    edited_uzbek: str,
    pipeline_uzbek: str = "",
    page: str = "unknown",
    sim_threshold: float = 0.65
) -> Tuple[bool, str]:
    """Captures user-edited sample_line into character memory via the persistent store."""
    return get_memory_store().record_user_edited_sample_line(
        character_name=character_name,
        en_text=en_text,
        edited_uzbek=edited_uzbek,
        pipeline_uzbek=pipeline_uzbek,
        page=page,
        sim_threshold=sim_threshold
    )


def update_visual_signature(
    character_name: str,
    new_colors: List[List[int]],
    new_weights: List[float],
    page: str = "unknown"
) -> bool:
    """Updates the character's visual signature in persistent memory."""
    return get_memory_store().update_visual_signature(
        character_name=character_name,
        new_colors=new_colors,
        new_weights=new_weights,
        page=page
    )


class _ProfileProxy:
    """Provides transparent attribute access to the active Superior Spider-Man profile."""
    def __getattr__(self, item):
        p = get_character_profile("Superior Spider-Man")
        if p is not None:
            return getattr(p, item)
        return getattr(SEED_SUPERIOR_SPIDERMAN, item)

# Backward-compatibility alias
SUPERIOR_SPIDERMAN_PROFILE = _ProfileProxy()


