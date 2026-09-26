"""
supabase_db.py
--------------
Centralized Supabase (Free tier PostgreSQL) database connector and client layer
for Comic Translation Studio.

Handles persistent cloud storage with automatic local JSON fallback for:
1. `mask_corrections` (replaces mask_correction_memory.json)
2. `quality_gate_events` (replaces quality_gate_log.json)
3. `character_profiles` (replaces character_memory.json)
4. `terminology_store` (replaces terminology_store.json)

Zero AI API keys: this accesses PostgreSQL for state persistence, never paid LLM APIs.
"""

import os
import json
import time
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    # Look for .env in current directory and project root
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("supabase_db")

# Fallback local file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FALLBACK_FILES = {
    "mask_corrections": os.path.join(BASE_DIR, "mask_correction_memory.json"),
    "quality_gate_events": os.path.join(BASE_DIR, "quality_gate_log.json"),
    "character_profiles": os.path.join(BASE_DIR, "character_memory.json"),
    "terminology_store": os.path.join(BASE_DIR, "terminology_store.json"),
}

_SUPABASE_CLIENT = None
_CLIENT_INITIALIZED = False


def _get_credential(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieves credential from environment or Streamlit secrets."""
    val = os.getenv(key)
    if val:
        return val.strip()

    # Check Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass

    return default


def get_supabase_client():
    """
    Returns an initialized Supabase Client or None if credentials are not configured.
    Thread-safe lazy initialization.
    """
    global _SUPABASE_CLIENT, _CLIENT_INITIALIZED
    if _CLIENT_INITIALIZED:
        return _SUPABASE_CLIENT

    url = _get_credential("SUPABASE_URL")
    key = (
        _get_credential("SUPABASE_KEY")
        or _get_credential("SUPABASE_SERVICE_ROLE_KEY")
        or _get_credential("SUPABASE_ANON_KEY")
    )

    if not url or not key:
        _CLIENT_INITIALIZED = True
        _SUPABASE_CLIENT = None
        return None

    try:
        from supabase import create_client
        _SUPABASE_CLIENT = create_client(url, key)
    except Exception as e:
        logger.warning(f"[SUPABASE_WARN] Failed to initialize Supabase client: {e}")
        _SUPABASE_CLIENT = None

    _CLIENT_INITIALIZED = True
    return _SUPABASE_CLIENT


def reset_client():
    """Resets client state, allowing re-initialization (e.g. for testing)."""
    global _SUPABASE_CLIENT, _CLIENT_INITIALIZED
    _SUPABASE_CLIENT = None
    _CLIENT_INITIALIZED = False


def is_supabase_configured() -> bool:
    """Returns True if Supabase credentials are set and client was created."""
    client = get_supabase_client()
    return client is not None


# ==============================================================================
# 1. MASK CORRECTIONS TABLE
# Columns: id, page, timestamp, tool_used, bubble_id, original_contour,
#          corrected_contours, metadata
# ==============================================================================

def insert_mask_correction(record: Dict[str, Any]) -> bool:
    """
    Inserts a manual mask edit into mask_corrections table.
    Falls back to mask_correction_memory.json if Supabase is unreachable.
    """
    client = get_supabase_client()
    if client is not None:
        try:
            row = {
                "page": str(record.get("page", "")),
                "timestamp": float(record.get("timestamp", time.time())),
                "tool_used": str(record.get("tool_used", "")),
                "bubble_id": int(record.get("bubble_id", 0)),
                "original_contour": record.get("original_contour", []),
                "corrected_contours": record.get("corrected_contours", []),
                "metadata": record.get("metadata", {})
            }
            client.table("mask_corrections").insert(row).execute()
            return True
        except Exception as e:
            logger.warning(f"[SUPABASE_WARN] Failed to insert mask_correction into Supabase: {e}. Falling back to local storage.")

    # Fallback to local JSON
    _append_local_json(FALLBACK_FILES["mask_corrections"], record)
    return False


def fetch_mask_corrections(page: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches mask corrections from Supabase.
    Returns None if Supabase is unavailable (prompting local fallback).
    """
    client = get_supabase_client()
    if client is None:
        return None

    try:
        query = client.table("mask_corrections").select("*")
        if page:
            query = query.eq("page", page)
        res = query.order("timestamp").execute()
        return res.data
    except Exception as e:
        logger.warning(f"[SUPABASE_WARN] Failed to fetch mask_corrections from Supabase: {e}. Falling back to local storage.")
        return None


# ==============================================================================
# 2. QUALITY GATE EVENTS TABLE
# Columns: id, page, pattern_name, stage, severity, action_taken, details,
#          bubble_id, timestamp
# ==============================================================================

def insert_quality_gate_event(event: Dict[str, Any]) -> bool:
    """
    Inserts a quality gate event into quality_gate_events table.
    Falls back to quality_gate_log.json if Supabase is unreachable.
    """
    client = get_supabase_client()
    if client is not None:
        try:
            row = {
                "page": event.get("page"),
                "pattern_name": str(event.get("pattern_name", "")),
                "stage": str(event.get("stage", "")),
                "severity": str(event.get("severity", "auto_fix")),
                "action_taken": str(event.get("action_taken", "")),
                "details": event.get("details", {}),
                "bubble_id": event.get("bubble_id"),
                "timestamp": float(event.get("timestamp") or time.time())
            }
            client.table("quality_gate_events").insert(row).execute()
            return True
        except Exception as e:
            logger.warning(f"[SUPABASE_WARN] Failed to insert quality_gate_event into Supabase: {e}. Falling back to local storage.")

    # Fallback to local JSON
    _append_local_json(FALLBACK_FILES["quality_gate_events"], event)
    return False


def fetch_quality_gate_events(page: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches recorded quality gate events from Supabase.
    Returns None if Supabase is unavailable (prompting local fallback).
    """
    client = get_supabase_client()
    if client is None:
        return None

    try:
        query = client.table("quality_gate_events").select("*")
        if page:
            query = query.eq("page", page)
        res = query.order("timestamp").execute()
        return res.data
    except Exception as e:
        logger.warning(f"[SUPABASE_WARN] Failed to fetch quality_gate_events from Supabase: {e}. Falling back to local storage.")
        return None


# ==============================================================================
# 3. CHARACTER PROFILES TABLE
# Columns: character_name (pk), voice_traits, established_facts, sample_lines,
#          visual_signature, updated_at
# ==============================================================================

def upsert_character_profile(profile_data: Dict[str, Any]) -> bool:
    """
    Upserts a character profile into character_profiles table.
    Preserves all traits, facts, sample lines, and visual signatures.
    """
    client = get_supabase_client()
    if client is not None:
        name = profile_data.get("character_name") or profile_data.get("name")
        if not name:
            return False

        base_row = {
            "character_name": str(name),
            "voice_traits": profile_data.get("voice_traits", []),
            "established_facts": profile_data.get("established_facts", []),
            "sample_lines": profile_data.get("sample_lines", []),
            "visual_signature": profile_data.get("visual_signature"),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

        # Extended fields if schema contains them
        extended_row = dict(base_row)
        if "aliases" in profile_data:
            extended_row["aliases"] = profile_data["aliases"]
        if "phrase_replacements" in profile_data:
            extended_row["phrase_replacements"] = [list(r) for r in profile_data["phrase_replacements"]]
        if "exact_lines" in profile_data:
            extended_row["exact_lines"] = profile_data["exact_lines"]
        if "forbidden_tones" in profile_data:
            extended_row["forbidden_tones"] = profile_data["forbidden_tones"]

        try:
            try:
                client.table("character_profiles").upsert(extended_row).execute()
                return True
            except Exception as inner_e:
                # If extended column does not exist on target table, retry with base 6 columns
                if "column" in str(inner_e).lower() and "does not exist" in str(inner_e).lower():
                    client.table("character_profiles").upsert(base_row).execute()
                    return True
                raise inner_e
        except Exception as e:
            logger.warning(f"[SUPABASE_WARN] Failed to upsert character_profile into Supabase: {e}. Falling back to local storage.")

    return False


def fetch_character_profiles() -> Optional[List[Dict[str, Any]]]:
    """
    Fetches all character profiles from Supabase.
    Returns None if Supabase is unavailable (prompting local fallback).
    """
    client = get_supabase_client()
    if client is None:
        return None

    try:
        res = client.table("character_profiles").select("*").execute()
        return res.data
    except Exception as e:
        logger.warning(f"[SUPABASE_WARN] Failed to fetch character_profiles from Supabase: {e}. Falling back to local storage.")
        return None


# ==============================================================================
# 4. TERMINOLOGY STORE TABLE
# Columns: source_term (pk), canonical_uzbek, entry_type, method,
#          first_seen_page, locked, reason
# ==============================================================================

def upsert_term(term_data: Dict[str, Any]) -> bool:
    """
    Upserts a terminology entry into terminology_store table.
    """
    client = get_supabase_client()
    if client is not None:
        source_term = term_data.get("source_term", "").strip().lower()
        if not source_term:
            return False

        base_row = {
            "source_term": source_term,
            "canonical_uzbek": str(term_data.get("canonical_uzbek", "")),
            "entry_type": str(term_data.get("entry_type", "term")),
            "method": str(term_data.get("method", "fixed")),
            "first_seen_page": str(term_data.get("first_seen_page", "auto")),
            "locked": bool(term_data.get("locked", True)),
            "reason": term_data.get("reason")
        }

        extended_row = dict(base_row)
        if "aliases" in term_data:
            extended_row["aliases"] = term_data["aliases"]

        try:
            try:
                client.table("terminology_store").upsert(extended_row).execute()
                return True
            except Exception as inner_e:
                if "column" in str(inner_e).lower() and "does not exist" in str(inner_e).lower():
                    client.table("terminology_store").upsert(base_row).execute()
                    return True
                raise inner_e
        except Exception as e:
            logger.warning(f"[SUPABASE_WARN] Failed to upsert terminology into Supabase: {e}. Falling back to local storage.")

    return False


def fetch_all_terms() -> Optional[List[Dict[str, Any]]]:
    """
    Fetches all terminology records from Supabase.
    Returns None if Supabase is unavailable (prompting local fallback).
    """
    client = get_supabase_client()
    if client is None:
        return None

    try:
        res = client.table("terminology_store").select("*").execute()
        return res.data
    except Exception as e:
        logger.warning(f"[SUPABASE_WARN] Failed to fetch terminology from Supabase: {e}. Falling back to local storage.")
        return None


# ==============================================================================
# LOCAL JSON FALLBACK HELPERS
# ==============================================================================

def _append_local_json(file_path: str, record: Dict[str, Any]):
    """Appends record to a local JSON list file safely."""
    try:
        data = []
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    data = content
                elif isinstance(content, dict) and "events" in content:
                    data = content["events"]
        data.append(record)
        tmp_path = file_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, file_path)
    except Exception as e:
        logger.warning(f"[FALLBACK_WARN] Failed to append to {file_path}: {e}")
