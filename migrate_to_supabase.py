"""
migrate_to_supabase.py
----------------------
One-time data migration script to copy all existing persistent JSON records
into Supabase (PostgreSQL free tier):

1. mask_correction_memory.json  -> mask_corrections table
2. quality_gate_log.json        -> quality_gate_events table
3. character_memory.json        -> character_profiles table
4. terminology_store.json       -> terminology_store table

Usage:
  python migrate_to_supabase.py

Ensure SUPABASE_URL and SUPABASE_KEY are set in .env or the system environment.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List

import supabase_db

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("migration")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def migrate_mask_corrections(client) -> int:
    """Migrates mask_correction_memory.json records to mask_corrections table."""
    path = os.path.join(BASE_DIR, "mask_correction_memory.json")
    if not os.path.exists(path):
        logger.info("mask_correction_memory.json not found. Skipping.")
        return 0

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        logger.warning("mask_correction_memory.json is not a list. Skipping.")
        return 0

    count = 0
    # Batch in chunks of 50 to avoid request payload limits
    batch_size = 50
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        rows = []
        for r in chunk:
            rows.append({
                "page": str(r.get("page", "")),
                "timestamp": float(r.get("timestamp", 0.0)),
                "tool_used": str(r.get("tool_used", "")),
                "bubble_id": int(r.get("bubble_id", 0)),
                "original_contour": r.get("original_contour", []),
                "corrected_contours": r.get("corrected_contours", []),
                "metadata": r.get("metadata", {})
            })
        try:
            client.table("mask_corrections").insert(rows).execute()
            count += len(rows)
        except Exception as e:
            logger.error(f"Failed to insert mask_corrections batch [{i}:{i+len(chunk)}]: {e}")

    logger.info(f"Successfully migrated {count}/{len(records)} records to mask_corrections.")
    return count


def migrate_quality_gate_events(client) -> int:
    """Migrates quality_gate_log.json records to quality_gate_events table."""
    path = os.path.join(BASE_DIR, "quality_gate_log.json")
    if not os.path.exists(path):
        logger.info("quality_gate_log.json not found. Skipping.")
        return 0

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("events", [])
    if not isinstance(events, list) or not events:
        logger.info("No quality gate events to migrate.")
        return 0

    count = 0
    batch_size = 50
    for i in range(0, len(events), batch_size):
        chunk = events[i:i + batch_size]
        rows = []
        for e in chunk:
            rows.append({
                "page": e.get("page"),
                "pattern_name": str(e.get("pattern_name", "")),
                "stage": str(e.get("stage", "")),
                "severity": str(e.get("severity", "auto_fix")),
                "action_taken": str(e.get("action_taken", "")),
                "details": e.get("details", {}),
                "bubble_id": e.get("bubble_id"),
                "timestamp": float(e.get("timestamp", 0.0))
            })
        try:
            client.table("quality_gate_events").insert(rows).execute()
            count += len(rows)
        except Exception as err:
            logger.error(f"Failed to insert quality_gate_events batch [{i}:{i+len(chunk)}]: {err}")

    logger.info(f"Successfully migrated {count}/{len(events)} records to quality_gate_events.")
    return count


def migrate_character_profiles(client) -> int:
    """Migrates character_memory.json to character_profiles table."""
    path = os.path.join(BASE_DIR, "character_memory.json")
    if not os.path.exists(path):
        logger.info("character_memory.json not found. Skipping.")
        return 0

    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    characters = data.get("characters", [])
    count = 0
    for c in characters:
        name = c.get("name")
        if not name:
            continue
        success = supabase_db.upsert_character_profile(c)
        if success:
            count += 1
        else:
            logger.warning(f"Failed to migrate character profile: '{name}'")

    logger.info(f"Successfully migrated {count}/{len(characters)} character profiles to character_profiles.")
    return count


def migrate_terminology_store(client) -> int:
    """Migrates terminology_store.json to terminology_store table."""
    path = os.path.join(BASE_DIR, "terminology_store.json")
    if not os.path.exists(path):
        logger.info("terminology_store.json not found. Skipping.")
        return 0

    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    terms = data.get("terms", [])
    count = 0
    for t in terms:
        source_term = t.get("source_term")
        if not source_term:
            continue
        success = supabase_db.upsert_term(t)
        if success:
            count += 1
        else:
            logger.warning(f"Failed to migrate term: '{source_term}'")

    logger.info(f"Successfully migrated {count}/{len(terms)} terms to terminology_store.")
    return count


def run_migration() -> bool:
    """Runs full migration against Supabase."""
    logger.info("Starting one-time Supabase migration...")
    client = supabase_db.get_supabase_client()
    if client is None:
        msg = (
            "----------------------------------------------------------------------\n"
            "SUPABASE CREDENTIALS NOT CONFIGURED!\n"
            "To migrate local data to Supabase cloud:\n"
            "1. Create a free Supabase project at https://supabase.com\n"
            "2. Execute the schema from 'supabase_schema.sql' in the Supabase SQL Editor\n"
            "3. Set SUPABASE_URL and SUPABASE_KEY in your .env file or environment:\n"
            "   SUPABASE_URL=https://<your-project-id>.supabase.co\n"
            "   SUPABASE_KEY=<your-anon-or-service-role-key>\n"
            "4. Run: python migrate_to_supabase.py\n"
            "----------------------------------------------------------------------"
        )
        logger.warning(msg)
        return False

    mc_count = migrate_mask_corrections(client)
    qg_count = migrate_quality_gate_events(client)
    cp_count = migrate_character_profiles(client)
    ts_count = migrate_terminology_store(client)

    logger.info("\n" + "=" * 50)
    logger.info("MIGRATION COMPLETE:")
    logger.info(f" - Mask corrections:     {mc_count}")
    logger.info(f" - Quality gate events:  {qg_count}")
    logger.info(f" - Character profiles:   {cp_count}")
    logger.info(f" - Terminology terms:    {ts_count}")
    logger.info("=" * 50)
    return True


if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
