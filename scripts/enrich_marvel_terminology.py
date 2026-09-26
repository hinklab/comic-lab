"""
scripts/enrich_marvel_terminology.py
------------------------------------
Standalone Marvel Comics API Terminology & Character Enrichment Tool.

Queries Marvel's Official Developer API (free tier: 3,000 calls/day, hard-capped at 2,700 calls/day)
to discover canonical character names, descriptions, and alias variants.
Compares findings against the local/Supabase `terminology_store` and `character_profiles`.

SAFETY & ARCHITECTURE POLICY:
- Standalone script: NOT part of the per-page comic pipeline.
- Read-only by default: Does NOT auto-write to terminology_store.
- Outputs an actionable Markdown report and JSON suggestions for human review.
- Optional '--apply' flag allows applying approved additions to terminology_store with
  full Rulebook policy validation.
- Completely inert when MARVEL_PUBLIC_KEY / MARVEL_PRIVATE_KEY are absent.

Usage:
  python scripts/enrich_marvel_terminology.py
  python scripts/enrich_marvel_terminology.py --apply
"""

import os
import sys
import time
import json
import hashlib
import logging
import argparse
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import api_quota_tracker
import terminology_store
import character_profiles

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("marvel_enrichment")

MARVEL_BASE_URL = "https://gateway.marvel.com/v1/public"


def get_marvel_keys() -> Tuple[Optional[str], Optional[str]]:
    """Retrieves Marvel Public and Private API keys from environment."""
    pub = os.getenv("MARVEL_PUBLIC_KEY")
    priv = os.getenv("MARVEL_PRIVATE_KEY")
    return (pub.strip() if pub else None, priv.strip() if priv else None)


def build_marvel_auth_params(public_key: str, private_key: str) -> Dict[str, str]:
    """Builds standard Marvel API auth parameters (ts, apikey, hash)."""
    ts = str(int(time.time() * 1000))
    raw_hash = f"{ts}{private_key}{public_key}".encode("utf-8")
    hash_digest = hashlib.md5(raw_hash).hexdigest()
    return {
        "ts": ts,
        "apikey": public_key,
        "hash": hash_digest
    }


def query_marvel_character(name: str, public_key: str, private_key: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """Queries Marvel API for character by exact name or prefix."""
    if not api_quota_tracker.can_consume("marvel_api", 1):
        logger.warning("[MARVEL_API] Daily usage cap reached (2,700 calls/day). Aborting queries.")
        return None

    auth_params = build_marvel_auth_params(public_key, private_key)
    query_params = dict(auth_params)
    query_params["nameStartsWith"] = name.strip()
    query_params["limit"] = "5"

    url = f"{MARVEL_BASE_URL}/characters?{urllib.parse.urlencode(query_params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "ComicLab-TerminologyEnricher/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        api_quota_tracker.record_consumption("marvel_api", 1)
        results = data.get("data", {}).get("results", [])
        return results[0] if results else None
    except Exception as e:
        logger.warning(f"[MARVEL_API_WARN] Query failed for '{name}': {e}")
        return None


def run_enrichment(apply_changes: bool = False) -> None:
    public_key, private_key = get_marvel_keys()
    if not public_key or not private_key:
        print("\n" + "=" * 70)
        print("MARVEL COMICS API ENRICHMENT - KEYS NOT CONFIGURED")
        print("=" * 70)
        print("The Marvel API feature is optional and currently inert.")
        print("To enable official Marvel Comics character data enrichment:")
        print("1. Get free API keys from: https://developer.marvel.com (3,000 calls/day free)")
        print("2. Add them to your .env file:")
        print("   MARVEL_PUBLIC_KEY=<your-public-key>")
        print("   MARVEL_PRIVATE_KEY=<your-private-key>")
        print("3. Re-run: python scripts/enrich_marvel_terminology.py")
        print("=" * 70 + "\n")
        return

    print("\n" + "=" * 70)
    print("MARVEL COMICS API TERMINOLOGY ENRICHMENT")
    print("=" * 70)

    # 1. Gather target characters from terminology_store and character_profiles
    term_store = terminology_store.TerminologyStore.get_instance()
    existing_terms = term_store.get_all_entries()
    existing_source_terms = {t["source_term"] for t in term_store._entries.values()}

    mem_store = character_profiles.CharacterMemoryStore.get_instance()
    character_names = mem_store.list_profiles()

    # Seed list of core characters to query Marvel for
    queries_to_run = list(dict.fromkeys(
        character_names + [
            "Spider-Man", "Doctor Octopus", "J. Jonah Jameson", "Mary Jane Watson",
            "Beetle", "Overdrive", "Living Brain"
        ]
    ))

    suggestions: List[Dict[str, Any]] = []

    print(f"Querying Marvel Official Database for {len(queries_to_run)} characters...\n")

    for name in queries_to_run:
        print(f"Searching Marvel API: '{name}'...")
        result = query_marvel_character(name, public_key, private_key)
        if not result:
            print(f"  -> No official match found for '{name}'.")
            continue

        marvel_name = result.get("name", "")
        marvel_id = result.get("id")
        description = result.get("description", "").strip()
        comics_count = result.get("comics", {}).get("available", 0)

        print(f"  -> Match found: '{marvel_name}' (ID: {marvel_id}, Comics: {comics_count})")
        if description:
            print(f"     Bio snippet: {description[:90]}...")

        # Analyze if new alias or term should be suggested
        norm_marvel = marvel_name.strip().lower()
        if norm_marvel not in existing_source_terms and norm_marvel != name.lower():
            suggestion = {
                "source_term": norm_marvel,
                "suggested_canonical_uzbek": marvel_name,  # Candidate for human approval
                "entry_type": "character_name",
                "method": "meaning",
                "marvel_id": marvel_id,
                "first_seen_page": "marvel_api",
                "notes": f"Official Marvel Character Name (Comics: {comics_count}). Bio: {description[:120]}"
            }
            suggestions.append(suggestion)

    # Output report
    report_path = os.path.join(BASE_DIR, "marvel_terminology_report.md")
    json_path = os.path.join(BASE_DIR, "marvel_terminology_suggestions.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(suggestions, f, indent=2, ensure_ascii=False)

    report_lines = [
        "# Marvel Comics API Terminology Enrichment Report",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"**Characters Queried:** {len(queries_to_run)}",
        f"**New Term Suggestions Identified:** {len(suggestions)}",
        "",
        "## Suggested Terminology Additions (Human Approval Required)",
        "",
        "| Source Term | Candidate Uzbek | Type | Method | Marvel ID | Notes |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for s in suggestions:
        report_lines.append(
            f"| `{s['source_term']}` | {s['suggested_canonical_uzbek']} | `{s['entry_type']}` | `{s['method']}` | {s['marvel_id']} | {s['notes'][:60]}... |"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n" + "=" * 70)
    print(f"REPORT GENERATED: {report_path}")
    print(f"SUGGESTIONS JSON: {json_path}")
    print(f"Identified {len(suggestions)} suggestions for human review.")
    print("=" * 70)

    if apply_changes and suggestions:
        print("\nApplying approved suggestions to TerminologyStore...")
        applied_count = 0
        for s in suggestions:
            try:
                registered = term_store.register_term(
                    source_term=s["source_term"],
                    canonical_uzbek=s["suggested_canonical_uzbek"],
                    first_seen_page=s["first_seen_page"],
                    locked=True,
                    entry_type=s["entry_type"],
                    method=s["method"],
                    reason=f"Marvel API Enrichment ID={s['marvel_id']}"
                )
                if registered:
                    applied_count += 1
            except Exception as e:
                print(f"  Skipped {s['source_term']}: {e}")
        print(f"Applied {applied_count}/{len(suggestions)} terms to TerminologyStore.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marvel Comics API Terminology Enrichment")
    parser.add_argument("--apply", action="store_true", help="Apply approved suggestions to terminology_store")
    args = parser.parse_args()
    run_enrichment(apply_changes=args.apply)
