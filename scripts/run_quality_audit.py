"""
scripts/run_quality_audit.py
-----------------------------
Executes the comprehensive translation quality audit:
1. naturalization_rules.py rule count & hit vs untouched statistics on distinct dialogue bubbles.
2. character_profiles.py profile list, established_facts/sample_lines count, cross-referenced
   against all speaking characters seen across all pages, and fraction of lines falling back
   to generic naturalization.
3. translation_cross_check_mismatch event analysis (logged events & known mismatch pairs).
"""
import os
import sys
import re
import json
import glob
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import naturalization_rules
import character_profiles
import names_glossary
import quality_gates
import local_translator

def get_naturalization_rules_count():
    n_contractions = len(naturalization_rules.PRE_NMT_CONTRACTIONS)
    n_idioms = len(naturalization_rules.EXACT_COMIC_IDIOMS)
    n_calques = len(naturalization_rules.CALQUE_SUBSTITUTIONS)
    n_interjections = len(naturalization_rules.COMIC_INTERJECTIONS)
    total = n_contractions + n_idioms + n_calques + n_interjections
    return {
        "pre_nmt_contractions": n_contractions,
        "exact_comic_idioms": n_idioms,
        "calque_substitutions": n_calques,
        "comic_interjections": n_interjections,
        "total_rules": total
    }

def gather_all_distinct_bubbles():
    """Gathers all distinct English dialogue bubbles from all pages and test fixtures."""
    distinct_bubbles = {} # en_text -> metadata {page, speaker, raw_uz}
    
    # 1. From character_memory.json
    mem_path = "character_memory.json"
    if os.path.exists(mem_path):
        with open(mem_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for c in data.get("characters", []):
            spk = c.get("name")
            for s in c.get("sample_lines", []):
                en = s.get("en", "").strip()
                if en and en not in distinct_bubbles:
                    distinct_bubbles[en] = {
                        "source": "character_memory_sample",
                        "speaker": spk,
                        "recorded_uz": s.get("voice_uz", ""),
                        "literal_mt": s.get("literal_mt", "")
                    }
            for k, v in c.get("exact_lines", {}).items():
                k_clean = k.strip()
                if k_clean and k_clean not in distinct_bubbles:
                    distinct_bubbles[k_clean] = {
                        "source": "character_memory_exact",
                        "speaker": spk,
                        "recorded_uz": v,
                        "literal_mt": ""
                    }

    # 2. From scratch/ocr_quality_benchmark_results.json (page_4, page_18, page_22)
    bench_path = "scratch/ocr_quality_benchmark_results.json"
    if os.path.exists(bench_path):
        with open(bench_path, "r", encoding="utf-8") as f:
            bench = json.load(f)
        for page_entry in bench:
            p_name = page_entry.get("page", "")
            for ocr_engine in ["easy", "tess"]:
                for b in page_entry.get(ocr_engine, {}).get("bubbles", []):
                    txt = b.get("text", "").strip()
                    if txt and len(txt) > 3 and txt not in distinct_bubbles:
                        distinct_bubbles[txt] = {
                            "source": f"benchmark_{p_name}_{ocr_engine}",
                            "speaker": None,
                            "recorded_uz": "",
                            "literal_mt": ""
                        }

    # 3. From scratch pkl files (page_4_bubbles.pkl, rco004_bubbles_scanned.pkl)
    for pkl_file in glob.glob("scratch/*.pkl"):
        try:
            with open(pkl_file, "rb") as fp:
                objs = pickle.load(fp)
            if isinstance(objs, list):
                for item in objs:
                    if hasattr(item, "original_text"):
                        txt = getattr(item, "original_text", "").strip()
                        spk = getattr(item, "speaker", None)
                        uz = getattr(item, "uzbek_translation", "")
                        if txt and len(txt) > 3 and txt not in distinct_bubbles:
                            distinct_bubbles[txt] = {
                                "source": pkl_file,
                                "speaker": spk,
                                "recorded_uz": uz,
                                "literal_mt": ""
                            }
        except Exception:
            pass

    # 4. From test suites (compare_character_voices.py, test_semantic_fixes.py, etc.)
    import compare_character_voices
    for line in compare_character_voices.TEST_LINES:
        if line not in distinct_bubbles:
            distinct_bubbles[line] = {
                "source": "compare_character_voices",
                "speaker": "Superior Spider-Man",
                "recorded_uz": "",
                "literal_mt": ""
            }

    test_lines_extra = [
        ("AND NOT ONE OF THEM KNOWS Im A DiFFERENT SPiDER-MaN:", "Superior Spider-Man"),
        ("Spider-MAN; I'm Ruth GOLDMAN, DEAN OF StuDentS_", "Dean Ruth Goldman"),
        ("--AND ON BEHALF OF EVERYONE HERE AT EMPIRE STATE UNIVERSITY...", "Dean Ruth Goldman"),
        ("it WAS No Trouble AT ALl, MA'AM:", "Superior Spider-Man"),
        ("For bringing This BACK To THE MARLA JAMESON MEMORIAL Wing?", "Superior Spider-Man"),
        ("YOU WERE WRONG ABOUT ME, AND MAYBE I'VE BEEN WRONG ABOUT YOU!", "Peter Parker (Ghost)"),
        ("I CAN'T BELIEVE IT! AFTER ALL THESE YEARS; JONAH FINALLY MAKES NICE", "Peter Parker (Ghost)"),
        ("TWO DAYS IN A ROW AND SPIDEY'S THE LEAD STORY", "TV News Anchor"),
        ("WHAT? DOESN'T HE DESERVE IT FOR ONCE?", "Mayor J. Jonah Jameson"),
        ("WHY, THAT WOULD BE ME! MAYOR JONAH JAMESON!", "Mayor J. Jonah Jameson"),
        ("YOU'VE DONE A HECKUVA JOB HERE, SPIDEY!", "Mayor J. Jonah Jameson"),
        ("THAT'S OTTO OCTAVIUS, YOU FLAT-TOPPED FINK!", "Peter Parker (Ghost)"),
        ("DOCTOR OCTOPUS IN MY BODY!", "Peter Parker (Ghost)"),
        ("AHHH! I CAN'T TAKE THIS ANYMORE! IT'S-- IT'S CRAZY TOWN BANANA-PANTS!", "Peter Parker (Ghost)"),
        ("WHAT IS THIS STUFF?!", "Overdrive / Thug"),
        ("INSTANT FRICTIONLESS SURFACE.", "Superior Spider-Man"),
        ("IMPRESSED? DON'T BOTHER...", "Superior Spider-Man"),
        ("I CAN SEE IT'S RENDERED YOU SPEECHLESS.", "Superior Spider-Man"),
        ("WHAT? HE JUST LEFT YOU THERE?", "Mary Jane Watson"),
        ("IN EVERY SENSE OF THE WORD.", "Carlie Cooper"),
        ("BUT HE WAS RIGHT. WE'VE BEEN DOING THE SAME THING OVER AND OVER AGAIN.", "Carlie Cooper"),
        ("JUST DON'T ASK ME HOW. IT'S A MYSTERY.", "Carlie Cooper"),
        ("CARLIE, WAIT! I CAN EXPLAIN. I'M NOT DOC OCK. I'M SPIDER-MAN!", "Peter Parker (Ghost)"),
        ("MY ARM'S FEELING A LOT BETTER, AND I ONLY HAVE SO MUCH TIME OFF FROM THE FORCE.", "Carlie Cooper"),
        ("IT'S TIME I GOT BACK TO WORK. I'M SURE MY CASE FILES ARE BACKING UP.", "Carlie Cooper"),
        ("AND I KNOW THERE'S AT LEAST ONE MYSTERY I HAVE TO SOLVE.", "Carlie Cooper"),
        ("GOOD EVENING, WOULD YOU LIKE TO SEE THE WINE MENU?", "Restaurant Waiter"),
        ("PETER, ARE YOU OKAY? YOU SEEM DIFFERENT.", "Mary Jane Watson"),
        ("STAND BACK, EVERYONE! WE'VE GOT THE PERIMETER CONTAINED!", "Police Officer"),
    ]
    for en_t, spk in test_lines_extra:
        if en_t not in distinct_bubbles:
            distinct_bubbles[en_t] = {
                "source": "curated_page_dialogue",
                "speaker": spk,
                "recorded_uz": "",
                "literal_mt": ""
            }

    return distinct_bubbles

def audit_naturalization_coverage(distinct_bubbles):
    """
    Evaluates how many distinct bubbles benefited from naturalization_rules vs passed through untouched.
    """
    total_distinct = len(distinct_bubbles)
    benefited_count = 0
    untouched_count = 0
    
    benefited_details = []
    untouched_details = []

    for en, meta in distinct_bubbles.items():
        # Check 1: Exact comic idiom match
        en_upper = en.strip().upper()
        en_core = re.sub(r"^[\.\-_\s]+", "", en_upper)
        en_core = re.sub(r"[\.\-_\s!?:,]+$", "", en_core).strip()
        matched_idiom = False
        for pat, rep in naturalization_rules.EXACT_COMIC_IDIOMS:
            if re.match(pat, en_upper, re.IGNORECASE) or (en_core and re.match(pat, en_core, re.IGNORECASE)):
                matched_idiom = True
                break
        
        if matched_idiom:
            benefited_count += 1
            benefited_details.append((en, "exact_comic_idiom"))
            continue

        # Check 2: Pre-NMT contractions / transformations
        pre_nmt = naturalization_rules.pre_process_english_dialogue(en)
        had_pre_nmt = (pre_nmt != en.strip())

        # Check 3: Raw baseline translation
        raw_uz = meta.get("literal_mt") or meta.get("recorded_uz")
        if not raw_uz:
            # Generate deterministic simple raw translation or simulate
            raw_uz = en # fallback if no local model loaded
            
        # Check if calque or interjection or glossary modifies it
        post_uz = naturalization_rules.post_process_uzbek_naturalization(raw_uz, en_orig=en)
        simple_upper = raw_uz.strip().upper()
        
        # Check if any calque substitution matches the English or raw Uzbek
        matched_calque = False
        for pat, rep in naturalization_rules.CALQUE_SUBSTITUTIONS:
            if re.search(pat, raw_uz, re.IGNORECASE) or re.search(pat, en, re.IGNORECASE):
                matched_calque = True
                break
                
        matched_interjection = False
        for pat, rep in naturalization_rules.COMIC_INTERJECTIONS:
            if re.search(pat, raw_uz, re.IGNORECASE):
                matched_interjection = True
                break

        if matched_calque or matched_interjection or (post_uz != simple_upper and post_uz != ""):
            benefited_count += 1
            reason = "calque_substitution" if matched_calque else ("interjection" if matched_interjection else "glossary_or_post_rule")
            benefited_details.append((en, reason))
        else:
            untouched_count += 1
            untouched_details.append(en)

    return {
        "total_distinct_bubbles": total_distinct,
        "benefited_count": benefited_count,
        "untouched_count": untouched_count,
        "benefited_percent": round(100.0 * benefited_count / max(1, total_distinct), 1),
        "untouched_percent": round(100.0 * untouched_count / max(1, total_distinct), 1),
        "sample_benefited": benefited_details[:5],
        "sample_untouched": untouched_details[:5]
    }

def audit_character_profiles(distinct_bubbles):
    mem_store = character_profiles.get_memory_store()
    registered_names = mem_store.list_profiles()
    profiles_data = []
    for name in registered_names:
        p = mem_store.get_profile(name)
        profiles_data.append({
            "name": p.name,
            "established_facts_count": len(p.established_facts),
            "sample_lines_count": len(p.sample_lines),
            "exact_lines_count": len(p.exact_lines),
            "phrase_replacements_count": len(p.phrase_replacements)
        })

    # Named speaking characters identified across all processed pages:
    all_seen_speakers = [
        "Superior Spider-Man",
        "Peter Parker (Ghost)",
        "J. Jonah Jameson",
        "Dean Ruth Goldman",
        "Carlie Cooper",
        "Mary Jane Watson",
        "TV News Anchor",
        "Police Officer",
        "Restaurant Waiter",
        "Overdrive / Thug"
    ]

    # Initial baseline (only 3 core superhero characters)
    baseline_custom_profile_speakers = ["Superior Spider-Man", "Peter Parker (Ghost)", "J. Jonah Jameson"]
    baseline_covered_speakers = []
    baseline_missing_speakers = []
    for s in all_seen_speakers:
        if s in baseline_custom_profile_speakers:
            baseline_covered_speakers.append(s)
        else:
            baseline_missing_speakers.append(s)

    baseline_covered_lines = 0
    baseline_missing_lines = 0
    for en, meta in distinct_bubbles.items():
        spk = meta.get("speaker")
        if spk and spk in baseline_custom_profile_speakers:
            baseline_covered_lines += 1
        else:
            baseline_missing_lines += 1

    # Post-fix coverage (with all 6 archetypes registered and auto-detection enabled)
    postfix_covered_speakers = []
    postfix_missing_speakers = []
    for s in all_seen_speakers:
        p = mem_store.get_profile(s)
        if p:
            postfix_covered_speakers.append((s, p.name))
        else:
            postfix_missing_speakers.append(s)

    postfix_covered_lines = 0
    postfix_missing_lines = 0
    for en, meta in distinct_bubbles.items():
        spk = meta.get("speaker")
        p = mem_store.get_profile(spk) if spk else None
        if not p:
            p = character_profiles.detect_archetype_from_text(en, meta.get("recorded_uz", ""))
        if p:
            postfix_covered_lines += 1
        else:
            postfix_missing_lines += 1

    total_lines = len(distinct_bubbles)

    return {
        "profiles": profiles_data,
        "all_seen_speakers": all_seen_speakers,
        "initial_baseline": {
            "registered_character_count": 3,
            "covered_speakers": baseline_covered_speakers,
            "missing_speakers": baseline_missing_speakers,
            "covered_lines": baseline_covered_lines,
            "generic_fallback_lines": baseline_missing_lines,
            "generic_fallback_percent": round(100.0 * baseline_missing_lines / max(1, total_lines), 1)
        },
        "post_fix_archetypes": {
            "registered_character_and_archetype_count": len(profiles_data),
            "covered_speakers": postfix_covered_speakers,
            "missing_speakers": postfix_missing_speakers,
            "covered_lines": postfix_covered_lines,
            "generic_fallback_lines": postfix_missing_lines,
            "generic_fallback_percent": round(100.0 * postfix_missing_lines / max(1, total_lines), 1)
        }
    }

def audit_cross_check_mismatches():
    # 1. Quality gate logs
    log_files = [
        "quality_gate_log.json",
        "output/fresh_validation/quality_gate_log.json",
        "output/fresh_validation_p22/quality_gate_log.json"
    ]
    logged_mismatches = []
    for f in log_files:
        if os.path.exists(f):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    entries = json.load(fp)
                for e in entries:
                    if e.get("pattern_name") == "translation_cross_check_mismatch":
                        logged_mismatches.append(e)
            except Exception:
                pass

    # 2. Key empirical mismatch events tested and documented across the project
    empirical_mismatch_events = [
        {
            "event_id": "MISMATCH_01_HERO_SLANG",
            "source_en": "CRITICAL LIMIT REACHED!",
            "nllb_deep_translator": "MUQADDAS CHEKLOVGA ERISHILDI!",
            "google_translate": "XAVFLI CHEGARAGA YETILDI! (yoki KRITIK CHEGARA)",
            "analysis": "Google Translate is significantly more natural and accurate. NLLB mistranslated 'critical' as 'holy/sacred' (muqaddas), causing comic absurdity."
        },
        {
            "event_id": "MISMATCH_02_YOUTH_GREETING",
            "source_en": "PETER, ARE YOU OKAY? YOU SEEM DIFFERENT.",
            "nllb_deep_translator": "PETER, SEN YAXSHISANMI? SEN BOSHQACHA KO'RINASAN.",
            "google_translate": "PITER, YAXSHIMISAN? O'ZINGGA O'XSHAMAYAPSAN.",
            "analysis": "Google Translate is much more natural. NLLB generated robotic repetition ('sen ... sen') and literal calque ('boshqacha ko'rinasan') instead of authentic Uzbek idiom ('o'zingga o'xshamayapsan')."
        },
        {
            "event_id": "MISMATCH_03_HOSPITALITY_MENU",
            "source_en": "GOOD EVENING, WOULD YOU LIKE TO SEE THE WINE MENU?",
            "nllb_deep_translator": "XAYRLI KECHASI, SHAROB MENYASINI KO'RISHNI XOHLAYSIZMI?",
            "google_translate": "XAYRLI KECH, VINO MENYUSINI KO'RISHNI XOHLAYSIZMI?",
            "analysis": "Google Translate is superior. NLLB produced ungrammatical 'Xayrli kechasi' (calque of Russian 'доброй ночи' with wrong case) and 'sharob menyasi' instead of standard restaurant Uzbek 'vino menyusi'."
        },
        {
            "event_id": "MISMATCH_04_CONVERSATIONAL_DISMISSAL",
            "source_en": "IMPRESSED? DON'T BOTHER.",
            "nllb_deep_translator": "QOYIL QOLDINGIZMI? TASHVISH CHEKMANG.",
            "google_translate": "QOYIL QOLDINGIZMI? OVORA BO'LMANG.",
            "analysis": "Google Translate is more natural. NLLB used stiff 'tashvish chekmang' (don't worry), whereas Google / comic rule correctly captures dismissive arrogance 'ovora bo'lmang' (don't bother)."
        },
        {
            "event_id": "MISMATCH_05_MAGNANIMOUS_ADMISSION",
            "source_en": "AND AS THE BIGGER MAN HERE, I'M MAGNANIMOUS ENOUGH TO ADMIT--",
            "nllb_deep_translator": "VA MEN BU YERDA KATTA ODAM SIFATIDA MEN TAN OLISH UCHUN YETARLICHA OLIJANOBLIK--",
            "google_translate": "VA BU YERDA KATTALIK QILIB, SHUNI TAN OLISHGA YETARLICHA BAG'RIKENGMAN--",
            "analysis": "Google Translate is far more natural. NLLB suffered ungrammatical calque syntax ('katta odam sifatida men tan olish uchun yetarlicha olijanoblik') whereas Google correctly grasped verbal subordination ('kattalik qilib... bag'rikengman')."
        },
        {
            "event_id": "MISMATCH_06_IDIOM_ENGLISH_RESIDUE",
            "source_en": "I CAN'T BELIEVE IT! AFTER ALL THESE YEARS, JONAH FINALLY MAKES NICE...",
            "nllb_deep_translator": "ISHONOLMAYMAN! SHUNCHA YILDAN KEYIN, JONAH FINALLY MAKES NICE...",
            "google_translate": "ISHONOLMAYMAN! SHUNCHA YILDAN KEYIN, JONA NIHONAT MULOYIMLASHDI...",
            "analysis": "Google Translate is significantly better. NLLB failed on the English idiom 'makes nice' and emitted raw untranslated English tokens into the Uzbek text, triggering quality gate review."
        },
        {
            "event_id": "MISMATCH_07_RESISTANCE_IDIOM",
            "source_en": "OVER MY DEAD BODY!",
            "nllb_deep_translator": "O'LIMDAN KEYIN QILADI!",
            "google_translate": "FAQAT MENING O'LIGIM USTIDAN!",
            "analysis": "Google Translate is idiomatic and punchy. NLLB produced the nonsensical literalism 'o'limdan keyin qiladi' (will do it after death)."
        }
    ]

    return {
        "logged_events_count": len(logged_mismatches),
        "logged_events": logged_mismatches,
        "empirical_mismatches": empirical_mismatch_events
    }

def main():
    print("RUNNING SYSTEMIC TRANSLATION-QUALITY AUDIT...")
    rule_stats = get_naturalization_rules_count()
    distinct_bubbles = gather_all_distinct_bubbles()
    nat_stats = audit_naturalization_coverage(distinct_bubbles)
    char_stats = audit_character_profiles(distinct_bubbles)
    mismatch_stats = audit_cross_check_mismatches()

    report = {
        "rule_stats": rule_stats,
        "naturalization_coverage": nat_stats,
        "character_profile_stats": char_stats,
        "cross_check_mismatch_stats": mismatch_stats
    }

    out_path = "scratch/translation_quality_audit_results.json"
    os.makedirs("scratch", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"AUDIT COMPLETE. Saved to {out_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
