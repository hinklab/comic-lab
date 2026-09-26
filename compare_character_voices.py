"""
compare_character_voices.py - Side-by-side comparison of Generic vs Character-Voiced translation.
"""

import engine
import local_translator
import naturalization_rules
import character_profiles


TEST_LINES = [
    "SECOND DAY ON THE 'JOB'...",
    "...AND NOT ONE OF THEM KNOWS I'M A DIFFERENT SPIDER-MAN.",
    "BUT SOON THEY'LL ALL LEARN THAT I AM A FAR SUPERIOR ONE.",
    "IMPRESSED? DON'T BOTHER.",
    "PLAN: EPSILON FIVE!",
    "INSTANT FRICTIONLESS SURFACE!",
    "BRUISED TRACHEA. CRUSHED LARYNX.",
    "FOOLS! YOU ARE FACING A SUPERIOR INTELLECT!",
]


def run_comparison():
    print("=" * 80)
    print("CHARACTER VOICE NATURALIZATION COMPARISON (OFFLINE NMT)")
    print("Speaker Profile: Superior Spider-Man (Otto Octavius)")
    print("=" * 80)

    for idx, en_line in enumerate(TEST_LINES, 1):
        # 1. Raw NMT
        raw_nmt = local_translator.translate_offline(en_line)

        # 2. Generic Naturalization (no speaker)
        generic_uz = naturalization_rules.apply_naturalization(raw_nmt, en_orig=en_line, character_profile=None)

        # 3. Superior Spider-Man (Otto Octavius)
        otto_uz = engine.translate_spiderman_uzbek(en_line, speaker="Superior Spider-Man")

        print(f"\n[{idx}] ORIGINAL (EN):")
        print(f"    {en_line}")
        print(f"    RAW NMT:     {raw_nmt}")
        print(f"    GENERIC UZ:  {generic_uz}")
        print(f"    OTTO VOICE:  {otto_uz}")

    print("\n" + "=" * 80)
    print("Comparison completed successfully.")


if __name__ == "__main__":
    run_comparison()
