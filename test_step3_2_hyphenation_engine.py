"""
Unit tests for Big Update 3-Bosqich: Step 3.2: O'zbek Tili Bo'g'in Ko'chirish Qoidasi (Hyphenation Engine).
Verifies:
1. Vowel-consonant syllabification rules (bo-la, mak-tab, qah-ra-mon, hukm-dor-lar-ning).
2. Digraph atomic preservation (sh, ch, ng, g', o').
3. 2-Letter Boundary Constraint (no single letter 'a-' or '-a' left alone).
4. Protected character names and proper nouns (Spider-Man, Jonni, Doom) are never broken.
5. Dynamic hyphen fitting and line wrapping with hyphenation.
"""

import unittest
from uzbek_hyphenator import (
    syllabify_uzbek,
    get_hyphenation_points,
    fit_word_with_hyphen,
    is_protected_proper_noun,
    wrap_words_with_hyphenation
)


class TestStep32HyphenationEngine(unittest.TestCase):

    def test_syllabification_standard_patterns(self):
        """Verifies accurate grammatical syllable division across diverse word structures."""
        # V - C - V pattern
        self.assertEqual(syllabify_uzbek("bolalar"), ["bo", "la", "lar"])
        self.assertEqual(syllabify_uzbek("kitob"), ["ki", "tob"])
        self.assertEqual(syllabify_uzbek("qahramon"), ["qah", "ra", "mon"])

        # V - C1 C2 - V pattern
        self.assertEqual(syllabify_uzbek("maktab"), ["mak", "tab"])
        self.assertEqual(syllabify_uzbek("katta"), ["kat", "ta"])
        self.assertEqual(syllabify_uzbek("ishchi"), ["ish", "chi"])

        # Complex morphology & multi-syllable suffixes
        self.assertEqual(syllabify_uzbek("hukmdorlarning"), ["hukm", "dor", "lar", "ning"])
        self.assertEqual(syllabify_uzbek("tarjimalarimizni"), ["tar", "ji", "ma", "la", "ri", "miz", "ni"])

    def test_digraph_and_apostrophe_preservation(self):
        """Digraphs (sh, ch, ng, g', o') and tutuq marks must never be split across syllables."""
        # 'ch' digraph
        self.assertEqual(syllabify_uzbek("pichoq"), ["pi", "choq"])
        # 'sh' digraph
        self.assertEqual(syllabify_uzbek("shahar"), ["sha", "har"])
        # 'ng' digraph (kept intact)
        self.assertEqual(syllabify_uzbek("singil"), ["si", "ngil"])
        # 'o'' atomic vowel (o'r-ga-nish)
        self.assertEqual(syllabify_uzbek("o'rganish"), ["o'r", "ga", "nish"])
        # Apostrophe / tutuq mark
        self.assertEqual(syllabify_uzbek("ta'lim"), ["ta'", "lim"])

    def test_two_letter_boundary_constraint(self):
        """Single-letter syllable splits (e.g. a-ka, e-shik, o-ta) must be strictly forbidden."""
        # 'aka' has 2 syllables (a-ka), but leaving 'a-' is prohibited
        points_aka = get_hyphenation_points("aka")
        self.assertEqual(points_aka, [], "Single-letter prefix 'a-' must be rejected")

        # 'ota' has 2 syllables (o-ta), but leaving 'o-' is prohibited
        points_ota = get_hyphenation_points("ota")
        self.assertEqual(points_ota, [])

        # Valid 3+ syllable word (> 7 chars) should allow multi-letter splits
        points_qahramon = get_hyphenation_points("qahramon")
        self.assertIn(("qah-", "ramon"), points_qahramon)
        self.assertIn(("qahra-", "mon"), points_qahramon)

    def test_words_with_seven_or_fewer_letters_no_hyphenation(self):
        """Words with 7 or fewer letters must never be hyphenated; only words > 7 letters qualify."""
        # 7 or fewer letters: prohibited
        self.assertEqual(get_hyphenation_points("bolalar"), [], "7-letter word must NOT be hyphenated")
        self.assertEqual(get_hyphenation_points("maktab"), [], "6-letter word must NOT be hyphenated")
        self.assertEqual(get_hyphenation_points("kitob"), [], "5-letter word must NOT be hyphenated")
        self.assertEqual(get_hyphenation_points("dastlab"), [], "7-letter word must NOT be hyphenated")

        # More than 7 letters: allowed
        self.assertTrue(len(get_hyphenation_points("qahramon")) > 0, "8-letter word must be allowed to hyphenate")
        self.assertTrue(len(get_hyphenation_points("hukmdorlar")) > 0, "10-letter word must be allowed to hyphenate")

    def test_protected_proper_nouns_no_hyphenation(self):
        """Hero names and comic characters must never be split by hyphenation."""
        self.assertTrue(is_protected_proper_noun("Spider-Man"))
        self.assertTrue(is_protected_proper_noun("Jonni"))
        self.assertTrue(is_protected_proper_noun("Doom"))
        self.assertTrue(is_protected_proper_noun("Parker"))
        self.assertTrue(is_protected_proper_noun("Betmen"))

        # get_hyphenation_points must return empty list for protected entities
        self.assertEqual(get_hyphenation_points("Jonni"), [])
        self.assertEqual(get_hyphenation_points("Parker"), [])
        self.assertEqual(get_hyphenation_points("Betmen"), [])

    def test_fit_word_with_hyphen(self):
        """fit_word_with_hyphen selects the longest valid prefix fitting under max_w."""
        def mock_measure(text: str) -> float:
            return len(text) * 10.0

        # 'qahramon' has split points:
        # 'qah-' (4 chars = 40px) and 'qahra-' (6 chars = 60px)
        
        # When max_w is 50px: only 'qah-' fits
        fit1 = fit_word_with_hyphen("qahramon", 50.0, mock_measure)
        self.assertIsNotNone(fit1)
        self.assertEqual(fit1, ("qah-", "ramon"))

        # When max_w is 70px: 'qahra-' fits (longer prefix preferred)
        fit2 = fit_word_with_hyphen("qahramon", 70.0, mock_measure)
        self.assertIsNotNone(fit2)
        self.assertEqual(fit2, ("qahra-", "mon"))

        # When max_w is 30px: neither fits
        fit3 = fit_word_with_hyphen("qahramon", 30.0, mock_measure)
        self.assertIsNone(fit3)

    def test_wrap_words_with_hyphenation_overflow_recovery(self):
        """Words that would otherwise overflow or cause massive line gaps are smoothly hyphenated."""
        def mock_measure(text: str) -> float:
            return len(text) * 10.0

        # Target: 2 lines of width 90px each
        # Sentence: "Bizning qahramonimiz"
        # Without hyphenation: "Bizning" is 70px. "qahramonimiz" is 120px (cannot fit on line 1, cannot fit on line 2!)
        words = ["Bizning", "qahramonimiz"]
        row_widths = [130.0, 130.0]

        lines = wrap_words_with_hyphenation(words, row_widths, mock_measure)
        self.assertIsNotNone(lines, "Should wrap successfully with hyphenation")
        self.assertEqual(len(lines), 2)
        # Line 1 should contain 'Bizning' and a hyphenated prefix of 'qahramonimiz'
        self.assertTrue(any("-" in w for w in lines[0]), f"Line 0 must have hyphenated word, got {lines[0]}")

    def test_bubble_lettering_hyphenation_integration(self):
        """fit_text_to_bubble in bubble_lettering imports and executes hyphenation correctly."""
        from bubble_lettering import fit_text_to_bubble, fit_word_with_hyphen as bl_hyphen
        self.assertIsNotNone(bl_hyphen, "bubble_lettering must have fit_word_with_hyphen imported")


if __name__ == "__main__":
    unittest.main()

