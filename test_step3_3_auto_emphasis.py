"""
Unit tests for Big Update 3-Bosqich: Step 3.3: Avtomatik Semantik Urg'u (Auto-Emphasis Prediction).
Verifies:
1. Tier 1 Negation and absolute phrase emphasis (hech qachon, aslo, mutlaqo).
2. Tier 2 Intensifier emphasis (albatta, juda, darhol, faqat).
3. Tier 3 Action imperative emphasis (to'xta, qara, qoch, yordam).
4. Tier 4 Modal & strong verb emphasis (shart, kerak, bilolmayman).
5. Erroneous 100% full-sentence bold stripping and focal replacement.
6. Preservation of valid pre-existing emphasis.
7. English source emphasis projection to Uzbek.
8. Clitic protection (stop words like va, ham, bu are never bolded alone).
"""

import unittest
from auto_emphasis import (
    apply_auto_emphasis,
    is_fully_bolded,
    has_existing_emphasis,
    strip_all_bold_tags,
    project_english_emphasis
)


class TestStep33AutoEmphasis(unittest.TestCase):

    def test_tier_1_negation_emphasis(self):
        """Tier 1 negations ('hech qachon', 'aslo') receive primary comic emphasis."""
        res = apply_auto_emphasis("MEN HECH QACHON TASLIM BO'LMAYMAN!")
        self.assertIn("**HECH QACHON**", res)
        # Verify whole sentence is not bold
        self.assertFalse(res.startswith("**MEN"))

    def test_tier_2_intensifier_emphasis(self):
        """Tier 2 intensifiers ('albatta', 'juda') receive emphasis."""
        res = apply_auto_emphasis("BIZ BUNI ALBATTA UDDALAYMIZ.")
        self.assertIn("**ALBATTA**", res)

        res2 = apply_auto_emphasis("BU JUDA XAVFLI BO'LIB QOLDI.")
        self.assertIn("**JUDA**", res2)

    def test_tier_3_imperative_action_emphasis(self):
        """Action calls and urgent commands ('to'xta', 'qara') receive emphasis."""
        res = apply_auto_emphasis("TO'XTA, NIMA QILYAPSAN?!")
        self.assertIn("**TO'XTA**", res)

    def test_tier_4_modal_emphasis(self):
        """Modals like 'shart' or 'kerak' receive emphasis when no tier 1/2 word is present."""
        res = apply_auto_emphasis("UNI TO'XTATISHIMIZ SHART!")
        self.assertIn("**SHART**", res)

    def test_strip_full_bubble_bolding(self):
        """If a bubble is 100% wrapped in bold, it must be unwrapped and key focal word emphasized."""
        bad_bubble = "**BU BUTUNLAY NOTO'G'RI BO'LGAN ISH**"
        self.assertTrue(is_fully_bolded(bad_bubble))

        cleaned = apply_auto_emphasis(bad_bubble)
        self.assertFalse(cleaned.startswith("**BU"))
        self.assertFalse(cleaned.endswith("ISH**"))
        # Should emphasize 'BUTUNLAY' or similar focal word
        self.assertIn("**BUTUNLAY**", cleaned)

    def test_existing_valid_emphasis_preserved(self):
        """Well-placed existing bold emphasis is preserved without tampering."""
        good_text = "U **AYNAN** O'SHA ODAM EDI."
        res = apply_auto_emphasis(good_text)
        self.assertEqual(res, good_text)

    def test_english_source_projection(self):
        """Uppercase or bold emphasis in English is mapped directly to the Uzbek word."""
        en_orig = "I will NEVER surrender!"
        uz_text = "MEN HECH QACHON TASLIM BO'LMAYMAN!"
        res = apply_auto_emphasis(uz_text, en_orig=en_orig)
        self.assertIn("**HECH QACHON**", res)

    def test_clitics_and_glue_words_protected(self):
        """Common clitics (va, ham, bu, bilan, uchun) must never be bolded alone."""
        res = apply_auto_emphasis("BU BIZ UCHUN JUDA MUHIM.")
        # 'juda' or 'muhim' should be bold, NOT 'va', 'ham', 'bu', 'uchun'
        self.assertNotIn("**BU**", res)
        self.assertNotIn("**UCHUN**", res)
        self.assertIn("**JUDA**", res)

    def test_short_non_urgent_utterance(self):
        """Short 1-2 word casual utterances ('Rahmat.', 'Xayr.') should not be needlessly bolded."""
        res = apply_auto_emphasis("Katta rahmat.")
        self.assertEqual(res, "Katta rahmat.")


if __name__ == "__main__":
    unittest.main()
