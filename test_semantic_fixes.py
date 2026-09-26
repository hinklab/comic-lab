"""
test_semantic_fixes.py - Verification for all 6 audit lines.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
import names_glossary
import naturalization_rules


class TestSemanticFixes(unittest.TestCase):

    def test_line1_none_knows_different_spiderman(self):
        # Raw OCR text with missing apostrophe and ending colon
        ocr_text = "AND NOT ONE OF THEM KNOWS Im A DiFFERENT SPiDER-MaN:"
        res = engine.translate_spiderman_uzbek(ocr_text, speaker="Superior Spider-Man")
        print("\n[Line 1] Output:", res)
        self.assertTrue("O'RGIMCHAK" in res or "SPIDER-MAN" in res)
        self.assertTrue("BILMAYDI" in res or "ANGLAB" in res)
        # Verify self-referential clause is present
        self.assertNotIn("O'RINNI", res)

    def test_line2_dean_ruth_goldman(self):
        # Raw OCR text with semicolon and trailing underscore
        ocr_text = "Spider-MAN; I'm Ruth GOLDMAN, DEAN OF StuDentS_"
        res = engine.translate_spiderman_uzbek(ocr_text)
        print("\n[Line 2] Output:", res)
        self.assertTrue("O'RGIMCHAK-ODAM" in res or "SPIDER-MAN" in res)
        self.assertIn("RUT GOLDMAN", res)
        self.assertIn("TALABALAR DEKANI", res)
        self.assertNotIn("O'G'LIM", res)
        self.assertNotIn("GOLDAMON", res)

    def test_line3_on_behalf_of_esu(self):
        # Ground truth assembled in correct order by sort_reading_order
        assembled_text = "--AND ON BEHALF OF EVERYONE HERE AT EMPIRE STATE UNIVERSITY..."
        res = engine.translate_spiderman_uzbek(assembled_text)
        print("\n[Line 3] Output:", res)
        self.assertIn("EMPIRE STATE UNIVERSITETI", res)
        self.assertIn("NOMIDAN", res)
        self.assertNotIn("YARMISIGA", res)
        self.assertNotIn("HAQIDA", res)

    def test_line4_no_trouble_at_all_maam(self):
        # Raw OCR text with colon at end
        ocr_text = "it WAS No Trouble AT ALl, MA'AM:"
        res = engine.translate_spiderman_uzbek(ocr_text)
        print("\n[Line 4] Output:", res)
        self.assertIn("ARZIMAYDI", res)
        self.assertIn("XONIM", res)
        self.assertNotIn("HAMMAGA", res)

    def test_line5_grammatical_person_agreement(self):
        # Raw OCR text for Bubble 14
        ocr_text = "For bringing This BACK To THE MARLA JAMESON MEMORIAL Wing?"
        res = engine.translate_spiderman_uzbek(ocr_text, speaker="Superior Spider-Man")
        print("\n[Line 5] Output:", res)
        self.assertIn("MARLA JEYMSON", res)
        self.assertIn("XOTIRA BINOSIGA", res)
        # Must be 1st-person (-ganim), NOT 2nd-person (-ganingiz)
        self.assertIn("QAYTARGANIM", res)
        self.assertNotIn("QAYTARGANINGIZ", res)

    def test_line6_names_glossary_consistency(self):
        cases = [
            ("Marla Jameson", "MARLA JEYMSON"),
            ("Ruth Goldman", "RUT GOLDMAN"),
            ("Otto Octavius", "OTTO OKTAVIUS"),
            ("J. Jonah Jameson", "J. JONA JEYMSON"),
            ("Empire State University", "EMPIRE STATE UNIVERSITETI"),
        ]
        for en, expected_uz in cases:
            masked, unmask = names_glossary.mask_proper_nouns(en)
            unmasked = names_glossary.unmask_proper_nouns(masked, unmask)
            self.assertEqual(unmasked.upper(), expected_uz)


if __name__ == "__main__":
    unittest.main()
