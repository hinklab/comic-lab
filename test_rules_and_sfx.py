"""
test_rules_and_sfx.py - Unit tests for Translation Rulebook and SFX Pipeline.
Tests Rules 1-8 as enforced code policies.
"""

import os
import unittest
import numpy as np
from PIL import Image

import terminology_store
import bubble_lettering
import engine
import sfx_engine


class TestTranslationRulebookPolicy(unittest.TestCase):

    def setUp(self):
        self.store = terminology_store.get_terminology_store()

    def test_rule1_title_by_meaning(self):
        """Rule 1: Titles must be translated by meaning."""
        entry = self.store._entries.get("the superior spider-man")
        self.assertIsNotNone(entry, "The Superior Spider-Man must exist in store")
        self.assertEqual(entry["entry_type"], "title")
        self.assertEqual(entry["method"], "meaning")
        self.assertEqual(entry["canonical_uzbek"], "Yuksak O'rgimchak-Odam")

        # Test policy rejection of non-meaning title
        bad_entry = {"source_term": "bad title", "canonical_uzbek": "...", "entry_type": "title", "method": "transliteration"}
        valid, err = terminology_store.validate_entry(bad_entry)
        self.assertFalse(valid)
        self.assertIn("Rule 1 violation", err)

    def test_rule2_character_names_by_meaning(self):
        """Rule 2: Character names must be translated by closest meaning."""
        for char_term, expected_uz in [
            ("spider-man", "O'rgimchak-Odam"),
            ("beetle", "Qo'ng'iz"),
            ("living brain", "Tirik Miya"),
            ("doctor octopus", "Doktor Oktopus"),
        ]:
            entry = self.store._entries.get(char_term)
            self.assertIsNotNone(entry, f"{char_term} must be seeded in store")
            self.assertEqual(entry["entry_type"], "character_name")
            self.assertEqual(entry["method"], "meaning")
            self.assertEqual(entry["canonical_uzbek"], expected_uz)

    def test_rule3_real_persons_by_transliteration(self):
        """Rule 3: Real people must be transliterated by pronunciation only (never meaning)."""
        entry = self.store._entries.get("steven strange")
        self.assertIsNotNone(entry, "Steven Strange must be in store")
        self.assertEqual(entry["entry_type"], "real_person")
        self.assertEqual(entry["method"], "transliteration")
        self.assertEqual(entry["canonical_uzbek"], "Stivn-Strenj")

        # Test policy rejection of real_person translated by meaning
        bad_entry = {"source_term": "john smith", "canonical_uzbek": "Temirchi", "entry_type": "real_person", "method": "meaning"}
        valid, err = terminology_store.validate_entry(bad_entry)
        self.assertFalse(valid)
        self.assertIn("Rule 3 violation", err)

    def test_rule4_character_name_override(self):
        """Rule 4: Exceptions breaking Rule 2 require transliteration_override and a reason."""
        entry = self.store._entries.get("overdrive")
        self.assertIsNotNone(entry, "Overdrive must be in store")
        self.assertEqual(entry["entry_type"], "character_name")
        self.assertEqual(entry["method"], "transliteration_override")
        self.assertEqual(entry["canonical_uzbek"], "Overdrayv")
        self.assertTrue(bool(entry.get("reason")), "Override must have explicit self-documenting reason")

        # Test rejection if reason is omitted
        bad_override = {"source_term": "speedster", "canonical_uzbek": "Spidster", "entry_type": "character_name", "method": "transliteration_override"}
        valid, err = terminology_store.validate_entry(bad_override)
        self.assertFalse(valid)
        self.assertIn("Rule 4 violation", err)

    def test_rule5_fixed_terminology(self):
        """Rule 5: Fixed terminology ('Oscillator' -> 'Tebranish generatori') locked without variation."""
        entry = self.store._entries.get("oscillator")
        self.assertIsNotNone(entry, "Oscillator must be in store")
        self.assertEqual(entry["entry_type"], "term")
        self.assertEqual(entry["method"], "fixed")
        self.assertEqual(entry["canonical_uzbek"], "Tebranish generatori")
        self.assertTrue(entry["locked"])


class TestLetteringFormattingEnforcement(unittest.TestCase):

    def test_rule6_quote_stripping(self):
        """Rule 6: Translated dialogue lines must never be wrapped in quotation marks."""
        samples = [
            ('"BU JUDA QIZIQ!"', 'BU JUDA QIZIQ!'),
            ('«O\'RGIMCHAK-ODAM KELDI!»', "O'RGIMCHAK-ODAM KELDI!"),
            ('“HAMMA NARSANI BILAMAN”', 'HAMMA NARSANI BILAMAN'),
            ("'SALOM, DO\'STLAR!'", "SALOM, DO'STLAR!"),
        ]
        for inp, expected in samples:
            cleaned = bubble_lettering.strip_dialogue_quotes(inp)
            self.assertEqual(cleaned, expected)

    def test_rule6_sentence_per_line_and_uzbek_apostrophe_preservation(self):
        """Rule 6: Each sentence separated onto its own line while preserving Uzbek apostrophes (O', G')."""
        multi_sentence = '"BU JUDA QIZIQ. MEN BUNI BILMASDIM! ROSTMI?"'
        formatted = bubble_lettering.format_dialogue_sentences(multi_sentence)
        expected_lines = ["BU JUDA QIZIQ.", "MEN BUNI BILMASDIM!", "ROSTMI?"]
        self.assertEqual(formatted.splitlines(), expected_lines)

        # Ensure Uzbek apostrophe in O' and G' is NOT stripped
        spidey_line = "O'RGIMCHAK-ODAM G'AZABDA. U QO'NG'IZNI KO'RDI!"
        formatted_spidey = bubble_lettering.format_dialogue_sentences(spidey_line)
        self.assertIn("O'RGIMCHAK-ODAM G'AZABDA.", formatted_spidey)
        self.assertIn("U QO'NG'IZNI KO'RDI!", formatted_spidey)

    def test_rule6_initials_and_decimals_not_split(self):
        """Rule 6: Initials (J.) and decimal numbers (3.5) must NOT be split as sentence ends."""
        text = "J. JONA JEYMSON 3.5 SOAT KUTDI. U G'AZABDA!"
        formatted = bubble_lettering.format_dialogue_sentences(text)
        lines = formatted.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], "J. JONA JEYMSON 3.5 SOAT KUTDI.")
        self.assertEqual(lines[1], "U G'AZABDA!")


class TestSFXEngine(unittest.TestCase):

    def test_rule8_all_sfx_words_covered_in_map(self):
        """Rule 8: Every SFX word in engine.SFX_WORDS must have a transliteration mapping."""
        for sw in engine.SFX_WORDS:
            translit = sfx_engine.transliterate_sfx(sw)
            self.assertTrue(len(translit) > 0, f"SFX word {sw} must have non-empty transliteration")
            self.assertNotEqual(translit, "", f"SFX {sw} must not be empty")

    def test_rule8_canonical_sfx_transliterations(self):
        """Rule 8: Specific sound effects match official transliterations."""
        expected = [
            ("BOOM", "BUM"),
            ("BOOM!", "BUM!"),
            ("THWIP", "TVIP"),
            ("THWIP!!", "TVIP!!"),
            ("POW", "POU"),
            ("KAPOW", "KA-POU"),
            ("KA-BOOM", "KA-BUM"),
            ("TEKK", "TEKK"),
            ("CHUK:", "CHUK:"),
            ("CHIK", "CHIK"),
            ("KOFF", "KXA"),
            ("CRASH", "QARS"),
            ("SWOOSH", "VUSH"),
        ]
        for en, expected_uz in expected:
            self.assertEqual(sfx_engine.transliterate_sfx(en), expected_uz)

    def test_rule8_elongated_sfx(self):
        """Rule 8: Elongated repeated sounds (BOOOOM) transliterate with elongation (BUUUUM)."""
        self.assertEqual(sfx_engine.transliterate_sfx("BOOOOM"), "BUUUUM")

    def test_rule8_sfx_inpainting_and_relettering(self):
        """Rule 8: Inpainting and relettering runs and produces modified image without error."""
        canvas = np.full((150, 300, 3), (120, 80, 50), dtype=np.uint8)
        sfx = sfx_engine.SFXElement(
            bbox=[[30, 40], [150, 40], [150, 90], [30, 90]],
            original_text="BOOM",
            transliterated_text="BUM",
            conf=0.9,
            center=(90.0, 65.0),
            angle_deg=0.0,
            width=120.0,
            height=50.0,
            fill_color=(255, 240, 150),
            stroke_color=(200, 30, 20),
            stroke_width=2
        )
        cleaned = sfx_engine.inpaint_sfx_elements(canvas, [sfx])
        self.assertEqual(cleaned.shape, canvas.shape)

        lettered = sfx_engine.render_sfx_elements(cleaned, [sfx])
        self.assertEqual(lettered.size, (300, 150))


if __name__ == "__main__":
    unittest.main()