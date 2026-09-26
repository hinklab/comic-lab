"""
test_translation_quality_enhancements.py - Unit tests for the 4 translation quality and safety enhancements.

Tests:
1. Multi-lobe / multi-line unified cluster translation and syntax-aware partitioning with SIZ formality.
2. Real 'MAAKES NICE' input and truncated translation quality gates (with agglutinative exception safety).
3. J. Jonah Jameson & Peter Parker Ghost voice profiles.
4. Export safety guard (preventing raw OCR from corrupting canvas).
"""

import unittest
import numpy as np
from PIL import Image

import engine
import quality_gates
import character_profiles
import naturalization_rules


class TestTranslationQualityEnhancements(unittest.TestCase):

    def setUp(self):
        self.mem_store = character_profiles.get_memory_store()

    # -------------------------------------------------------------------------
    # 1. Multi-Lobe Unified Cluster Translation & Formality
    # -------------------------------------------------------------------------
    def test_multi_lobe_unified_translation_and_partitioning(self):
        """
        Verifies that 'YOU WERE WRONG ABOUT ME, AND MAYBE I'VE BEEN WRONG ABOUT YOU!'
        is translated as ONE unified sentence first, preserving subject/pronoun order,
        and partitioned cleanly across lobes at the clause boundary.
        """
        full_text = "YOU WERE WRONG ABOUT ME, AND MAYBE I'VE BEEN WRONG ABOUT YOU!"
        uz_full = engine.translate_spiderman_uzbek(full_text, speaker="Peter Parker (Ghost)")

        # Pronoun & subject agreement check
        self.assertIn("SIZ", uz_full, "Must use respectful SIZ address for Jameson")
        self.assertIn("MEN", uz_full, "Must retain 1st-person MEN")
        self.assertNotIn("SEN", uz_full, "Must not mix SEN in this formal address")
        self.assertEqual(uz_full, "--SIZ MEN HAQIMDA ADASHGANSIZ, EHTIMOL MEN HAM SIZ HAQINGIZDA ADASHGANDIRMAN!")

        # Syntax-aware lobe partitioning check
        lobe_weights = [5, 7]  # e.g. 5 words in Lobe 1, 7 words in Lobe 2
        parts = engine.partition_translated_text_for_lobes(uz_full, lobe_weights)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], "--SIZ MEN HAQIMDA ADASHGANSIZ,")
        self.assertEqual(parts[1], "EHTIMOL MEN HAM SIZ HAQINGIZDA ADASHGANDIRMAN!")

    def test_multi_lobe_speech_bubble_cluster_translation(self):
        """
        Verifies translate_bubbles_list translates a multi-lobe cluster via single call
        and assigns partitioned clauses to the respective sub-bubbles.
        """
        b1 = engine.SpeechBubble(
            bubble_id=1, x0=10, y0=10, x1=100, y1=50,
            original_text="YOU WERE WRONG ABOUT ME,",
            cluster_id="cluster_ghost_1",
            cluster_text="YOU WERE WRONG ABOUT ME, AND MAYBE I'VE BEEN WRONG ABOUT YOU!",
            lobe_index=0, total_lobes=2,
            speaker="Peter Parker (Ghost)"
        )
        b2 = engine.SpeechBubble(
            bubble_id=2, x0=10, y0=60, x1=100, y1=100,
            original_text="AND MAYBE I'VE BEEN WRONG ABOUT YOU!",
            cluster_id="cluster_ghost_1",
            cluster_text="YOU WERE WRONG ABOUT ME, AND MAYBE I'VE BEEN WRONG ABOUT YOU!",
            lobe_index=1, total_lobes=2,
            speaker="Peter Parker (Ghost)"
        )

        res = engine.translate_bubbles_list([b1, b2])
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].uzbek_translation, "--SIZ MEN HAQIMDA ADASHGANSIZ,")
        self.assertEqual(res[1].uzbek_translation, "EHTIMOL MEN HAM SIZ HAQINGIZDA ADASHGANDIRMAN!")
        self.assertFalse(res[0].needs_review)
        self.assertFalse(res[1].needs_review)

    # -------------------------------------------------------------------------
    # 2. Quality Gates: 'MAAKES NICE' & Truncated Translation Detection
    # -------------------------------------------------------------------------
    def test_real_maakes_nice_input_flagged(self):
        """
        Verifies that the exact real broken output containing 'MAAKES NICE' is flagged.
        """
        orig_text = "I CAN'T BELIEVE IT! AFTER ALL THESE YEARS; JONAH FINALLY MAKES NICE"
        broken_uz = "BU YILLAR O'TIB, JONA OXIRI MAAKES NICE"

        val = quality_gates.post_translation_validation(broken_uz, orig_text)
        self.assertTrue(val["needs_review"], "Must flag broken translation with English residue")
        self.assertIn("NICE", val["flagged_tokens"])

    def test_dropped_sentence_truncation_flagged(self):
        """
        Verifies that dropping an entire sentence from a multi-sentence source is flagged,
        even without English words in the output.
        """
        orig_text = "I CAN'T BELIEVE IT! AFTER ALL THESE YEARS, JONAH FINALLY MAKES NICE."
        truncated_uz = "Jona oxiri muloyimlashdi."  # 'I can't believe it' sentence dropped completely

        val = quality_gates.post_translation_validation(truncated_uz, orig_text)
        self.assertTrue(val["needs_review"], "Must flag dropped sentence truncation")
        self.assertIn("Qisman kesilgan tarjima", val["reason"])

    def test_agglutinative_single_word_not_flagged(self):
        """
        Verifies that legitimate 1-word Uzbek agglutinative translations are NOT flagged.
        """
        orig_text = "I CAN'T BELIEVE IT!"  # 4 words
        valid_uz = "Ishonmayman!"           # 1 word (25% length)

        val = quality_gates.post_translation_validation(valid_uz, orig_text)
        self.assertFalse(val["needs_review"], "Agglutinative 1-word Uzbek must NOT be false-flagged")

    def test_raw_ocr_exact_match_flagged(self):
        """
        Verifies that when translation returns raw OCR untouched, it is flagged.
        """
        raw_ocr = "MAAKES NICE"
        val = quality_gates.post_translation_validation(raw_ocr, raw_ocr)
        self.assertTrue(val["needs_review"])
        self.assertIn("xom OCR", val["reason"])

    def test_proper_nouns_from_glossary_exempted(self):
        """
        Verifies that comic proper nouns like Spidey, Spider-Man, Otto, Jameson are permitted.
        """
        orig_text = "YOU'VE DONE A HECKUVA JOB HERE, SPIDEY!"
        valid_uz = "BU YERDA QOYILMAQOM ISH QILDING, O'RGIMCHAK!"

        val = quality_gates.post_translation_validation(valid_uz, orig_text)
        self.assertFalse(val["needs_review"])

    # -------------------------------------------------------------------------
    # 3. J. Jonah Jameson & Peter Parker Ghost Voice Profiles
    # -------------------------------------------------------------------------
    def test_jonah_jameson_profile_loaded(self):
        """Verifies J. Jonah Jameson profile is registered in character memory."""
        profile = self.mem_store.get_profile("J. Jonah Jameson")
        self.assertIsNotNone(profile)
        self.assertTrue(profile.matches("Mayor Jonah Jameson"))
        self.assertTrue(profile.matches("Jameson"))
        self.assertTrue(profile.matches("Mer J. Jona Jeymson"))

    def test_jonah_jameson_voice_translation(self):
        """Verifies Jameson's distinct comic voice styling."""
        line = "YOU'VE DONE A HECKUVA JOB HERE, SPIDEY!"
        uz = engine.translate_spiderman_uzbek(line, speaker="J. Jonah Jameson")
        self.assertEqual(uz, "BU YERDA QOYILMAQOM ISH QILDING, O'RGIMCHAK!")

    def test_peter_parker_ghost_profile_loaded(self):
        """Verifies Peter Parker (Ghost) profile is registered and matches."""
        profile = self.mem_store.get_profile("Peter Parker (Ghost)")
        self.assertIsNotNone(profile)
        self.assertTrue(profile.matches("Ghost Peter"))
        self.assertTrue(profile.matches("Peter Ghost"))

    # -------------------------------------------------------------------------
    # 4. Export Safety Guard
    # -------------------------------------------------------------------------
    def test_export_safety_guard_skips_raw_ocr(self):
        """
        Verifies that typeset_lettering_on_page does not write raw OCR onto the canvas
        when needs_review is True and text is untranslated.
        """
        canvas = Image.new("RGB", (300, 200), (255, 255, 255))
        bubble = engine.SpeechBubble(
            bubble_id=1, x0=50, y0=50, x1=250, y1=150,
            original_text="CORRUPTED RAW OCR",
            uzbek_translation="CORRUPTED RAW OCR",
            needs_review=True,
            review_reason="Tarjima qilinmagan xom OCR"
        )
        font_path = next(iter(engine.get_available_fonts().values()))

        out = engine.typeset_lettering_on_page(canvas, [bubble], font_path=font_path)

        # Output canvas should remain pristine white (no black ink drawn!)
        out_np = np.array(out)
        self.assertTrue(np.all(out_np == 255), "Raw OCR must NOT be drawn onto canvas when needs_review is True")


if __name__ == "__main__":
    unittest.main()
