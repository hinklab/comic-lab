import unittest
import sfx_engine
import engine


class TestStep22SFXClassification(unittest.TestCase):
    def test_sfx_semantic_categorization(self):
        """Verify each category of comic sound effects is classified into its authentic taxonomy."""
        cases = [
            ("BOOM!", "impact_explosion", "Zarba / Portlash", "BUM!"),
            ("KABOOM", "impact_explosion", "Zarba / Portlash", "KA-BUM"),
            ("THWIP!", "web_speed", "Tezlik / To'r ovozi", "TVIP!"),
            ("SWOOSH", "web_speed", "Tezlik / To'r ovozi", "VUSH"),
            ("CLANG!", "mechanical_tech", "Mexanik / Texnologiya", "JANG!"),
            ("ZAP!", "mechanical_tech", "Mexanik / Texnologiya", "ZAP!"),
            ("TEKK", "mechanical_tech", "Mexanik / Texnologiya", "TEKK"),
            ("KRAK!", "destruction_sharp", "Sinish / Qirsillash", "QARS!"),
            ("SNIKT", "destruction_sharp", "Sinish / Qirsillash", "SNIKT"),
            ("ROAR!", "creature_vocal", "Maxluq / Hayvon ovozi", "NA'RA!"),
            ("GRRR", "creature_vocal", "Maxluq / Hayvon ovozi", "GRRR"),
            ("UGH...", "human_vocal_bodily", "Insoniy tovush / Qichqiriq", "UX..."),
            ("KOFF", "human_vocal_bodily", "Insoniy tovush / Qichqiriq", "KXA"),
        ]
        for text, exp_cat, exp_lbl, exp_translit in cases:
            is_sfx, cat, lbl, translit, conf = sfx_engine.classify_sfx(text)
            self.assertTrue(is_sfx, f"Failed to detect SFX for '{text}'")
            self.assertEqual(cat, exp_cat, f"Mismatch category for '{text}': got {cat}, expected {exp_cat}")
            self.assertEqual(lbl, exp_lbl, f"Mismatch label for '{text}': got {lbl}, expected {exp_lbl}")
            self.assertEqual(translit, exp_translit, f"Mismatch transliteration for '{text}': got {translit}, expected {exp_translit}")
            self.assertGreaterEqual(conf, 0.90)

    def test_elongated_vowel_sfx(self):
        """Verify stretched vowels in comic sound effects are properly classified and transliterated."""
        elongations = [
            ("BOOOOM!", "impact_explosion", "BUUUUM!"),
            ("THWIIIP!", "web_speed", "TVIIIP!"),
            ("KRAAAK!", "destruction_sharp", "QAAARS!"),
        ]
        for text, exp_cat, exp_translit in elongations:
            is_sfx, cat, lbl, translit, conf = sfx_engine.classify_sfx(text)
            self.assertTrue(is_sfx)
            self.assertEqual(cat, exp_cat)
            self.assertEqual(translit, exp_translit)

    def test_dialogue_exemption_from_sfx(self):
        """Verify normal spoken dialogue and exclamations are NEVER classified as sound effects."""
        dialogues = [
            "Hello, Peter.",
            "STOP!",
            "PLEASE!",
            "WAIT!",
            "LOOK!",
            "HELP!",
            "I am ready.",
            "What happened?",
        ]
        for text in dialogues:
            is_sfx, cat, lbl, translit, conf = sfx_engine.classify_sfx(text)
            self.assertFalse(is_sfx, f"Spoken dialogue '{text}' was falsely classified as SFX!")
            self.assertIsNone(cat)
            self.assertIsNone(translit)

    def test_speech_bubble_auto_classification(self):
        """Verify SpeechBubble instances automatically detect SFX and set shape_type to 'sfx'."""
        bubble = engine.SpeechBubble(
            bubble_id=1,
            x0=50, y0=50, x1=120, y1=80,
            original_text="THWIP!",
            confidence=0.98
        )
        self.assertTrue(bubble.is_sfx)
        self.assertEqual(bubble.shape_type, "sfx")
        self.assertEqual(bubble.sfx_category, "web_speed")
        self.assertEqual(bubble.sfx_category_label, "Tezlik / To'r ovozi")
        self.assertEqual(bubble.sfx_transliteration, "TVIP!")
        self.assertEqual(bubble.uzbek_translation, "TVIP!")

    def test_sfx_element_extraction_with_category(self):
        """Verify extract_sfx_elements returns elements with category and category_label populated."""
        raw_ocr = [
            ([[10, 10], [50, 10], [50, 30], [10, 30]], "BOOM", 0.95),
            ([[60, 60], [100, 60], [100, 80], [60, 80]], "Good day", 0.99)
        ]
        sfx_items = sfx_engine.extract_sfx_elements(raw_ocr, (100, 100))
        self.assertEqual(len(sfx_items), 1)
        item = sfx_items[0]
        self.assertEqual(item.original_text, "BOOM")
        self.assertEqual(item.transliterated_text, "BUM")
        self.assertEqual(item.category, "impact_explosion")
        self.assertEqual(item.category_label, "Zarba / Portlash")


if __name__ == "__main__":
    unittest.main()
