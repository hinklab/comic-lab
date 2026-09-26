"""
test_offline_translation.py - Unit test suite for 100% offline translation & naturalization.
Confirms zero network dependency by blocking socket calls.
"""

import unittest
import socket
import time
import local_translator
import naturalization_rules
import engine


class TestOfflineTranslation(unittest.TestCase):
    def setUp(self):
        # Programmatically disable socket connection to enforce 100% offline execution
        self._orig_socket = socket.socket

        def _offline_guard(*args, **kwargs):
            raise ConnectionError("Network call blocked! Translation must be 100% offline.")

        socket.socket = _offline_guard

    def tearDown(self):
        # Restore original socket
        socket.socket = self._orig_socket

    def test_local_translator_offline_zero_network(self):
        """Verifies local translator runs with zero network connectivity."""
        text = "Hello, world!"
        res = local_translator.translate_offline(text)
        self.assertIsInstance(res, str)
        self.assertTrue(len(res) > 0)
        # Verify it did not crash with ConnectionError
        self.assertNotIn("blocked", res.lower())

    def test_naturalization_idioms_and_calques(self):
        """Verifies naturalization rules replace robotic calques with natural comic Uzbek."""
        # 1. Interjection: Here he comes!
        raw_here = "Mana u keladi!"
        pol_here = naturalization_rules.apply_naturalization(raw_here, en_orig="Here he comes!")
        self.assertEqual(pol_here, "ANA, KELYAPTI!")

        # 2. Honorific & Calque: It was no trouble at all, ma'am.
        raw_trouble = "Bu umuman muammo emas edi, madam."
        pol_trouble = naturalization_rules.apply_naturalization(raw_trouble, en_orig="It was no trouble at all, ma'am.")
        self.assertEqual(pol_trouble, "ARZIMAYDI, XONIM.")

        # 3. Comic Calque: Lead story
        raw_lead = "Spidey asosiy hikoya:"
        pol_lead = naturalization_rules.apply_naturalization(raw_lead, en_orig="Two days in a row and Spidey's the lead story:")
        self.assertEqual(pol_lead, "IKKI KUNDAN BERI O'RGIMCHAK BOSH MAVZU:")

        # 4. Superhero proper name: Spider-Man (Rule 2: closest meaning -> O'rgimchak-Odam)
        raw_spidey = "Qo'riqchi Minorasi Jamiyati, men Rut Goldmanmanman, talabalar dekani."
        pol_spidey = naturalization_rules.apply_naturalization(raw_spidey, en_orig="Spider-Man, I'm Ruth Goldman, Dean of students.")
        self.assertTrue("O'RGIMCHAK-ODAM" in pol_spidey or "SPIDER-MAN" in pol_spidey)
        self.assertIn("TALABALAR DEKANI", pol_spidey)

        # 5. Circuit breaker
        raw_breaker = "Panikaga tushmang, men to'qnashuvni tuzataman!"
        pol_breaker = naturalization_rules.apply_naturalization(raw_breaker, en_orig="Don't panic! I can fix the circuit breaker!")
        self.assertIn("VAHIMAGA TUSHMANG", pol_breaker)
        self.assertIn("SAQLAGICHNI", pol_breaker)

    def test_engine_spiderman_uzbek_integration(self):
        """Verifies engine.translate_spiderman_uzbek runs end-to-end fully offline."""
        test_phrases = [
            ("Spider-Man, I'm Ruth Goldman, Dean of students.", "O'RGIMCHAK-ODAM"),
            ("It was no trouble at all, ma'am.", "ARZIMAYDI, XONIM"),
            ("Two days in a row and Spidey's the lead story:", "BOSH MAVZU"),
            ("What the--?!", "BU NIMA YANA--?!"),
            ("Look out! The reactor is about to explode!", "EHTIYOT BO'LING"),
        ]
        # Warmup to separate one-time model initialization from inference latency
        engine.translate_spiderman_uzbek("Warmup")
        for en, expected_keyword in test_phrases:
            t0 = time.time()
            uz = engine.translate_spiderman_uzbek(en)
            dur = (time.time() - t0) * 1000
            self.assertTrue(
                expected_keyword in uz or ("O'RGIMCHAK-ODAM" == expected_keyword and "SPIDER-MAN" in uz),
                f"Expected '{expected_keyword}' in translated '{uz}'"
            )
            # Confirm inference latency is practical for CPU
            self.assertLess(dur, 8000, f"Translation took too long: {dur:.1f}ms")


if __name__ == "__main__":
    unittest.main()
