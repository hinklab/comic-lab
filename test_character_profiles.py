"""
test_character_profiles.py - Unit tests for Character Voice Profiles & Persona Naturalization.
"""

import unittest
import character_profiles
import naturalization_rules


class TestCharacterProfiles(unittest.TestCase):

    def test_registry_lookup_canonical(self):
        profile = character_profiles.get_character_profile("Superior Spider-Man")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "Superior Spider-Man")

    def test_registry_lookup_aliases(self):
        for alias in ["superior spiderman", "otto octavius", "otto", "doc ock", "spiderman", "spider-man"]:
            profile = character_profiles.get_character_profile(alias)
            self.assertIsNotNone(profile, f"Failed lookup for alias: {alias}")
            self.assertEqual(profile.name, "Superior Spider-Man")

    def test_registry_lookup_unknown(self):
        self.assertIsNone(character_profiles.get_character_profile("Random civilian"))
        self.assertIsNone(character_profiles.get_character_profile(""))
        self.assertIsNone(character_profiles.get_character_profile(None))

    def test_exact_lines_otto(self):
        profile = character_profiles.get_character_profile("Superior Spider-Man")
        self.assertIsNotNone(profile)

        cases = [
            ("PLAN: EPSILON FIVE!", "REJA: EPSILON-BESH!"),
            ("IMPRESSED? DON'T BOTHER.", "QOYIL QOLDINGIZMI? OVORA BO'LMANG."),
            ("INSTANT FRICTIONLESS SURFACE!", "ISHQALANISHSIZ TEZKOR SIRT!"),
            ("BRUISED TRACHEA.", "LAT YEGAN NAFAS YO'LI."),
            ("CRUSHED LARYNX.", "EZILGAN HIQILDOQ."),
            ("SECOND DAY ON THE 'JOB'...", "ISHDAGI IKKINCHI KUN..."),
            (
                "BUT SOON THEY'LL ALL LEARN THAT I AM A FAR SUPERIOR ONE.",
                "BIROQ TEZ ORADA BARCHALARI MENING ANCHA USTUN SPIDER-MAN EKANIMNI ANGLAB YETISHADI!"
            ),
        ]

        for en_line, expected_uz in cases:
            res = profile.naturalize("Generic text", en_orig=en_line)
            self.assertEqual(res, expected_uz, f"Mismatch for '{en_line}'")

    def test_phrase_replacements(self):
        profile = character_profiles.get_character_profile("Superior Spider-Man")
        self.assertIsNotNone(profile)

        # "kuchli aql" -> "ustun tafakkur"
        text = "Siz kuchli aql bilan to'qnashdingiz"
        res = profile.naturalize(text, en_orig="")
        self.assertIn("ustun tafakkur", res)

        # "bilib oladi" -> "anglab yetishadi"
        text = "Ular mening kimligimni bilib oladi"
        res = profile.naturalize(text, en_orig="")
        self.assertIn("anglab yetishadi", res)

        # "tashvish chekmang" -> "ovora bo'lmang"
        text = "Tashvish chekmang, men bor"
        res = profile.naturalize(text, en_orig="")
        self.assertIn("ovora bo'lmang", res)

    def test_naturalization_rules_integration(self):
        # Without character profile (generic)
        generic_res = naturalization_rules.apply_naturalization(
            "Ular tez orada buni bilib oladi.",
            en_orig="They will soon learn this."
        )
        self.assertIn("BILIB OLADI", generic_res)

        # With Superior Spider-Man profile
        otto_res = naturalization_rules.apply_naturalization(
            "Ular tez orada buni bilib oladi.",
            en_orig="They will soon learn this.",
            character_profile="Superior Spider-Man"
        )
        self.assertIn("ANGLAB YETISHADI", otto_res)

    def test_list_available_profiles(self):
        profiles = character_profiles.list_available_profiles()
        self.assertIn("Superior Spider-Man", profiles)


if __name__ == "__main__":
    unittest.main()
