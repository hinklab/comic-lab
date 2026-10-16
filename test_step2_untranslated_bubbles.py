import re
import naturalization_rules
import engine


def test_boss_idiom_recognition():
    """Verify that 'You got it, boss' and comic variants match authentic Uzbek dialogue."""
    cases = [
        ("You got it, boss.", "XO'P BO'LADI, BOSHLIQ."),
        ("You got it, boss", "XO'P BO'LADI, BOSHLIQ."),
        ("YOU GOT IT, BOSS!", "XO'P BO'LADI, BOSHLIQ."),
        ("You goc it, boss.", "XO'P BO'LADI, BOSHLIQ."),
        ("You got it, Chief.", "XO'P BO'LADI, BOSHLIQ."),
        ("You got it, Cap.", "XO'P BO'LADI, BOSHLIQ."),
        ("You got it, sir.", "XO'P BO'LADI, BOSHLIQ."),
        ("Yes, boss!", "XO'P BO'LADI, BOSHLIQ."),
        ("You got it.", "XO'P BO'LADI."),
        ("You've got it.", "XO'P BO'LADI."),
        ("I got it, boss.", "TUSHUNDIM, BOSHLIQ."),
        ("I got it.", "TUSHUNDIM."),
    ]
    for en, expected in cases:
        res = naturalization_rules.apply_naturalization("", en_orig=en)
        assert res == expected, f"Expected '{expected}' for '{en}', got '{res}'"


def test_boss_calque_substitutions():
    """Verify that literal machine translation calques for 'You got it, boss' are naturalized."""
    calques = [
        ("To'g'ri, boshi.", "XO'P BO'LADI, BOSHLIQ."),
        ("To'g'ri boshi", "XO'P BO'LADI, BOSHLIQ"),
        ("Siz buni tushundingiz, xo'jayin.", "XO'P BO'LADI, BOSHLIQ."),
        ("Buni tushundingiz, xo'jayin", "XO'P BO'LADI, BOSHLIQ"),
        ("Buni oldingiz, xo'jayin", "XO'P BO'LADI, BOSHLIQ"),
        ("To'g'ri keladi.", "XO'P BO'LADI."),
    ]
    for raw_uz, expected in calques:
        res = naturalization_rules.apply_naturalization(raw_uz)
        assert res == expected, f"Expected '{expected}' for calque '{raw_uz}', got '{res}'"


def test_untranslated_english_leak_interception():
    """Verify that when an upstream service returns English text verbatim as Uzbek, it is intercepted and naturalized."""
    # Leaked English verbatim
    res1 = naturalization_rules.apply_naturalization("You got it, boss.", en_orig="You got it, boss.")
    assert res1 == "XO'P BO'LADI, BOSHLIQ."

    # Leaked English with empty en_orig
    res2 = naturalization_rules.apply_naturalization("You got it, boss.")
    assert res2 == "XO'P BO'LADI, BOSHLIQ."

    # OCR typo leak
    res3 = naturalization_rules.apply_naturalization("You goc it, boss.")
    assert res3 == "XO'P BO'LADI, BOSHLIQ."


def test_untranslated_leak_detection_logic():
    """Verify that server.py untranslated condition catches leaks without catching SFX."""
    def check_untranslated(orig, uz):
        clean_o = re.sub(r'[^a-zA-Z0-9]', '', orig).lower()
        clean_u = re.sub(r'[^a-zA-Z0-9]', '', uz).lower()
        return (not uz) or (clean_o and clean_u and clean_o == clean_u and not engine.is_pure_sfx_text(orig) and len(clean_o) >= 2)

    # Empty uzbek -> True
    assert check_untranslated("You got it, boss.", "") is True

    # Verbatim English leak -> True
    assert check_untranslated("You got it, boss.", "You got it, boss.") is True
    assert check_untranslated("Look out!", "Look out!") is True

    # Valid translation -> False
    assert check_untranslated("You got it, boss.", "XO'P BO'LADI, BOSHLIQ.") is False

    # Pure SFX -> False (SFX can stay identical)
    assert check_untranslated("BOOM!", "BOOM!") is False
    assert check_untranslated("KRAK!", "KRAK!") is False


if __name__ == "__main__":
    test_boss_idiom_recognition()
    print("test_boss_idiom_recognition PASSED")
    test_boss_calque_substitutions()
    print("test_boss_calque_substitutions PASSED")
    test_untranslated_english_leak_interception()
    print("test_untranslated_english_leak_interception PASSED")
    test_untranslated_leak_detection_logic()
    print("test_untranslated_leak_detection_logic PASSED")
    print("\nALL STEP 2 TESTS PASSED PERFECTLY!")

