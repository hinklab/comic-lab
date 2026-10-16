import re
import naturalization_rules


def test_hayvonning_calque_elimination():
    """Verify that 'HAYVONNING' or '...hayvonning.' is replaced with natural Uzbek dialogue continuation."""
    cases = [
        ("HAYVONNING", "...MAXLUQ TABIATI HAQIDA."),
        ("...hayvonning.", "...MAXLUQ TABIATI HAQIDA."),
        ("hayvonning", "...MAXLUQ TABIATI HAQIDA."),
        ("HAYVONNING.", "...MAXLUQ TABIATI HAQIDA."),
        ("--hayvonning", "...MAXLUQ TABIATI HAQIDA."),
    ]
    for raw, expected in cases:
        res = naturalization_rules.apply_naturalization(raw)
        assert res == expected, f"Expected '{expected}' for '{raw}', got '{res}'"


def test_dangling_genitive_general_resolution():
    """Verify that isolated single-word genitives (-ning) in dialogue bubbles are cleanly resolved."""
    cases = [
        ("shaharning", "...SHAHAR HAQIDA."),
        ("...odamning.", "...ODAM HAQIDA."),
        ("dushmanning", "...DUSHMAN HAQIDA."),
        ("maxluqning", "...MAXLUQ TABIATI HAQIDA."),
    ]
    for raw, expected in cases:
        res = naturalization_rules.apply_naturalization(raw)
        assert res == expected, f"Expected '{expected}' for '{raw}', got '{res}'"


def test_beast_idiom_recognition():
    """Verify that English 'of the beast' / 'nature of the beast' is correctly translated into authentic Uzbek."""
    cases = [
        ("of the beast", "...MAXLUQ TABIATI HAQIDA."),
        ("...of the beast.", "...MAXLUQ TABIATI HAQIDA."),
        ("the nature of the beast", "...MAXLUQ TABIATI HAQIDA."),
        ("...nature of the beast.", "...MAXLUQ TABIATI HAQIDA."),
        ("nature of the beast", "...MAXLUQ TABIATI HAQIDA."),
        ("beast", "...MAXLUQ."),
    ]
    for en, expected in cases:
        res = naturalization_rules.apply_naturalization("", en_orig=en)
        assert res == expected, f"Expected '{expected}' for '{en}', got '{res}'"


def test_podcast_dialogue_naturalization():
    """Verify that 'Sorry, I was listening to a podcast...' is naturalized without awkward possessive suffixes."""
    # From English
    res1 = naturalization_rules.apply_naturalization("", en_orig="Sorry, I was listening to a podcast...")
    assert res1 == "KECHIRASIZ, PODKAST TINGLAYOTGAN EDIM..."

    # From awkward NLLB Uzbek calque
    res2 = naturalization_rules.apply_naturalization("Kechirasiz, podkastini tinglayotgan edim.")
    assert res2 == "KECHIRASIZ, PODKAST TINGLAYOTGAN EDIM..."


def test_multi_bubble_continuation_detection():
    """Verify that sentence continuation logic detects chained dialogues."""
    def is_continuation(prev_t, curr_t):
        p = prev_t.strip()
        c = curr_t.strip()
        if not p or not c:
            return False
        if p.endswith("...") or p.endswith("…") or p.endswith("--") or c.startswith("...") or c.startswith("…") or c.startswith("--"):
            return True
        if not p.endswith((".", "!", "?")) and (c[0].islower() or c.split()[0].lower() in ("of", "the", "and", "about", "for", "to", "in", "on", "with")):
            return True
        return False

    assert is_continuation("Sorry, I was listening to a podcast...", "...of the beast.") is True
    assert is_continuation("Sorry, I was listening to a podcast", "of the beast") is True
    assert is_continuation("I have to tell you something--", "it's about Peter.") is True
    assert is_continuation("Stop right there!", "Who are you?") is False


if __name__ == "__main__":
    test_hayvonning_calque_elimination()
    print("test_hayvonning_calque_elimination PASSED")
    test_dangling_genitive_general_resolution()
    print("test_dangling_genitive_general_resolution PASSED")
    test_beast_idiom_recognition()
    print("test_beast_idiom_recognition PASSED")
    test_podcast_dialogue_naturalization()
    print("test_podcast_dialogue_naturalization PASSED")
    test_multi_bubble_continuation_detection()
    print("test_multi_bubble_continuation_detection PASSED")
    print("\nALL STEP 3 TESTS PASSED PERFECTLY!")
