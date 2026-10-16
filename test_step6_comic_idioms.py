import naturalization_rules


def test_put_him_on_ice():
    """Verify 'Put him on ice' is correctly recognized as an idiom and calques are fixed."""
    en_cases = [
        ("Put him on ice.", "UNI MUZLATIB QO'YING."),
        ("PUT HIM ON ICE!", "UNI MUZLATIB QO'YING."),
        ("Put her on ice.", "UNI MUZLATIB QO'YING."),
        ("Let's put him on ice.", "UNI MUZLATIB QO'YING."),
        ("Put them on ice", "UNI MUZLATIB QO'YING."),
    ]
    for en, expected in en_cases:
        res = naturalization_rules.apply_naturalization("", en_orig=en)
        assert res == expected, f"Failed for en '{en}': got '{res}', expected '{expected}'"

    calque_cases = [
        ("Unga muzlab turish kerak.", "UNI MUZLATIB QO'YING."),
        ("unga muzlab turish kerak", "UNI MUZLATIB QO'YING"),
        ("Muzlab turish kerak.", "MUZLATIB QO'YISH KERAK."),
    ]
    for calque, expected in calque_cases:
        res = naturalization_rules.apply_naturalization(calque)
        assert res == expected, f"Failed for calque '{calque}': got '{res}', expected '{expected}'"


def test_hook_him_up():
    """Verify 'Hook him up' and OCR quotation artifacts are naturalized to authentic comic commands."""
    en_cases = [
        ("Hook him up.", "UNI KISHANLANG."),
        ("HOOK HIM UP!", "UNI KISHANLANG."),
        ("Lock him up.", "UNI QAMAB QO'YING."),
        ("Take him in.", "UNI QO'LGA OLING."),
    ]
    for en, expected in en_cases:
        res = naturalization_rules.apply_naturalization("", en_orig=en)
        assert res == expected, f"Failed for en '{en}': got '{res}', expected '{expected}'"

    calque_cases = [
        ("'NI QO'LGA QO'YING.", "UNI KISHANLANG."),
        ("DNI QO'LGA QO'YING _", "UNI KISHANLANG."),
        ("Uni qo'lga qo'ying.", "UNI KISHANLANG."),
        ("Qo'lga qo'ying.", "QO'LGA OLING."),
    ]
    for calque, expected in calque_cases:
        res = naturalization_rules.apply_naturalization(calque)
        assert res == expected, f"Failed for calque '{calque}': got '{res}', expected '{expected}'"


def test_one_and_the_same():
    """Verify 'One and the same' idiom and 'bir jondir' literalism are naturalized."""
    en_cases = [
        ("Your fears and our safety are one and the same.", "SIZNING QO'RQUVLARINGIZ VA XAVFSIZLIGIMIZ BITTA NARSADIR."),
        ("Your fears and securities are one and the same.", "SIZNING QO'RQUVLARINGIZ VA XAVFSIZLIGIMIZ BITTA NARSADIR."),
        ("Your fears and our security are one and the same.", "SIZNING QO'RQUVLARINGIZ VA XAVFSIZLIGIMIZ BITTA NARSADIR."),
        ("One and the same.", "BITTA NARSADIR."),
        ("ONE AND THE SAME", "BITTA NARSADIR."),
    ]
    for en, expected in en_cases:
        res = naturalization_rules.apply_naturalization("", en_orig=en)
        assert res == expected, f"Failed for en '{en}': got '{res}', expected '{expected}'"

    calque_cases = [
        ("SIZNING QO'RQUVLARINGIZ VA XAVFSIZLIKLARINGIZ BIR JONDIR.", "SIZNING QO'RQUVLARINGIZ VA XAVFSIZLIGIMIZ BITTA NARSADIR."),
        ("Bu bir jondir.", "BU BITTA NARSADIR."),
    ]
    for calque, expected in calque_cases:
        res = naturalization_rules.apply_naturalization(calque)
        assert res == expected, f"Failed for calque '{calque}': got '{res}', expected '{expected}'"


def test_infusion_and_wake_him_up():
    """Verify medical infusion and patient waking dialogue is properly structured."""
    en_cases = [
        ("Let's schedule a 48-hour infusion and wake him up.", "48 SOATLIK INFUZIYA TAYINLAB, KEYIN UNI UYG'OTAMIZ."),
        ("Schedule a 48-hour infusion and wake him up.", "48 SOATLIK INFUZIYA TAYINLAB, KEYIN UNI UYG'OTAMIZ."),
        ("Schedule a 48-hour infusion.", "48 SOATLIK INFUZIYA TAYINLANG."),
    ]
    for en, expected in en_cases:
        res = naturalization_rules.apply_naturalization("", en_orig=en)
        assert res == expected, f"Failed for en '{en}': got '{res}', expected '{expected}'"

    calque_cases = [
        ("KELING, 48 SOATLIK REJALASHTIRIB, INFUTSIYANI UYG'OTAMIZ.", "48 SOATLIK INFUZIYA TAYINLAB, KEYIN UNI UYG'OTAMIZ."),
        ("48 soatlik rejalashtirib, infutsiyani uyg'otamiz.", "48 SOATLIK INFUZIYA TAYINLAB, KEYIN UNI UYG'OTAMIZ."),
    ]
    for calque, expected in calque_cases:
        res = naturalization_rules.apply_naturalization(calque)
        assert res == expected, f"Failed for calque '{calque}': got '{res}', expected '{expected}'"


def test_untranslated_english_leak_interception():
    """Verify that if English text leaks directly into uz_text without en_orig, it is caught."""
    leaks = [
        ("Put him on ice.", "UNI MUZLATIB QO'YING."),
        ("Hook him up.", "UNI KISHANLANG."),
        ("One and the same.", "BITTA NARSADIR."),
        ("Your fears and our safety are one and the same.", "SIZNING QO'RQUVLARINGIZ VA XAVFSIZLIGIMIZ BITTA NARSADIR."),
        ("Let's schedule a 48-hour infusion and wake him up.", "48 SOATLIK INFUZIYA TAYINLAB, KEYIN UNI UYG'OTAMIZ."),
    ]
    for leak, expected in leaks:
        res = naturalization_rules.apply_naturalization(leak)
        assert res == expected, f"Failed for leaked text '{leak}': got '{res}', expected '{expected}'"


if __name__ == "__main__":
    test_put_him_on_ice()
    print("PASS: test_put_him_on_ice")
    test_hook_him_up()
    print("PASS: test_hook_him_up")
    test_one_and_the_same()
    print("PASS: test_one_and_the_same")
    test_infusion_and_wake_him_up()
    print("PASS: test_infusion_and_wake_him_up")
    test_untranslated_english_leak_interception()
    print("PASS: test_untranslated_english_leak_interception")
    print("\nALL 5 STEP 6 IDIOM & NATURALIZATION TESTS PASSED PERFECTLY!")
