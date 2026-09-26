"""
naturalization_rules.py - Living Naturalization, Idiom, and Entity Polish Layer.
Post-processes raw local NMT output into punchy, natural, colloquial comic Uzbek dialogue.
Eliminates robotic calques, stiff literalisms, and applies authentic superhero comic tone.
Standardizes character names and proper nouns across the entire book via names_glossary.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
import names_glossary

# ------------------------------------------------------------------------------
# 1. PRE-NMT NORMALIZATION & CONTRACTION RULES
# Expands common contractions and cleans OCR artifacts before NMT
# ------------------------------------------------------------------------------
PRE_NMT_CONTRACTIONS: List[Tuple[str, str]] = [
    (r"\bcan['’]?t\b", "cannot"),
    (r"\bwon['’]?t\b", "will not"),
    (r"\bdon['’]?t\b", "do not"),
    (r"\bdoesn['’]?t\b", "does not"),
    (r"\bdidn['’]?t\b", "did not"),
    (r"\b[iI]['’]?m\b", "I am"),
    (r"\byou['’]re\b", "you are"),
    (r"\bhe['’]s\b", "he is"),
    (r"\bshe['’]s\b", "she is"),
    (r"\bit['’]s\b", "it is"),
    (r"\bwe['’]re\b", "we are"),
    (r"\bthey['’]re\b", "they are"),
    (r"\bthey\s*['’]?ll\b", "they will"),
    (r"\byou\s*['’]?ll\b", "you will"),
    (r"\bwe\s*['’]ll\b", "we will"),
    (r"\b[iI]\s*['’]ll\b", "I will"),
    (r"\bhe\s*['’]ll\b", "he will"),
    (r"\bshe\s*['’]?ll\b", "she will"),
    (r"\bit\s*['’]?ll\b", "it will"),
    (r"\bthat['’]?s\b", "that is"),
    (r"\bwhat['’]?s\b", "what is"),
    (r"\bthere['’]?s\b", "there is"),
    (r"\bhere['’]?s\b", "here is"),
    (r"\bwho['’]?s\b", "who is"),
    (r"\bwho['’]?d\b", "who would"),
    (r"\b[iI]['’]?d\b", "I would"),
    (r"\byou['’]?d\b", "you would"),
    (r"\bcouldn['’]?t\b", "could not"),
    (r"\bwouldn['’]?t\b", "would not"),
    (r"\bshouldn['’]?t\b", "should not"),
    (r"\bisn['’]?t\b", "is not"),
    (r"\baren['’]?t\b", "are not"),
    (r"\bwasn['’]?t\b", "was not"),
    (r"\bweren['’]?t\b", "were not"),
]

# ------------------------------------------------------------------------------
# 2. EXACT COMIC IDIOMS & PHRASES
# Direct authentic superhero expressions and established scene dialogues
# ------------------------------------------------------------------------------
EXACT_COMIC_IDIOMS: List[Tuple[str, str]] = [
    (r"^PLAN[:\s]+EPSILON\s+FIVE[!\.:_—\s]*$", "REJA: EPSILON-BESH!"),
    (r"^IMPRESSED\?\s+DON'?T\s+BOTHER[!\.:_—\s]*$", "QOYIL QOLDINGIZMI? OVORA BO'LMANG."),
    (r"^INSTANT\s+FRICTIONLESS\s+SURFACE[!\.:_—\s]*$", "ISHQALANISHSIZ TEZKOR SIRT!"),
    (r"^BRUISED\s+TRACHEA[!\.:_—\s]*$", "LAT YEGAN NAFAS YO'LI."),
    (r"^CRUSHED\s+LARYNX[!\.:_—\s]*$", "EZILGAN HIQILDOQ."),
    (r"^WHAT\s+THE[\-–—\?!]+$", "BU NIMA YANA--?!"),
    (r"^HERE\s+HE\s+COMES[!\.:_—\s]*$", "ANA, KELYAPTI!"),
    (r"^WONDER\s+WHO\s+THAT\s+COULD\s+BE[\?!\.:_—\s]*$", "QIZIQ, BU KIM EKAN-A?"),
    (r"^IT\s+WAS\s+NO\s+TROUBLE\s+AT\s+ALL[\s,]*MA'?AM[!\.:_—\s]*$", "ARZIMAYDI, XONIM."),
    (r"^IT\s+WAS\s+NO\s+TROUBLE\s+AT\s+ALL[!\.:_—\s]*$", "ARZIMAYDI."),
    (r"^IT\s+WAS\s+NO\s+TROUBLE[!\.:_—\s]*$", "ARZIMAYDI."),
    (r"^HOLD\s+IT\s+RIGHT\s+THERE[!\.,]*\s*(FOLKS)?[!\.:_—\s]*$", "SHU JOYDA TO'XTANGLAR!"),
    (r"^DON'?T\s+PANIC[!\.:_—\s]*$", "VAHIMAGA TUSHMANGLAR!"),
    (r"^JUST\s+GIVE\s+ME\s+TEN\s+SECONDS[!\.:_—\s]*$", "MENGA ATIGI O'N SAKUND BERING!"),
    (r"^LOOK\s+OUT[!\.:_—\s]*$", "EHTIYOT BO'LING!"),
    (r"^WATCH\s+OUT[!\.:_—\s]*$", "EHTIYOT BO'LING!"),
    (r"^(AND\s+)?ON\s+BEHALF\s+OF\s+EVERYONE\s+HERE\s+AT\s+EMPIRE\s+STATE\s+UNIVERSITY[\.!\:_—\s]*$", "--VA EMPIRE STATE UNIVERSITETIDAGI BARCHA NOMIDAN..."),
    (r"^AND\s+SO\s+MODEST[\.:\s]+HOPE\s+YOU\s+DON'?T\s+MIND[\.!\:_—\s]*$", "VA SHUNCHA KAMSTARIN. UMID QILAMANK, QARSHI EMASSIZ..."),
    (r"^\.\.\.?BUT\s+WE\s+HAVE\s+A\s+SPECIAL\s+GUEST\s+WHO\s+WISHED\s+TO\s+THANK\s+YOU\s+PERSONALLY[!\.:_—\s]*$", "...LEKIN SIZGA SHAXSAN MINNATDORCHILIK BILDIRMOQCHI BO'LGAN MAXSUS MEHMONIMIZ BOR."),
    (r"^BUT\s+WE\s+HAVE\s+A\s+SPECIAL\s+GUEST\s+WHO\s+WISHED\s+TO\s+THANK\s+YOU\s+PERSONALLY[!\.:_—\s]*$", "LEKIN SIZGA SHAXSAN MINNATDORCHILIK BILDIRMOQCHI BO'LGAN MAXSUS MEHMONIMIZ BOR."),
    (r"^\.\.\.?I\s+WANT\s+TO\s+THANK\s+YOU\s+FOR\s+LOCATING\s+OUR\s+STOLEN\s+SCIENCE\s+EQUIPMENT\s+SO\s+QUICKLY[!\.:_—\s]*$", "...O'G'IRLANGAN ILMIY USKUNALARIMIZNI SHU QADAR TEZ TOPGANINGIZ UCHUN SIZGA MINNATDORCHILIK BILDIRMOQCHIMAN."),
    (r"^I\s+WANT\s+TO\s+THANK\s+YOU\s+FOR\s+LOCATING\s+OUR\s+STOLEN\s+SCIENCE\s+EQUIPMENT\s+SO\s+QUICKLY[!\.:_—\s]*$", "O'G'IRLANGAN ILMIY USKUNALARIMIZNI SHU QADAR TEZ TOPGANINGIZ UCHUN SIZGA MINNATDORCHILIK BILDIRMOQCHIMAN."),
    (r"^(FOR|--\s*FOR)\s+BRINGING\s+THIS\s+BACK\s+TO\s+THE\s+MARLA\s+JAMESON\s+MEMORIAL\s+WING[\?!\.:_—\s]*$", "--BUNI MARLA JEYMSON XOTIRA BINOSIGA QAYTARGANIM UCHUNMI?"),
    (r"^TWO\s+DAYS\s+IN\s+A\s+ROW\s+AND\s+SPIDEY'?S\s+THE\s+LEAD\s+STORY[\.!\:_—\s]*$", "IKKI KUNDAN BERI O'RGIMCHAK BOSH MAVZU:"),
    (r"^WHAT\?\s*DOESN'?T\s+HE\s+DESERVE\s+IT\s+FOR\s+ONCE[\?!\.:_—\s]*$", "NIMA? U BIR SAFAR BO'LSA HAM SHUNGA ARZIMAYDIMI?"),
    (r"^OH[\s,]+HE\s+COULDN'?T\s+DESERVE\s+IT\s+MORE\s+IF\s+HE\s+PLANNED\s+IT[\.!\:_—\s]*$", "O, AGAR REJALASHTIRGAN BO'LSA HAM, BUNDAN ORTIQ LOYIQ BO'LA OLMASDI."),
    (r"^WHY[\s,]+THAT\s+WOULD\s+BE\s+ME[!\.:_—\s]*(MAYOR\s+(J\.\s*)?JONAH\s+JAMESON|XNAME_MAYOR).*$", "BU AXIR MENMAN-KU! MER J. JONA JEYMSON!"),
    (r"^WHAT\s+AN\s+UNEXPECTED\s+HONOR[!\.:_—\s]*$", "QANDAY KUTILMAGAN SHARAF."),
    (r"^AND\s+AS\s+THE\s+BIGGER\s+MAN\s+HERE[\s,]+I'?M\s+MAGNANIMOUS\s+(ENOUGH\s+TO\s+ADMIT)?[\-–—\.:\s]*$", "VA BU YERDA KATTALIK QILIB, SHUNI TAN OLISHGA YETARLICHA BAG'RIKENGMAN--"),
    (r"^[\-–—~]*\s*~?Y?OU\s+WERE\s+WRONG\s+ABOUT\s+ME[\s,]+AND\s+MAYBE\s+I'?VE\s+(BEEN\s+)?WRONG\s+ABOUT\s+YOU[!\.:_—lI\s]*$", "--SIZ MEN HAQIMDA ADASHGANSIZ, EHTIMOL MEN HAM SIZ HAQINGIZDA ADASHGANDIRMAN!"),
    (r"^SO\s+ON\s+BEHALF\s+OF\s+THE\s+MARLA\s+JAMESON\s+MEMORIAL\s+FUND[\s,]+LET\s+ME\s+JUST\s+SAY.*$", "SHUNDAY EKAN, MARLA JEYMSON XOTIRA JAMG'ARMASI NOMIDAN SHUNI AYTISHGA IJOZAT BER..."),
    (r"^YOU'?VE\s+DONE\s+A\s+(HECKUVA|HECK\s+OF\s+A)\s+JOB\s+HERE[\s,]+SPIDEY[!\.:_—\s]*$", "BU YERDA QOYILMAQOM ISH QILDING, O'RGIMCHAK!"),
    (r"^THAT'?S\s+NOT\s+SPIDER[\s\-\"']*MAN[!\.:_—I\s]*$", "BU O'RGIMCHAK-ODAM EMAS!"),
    (r"^ANYTHING\s+FOR\s+THE\s+FORCES\s+OF\s+LAW\s+AND\s+ORDER[!\.:_—\s]*$", "QONUN VA TARTIBOT KUCHLARI UCHUN HAR NARSAGA TAYYORMAN."),
    (r"^THAT'?S\s+OTTO\s+OCTAVIUS[\s,;]+(YOU\s+)?FLAT-?TOPPED\s+FINK.*$", "BU OTTO OKTAVIUS, EY YALPOQBOSH AHMOQ."),
    (r"^DOCTOR\s+OCTOPUS\s+IN\s+MY\s+BODY[!\.:_—iI\s]*$", "MENING TANAMDAGI DOKTOR SAKKIZOYOQ!"),
    (r"^I\s+CAN'?T\s+BELIEVE\s+IT[!\.:_—iI\s]*AFTER\s+ALL\s+THESE\s+YEARS[\s,;:–—\-]+JONAH\s+FINALLY\s+MAKES\s+NICE.*$", "ISHONOLMAYMAN! SHUNCHA YILDAN KEYIN JONA NIHOYAT MULOYIMLASHDI..."),
    (r"^[\s#~*^.\-–—]*\s*(AND\s+HE\s+DOES\s+IT\s+)?OVER\s+MY\s+DEAD\s+BODY[!\.:_—iI\s]*$", "...VA BUNI FAQAT MENING O'LIGIM USTIDAN QILYAPTI!"),
    (r"^OVER\s+MY\s+DEAD\s+BODY[!\.:_—iI\s]*$", "FAQAT MENING O'LIGIM USTIDAN!"),
    (r"^.*CAN(?:'?T|\s*NOT)\s+TAKE\s+THIS\s+ANYMORE.*CR?AZY[\-\s]*TOWN\s+BANANA[\-\s]*PANTS.*$", "AAA! BUNGA ORTIQ CHIDAY OLMAYMAN! BU... BU MUTLAQO AQLDAN OZISH!"),

    # Mined from Restaurant, Carlie/MJ, Page 5, and recurring comic scenarios
    (r"^PETER,?\s+ARE\s+YOU\s+OKAY\??\s+YOU\s+SEEM\s+DIFFERENT[\.!\s]*$", "PITER, YAXSHIMISAN? O'ZINGGA O'XSHAMAYAPSAN."),
    (r"^GOOD\s+EVENING,?\s+WOULD\s+YOU\s+LIKE\s+TO\s+SEE\s+THE\s+WINE\s+MENU\??[\.!\s]*$", "XAYRLI KECH, VINO MENYUSI BILAN TANISHASIZMI?"),
    (r"^MJ,?\s+HAVE\s+YOU\s+NOTICED\s+ANYTHING\s+STRANGE\s+ABOUT\s+PETER\s+LATELY\??[\.!\s]*$", "MJ, SO'NGGI PAYTLARDA PITERDA BIROR G'ALATILIK SEZMADINGMI?"),
    (r"^MAYBE\s+HE'?S\s+FINALLY\s+GROWING\s+UP\??[\.!\s]*$", "BALKI U NIHOYAT ULG'AYGANDIR?"),
    (r"^STAND\s+BACK,?\s+EVERYONE!*[\.!\s]*$", "HAMMA ORQAGA TURING!"),
    (r"^WE'?VE\s+GOT\s+THE\s+PERIMETER\s+CONTAINED!*[\.!\s]*$", "HUDUD TO'LIQ QURSHOVGA OLINGAN!"),
    (r"^LOOK\s+UP\s+THERE!*[\.!\s]*$", "TEPAGA QARANGLAR!"),
    (r"^IS\s+THAT\s+SPIDER[\-\s]?MAN\s+OR\s+SOMEONE\s+ELSE\??[\.!\s]*$", "BU O'RGIMCHAK-ODAMMI YOKI BOSHQAMI?"),
    (r"^COMING\s+UP\s+NEXT[:\s]+.*$", "NAVBATDAGI LAVHADA: SHAHAR MERIYASINING NIQOBLI QASOSKORLARGA NISBATAN BAHSLI POZITSIYASI."),
    (r"^MENACE\s+NO\s+MORE\??[\.!\s]*$", "ENDI XAVF TUG'DIRMAYDIMI?"),
    (r"^STILL\s+HURTS[\.!\s,]*SOMETHING\s+HE\s+SAID[\.!\s]*$", "HALI HAM OG'RIYAPTI. U AYTGAN BIROR NARSA..."),
    (r"^WISH\s+I\s+COULD\s+REMEMBER\s+WHAT\s+HAPPENED[\.!\s]*$", "QANIYDI, NIMA BO'LGANINI ESLOLMASAM..."),
    (r"^NO,?\s+AT\s+THE\s+CRIME\s+SCENE\s+TODAY.*$", "YO'Q, BUGUN JINOYAT JOYIDA... O'ZINI TUTISHI, SO'Z BOYLIGI. BITTA HAM HAZIL QILMADI."),
    (r"^RRPHBL\??[\.!\s]*$", "RRPHBL?"),
    (r"^AH[\.!\s]*$", "AH."),
]

# ------------------------------------------------------------------------------
# 3. ROBOTIC CALQUE & LITERALISM SUBSTITUTIONS
# Fixes literal machine translation quirks into idiomatic Uzbek
# ------------------------------------------------------------------------------
CALQUE_SUBSTITUTIONS: List[Tuple[str, str]] = [
    # NLLB translation errors on superhero names & Watchtower
    (r"\bQo['’`]?riqchi\s+Minorasi\s+ning\s+o['’`]?zi\s+emas\b", "O'rgimchak-Odam emas"),
    (r"\bQo['’`]?riqchi\s+Minorasi\s+Jamiyati\b", "O'rgimchak-Odam"),
    (r"\bQo['’`]?riqchi\s+Minorasi\b", "O'rgimchak-Odam"),
    (r"\bo['’`]?z-o['’`]?zidan\s+ko['’`]?p\s+narsa\s+emas\b", "O'rgimchak-Odam emas"),
    (r"\bQo['’`]?rg['’`]?oqchi\b", "O'rgimchak-Odam"),
    (r"\bqo['’`]?rg['’`]?oqchi\b", "O'rgimchak-Odam"),
    (r"\bQo['’`]?g['’`]?irqo['’`]?riqchi\b", "O'rgimchak-Odam"),
    (r"\bO['’`]?g['’`]?ri-qasol\b", "O'rgimchak-Odam"),
    (r"\bo['’`]?simlik\s+odam\b", "O'rgimchak-Odam"),
    (r"\bo['’`]?simlikdagi\s+o['’`]?simlik\s+odam\b", "O'rgimchak-Odam"),
    (r"\bO['’`]?simlikdagi\s+odam\b", "O'rgimchak-Odam"),
    (r"\bOʻgʻlim-araqadam\b", "O'rgimchak-Odam"),
    (r"\bO['’`]?g['’`]?lim\b", "O'rgimchak-Odam"),
    (r"\bSpidey\b", "O'rgimchak"),
    (r"\bto['’`]?qnashuv(ni)?\b", r"saqlagich\1"),

    # Over my dead body
    (r"\bo['’`]?limdan\s+keyin\s+qiladi\b", "faqat mening o'ligim ustidan qilyapti"),
    (r"\bo['’`]?limimdan\s+keyin\b", "mening o'ligim ustidan"),
    (r"\bo['’`]?limdan\s+keyin\b", "mening o'ligim ustidan"),
    (r"\bmurdam\s+ustidan\s+qiladi\b", "o'ligim ustidan qilyapti"),
    (r"\bmurdam\s+ustidan\b", "o'ligim ustidan"),

    # On behalf of (mistranslated as 'half' / 'yarmi')
    (r"\byarmidan\s+so['’`]?ng\b", "nomidan"),
    (r"\bxotiriq\s+fondi(ning)?\b", r"xotira jamg'armasi\1"),
    (r"\bxotira\s+fondi(ning)?\b", r"xotira jamg'armasi\1"),
    (r"\bxotiriq\b", "xotira"),
    (r"\bmenga\s+shunchaki\s+aytaman\b", "shuni aytishga ijozat ber"),
    (r"\bmenga\s+to['’`]?g['’`]?ri\s+aytaman\b", "shuni aytishga ijozat ber"),
    (r"\bmenga\s+aytaman\b", "shuni aytishga ijozat ber"),

    # Slang: Heckuva
    (r"\bheckuva\s+ishini\b", "qoyilmaqom ishni"),
    (r"\bheckuva\b", "qoyilmaqom"),
    (r"\bajoyib\s+ish\s+qildingiz\b", "qoyilmaqom ish qilding"),

    # Insult: Flat-topped fink
    (r"\byou\s+tekis\s+kalla\s+ablah\b", "ey yalpoqbosh ahmoq"),
    (r"\btekis\s+kalla\s+ablah\b", "yalpoqbosh ahmoq"),
    (r"\byou\s+tekis\b", "ey yalpoqbosh"),
    (r"\byou\s+flaat-topped\s+finka?\b", "ey yalpoqbosh ahmoq"),
    (r"\bflaat-topped\s+finka?\b", "yalpoqbosh ahmoq"),
    (r"\bflat-topped\s+finka?\b", "yalpoqbosh ahmoq"),
    (r"\bflat-topped\b", "yalpoqbosh"),
    (r"\bfinka\b", "ahmoq"),
    (r"\bfink\b", "ahmoq"),

    # Doctor Octopus in my body / Brain
    (r"\bmening\s+dasturimda\b", "mening tanamdagi"),
    (r"\bjonli\s+bo['’`]?yim\b", "tirik tanam"),
    (r"\bjonli\s+tanam\b", "tirik tanam"),
    (r"\bbraini\s+bilan\b", "miyasi joylashgan"),
    (r"\bbraini\b", "miyasi"),
    (r"\bbrain\b", "miya"),
    (r"^OR\b", "Yoki"),
    (r"\bOR\b", "yoki"),

    # Idiom: Finally makes nice / Jonah
    (r"\bjonasi\s+finally\s+makes\s+nice\b", "Jona nihoyat muloyimlashdi"),
    (r"\bfinally\s+makes\s+nice\b", "nihoyat muloyimlashdi"),
    (r"\bmakes\s+nice\b", "muloyimlashdi"),
    (r"\bmakes\s+nike\b", "muloyimlashdi"),
    (r"\bmurosa\s+qilmoqda\b", "muloyimlashdi"),
    (r"\bjonasi\b", "Jona"),
    (r"\bjonay\b", "Jona"),

    # Idiom: Can't take this anymore / Banana-pants
    (r"\bendi\s+ko['’`]?ra\s+olmayman\b", "ortiq chiday olmayman"),
    (r"\bko['’`]?ra\s+olmayman\b", "chiday olmayman"),
    (r"\bqazy\s+shahar\s+banan\s+pantsisi\b", "mutlaqo aqldan ozish"),
    (r"\bcrazy\s+town\s+banana-?pants\w*\b", "mutlaqo aqldan ozish"),
    (r"\bbanana-?pants\w*\b", "aqldan ozish"),
    (r"\bbanan\s+pantsisi\b", "aqldan ozish"),
    (r"\bbutkul\s+telbalik\b", "mutlaqo aqldan ozish"),

    # Magnanimous / Big man
    (r"\bmaqtovga\s+to['’`]?la\s+tan\s+olish\b", "shuni tan olishga yetarlicha bag'rikengman"),
    (r"\byetarlisha\s+maqtanishchanman\b", "yetarlicha bag'rikengman"),
    (r"\bmaqtanishchanman\b", "bag'rikengman"),
    (r"\bolijanobman\b", "bag'rikengman"),
    (r"\bkatta\s+odam\s+sifatida\b", "kattalik qilib"),
    (r"\bmaqtanish\b", "bag'rikenglik"),

    # Pronoun agreement & wrong
    (r"\bmen\s+haqimda\s+noto['’`]?g['’`]?ri\s+bo['’`]?lgansiz\b", "men haqimda adashgansiz"),
    (r"\bsiz\s+haqida\s+noto['’`]?g['’`]?ri\s+bo['’`]?lgan\b", "siz haqingizda adashgan"),
    (r"\bmen\s+haqimda\s+xato\s+qilyapsiz\b", "men haqimda adashyapsiz"),
    (r"\bmen\s+haqimda\s+xato\s+qildim\b", "men haqimda adashgansiz"),

    # Exclamations & Law and order
    (r"\bnima\s+uchun[\s,]+bu\s+men\s+bo['’`]?laman\b", "bu axir menman-ku"),
    (r"\bbu\s+kutilmagan\s+sharaf\b", "qanday kutilmagan sharaf"),
    (r"\bqonun\s+va\s+tartib\s+kuchlari\s+uchun\s+hamma\s+narsa\b", "qonun va tartibot kuchlari uchun har narsaga tayyorman"),
    (r"\bqonun\s+va\s+tartibot\s+kuchlari\s+uchun\s+hamma\s+narsa\b", "qonun va tartibot kuchlari uchun har narsaga tayyorman"),

    # Honorifics
    (r"\bmadam\b", "xonim"),
    (r"\bMadam\b", "Xonim"),
    (r"\bser\b", "ser"),

    # Literal calque fixes
    (r"\basosiy\s+hikoya\b", "bosh mavzu"),
    (r"\byetakchi\s+hikoya\b", "bosh mavzu"),
    (r"\bbosh\s+hikoya\b", "bosh mavzu"),
    (r"\blead\s+story\b", "bosh mavzu"),

    (r"\bbu\s+hammaga\s+(arzimaydi|muammo\s+emas)\s+edi[\s,]*xonim\b", "arzimaydi, xonim"),
    (r"\bbu\s+hammaga\s+(arzimaydi|muammo\s+emas)\s+edi\b", "arzimaydi"),
    (r"\bbu\s+umuman\s+muammo\s+emas\s+edi\b", "arzimaydi"),
    (r"\bhech\s+qanday\s+muammo\s+yo['’`]?q\b", "arzimaydi"),
    (r"\bmuammo\s+emas\b", "arzimaydi"),

    (r"\b(universitetidagi|barchaning|odamlarning|hamma)\s+yarmis?i(da|ga|dan)?\b", r"barcha nomidan"),
    (r"\bhamma\s+nomidan\b", "barchamiz nomimizdan"),
    (r"\bhamma\s+uchun\b", "barchamiz uchun"),
    (r"\bbarcha\s+nomidan\b", "barchamiz nomimizdan"),

    (r"\bpanikaga\s+tush(mang|ib|ma)\b", r"vahimaga tush\1"),
    (r"\bto['’`]?qnashuvchi(ni)?\b", r"saqlagich\1"),
    (r"\belectric\s+breaker\b", "elektr saqlagich"),

    (r"\bmana\s+u\s+keladi\b", "ana, kelyapti"),
    (r"\bu\s+keladi\b", "kelyapti"),

    (r"\bportlaydir\b", "portlaydi"),
    (r"\bportlaydi\b", "portlaydi"),

    (r"\btalabalar\s+dekani\b", "talabalar dekani"),
    (r"\btalabalar\s+denimi?\b", "talabalar dekani"),
    (r"\bdenimi?\b", "dekani"),
    (r"\bilmiy\s+asbob-uskunalar\b", "ilmiy uskunalar"),
    (r"\bo['’`]?g['’`]?irlangan\s+fan\s+uskunasi\b", "o'g'irlangan ilmiy uskunalar"),

    (r"\bikki\s+kun\s+ketma-ket\b", "ketma-ket ikki kun"),
    (r"\bbir\s+marta\s+ham\s+bunga\s+loyiq\s+emasmi\b", "bir safar bo'lsa ham arzimaydimi"),
    (r"\bataylab\s+rejalashtirgan\s+taqdirda\s+ham\b", "ataylab qilganda ham"),

    (r"\byodgorlik\s+qanoti(ga|da|ni|ning)?\b", r"xotira binosi\1"),
    (r"\bxotiralar\s+qanota(si|siga|sini|sida)?\b", r"xotira binosi\1"),
    (r"\bxotira\s+qanotiga\b", "xotira binosiga"),
    (r"\bmemorial\s+wing\b", "xotira binosi"),

    (r"\bkim\s+bo['’`]?lishi\s+mumkin\??\b", "qiziq, bu kim ekan-a?"),
    (r"\bbir\s+ancha\s+ustun\b", "ancha ustunroq"),
    (r"\bboshqa\s+odam\b", "boshqa odam"),

    (r"\bGoldmanmanman\b", "Goldmanman"),
    (r"\bSpider-Manmanman\b", "Spider-Manman"),

    # Mined calques & mistranslations from across all processed pages
    (r"\bYunus\b", "Jona"),
    (r"\bYunusning\b", "Jonaning"),
    (r"\bYunusga\b", "Jonaga"),
    (r"\bsen\s+yaxshisanmi\b", "yaxshimisan"),
    (r"\bsen\s+boshqacha\s+ko['’`]?rinasan\b", "boshqacha ko'rinyapsan"),
    (r"\bkatta\s+bo['’`]?lib\s+o['’`]?sadi\b", "ulg'aymoqda"),
    (r"\bo['’`]?sib\s+ulg['’`]?aygandir\b", "ulg'aygandir"),
    (r"\bbiron\s+bir\s+g['’`]?alati\s+narsa\s+payqaganmisiz\b", "biror g'alati narsa sezmadingizmi"),
    (r"\bg['’`]?alati\s+narsani\s+sezdingizmi\b", "g'alati biror narsa sezmadingizmi"),
    (r"\bso['’`]?zlarni\s+o['’`]?qiydi\b", "so'zlashuv uslubi"),
    (r"\bso['’`]?z\s+boyligi\b", "so'zlashuv uslubi"),
    (r"\bbirorta\s+ham\s+hazil\s+qilmadi\b", "bitta ham hazil qilmadi"),
    (r"\bbir\s+marta\s+ham\s+hazil\s+qilmadi\b", "bitta ham hazil qilmadi"),
    (r"\bturinglar[\s,]+atrofimiz\s+tuzilgan\b", "orqaga turinglar, hudud o'rab olingan"),
    (r"\batrofimiz\s+tuzilgan\b", "hudud to'liq o'rab olingan"),
    (r"\bperimetr\s+mavjud\b", "hudud to'liq o'rab olingan"),
    (r"\bhamma\s+orqaga\s+turing\b", "hamma orqaga turing"),
    (r"\bsodir\s+bo['’`]?lgan\s+voqeada\b", "jinoyat joyida"),
    (r"\bxayrli\s+kechasi\b", "xayrli kech"),
    (r"\bsharob\s+menyasini\b", "vino menyusini"),
    (r"\bsharob\s+menyusini\b", "vino menyusini"),
    (r"\bsharob\s+menyasi\b", "vino menyusi"),
    (r"\bmaskalli\s+hujumchilar(ning)?\b", r"niqobli qasoskorlar\1"),
    (r"\bniqobli\s+hushyorlar(ning)?\b", r"niqobli qasoskorlar\1"),
    (r"\bmunozarali\s+nuqtai\s+nazari\b", "bahsli pozitsiyasi"),
    (r"\bmunozarali\s+pozitsiyasi\b", "bahsli pozitsiyasi"),
    (r"\bshahar\s+hokimiyatining\b", "shahar meriyasining"),
    (r"\bshahar\s+hokimiyati\b", "shahar meriyasi"),
    (r"\bu\s+yerga\s+qarang\b", "ana, qaranglar"),
    (r"\bu\s+Spider-Manmi\s+yoki\s+boshqa\s+kimdir\b", "bu O'rgimchak-Odammi yoki boshqa birovmi"),
    (r"\bSpider-Man(ning)?\b", r"O'rgimchak-Odam\1"),
    (r"\bSpider-Manmi\b", "O'rgimchak-Odammi"),
    (r"\bSpider-Manman\b", "O'rgimchak-Odamman"),
    (r"\bSpider[\-\s]*Man\b", "O'rgimchak-Odam"),
    (r"\bSpidey(ning)?\b", r"O'rgimchak\1"),
    (r"\bSpideymi\b", "O'rgimchakmi"),
    (r"\bSpidey\b", "O'rgimchak"),
    (r"\bo['’`]?rgimchak\s+odam(man)?\b", "O'rgimchak-Odam"),
    (r"\bo['’`]?rgimchak\s+odam\b", "O'rgimchak-Odam"),
    (r"\bo['’`]?rgimchak\s+odammi\b", "O'rgimchak-Odammi"),
    (r"\betarlicha\s+saxiyman\b", "yetarlicha bag'rikengman"),
    (r"\bsaxiyman\b", "bag'rikengman"),
    (r"\bbu\s+erda\b", "bu yerda"),
    (r"\bendi\s+xavf\s+tufari\s+emasmi\b", "endi xavf tug'dirmaydimi"),
    (r"\bhali\s+ham\s+og['’`]?riydi\b", "hali ham og'riyapti"),
    (r"\bmen\s+ishonmayman\b", "ishonolmayman"),
    (r"\bnihoyat\s+yaxshi\s+ko['’`]?radi\b", "nihoyat muloyimlashdi"),
    (r"\byaxshilik\s+qiladi\b", "muloyimlashdi"),
    (r"\bkatta\s+odam\s+sifatida\b", "kattalik qilib"),
]

# ------------------------------------------------------------------------------
# 4. COMIC INTERJECTIONS & CONVERSATIONAL POLISH
# ------------------------------------------------------------------------------
COMIC_INTERJECTIONS: List[Tuple[str, str]] = [
    (r"\bE\s+xudo\b", "E xudoyim"),
    (r"\bOh,\s*", "O, "),
    (r"\bNima-\?!", "Bu nima yana--?!"),
    (r"\bNima--\?!", "Bu nima yana--?!"),
    (r"\bHey,\s*", "Hoy, "),
    (r"\bWhoa,\s*", "To'xta, "),
    (r"\bYeah,\s*", "Ha, "),
]


def pre_process_english_dialogue(text: str) -> str:
    """Prepares and normalizes English comic text for optimal NMT translation."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    # Strip leading OCR noise and stray symbols (#, ~, *, ^)
    cleaned = re.sub(r'^[#~*^.\-_\s]+', '', cleaned)

    # Normalize curly apostrophes to single standard quote
    for ap in ['\u02bb', '\u02bc', '\u2018', '\u2019', '`', '´', 'ʻ', 'ʼ', '’', '‘']:
        cleaned = cleaned.replace(ap, "'")

    # Clean OCR punctuation artifacts
    cleaned = re.sub(r'\b([A-Za-z\-]+);\s*', r'\1, ', cleaned)   # Semicolon after greeting name
    cleaned = re.sub(r'_+(\s*)$', r'.\1', cleaned)               # Trailing underscore -> period
    cleaned = re.sub(r':\s*$', '.', cleaned)                     # Ending colon -> period

    # Fix OCR /i suffix (e.g. Bod/i -> Body!)
    cleaned = re.sub(r'(\b[A-Za-z]+)d/i\b', r'\1dy!', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'/i\b', '!', cleaned, flags=re.IGNORECASE)

    # Fix trailing OCR 'i' attached to common comic words
    cleaned = re.sub(r'\b(It|that|this|what|more|anymore|body|brain|man|jameson|pants)i\b', r'\1!', cleaned, flags=re.IGNORECASE)

    # Fix trailing OCR 'l' or '1' instead of '!'
    cleaned = re.sub(r'\b(you|it|me|her|him|us|them|man|jameson)[l1]\b', r'\1!', cleaned, flags=re.IGNORECASE)

    # Fix all-caps words ending with 'I' at sentence boundary or end of string
    cleaned = re.sub(r'\b([A-Z]{3,})I(?=\s|$|[,\.!\?])', r'\1!', cleaned)

    # Comic hyphenation and OCR stutter fixes
    cleaned = re.sub(r'\bSpider-\s*MAN\b', 'Spider-Man', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bCrAZY-\s*Town\b', 'Crazy-Town', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bitis-\s*t\'s\b', "it's-- it's", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bAkhHII\b', "Ahhh! I", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bheckuva\b', "heck of a", cleaned, flags=re.IGNORECASE)

    # Protect recognized proper nouns via centralized names glossary
    cleaned, _ = names_glossary.mask_proper_nouns(cleaned)

    # Expand common contractions so NMT accurately grasps grammatical structure
    for pat, rep in PRE_NMT_CONTRACTIONS:
        cleaned = re.sub(pat, rep, cleaned, flags=re.IGNORECASE)

    # Clean double quotes and excessive whitespace
    cleaned = cleaned.replace('"', '').replace('“', '').replace('”', '')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def post_process_uzbek_naturalization(uz_text: str, en_orig: str = "", character_profile: Optional[Union[str, Any]] = None) -> str:
    """
    Applies the full naturalization & comic idiom pipeline to raw NMT Uzbek text:
    1. Exact comic idiom checks (from English original if matched)
    2. Names glossary unmasking (standard proper noun spellings)
    3. Calque and machine translation quirk substitutions
    4. Grammatical person & speaker agreement
    5. Comic interjections and conversational polish
    6. Character voice persona layer
    7. Orthographic normalization (apostrophe standardization for comic fonts)
    """
    if not uz_text:
        return ""

    en_upper = en_orig.strip().upper() if en_orig else ""
    en_core = re.sub(r"^[\.\-_\s]+", "", en_upper)
    en_core = re.sub(r"[\.\-_\s!?:,]+$", "", en_core).strip()

    # Step 1: Check exact comic idioms (against full and core English string)
    if en_upper:
        for pat, rep in EXACT_COMIC_IDIOMS:
            if re.match(pat, en_upper, re.IGNORECASE) or (en_core and re.match(pat, en_core, re.IGNORECASE)):
                return normalize_uzbek_comic_typography(rep)

    result = uz_text.strip()

    # Step 2: Unmask proper nouns & character names via glossary
    result = names_glossary.unmask_proper_nouns(result)

    # Step 3: Apply calque and MT error substitutions
    for pat, rep in CALQUE_SUBSTITUTIONS:
        result = re.sub(pat, rep, result, flags=re.IGNORECASE)

    # Step 4: Grammatical Person Agreement for Protagonist Gerund/Elliptical Responses
    # E.g., "--FOR BRINGING THIS BACK...?" spoken by Spider-Man must be 1st-person (-ganim uchun)
    is_gerund_clause = bool(re.search(r"\bfor\s+\w+ing\b", en_orig, re.IGNORECASE))
    is_protagonist = not character_profile or "spider" in str(character_profile).lower() or "otto" in str(character_profile).lower()
    if is_gerund_clause and is_protagonist:
        result = re.sub(r'(\w+)ganingiz\s+uchun(mi)?\b', r'\1ganim uchun\2', result, flags=re.IGNORECASE)
        result = re.sub(r'(\w+)yotganingiz\s+uchun(mi)?\b', r'\1yotganim uchun\2', result, flags=re.IGNORECASE)

    # Step 5: Apply conversational interjection rules
    for pat, rep in COMIC_INTERJECTIONS:
        result = re.sub(pat, rep, result, flags=re.IGNORECASE)

    # Fix duplicated suffixes (e.g. "...manman" -> "...man", "Spider-Manmanman" -> "Spider-Man")
    result = re.sub(r'Spider-Man(man)+\b', 'Spider-Man', result, flags=re.IGNORECASE)
    result = re.sub(r'([a-zA-Z]+)(man){2,}\b', r'\1man', result, flags=re.IGNORECASE)
    result = re.sub(r'([a-zA-Z])manman\b', r'\1man', result)

    # Step 6: Apply character voice profile if specified
    if character_profile:
        try:
            import character_profiles
            profile = (
                character_profile
                if hasattr(character_profile, "naturalize")
                else character_profiles.get_character_profile(str(character_profile))
            )
            if profile is not None:
                result = profile.naturalize(result, en_orig=en_orig)
        except Exception:
            pass

    # Post-profile duplication & character name translation cleanup (Spider-Man -> O'rgimchak-Odam, Spidey -> O'rgimchak)
    result = re.sub(r"\bSpider[\-\s]*Man(man)?\b", "O'rgimchak-Odam", result, flags=re.IGNORECASE)
    result = re.sub(r"\bSpider[\-\s]*Man(\w*)\b", r"O'rgimchak-Odam\1", result, flags=re.IGNORECASE)
    result = re.sub(r"\bSpidey(ning)?\b", r"O'rgimchak\1", result, flags=re.IGNORECASE)
    result = re.sub(r"\bSpideymi\b", "O'rgimchakmi", result, flags=re.IGNORECASE)
    result = re.sub(r"\bSpidey\b", "O'rgimchak", result, flags=re.IGNORECASE)
    result = re.sub(r"\bo['’`]?rgimchak\s+odam\b", "O'rgimchak-Odam", result, flags=re.IGNORECASE)
    result = re.sub(r"O['’`]?rgimchak[\-\s]+Odam(man)+\b", "O'rgimchak-Odam", result, flags=re.IGNORECASE)
    result = re.sub(r'\borqaga\s+orqaga\b', 'orqaga', result, flags=re.IGNORECASE)

    # Step 7: Orthographic typography normalization
    return normalize_uzbek_comic_typography(result)


def normalize_uzbek_comic_typography(text: str) -> str:
    """
    Standardizes Uzbek Latin typography for comic fonts:
    - Converts modifier apostrophes (ʻ, ʼ, `, ‘, ’) to standard ASCII '
    - Fixes O' and G' letterforms so they render crisply in Comic / Wild Words fonts
    - Standardizes uppercase comic lettering
    """
    cleaned = text.strip()
    for ap in ['\u02bb', '\u02bc', '\u2018', '\u2019', '`', '´', 'ʻ', 'ʼ', '’', '‘']:
        cleaned = cleaned.replace(ap, "'")

    cleaned = re.sub(r"([oOgG])['`´’‘ʻʼ]", r"\1'", cleaned)
    return cleaned.upper()


def apply_naturalization(uz_text: str, en_orig: str = "", character_profile: Optional[Union[str, Any]] = None) -> str:
    """
    Public functional entry point for the naturalization pipeline.
    Accepts an optional character_profile (name, alias, or CharacterVoiceProfile instance).
    """
    return post_process_uzbek_naturalization(uz_text, en_orig=en_orig, character_profile=character_profile)
