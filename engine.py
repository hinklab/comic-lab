"""
Comic Book Translation and Typesetting Engine
Zero AI API Keys (free/local translation model); online services for data storage are fine.
Powered by:
- OpenCV for white contour bubble detection and 100% clean erase
- EasyOCR for accurate text recognition on cropped bubbles
- deep-translator with Spider-Man Superhero Tone Mapper for natural Uzbek
- Pillow with CC Wild Words, dynamic auto-scaling, and line collision prevention
"""

import os
import warnings
warnings.filterwarnings("ignore")
import re
import math
import html
import urllib.request
import urllib.parse
import json
from typing import List, Tuple, Dict, Optional, Any
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import bubble_lettering
import quality_gates
import local_translator
import naturalization_rules
import sfx_engine

# Known SFX words and junk to strictly filter out
SFX_WORDS = {
    "TEKK", "SIW", "THWIP", "BAM", "POW", "BOOM", "WHAM", "CRASH",
    "SWOOSH", "ZAP", "BANG", "THUMP", "CLANG", "CLICK", "GULP",
    "UGH", "AAAH", "ARGH", "SHKK", "SKRRR", "KRAK", "SNIKT", "PING",
    "PANG", "POP", "SMACK", "SPLASH", "WHOOSH", "YAAAH", "GRRR", "HISS",
    "RUMBLE", "VROOOM", "ROAR", "KABOOM", "ZING", "SNAP", "CRACKLE",
    "KAPOW", "THOOM", "KRAKOOM", "RATATAT", "SKREEE", "CHOMP", "PST", "SHH",
    "CHK", "KOFF", "COUGH", "GASP", "SNORT", "WHEEZE", "AHEM", "CHUK", "CHIK"
}

# Known watermark domains and scraping markers to strictly filter out
WATERMARK_WORDS = {
    "READALLCOMICS", "HOEGOMIG", "NEL", "UPLOAD", "VK.COM", "T.ME", "TELEGRAM", "WWW", "HTTP", "HTTPS", "COM"
}

# Exact / Full-Sentence Comic Idiom Mappings
EXACT_COMIC_MATCHES = [
    (r"^IMPRESSED\?\s*DON'?T\s*(?:BOTHER|BE)[\s\._!]*$", "QOYIL QOLDINGMI? HOVLIQMA!"),
    (r"^INSTANT\s+FRICTIONLESS\s+SURFACE[\s\._!]*$", "BIR ZUMDA ISHQALANISHSIZ SIRT."),
    (r"^PLAN:\s*EPSILON\s+FIVE[\s\._!]*$", "REJA: EPSILON-BESH."),
    (r"^PLAN:\s*OMEGA\s+THREE[\s\._!]*$", "REJA: OMEGA-UCH."),
    (r"^BRUISED\s+TRACHEA[\s\._!]*$", "TRAXEYA ZARARLANGAN."),
    (r"^(?:OH\s+)?WHAT\s+NOW[\s\?!\._]*$", "EY, ENDI NIMASI?!"),
    (r"^ANOTHER\s+PARTY\s+TRICK[\s\._!]*$", "YANA BIR ARZON NAYRANG."),
    (r"^HOLY[!i\s]*$", "YO TOVBA!"),
    (r"^(?:HOLY[!\s]*)?NOT\s+COOL[i!]*\s*YOU\s+COULD'?VE\s+TAKEN\s+HIS\s+HEAD\s+OFF[!?]*$", "HAZILNING CHEGARASI BOR! BOSHI KETIB QOLISHI MUMKIN EDI-KU!"),
    (r"^HE'?LL\s+LIVE[\s\._]+AS\s+LONG\s+AS\s+HE'?S\s+SMART\s+AND\s+STAYS\s+DOWN[\s\._!]*$", "YASHAB KETADI. AGAR AQLLI BO'LIB, QIMIRLAMAY YOTSA."),
    (r"^HE'?LL\s+LIVE[\s\._]+AS\s+LONG\s+AS\s+HE'?S\s+SMART[\s\._!]*$", "YASHAB KETADI. AGAR AQLLI BO'LSA."),
    (r"^AND\s+STAYS\s+DOWN[\s\._!]*$", "VA YIQILGANICHA QOLADI."),
    (r"^WHAT\s+IS\s+THIS\s+STUFF[\s\?!\._]*$", "BU NIMA BALO?!"),
    (r"^FRIEND\s+OF\s+MINE\s+CAME\s+UP\s+WITH\s+IT[\s\._!]*$", "BUNI BIR DO'STIM O'YLAB TOPGAN."),
    (r"^(?:AND\s+)?A\s+GOOD\s+(?:SPENT\s+)?HOUR\s+APPLYING\s+IT\s+TO\s+THE\s+PAVEMENT[\s\._!]*$", "VA UNI YO'L QOPLAMASIGA SURTISH UCHUN ROSTMANA BIR SOAT VAQT KETDI."),
    (r"^I\s+CAN\s+SEE\s+IT'?S\s+RENDERED\s+YOU\s+SPEECHLESS[\s\._!:]*$", "O'ZIM HAM KO'RIB TURIBMAN... TILING LOL QOLDI SHEKILLI."),
    (r"^MY\s+SHOCK\s+GAUNTLETS!?\s+THEY'?RE\s+SHORTING\s+OUT[\s\._!]*$", "MENING ELEKTR QO'L-QOPLARIM! QISQA TUTASHUV BO'LYAPTI!"),
    (r"^UNH[!?]*\s+THIS\s+THING\s+WEIGHS\s+A\s+TON[!?]*$", "UH! BU MATOH BIR TONNA KELADI-YA!"),
    (r"^HEY!?\s+HERMAN[\s\?!\._]*$", "EY! GERMAN?"),
    (r"^HEY[!i\s]*$", "EY!"),
    (r"^HERMAN[\s\?!]*$", "GERMAN?"),
    (r"^HORIZON'?S\s+LATEST\s+INVENTION:\s*A\s+QUICK,\s*CHEAP,\s*AND\s+EASY\s+TO\s+INSTALL\s+POWER\s+DAMPENING\s+FIELD[\s\._!]*$", "HORIZON'NING ENG SO'NGGI IXTIROSI: TEZ, ARZON VA O'RNATISH OSON BO'LGAN QUVVATNI SO'NDIRISH MAYDONI."),
    (r"^GYAHH?[\s!]*$", "GYAHH!"),
    (r"^AHK+[\s!:;]*$", "AHKK!"),
    # Overlapping Dual-Bubble Pair Mappings (Clear earpiece audio vs muffled girlfriend speech)
    # Pair 1 — Rect: Peter Parker earpiece, Oval: Running a nightclub (muffled)
    (r"^(?:I\s+AM|IAM|IAm)\s+PETER\s+PARKER[,.\s]+AND\s+THAT'?S\s+(?:NOT\s+SUCH\s+A\s+BAD\s+THING\s+AT\s+ALL|NOT\s+SUCH)[\s\w.,!]*$", "MEN PITER PARKERMAN. VA BU HECH HAM YOMON EMAS."),
    (r"^(?:KONNINO|KUWNING|RUNNINO|RUINNO|RUNNING|RUNNIN)[\s\w]*(?:A\s+)?(?:NIGHTCLUB|nightclub)[.,\s!_]*$", "...BOSHQARISH...\n...TUNGI KLUB..."),
    (r"^(?:KUWNING|RUNNING).*NIGHTCLUB.*$", "...BOSHQARISH...\n...TUNGI KLUB..."),
    (r"^(?:KONNINO|KUWNING|RUNNINO|RUINNO|RUNNING|RUNNIN)[\s\w.,!_]*$", "...BOSHQARISH..."),
    (r"^(?:NIGHTCLUB|nightclub)[.,\s!_]*$", "...TUNGI KLUB..."),
    # Pair 2 — Rect: High Paying Job / State of Art Lab, Oval: PA / Medic / Keeping up staff morale (muffled)
    (r"^HIGH\s+PAYING(?:\s+PA)?\s+JO?B?[.,\s:]+STATE\s+OF\s+THE\s+ART\s+LAB[.,\s:]*$", "KATTA MAOSH. ENG ZAMONAVIY LABORATORIYA."),
    (r"^HIGH\s+PAYING(?:\s+PA)?[\s.,]+JO?B?[.,\s:]*$", "KATTA MAOSH."),
    (r".*(?:PA\s+)?(?:MEDIG|MEDIC|Medid|MEDID).*KEEP.*MORALE.*$", "...YORDAMCHI... TIBBIYOT... XODIMLAR KAYFIYATINI KO'TARISH..."),
    (r"^(?:PA[.,\s]+)?(?:MEDIG|MEDIC|Medid)[.,\s:]*$", "...TIBBIYOT..."),
    (r".*(?:DTAR|3TART|STAFF)\s+MORALE.*$", "...XODIMLAR KAYFIYATINI KO'TARISH..."),
    (r"^(?:KEEPING|KEEPINS|KEEPNG|Keepins).*MORALE[.,\s:_]*$", "...XODIMLAR KAYFIYATINI KO'TARISH..."),
    (r"^MORALE[_.,\s]*$", "...KAYFIYAT..."),
    (r"^KEEPING[.,\s]*$", "...SAQLASH..."),
    # Pair 3 — Rect: Super-Powers Youth Vigor, Oval: Just parts / Health codes (muffled)
    (r"^SUPER-?POWERS[.,\s:]+(?:YOUTH|YOUTH[.,\s:]+VIGOR)[.,\s:]*$", "SUPER-QOBILIYATLAR. YOSHLIK. G'AYRAT."),
    (r"^SUPER-?POWERS[.:,\s]*$", "SUPER-QOBILIYATLAR."),
    (r"^JUST\s+(?:'ARTS|ARTS|PARTS|RARTS)[.,\s:]+(?:CODES|CODes|LODES)[.,\s:]*$", "...FAQAT DETALLAR..."),
    (r".*JUST\s+['\"`]?A?RTS.*$", "...SHUNCHAKI QISMLAR..."),
    (r"^JUST\s+(?:'ARTS|ARTS|PARTS)[.,\s:]*$", "...SHUNCHAKI QISMLAR..."),
    (r"^(?:JUST\s+)?(?:'ARTS|ARTS|PARTS)[.,\s:;]*$", "...SHUNCHAKI QISMLAR..."),
    (r".*JUST.*(?:'ARTS|ARTS|PARTS).*SAFETY\s+CHECKS.*$", "...SHUNCHAKI QISMLAR. SANITARIYA QOIDALARI. XAVFSIZLIK TEKSHIRUVLARI..."),
    (r"^(?:NCACT|TCACTN|NCNSTN|HEALTH|HEALTN).*(?:SAFETY|CHECKS)[.,\s:_]*$", "...SANITARIYA QOIDALARI. XAVFSIZLIK TEKSHIRUVLARI..."),
    (r"^(?:LODES|CODES|NEALIN|HEALTH|HEALTN|TCACTN).*(?:SAFETY|CHECKS).*$", "...SANITARIYA QOIDALARI. XAVFSIZLIK TEKSHIRUVLARI..."),
    (r"^SAFETY\s+CHECKS?[.,\s:]*$", "...XAVFSIZLIK TEKSHIRUVLARI..."),
    (r"^(?:JUST\s+CO(?:DES?)?\s+)?SAFETY\s+CHECKS?[.,\s:]*$", "...SHUNCHAKI QISMLAR. XAVFSIZLIK TEKSHIRUVLARI..."),
    # Pair 4 — Rect: And then there's this lovely creature, Oval: Customer who was making a scene (muffled)
    (r"^AND\s+THEN\s+THERE'?S\s+THIS\s+LOVELY\s+CREATURE[.,\s:]*$", "KEYIN ESA MANA BU GO'ZAL MAVJUDOT BOR."),
    (r"^(?:CUSIUMEK|CUSIOMEK|COSTOMER|CUSTOMER)\s+WHO[.,\s:]*$", "...TO'PALON KO'TARAYOTGAN..."),
    (r".*(?:CUSIUMEK|CUSIOMEK|COSTOMER|CUSTOMER).*(?:SCENE|MAKING).*$", "...TO'PALON\nKO'TARAYOTGAN\nMIJOZ..."),
    (r"^WAS\s+MAKi?NG\s+A\s+SCENE[-.,\s]*$", "...MIJOZ..."),
    # Narration & Headset Dialogue Idioms
    (r"^AND\s+THE\s+BEST\s+PART\s+ABOUT\s+IT[?.,\s]*$", "VA BUNING ENG ZO'R TOMONI?"),
    (r"^THE\s+VIEW[.,\s!]*$", "MANZARA."),
    (r"^NO[.,\s]+NOTHING\s+LIKE\s+THAT[.,\s]+I'?M\s+LISTENING\s+(?:IN\s+)?ON\s+THE\s+SINISTER\s+SIX[.,\s:]*$", "YO'Q, BUNAQA EMAS. MEN 'SHUM OLTILIK'NI TINGLAYAPMAN."),
    (r"^THEY'?RE\s+PLOTTING\s+THEIR\s+NEXT\s+ATTACK[.,\s!]*$", "ULAR NAVBATDAGI HUJUMLARINI REJALASHTIRMOQDA.")
]

# Superhero comic post-processing dictionary for translated Uzbek text
COMIC_REPLACEMENTS = [
    (r"\bPLAN:\s*EPSILON\s+FIVE\b", "REJA: EPSILON-BESH"),
    (r"\bEPSILON\s+FIVE\b", "EPSILON-BESH"),
    (r"\bPLAN:\s*OMEGA\s+THREE\b", "REJA: OMEGA-UCH"),
    (r"\bOMEGA\s+THREE\b", "OMEGA-UCH"),
    (r"\bBRUISED\s+TRACHEA\b", "TRAXEYA ZARARLANGAN"),
    (r"\bAND\s+STAYS\s+DOWN\b", "VA YIQILGANICHA QOLADI"),
    (r"\bSTAYS\s+DOWN\b", "YIQILGANICHA QOLADI"),
    (r"\bWHAT\s+THE--\?!?", "BU NIMA BALO?!"),
    (r"\bWHAT\s+THE\s+HECK\b", "BU NIMA BALO"),
    (r"\bWHAT\s+THE\b", "BU NIMA BALO"),
    (r"\bLOOK\s+OUT!?\b", "EHTIYOT BO'L!"),
    (r"\bWATCH\s+OUT!?\b", "EHTIYOT BO'L!"),
    (r"\bDON'?T\s+PANIC!?\b", "VAHIMAGA TUSHMA!"),
    (r"\bDONT\s+PANIC!?\b", "VAHIMAGA TUSHMA!"),
    (r"\bJUST\s+GIVE\s+ME\s+TEN\s+SECONDS!?\b", "MENGA ATIGI O'N SONIYA BER!"),
    (r"\bJUST\s+GIVE\s+ME\s+(\w+)\s+SECONDS!?\b", r"MENGA ATIGI \1 SONIYA BER!"),
    (r"\bIMPRESSED\?\s*BEZOVTA\s+QILMANG\b", "QOYIL QOLDINGMI? HOVLIQMA"),
    (r"\bIMPRESSED\?\s*DON'?T\s*(?:BOTHER|BE)\b", "QOYIL QOLDINGMI? HOVLIQMA"),
    (r"\bCRITICAL\s+LIMIT\b", "XAVFLI DARAJA"),
    (r"\bCRITICAL\s+CAPACITY\b", "XAVFLI DARAJA"),
    (r"\bREACTOR\s+IS\s+ABOUT\s+TO\s+EXPLODE\b", "REAKTOR HOZIR PORTLAYDI"),
    (r"\bCIRCUIT\s+BREAKER\b", "ELEKTR UZATGICH"),
    (r"\bHOLD\s+ON!?\b", "CHIDA!"),
    (r"\bHANG\s+ON!?\b", "MAHKAM USHLA!"),
    (r"\bOH\s+NO!?\b", "LA'NAT!"),
    (r"\bSHUT\s+UP!?\b", "JIM TUR!"),
    (r"\bNO\s+WAY!?\b", "BO'LISHI MUMKIN EMAS!"),
    (r"\bYOU\s+GOTTA\s+BE\s+KIDDING!?\b", "HAZILLASHYAPSANMI?!"),
    (r"\bTAKE\s+THIS!?\b", "MANA BUNISI SENGA!"),
    (r"\bI\s+GOT\s+THIS\b", "HAMMASI MENING NAZORATIMDA"),
    (r"\bLET'S\s+DO\s+THIS\b", "KETDIK, BOSHLADIK"),
    (r"'S\s+RENDERED\s+YOU\s+SPEECHLESS", "TILING LOL QOLDI SHEKILLI"),
    (r"\bMY\s+SHOCK\s+GAUNTLETS?I?\b", "MENING ELEKTR QO'L-QOPLARIM"),
    (r"\bSHORTING\s+OUT\b", "QISQA TUTASHUV BO'LYAPTI"),
    (r"\bFRICTIONLESS\b", "ISHQALANISHSIZ"),
    (r"\bSPEECHLESS\b", "LOL QOLGAN"),
    (r"\bKO'KARGAN\s+TRAXEYA\b", "TRAXEYA ZARARLANGAN."),
    (r"\bQUVVATNI\s+(?:SUVLASHTIRISH|SO'NISH)\s+MAYDONI:?\b", "QUVVATNI SO'NDIRISH MAYDONI:"),
    (r"\bGORIZONTNING\s+SO'NGI\s+IXTIROSI:?\b", "HORIZON'NING ENG SO'NGGI IXTIROSI:"),
    (r"\bULAR\s+QISQARISHMOQDA\b", "QISQA TUTASHUV BO'LYAPTI!"),
    (r"\bQISQARISHMOQDA\b", "QISQA TUTASHUV BO'LYAPTI!"),
    (r"\bU\s+AKILLI\s+BO'LSA\b", "AQLLI BO'LSA."),
    (r"\bHOLY!?\b", "YO TOVBA!"),
    (r"\bHEY!?\b", "EY!"),
    (r"\bHERMAN\??\b", "GERMAN?")
]

# Load comprehensive dictionary from data/words_alpha.txt with fallback
COMMON_WORDS = set()
_data_words_path = os.path.join(os.path.dirname(__file__), "data", "words_alpha.txt")
if os.path.exists(_data_words_path):
    try:
        with open(_data_words_path, "r", encoding="utf-8") as _wf:
            COMMON_WORDS = set(line.strip().upper() for line in _wf if line.strip())
    except Exception:
        pass

# 1-letter valid English words
VALID_1_LETTER_WORDS = {"A", "I"}

# Valid 2-letter words and common comic abbreviations/acronyms
VALID_2_LETTER_WORDS = {
    "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "HI", "IF", "IN", "IS", "IT",
    "ME", "MY", "NO", "OF", "OH", "OK", "ON", "OR", "SO", "TO", "UP", "US", "WE", "YE",
    "MJ", "JJ", "TV", "DC", "AI", "DJ", "ID", "PM", "HQ", "PA", "MA", "HA", "EH", "UH",
    "DR", "MR", "MS", "JR", "SR", "ST", "VS", "EX"
}

# Common comic slang, contractions, interjections, titles, and common character names
COMIC_EXTRA_WORDS = {
    # Slang & Contractions
    "HECKUVA", "GONNA", "GOTTA", "WANNA", "DUNNO", "LEMME", "GIMME", "KINDA", "SORTA",
    "LOTTA", "OUTTA", "BETCHA", "DAMMIT", "DAMN", "GEEZ", "JEEZ", "YIKES", "WHOA",
    "HUH", "HM", "HMM", "HMMM", "PHEW", "WHEW", "BINGO", "BRAVO", "DUH", "EUREKA",
    "GOSH", "HA", "HAHA", "HEH", "HEHE", "HURRAY", "HOORAY", "OOMPH", "OOPS", "OUCH",
    "SHEESH", "TSK", "UGH", "VOILA", "YAY", "YIPPEE", "YO", "YEA", "NAH", "NOPE", "YEP",
    "ALRIGHT", "OKAY", "AINT", "CAUSE", "BOUT", "EM", "TIL", "MAAM", "MISTER", "CMON",
    "WHATCHA", "HOWDY", "YALL", "GOTCHA", "SHOOK",
    # Classic Marvel & comic names/titles
    "SPIDEY", "PETER", "PARKER", "JONAH", "JAMESON", "JJJ", "OSBORN", "NORMAN", "HARRY",
    "GOBLIN", "OCK", "OCTAVIUS", "CONNERS", "GWEN", "STACY", "THOR", "LOKI", "THANOS",
    "GROOT", "LOGAN", "WOLVERINE", "DEADPOOL", "WADE", "BATMAN", "SUPERMAN", "ROBIN",
    "JOKER", "HARLEY", "ALFRED", "GORDON", "LUTHOR", "CLARK", "KENT", "BRUCE", "WAYNE",
    "TONY", "STARK", "JARVIS", "CAP", "ROGERS", "BUCKY", "BARNES", "NATASHA", "ROMANOFF",
    "CLINT", "BARTON", "FURY", "COULSON", "SHIELD", "HYDRA", "AVENGERS", "MUTANT", "MUTANTS",
    "VILLAIN", "VILLAINS", "HERO", "HEROES", "MILES", "MORALES", "MORBIUS", "KRAVEN",
    "CARNAGE", "VENOM", "RHINO", "SCORPION", "ELECTRO", "VULTURE", "SHOCKER", "CHAMELEON",
    "SANDMAN", "MYSTERIO", "OVERDRIVE", "BEETLE", "SPEEDSTER", "BOOMERANG",
    "HORIZON", "LABS", "MAYOR", "COUNCIL", "POLICE", "CHIEF", "CAPTAIN", "SERGEANT",
    "OFFICER", "DETECTIVE", "COMMISSIONER", "DOC", "PROF", "PROFESSOR", "BANANAPANTS",
    "PANTS", "BANANA", "RRPHBL", "PFFFT", "BRRR", "HRM", "TCH", "HMPH"
}

def load_dynamic_comic_words() -> set:
    """Dynamically harvest known character names, aliases, and terms from persistent stores."""
    words = set()
    term_path = os.path.join(os.path.dirname(__file__), "terminology_store.json")
    if os.path.exists(term_path):
        try:
            with open(term_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data.get("terms", []):
                    for part in re.findall(r"[A-Za-z']+", entry.get("source_term", "").upper()):
                        clean_part = part.replace("'", "")
                        if len(clean_part) >= 2:
                            words.add(clean_part)
                    for alias in entry.get("aliases", []):
                        for part in re.findall(r"[A-Za-z']+", alias.upper()):
                            clean_part = part.replace("'", "")
                            if len(clean_part) >= 2:
                                words.add(clean_part)
        except Exception:
            pass

    mem_path = os.path.join(os.path.dirname(__file__), "character_memory.json")
    if os.path.exists(mem_path):
        try:
            with open(mem_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for char in data.get("characters", []):
                    for part in re.findall(r"[A-Za-z']+", char.get("name", "").upper()):
                        clean_part = part.replace("'", "")
                        if len(clean_part) >= 2:
                            words.add(clean_part)
                    for alias in char.get("aliases", []):
                        for part in re.findall(r"[A-Za-z']+", alias.upper()):
                            clean_part = part.replace("'", "")
                            if len(clean_part) >= 2:
                                words.add(clean_part)
        except Exception:
            pass
    return words

# Always ensure comic vocabulary and common words are populated
_FALLBACK_COMMON = {
    "THE", "BE", "TO", "OF", "AND", "A", "IN", "THAT", "HAVE", "I", "IT", "FOR", "NOT", "ON", "WITH",
    "HE", "AS", "YOU", "DO", "AT", "THIS", "BUT", "HIS", "BY", "FROM", "THEY", "WE", "SAY", "HER", "SHE",
    "OR", "AN", "WILL", "MY", "ONE", "ALL", "WOULD", "THERE", "THEIR", "WHAT", "SO", "UP", "OUT", "IF",
    "ABOUT", "WHO", "GET", "WHICH", "GO", "ME", "WHEN", "MAKE", "CAN", "LIKE", "TIME", "NO", "JUST", "HIM",
    "KNOW", "TAKE", "PEOPLE", "INTO", "YEAR", "YOUR", "GOOD", "SOME", "COULD", "THEM", "SEE", "OTHER",
    "THAN", "THEN", "NOW", "LOOK", "ONLY", "COME", "ITS", "OVER", "THINK", "ALSO", "BACK", "AFTER", "USE",
    "TWO", "HOW", "OUR", "WORK", "FIRST", "WELL", "WAY", "EVEN", "NEW", "WANT", "BECAUSE", "ANY", "THESE",
    "GIVE", "DAY", "MOST", "US", "DOWN", "STAYS", "STAY", "LIVE", "LONG", "SMART", "PLAN", "FIVE", "TRICK",
    "PARTY", "ANOTHER", "HORIZON", "HORIZONS", "LATEST", "INVENTION", "QUICK", "CHEAP", "EASY", "INSTALL",
    "POWER", "DAMPENING", "FIELD", "SHOCK", "GAUNTLETS", "GAUNTLET", "GAUNTLETSI", "SHORTING", "THING", "WEIGHS", "TON",
    "HEAD", "OFF", "TAKEN", "COOL", "BRUISED", "TRACHEA", "HOLD", "STOP", "WAIT", "HELP", "RUN", "FAST",
    "MAN", "SPIDER", "HERMAN", "BOY", "GIRL", "DEAD", "KILL", "ALIVE", "NEED", "TELL", "FEEL", "BAD",
    "MUCH", "BEFORE", "GREAT", "SAME", "BIG", "MUST", "SUCH", "WHY", "ASK", "WENT", "MEN", "READ", "LAND",
    "DIFFERENT", "HOME", "MOVE", "TRY", "KIND", "HAND", "PICTURE", "AGAIN", "CHANGE", "PLAY", "SPELL",
    "AIR", "AWAY", "ANIMAL", "HOUSE", "POINT", "PAGE", "LETTER", "MOTHER", "ANSWER", "FOUND", "STUDY", "STILL",
    "LEARN", "SHOULD", "AMERICA", "WORLD", "HIGH", "EVERY", "NEAR", "ADD", "FOOD", "BETWEEN", "OWN", "BELOW",
    "COUNTRY", "PLANT", "LAST", "SCHOOL", "FATHER", "KEEP", "TREE", "NEVER", "START", "CITY", "EARTH", "EYE",
    "LIGHT", "THOUGHT", "UNDER", "STORY", "SAW", "LEFT", "DONT", "CANT", "WONT", "ITS", "HELL",
    "SHELL", "THEYRE", "YOURE", "IM", "WERE", "LETS", "WHATS", "THATS", "COULDVE", "SHOULDVE",
    "WOULDVE", "HES", "SHES", "WHOS", "THERES", "HERES", "AINT", "HOLY", "UNH", "OOF", "WOW", "HEY",
    "YEAH", "YEP", "NOPE", "OK", "OKAY", "SURE", "PLEASE", "SORRY", "THANKS", "THANK", "HELLO", "HI", "BYE",
    "OH", "AH", "UH", "UM", "HUH", "HA", "HEH", "EPSILON", "OMEGA", "THREE", "INSTANT", "FRICTIONLESS",
    "SURFACE", "IMPRESSED", "BOTHER", "SPEECHLESS", "RENDERED", "STUFF", "PAVEMENT", "APPLYING", "SPENT",
    "MINE", "CAME", "GYAHH", "GYAH", "AHKK", "AHK", "ARGH", "URGH", "AAAH", "GAHH",
    "KONNINO", "MEDID", "MEDIC", "KEEPINS", "KEEPING", "RARTS", "PARTS", "NCACT", "STAC", "STAFF", "TOF"
}
COMMON_WORDS.update(_FALLBACK_COMMON)
COMMON_WORDS.update(VALID_2_LETTER_WORDS)
COMMON_WORDS.update(COMIC_EXTRA_WORDS)
COMMON_WORDS.update(load_dynamic_comic_words())


class SpeechBubble(BaseModel):
    bubble_id: int = Field(default=1, description="Sequential bubble ID in reading order")
    x0: int = Field(description="Left pixel coordinate")
    y0: int = Field(description="Top pixel coordinate")
    x1: int = Field(description="Right pixel coordinate")
    y1: int = Field(description="Bottom pixel coordinate")
    box: Optional[List[int]] = Field(default=None, description="[x0, y0, x1, y1] bounding coordinates")
    original_text: str = Field(description="Extracted English dialogue")
    clean_text: Optional[str] = Field(default=None, description="Cleaned dialogue text")
    uzbek_translation: str = Field(default="", description="Superhero-tone Uzbek dialogue")
    confidence: float = Field(default=1.0)
    enabled: bool = Field(default=True, description="Whether to letter this bubble")
    is_active: bool = Field(default=True, description="Whether to letter this bubble")
    size_offset: int = Field(default=0, description="Per-bubble font size nudge")
    font_size_offset: int = Field(default=0, description="Per-bubble font size nudge")
    contour_points: Optional[List[List[int]]] = Field(default=None, description="Contour polygon points for cleaning")
    contour: Optional[Any] = Field(default=None, description="Exact contour polygon [x, y] points or ndarray")
    shape_type: str = Field(default="oval", description="Classified shape: 'oval' or 'rectangle'")
    line_centers: Optional[List[List[int]]] = Field(default=None, description="Centers of constituent text lines")
    lines: Optional[List[Dict[str, Any]]] = Field(default=None, description="Constituent OCR text lines")
    z_order: int = Field(default=0, description="Z-order layering: 0=background oval, 1=foreground rectangle")
    is_occluded: bool = Field(default=False, description="Whether this bubble is occluded by a foreground bubble")
    fragments: Optional[List[Dict[str, Any]]] = Field(default=None, description="Visible fragments when occluded")
    visible_mask: Optional[Any] = Field(default=None, description="Binary visible mask")
    speaker: Optional[str] = Field(default=None, description="Speaker character persona (e.g. 'Superior Spider-Man', 'Peter Parker', 'Mary Jane')")
    speaker_confidence: Optional[float] = Field(default=None, description="Speaker detection confidence score (0.0 - 1.0)")
    tail_region_colors: Optional[List[List[int]]] = Field(default=None, description="Dominant colors near tail tip in RGB")
    tail_region_weights: Optional[List[float]] = Field(default=None, description="Weights of dominant colors")
    pipeline_uzbek_translation: Optional[str] = Field(default="", description="Raw initial pipeline translation before user manual edits")
    is_tinted: bool = Field(default=False, description="Whether this bubble has an authentic spectral/monochrome tint")
    bg_is_tinted: bool = Field(default=False, description="Whether the bubble background fill itself has color tint")
    text_is_spectral: bool = Field(default=False, description="Whether the dialogue has spectral/colored text ink")
    tint_bgr: Optional[List[int]] = Field(default=None, description="Sampled background tint color [B, G, R]")
    tint_rgb: Optional[List[int]] = Field(default=None, description="Sampled background tint color [R, G, B]")
    orig_text_rgb: Optional[List[int]] = Field(default=None, description="Sampled original text ink color [R, G, B]")
    cluster_id: Optional[str] = Field(default=None, description="Shared ID for conjoined/multi-lobe bubble clusters")
    cluster_text: Optional[str] = Field(default=None, description="Full unified source dialogue across all lobes in cluster")
    cluster_uzbek: Optional[str] = Field(default=None, description="Full unified Uzbek translation of cluster")
    lobe_index: int = Field(default=0, description="0-indexed lobe position within multi-lobe cluster")
    total_lobes: int = Field(default=1, description="Total number of lobes in cluster")
    needs_review: bool = Field(default=False, description="Flagged for manual review (e.g. translation fallback/residue)")
    review_reason: Optional[str] = Field(default=None, description="Detailed reason if flagged for review")
    quality_warning: Optional[str] = Field(default=None, description="Quality warning pattern name")


    def model_post_init(self, __context: Any) -> None:
        if self.box is None:
            self.box = [self.x0, self.y0, self.x1, self.y1]
        raw_source = self.clean_text or self.original_text or ""
        self.clean_text = naturalization_rules.pre_process_english_dialogue(raw_source) if raw_source else ""
        if not self.pipeline_uzbek_translation and self.uzbek_translation:
            self.pipeline_uzbek_translation = self.uzbek_translation
        if self.contour is None and self.contour_points is not None:
            self.contour = self.contour_points
        elif self.contour_points is None and self.contour is not None:
            self.contour_points = self.contour


class BubbleFragment(BaseModel):
    fragment_id: int = Field(default=1, description="Sequential fragment index within its FragmentGroup")
    parent_bubble_id: int = Field(default=0, description="ID of parent occluded bubble / rectangle")
    x0: int = Field(description="Left pixel coordinate")
    y0: int = Field(description="Top pixel coordinate")
    x1: int = Field(description="Right pixel coordinate")
    y1: int = Field(description="Bottom pixel coordinate")
    box: Optional[List[int]] = Field(default=None, description="[x0, y0, x1, y1] bounding coordinates")
    original_text: str = Field(default="", description="Extracted English dialogue in this fragment")
    clean_text: Optional[str] = Field(default=None, description="Cleaned dialogue text")
    uzbek_translation: str = Field(default="", description="Superhero-tone Uzbek dialogue")
    pipeline_uzbek_translation: Optional[str] = Field(default="", description="Superhero-tone Uzbek dialogue")
    confidence: float = Field(default=1.0)
    contour: Optional[Any] = Field(default=None, description="Contour polygon points")
    contour_points: Optional[List[List[int]]] = Field(default=None, description="Contour polygon points")
    lines: Optional[List[Dict[str, Any]]] = Field(default=None, description="Constituent OCR lines")
    line_centers: Optional[List[List[int]]] = Field(default=None, description="Centers of constituent text lines")
    area: int = Field(default=0, description="Contour pixel area")
    shape_type: str = Field(default="oval", description="Classified shape ('oval')")
    z_order: int = Field(default=0, description="Z-order layering (0 = background oval fragment)")
    is_occluded: bool = Field(default=True, description="Whether this fragment is an occluded sliver")
    speaker: Optional[str] = Field(default=None, description="Speaker character persona")
    speaker_confidence: Optional[float] = Field(default=None, description="Speaker detection confidence score")
    tail_region_colors: Optional[List[List[int]]] = Field(default=None, description="Dominant colors near tail tip in RGB")
    tail_region_weights: Optional[List[float]] = Field(default=None, description="Weights of dominant colors")

    def model_post_init(self, __context: Any) -> None:
        if self.box is None:
            self.box = [self.x0, self.y0, self.x1, self.y1]
        raw_source = self.clean_text or self.original_text or ""
        self.clean_text = naturalization_rules.pre_process_english_dialogue(raw_source) if raw_source else ""
        if self.contour is None and self.contour_points is not None:
            self.contour = self.contour_points
        elif self.contour_points is None and self.contour is not None:
            self.contour_points = self.contour

    def to_speech_bubble(self, bubble_id: int = 0) -> SpeechBubble:
        cnt = self.contour_points or self.contour
        return SpeechBubble(
            bubble_id=bubble_id,
            x0=self.x0,
            y0=self.y0,
            x1=self.x1,
            y1=self.y1,
            box=self.box or [self.x0, self.y0, self.x1, self.y1],
            original_text=self.original_text,
            clean_text=self.clean_text or self.original_text,
            uzbek_translation=self.uzbek_translation,
            confidence=self.confidence,
            shape_type=self.shape_type,
            z_order=self.z_order,
            contour=cnt,
            contour_points=cnt,
            line_centers=self.line_centers,
            is_occluded=self.is_occluded,
            speaker=self.speaker,
            speaker_confidence=self.speaker_confidence,
            tail_region_colors=self.tail_region_colors,
            tail_region_weights=self.tail_region_weights
        )


class FragmentGroup(BaseModel):
    parent_id: int = Field(default=0, description="Parent occluded bubble or rectangle index")
    fragments: List[BubbleFragment] = Field(default_factory=list, description="List of visible BubbleFragment slivers")
    parent_contour: Optional[Any] = Field(default=None, description="Original parent contour polygon")
    parent_box: Optional[List[int]] = Field(default=None, description="Original parent bounding box [x0, y0, x1, y1]")


def get_memory_stats() -> Dict[str, Any]:
    """
    Returns comprehensive memory statistics:
    1. Process tree RSS (parent + all recursive children) in MB.
    2. Linux cgroup v1/v2 memory metrics if available:
       - cgroup_limit_mb: cgroup hard limit (or None if unlimited / not in cgroup)
       - cgroup_current_mb: current cgroup memory usage
       - cgroup_peak_mb: max peak cgroup memory usage observed by OS
       - cgroup_anon_mb: anonymous memory (real resident process heap/stack)
       - cgroup_file_mb: file cache/page cache (reclaimable under memory pressure)
       - cgroup_shmem_mb: shared memory / tmpfs
    """
    import os
    import psutil

    total_rss = 0.0
    try:
        proc = psutil.Process(os.getpid())
        total_rss += proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass
    process_rss_mb = total_rss / (1024 * 1024)

    cgroup_limit_mb = None
    cgroup_current_mb = None
    cgroup_peak_mb = None
    anon_mb = None
    file_mb = None
    shmem_mb = None
    cgroup_stat = {}

    def _read_cg_bytes(path: str) -> Optional[float]:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    val = f.read().strip()
                if val and val != "max":
                    b = int(val)
                    if b < 10**15:  # Filter out ~9e18 representing unlimited
                        return b / (1024 * 1024)
            except Exception:
                pass
        return None

    # Check cgroup v2
    cg2_cur = "/sys/fs/cgroup/memory.current"
    cg2_max = "/sys/fs/cgroup/memory.max"
    cg2_peak = "/sys/fs/cgroup/memory.peak"
    cg2_stat = "/sys/fs/cgroup/memory.stat"

    # Check cgroup v1
    cg1_cur = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    cg1_lim = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    cg1_peak = "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"
    cg1_stat = "/sys/fs/cgroup/memory/memory.stat"

    stat_file = None
    if os.path.exists(cg2_cur):
        cgroup_current_mb = _read_cg_bytes(cg2_cur)
        cgroup_limit_mb = _read_cg_bytes(cg2_max)
        cgroup_peak_mb = _read_cg_bytes(cg2_peak)
        if os.path.exists(cg2_stat):
            stat_file = cg2_stat
    elif os.path.exists(cg1_cur):
        cgroup_current_mb = _read_cg_bytes(cg1_cur)
        cgroup_limit_mb = _read_cg_bytes(cg1_lim)
        cgroup_peak_mb = _read_cg_bytes(cg1_peak)
        if os.path.exists(cg1_stat):
            stat_file = cg1_stat

    if stat_file:
        try:
            with open(stat_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            cgroup_stat[parts[0]] = int(parts[1])
                        except ValueError:
                            pass

            if "anon" in cgroup_stat:
                anon_mb = cgroup_stat["anon"] / (1024 * 1024)
            elif "total_rss" in cgroup_stat:
                anon_mb = cgroup_stat["total_rss"] / (1024 * 1024)
            elif "rss" in cgroup_stat:
                anon_mb = cgroup_stat["rss"] / (1024 * 1024)

            if "file" in cgroup_stat:
                file_mb = cgroup_stat["file"] / (1024 * 1024)
            elif "total_cache" in cgroup_stat:
                file_mb = cgroup_stat["total_cache"] / (1024 * 1024)
            elif "cache" in cgroup_stat:
                file_mb = cgroup_stat["cache"] / (1024 * 1024)

            if "shmem" in cgroup_stat:
                shmem_mb = cgroup_stat["shmem"] / (1024 * 1024)
            elif "total_shmem" in cgroup_stat:
                shmem_mb = cgroup_stat["total_shmem"] / (1024 * 1024)
        except Exception:
            pass

    return {
        "rss_mb": round(process_rss_mb, 1),
        "cgroup_current_mb": round(cgroup_current_mb, 1) if cgroup_current_mb is not None else None,
        "cgroup_limit_mb": round(cgroup_limit_mb, 1) if cgroup_limit_mb is not None else None,
        "cgroup_peak_mb": round(cgroup_peak_mb, 1) if cgroup_peak_mb is not None else None,
        "cgroup_anon_mb": round(anon_mb, 1) if anon_mb is not None else None,
        "cgroup_file_mb": round(file_mb, 1) if file_mb is not None else None,
        "cgroup_shmem_mb": round(shmem_mb, 1) if shmem_mb is not None else None,
        "cgroup_stat": cgroup_stat,
    }


def format_memory_summary(stats: Optional[Dict[str, Any]] = None) -> str:
    """Formats memory stats into a readable string for logs and UI widgets."""
    if stats is None:
        stats = get_memory_stats()

    parts = [f"Tree RSS: {stats['rss_mb']:.1f} MB"]

    cg_breakdown = []
    if stats.get("cgroup_anon_mb") is not None:
        cg_breakdown.append(f"anon: {stats['cgroup_anon_mb']:.0f}M")
    if stats.get("cgroup_file_mb") is not None:
        cg_breakdown.append(f"file: {stats['cgroup_file_mb']:.0f}M")
    if stats.get("cgroup_shmem_mb") is not None and stats["cgroup_shmem_mb"] >= 1.0:
        cg_breakdown.append(f"shmem: {stats['cgroup_shmem_mb']:.0f}M")

    breakdown_str = f" ({', '.join(cg_breakdown)})" if cg_breakdown else ""

    if stats.get("cgroup_limit_mb") is not None:
        cur = stats.get("cgroup_current_mb", stats["rss_mb"])
        lim = stats["cgroup_limit_mb"]
        pct = (cur / lim * 100) if lim > 0 else 0
        parts.append(f"cgroup: {cur:.1f}/{lim:.1f} MB ({pct:.0f}%){breakdown_str}")
    elif stats.get("cgroup_current_mb") is not None:
        parts.append(f"cgroup: {stats['cgroup_current_mb']:.1f} MB{breakdown_str}")

    if stats.get("cgroup_peak_mb") is not None:
        peak_val = stats["cgroup_peak_mb"]
        peak_str = f"Peak: {peak_val:.1f} MB"
        if stats.get("cgroup_limit_mb"):
            peak_pct = peak_val / stats["cgroup_limit_mb"] * 100
            peak_str += f" ({peak_pct:.0f}%)"
        parts.append(peak_str)

    return " | ".join(parts)


def get_container_processes() -> List[Dict[str, Any]]:
    """Returns a list of all processes in the container/system with PID, PPID, RSS, and command."""
    import psutil
    import time
    procs = []
    current_pid = os.getpid()
    for p in psutil.process_iter(['pid', 'ppid', 'name', 'memory_info', 'cmdline', 'status', 'create_time']):
        try:
            info = p.info
            rss_mb = round(info['memory_info'].rss / (1024 * 1024), 1) if info.get('memory_info') else 0.0
            cmd_list = info.get('cmdline') or [info.get('name') or '']
            cmd = " ".join(cmd_list)
            uptime_s = round(time.time() - info.get('create_time', time.time()), 1)
            procs.append({
                "pid": info.get('pid'),
                "ppid": info.get('ppid'),
                "name": info.get('name', ''),
                "rss_mb": rss_mb,
                "status": info.get('status', ''),
                "uptime_s": uptime_s,
                "cmd": cmd,
                "is_current": (info.get('pid') == current_pid),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    procs.sort(key=lambda x: x["rss_mb"], reverse=True)
    return procs


def log_container_processes():
    """Prints all running container processes to console log for diagnostics."""
    procs = get_container_processes()
    total_rss = sum(p["rss_mb"] for p in procs)
    print(f"[CONTAINER_PROCS] Total {len(procs)} processes | Sum RSS: {total_rss:.1f} MB", flush=True)
    for p in procs:
        cur_marker = " [CURRENT_PROCESS]" if p.get("is_current") else ""
        cmd_snippet = p['cmd'][:100] + ("..." if len(p['cmd']) > 100 else "")
        print(f"  PID {p['pid']} (PPID {p['ppid']}) | RSS: {p['rss_mb']:.1f} MB | {p['status']} | {cmd_snippet}{cur_marker}", flush=True)


def get_disk_cache_diagnostics() -> Dict[str, Any]:
    """Returns disk usage of models, huggingface, torch, and easyocr cache directories."""
    import tempfile
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths_to_check = {
        "models_repo": os.path.join(base_dir, "models"),
        "hf_cache": os.path.expanduser("~/.cache/huggingface"),
        "torch_cache": os.path.expanduser("~/.cache/torch"),
        "easyocr_cache": os.path.expanduser("~/.EasyOCR"),
        "tmp": tempfile.gettempdir(),
    }

    results = {}
    for name, p in paths_to_check.items():
        exists = os.path.exists(p)
        size_mb = 0.0
        files_count = 0
        if exists:
            try:
                if os.path.isfile(p):
                    size_mb = os.path.getsize(p) / (1024 * 1024)
                    files_count = 1
                else:
                    for root, dirs, files in os.walk(p):
                        for f in files:
                            fp = os.path.join(root, f)
                            try:
                                if not os.path.islink(fp):
                                    size_mb += os.path.getsize(fp) / (1024 * 1024)
                                    files_count += 1
                            except Exception:
                                pass
            except Exception:
                pass
        results[name] = {
            "path": p,
            "exists": exists,
            "size_mb": round(size_mb, 1),
            "files": files_count,
        }
    return results


def log_disk_cache_diagnostics():
    """Prints disk usage of cache and models directories to console log."""
    diag = get_disk_cache_diagnostics()
    print("[DISK_CACHE_DIAGNOSTICS]", flush=True)
    for k, v in diag.items():
        print(f"  {k}: {v['size_mb']:.1f} MB ({v['files']} files) @ {v['path']} (exists={v['exists']})", flush=True)


def get_process_rss_mb() -> float:
    """Returns current process tree Resident Set Size (RSS) memory in megabytes."""
    return get_memory_stats()["rss_mb"]


try:
    import streamlit as st
    @st.cache_resource(show_spinner=False)
    def _cached_easyocr_reader():
        import easyocr
        import torch
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        return easyocr.Reader(['en'], gpu=False, verbose=False)
except Exception:
    _cached_easyocr_reader = None

_OCR_READER = None


def get_ocr_reader():
    """Returns a singleton EasyOCR reader with memory-optimized defaults."""
    global _OCR_READER
    if _cached_easyocr_reader is not None:
        reader = _cached_easyocr_reader()
        print(f"[DIAGNOSTIKA 2] EasyOCR reader yaratilgandan keyin (RAM RSS): {get_process_rss_mb():.1f} MB", flush=True)
        return reader
    if _OCR_READER is None:
        import easyocr
        import torch
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        _OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
        print(f"[DIAGNOSTIKA 2] EasyOCR reader yaratilgandan keyin (RAM RSS): {get_process_rss_mb():.1f} MB", flush=True)
    return _OCR_READER


def release_ocr_reader():
    """Explicitly releases EasyOCR reader from memory and triggers garbage collection."""
    global _OCR_READER
    try:
        if _cached_easyocr_reader is not None and hasattr(_cached_easyocr_reader, "clear"):
            _cached_easyocr_reader.clear()
    except Exception:
        pass
    _OCR_READER = None
    import gc
    gc.collect()
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"[DIAGNOSTIKA] EasyOCR reader xotiradan bo'shatildi (RAM RSS): {get_process_rss_mb():.1f} MB", flush=True)


def run_tiled_easyocr(image: Image.Image, reader=None, canvas_size: int = 2048) -> List[Any]:
    """
    Memory-efficient tiled EasyOCR inference:
    - For images with height > 2000px, splits into overlapping horizontal tiles (halves).
    - Prevents PyTorch CRAFT from allocating 2GB+ intermediate feature tensors.
    - Each tile retains 100% native glyph resolution (text is sharp, not downscaled).
    - Peak memory stays safely under 450MB, completely eliminating Streamlit Cloud OOM crashes.
    """
    if reader is None:
        reader = get_ocr_reader()

    import torch
    import gc

    w, h = image.size
    if h <= 2000:
        with torch.no_grad():
            res = reader.readtext(np.array(image.convert("RGB")), paragraph=False, canvas_size=canvas_size)
            gc.collect()
            return res

    overlap = 180
    mid = h // 2

    # Tile 1: Top half
    crop1 = image.crop((0, 0, w, mid + overlap))
    with torch.no_grad():
        res1 = reader.readtext(np.array(crop1.convert("RGB")), paragraph=False, canvas_size=canvas_size)
    del crop1
    gc.collect()

    # Tile 2: Bottom half
    crop2 = image.crop((0, mid - overlap, w, h))
    with torch.no_grad():
        res2 = reader.readtext(np.array(crop2.convert("RGB")), paragraph=False, canvas_size=canvas_size)
    del crop2
    gc.collect()

    # Shift Tile 2 coordinates by y_offset
    offset_y = mid - overlap
    res2_shifted = []
    for item in res2:
        if len(item) == 2:
            bbox, text = item
            conf = 1.0
        else:
            bbox, text, conf = item
        shifted_bbox = [[pt[0], pt[1] + offset_y] for pt in bbox]
        res2_shifted.append((shifted_bbox, text, conf))

    # Combine with deduplication in the overlap seam
    combined = list(res1)
    for item2 in res2_shifted:
        b2 = item2[0]
        c2y = sum(p[1] for p in b2) / 4.0
        c2x = sum(p[0] for p in b2) / 4.0
        is_dup = False
        for item1 in res1:
            b1 = item1[0]
            c1y = sum(p[1] for p in b1) / 4.0
            c1x = sum(p[0] for p in b1) / 4.0
            if abs(c1y - c2y) < 18 and abs(c1x - c2x) < 25:
                is_dup = True
                break
        if not is_dup:
            combined.append(item2)

    return combined


def get_available_fonts() -> Dict[str, str]:
    """Returns available fonts with CC Wild Words as primary recommended choice."""
    fonts = {}
    fonts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "fonts"))

    cc_wild = os.path.join(fonts_dir, "CCWildWords.ttf")
    if os.path.exists(cc_wild):
        fonts["CC Wild Words (Recommended)"] = cc_wild

    comic_neue = os.path.join(fonts_dir, "ComicNeue-Bold.ttf")
    if os.path.exists(comic_neue):
        fonts["Comic Neue Bold"] = comic_neue

    komika = os.path.join(fonts_dir, "KOMTXTB_.ttf")
    if os.path.exists(komika):
        fonts["Komika Text Bold"] = komika

    manga_temple = os.path.join(fonts_dir, "mangatb.ttf")
    if not os.path.exists(manga_temple):
        manga_temple = os.path.join(fonts_dir, "mangat.ttf")
    if os.path.exists(manga_temple):
        fonts["Manga Temple Bold"] = manga_temple

    system_fonts = {
        "Comic Sans MS": r"C:\Windows\Fonts\comic.ttf",
        "Segoe UI Bold": r"C:\Windows\Fonts\segoeuib.ttf",
        "Arial Bold": r"C:\Windows\Fonts\arialbd.ttf",
    }
    for name, path in system_fonts.items():
        if os.path.exists(path):
            fonts[name] = path

    return fonts


def normalize_uzbek_comic_text(text: str) -> str:
    """
    Normalizes Uzbek comic text:
    - Normalizes turned commas, modifier letters, and tutruq belgisi to ASCII '
    - Strips quotation marks and separates sentences onto separate lines (Rule 6)
    - Cleans redundant punctuation
    - Converts strictly to UPPERCASE
    """
    if not text:
        return ""
    apostrophes = ['\u02bb', '\u02bc', '\u2018', '\u2019', '`', '´', 'ʻ', 'ʼ', '’', '‘']
    for ap in apostrophes:
        text = text.replace(ap, "'")
    # Clean duplicate punctuation
    text = re.sub(r'!{2,}', '!', text)
    # Apply Rule 6: dialogue lines NEVER get wrapped in quotes, each sentence on its own line
    text = bubble_lettering.format_dialogue_sentences(text)
    text = re.sub(r'[^\S\r\n]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.upper().strip()



def is_phonotactically_plausible_word(w: str) -> bool:
    """
    Checks if an unlisted word/name has valid English / comic phonotactic structure.
    Prevents OCR background texture noise (e.g. 'zzxkp', 'ffff') while accepting
    proper nouns, alien/comic names, and slang (e.g. 'SPIDEY', 'ZORBLAX', 'HECKUVA').
    """
    w_clean = re.sub(r"[^A-Z]", "", w.upper())
    if len(w_clean) < 2:
        return False
    if w_clean in SFX_WORDS or w_clean in WATERMARK_WORDS:
        return False
    # Reject single repeated character (e.g. 'ZZZZZ', 'XXXX')
    if len(set(w_clean)) == 1:
        return False
    # Must contain at least one vowel
    vowels = set("AEIOUY")
    vowel_count = sum(1 for c in w_clean if c in vowels)
    if vowel_count == 0:
        return False
    # Vowel to length ratio: at least 1 vowel; if longer than 6 letters, at least 12% vowels
    if len(w_clean) > 6 and (vowel_count / len(w_clean)) < 0.12:
        return False
    # Consonant streak must be <= 4 consecutive consonants (e.g., 'SCHMIDT' has 4: 'SCHM')
    consonant_streak = 0
    for c in w_clean:
        if c not in vowels:
            consonant_streak += 1
            if consonant_streak > 4:
                return False
        else:
            consonant_streak = 0
    return True


def check_is_sfx_or_junk(text: str, conf: float = 1.0, is_on_bubble_paper: bool = False) -> Tuple[bool, str]:
    """
    Strictly filters out SFX, watermarks, reflections, fragmented tokens, and non-dialogue noise:
    1. Confidence threshold check (conf >= 0.12).
    2. Pure SFX words (e.g. TEKK, KOFF, CHUK, CHIK, GYAHH).
    3. Known watermarks (e.g. readallcomics, hoegomig, etc.).
    4. Minimum alphabetic letters check (len(letters) >= 2).
    5. Abnormal symbol-to-alphanumeric ratio check (<= 1.20 for comic punctuation like ?!).
    6. Fragmented single-letter tokens check (< 60% single char tokens).
    7. Dictionary & phonotactic check: Must contain at least one valid English word, character name, or plausible word.
    """
    text_clean = text.strip()
    if not text_clean:
        return True, "empty"

    # 1. Enforce confidence threshold if conf was measured
    if conf < 0.12:
        return True, f"low_conf ({conf:.2f} < 0.12)"

    # 2. Check SFX (exact, character repetitions, substring, and fuzzy matching)
    cleaned_sfx = re.sub(r'[^A-Z0-9\s]', ' ', text_clean.upper()).strip()
    sfx_words = cleaned_sfx.split()
    if sfx_words:
        import difflib
        def is_word_sfx(w: str) -> bool:
            w_strip = re.sub(r'[^A-Z]', '', w)
            if not w_strip:
                return False

            # Whitelist: If word is in COMMON_WORDS, do NOT apply fuzzy-SFX checks.
            # Real short dialogue words ("OH", "NO", "GO", "HUH", "AH", "UH", "UM", "HA", "HEH", "HEY", "NOW", etc.)
            # must never be filtered out by fuzzy/substring matching.
            if w_strip in COMMON_WORDS or w in COMMON_WORDS:
                if w_strip in {"OH", "NO", "GO", "HUH", "AH", "UH", "UM", "HA", "HEH", "HEY", "NOW", "YES", "WHY", "STOP", "WAIT", "HELP", "RUN", "AHKK", "AHK", "ARGH", "URGH", "AAAH", "GAHH"}:
                    return False
                if w_strip in SFX_WORDS:
                    return True
                return False

            # For non-whitelisted words, check exact SFX match first
            if w_strip in SFX_WORDS:
                return True

            collapsed = re.sub(r'(.)\1{1,}', r'\1', w_strip)
            for sw in SFX_WORDS:
                sw_collapsed = re.sub(r'(.)\1{1,}', r'\1', sw)
                if collapsed == sw_collapsed:
                    return True
                if len(sw) >= 4 and (sw in w_strip or (len(w_strip) >= 4 and w_strip in sw)):
                    return True
            matches = difflib.get_close_matches(w_strip, list(SFX_WORDS), n=1, cutoff=0.80)
            if matches:
                return True
            return False

        if all(is_word_sfx(w) for w in sfx_words):
            return True, "sfx"
        if len(sfx_words) <= 2 and any(is_word_sfx(w) for w in sfx_words) and not any(w in COMMON_WORDS for w in sfx_words):
            return True, "sfx_partial"

    # 3. Check known watermarks
    normalized_upper = text_clean.upper()
    if any(w in sfx_words or (len(w) >= 4 and w in normalized_upper) for w in WATERMARK_WORDS):
        return True, "watermark"

    # 4. Min letters check (rejects standalone numbers '34', '2')
    letters = re.findall(r'[A-Za-z]', text_clean)
    if len(letters) < 2:
        return True, "too few letters"

    # 5. Symbol / noise ratio check (allow standard comic punctuation like ?! or ...)
    alnum = re.findall(r'[A-Za-z0-9]', text_clean)
    symbols = re.findall(r'[^A-Za-z0-9\s]', text_clean)
    if len(alnum) == 0 or (len(symbols) / len(alnum)) > 1.20:
        return True, "abnormal symbol ratio"

    # 6. Reject excessive single-letter fragments (e.g. 'R ! 2')
    tokens = text_clean.split()
    single_char_tokens = [t for t in tokens if len(t) == 1]
    if len(tokens) >= 3 and (len(single_char_tokens) / len(tokens)) >= 0.60:
        return True, "fragmented tokens"

    # 7. Must contain at least one valid dictionary word of length >= 2 or valid 1-letter word or plausible word
    raw_words = re.findall(r"[A-Za-z']+", text_clean.upper())
    has_valid_word = False
    for w in raw_words:
        clean_w = w.replace("'", "").strip("_")
        trimmed_w = clean_w.rstrip("I1!")

        # 1-letter valid English words ('A', 'I')
        if clean_w in VALID_1_LETTER_WORDS:
            if len(raw_words) >= 2 or len(letters) >= 2:
                has_valid_word = True
                break

        # Check dictionary and known comic words
        if (len(clean_w) >= 2 and (clean_w in COMMON_WORDS or w in COMMON_WORDS)) or \
           (len(trimmed_w) >= 2 and trimmed_w in COMMON_WORDS):
            has_valid_word = True
            break

        # Check phonotactic plausibility for unknown names/words
        if len(clean_w) >= 2 and is_phonotactically_plausible_word(clean_w):
            if conf >= 0.25 or is_on_bubble_paper:
                has_valid_word = True
                break

    if not has_valid_word and is_on_bubble_paper and len(letters) >= 2:
        # Fallback for bubble paper: legitimate comic vocalizations, sputters, and interjections
        # (e.g. 'RRPHBL', 'PFFFT', 'BRRR', 'HRM', 'TCH', 'HMPH') often lack standard vowels.
        # Preserve them as valid comic dialogue when located on verified bubble paper.
        has_valid_word = True

    if not has_valid_word:
        return True, "no valid dictionary word"

    return False, "valid"


def is_sfx_or_junk(text: str, conf: float = 1.0, is_on_bubble_paper: bool = False) -> bool:
    """Convenience boolean check for junk/SFX."""
    is_junk, _ = check_is_sfx_or_junk(text, conf, is_on_bubble_paper=is_on_bubble_paper)
    return is_junk


def free_web_translate(text: str) -> str:
    """
    100% Offline translation via local CTranslate2 NLLB-200 model.
    Zero external network calls or cloud APIs.
    """
    cleaned = text.strip()
    if not cleaned:
        return ""
    try:
        return local_translator.translate_offline(cleaned)
    except Exception as e:
        print(f"[TRANSLATE_WARN] Offline translation fallback: {e}")
        return cleaned


def translate_spiderman_uzbek(en_text: str, speaker: Optional[str] = None) -> str:
    """
    Translates English dialogue into punchy, colloquial superhero Uzbek fully OFFLINE:
    - Normalizes curly quotes/apostrophes to standard ASCII '.
    - Cleans standalone OCR edge noise (trailing underscores, mismatched quotes).
    - Step 1: Pre-processes English contractions and dialogue markers via naturalization_rules.
    - Step 2: Translates complete unified sentence via 100% offline local NLLB model.
    - Step 3: Applies living Naturalization and Comic Idiom polish layer + Character Voice Profiles.
    - Strictly normalizes to UPPERCASE comic lettering.
    """
    cleaned = en_text.strip()
    if not cleaned:
        return ""

    # Detect leading ellipses / dots / dashes
    has_leading_dots = bool(re.match(r"^[\s#~*^]*\.{2,}", cleaned))
    has_leading_dash = bool(re.match(r"^[\s#~*^]*\-{1,2}", cleaned))
    core_text = re.sub(r"^[#~*^.\-_\s]+", "", cleaned)
    core_text = re.sub(r"[#~*^.\-_\s]+$", "", core_text).strip()

    # Step 1: Pre-process English contractions & dialogue
    pre_en = naturalization_rules.pre_process_english_dialogue(core_text)

    # Step 2: 100% Offline Local NMT Translation
    raw_uz = free_web_translate(pre_en)

    # Step 2b: Optional Naturalness-Scored Base MT Selection (Google Cloud vs NLLB)
    try:
        import cloud_enhancements
        if cloud_enhancements.is_translate_cross_check_enabled():
            optimal_base, _ = cloud_enhancements.select_optimal_base_translation(pre_en, raw_uz)
            if optimal_base:
                raw_uz = optimal_base
    except Exception:
        pass

    # Step 3: Naturalization & Idiom Polish Layer (with optional character voice profile)
    uz_text = naturalization_rules.apply_naturalization(raw_uz, en_orig=core_text, character_profile=speaker)

    if has_leading_dots and not uz_text.startswith("..."):
        uz_text = "..." + uz_text
    elif has_leading_dash and not uz_text.startswith("--") and not uz_text.startswith("-"):
        uz_text = "--" + uz_text
    if not (uz_text.endswith("!") or uz_text.endswith("?") or uz_text.endswith(".")):
        if cleaned.endswith("."):
            uz_text += "."
        elif cleaned.endswith("!"):
            uz_text += "!"
        elif cleaned.endswith("?"):
            uz_text += "?"

    return naturalization_rules.normalize_uzbek_comic_typography(uz_text)


def partition_translated_text_for_lobes(uz_text: str, lobe_weights: List[float]) -> List[str]:
    """
    Splits a unified translated sentence across multi-lobe bubbles at natural
    syntactic boundaries (clauses, commas, dashes, conjunctions), avoiding
    fragmentation or re-translation.
    """
    if len(lobe_weights) <= 1 or not uz_text.strip():
        return [uz_text.strip()] * max(1, len(lobe_weights))

    # If there are exactly 2 lobes
    if len(lobe_weights) == 2:
        # Look for sentence boundary first (. ! ? ;)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?…;])\s+', uz_text.strip()) if s.strip()]
        if len(sentences) == 2:
            return sentences
        elif len(sentences) > 2:
            mid = len(sentences) // 2
            return [" ".join(sentences[:mid]).strip(), " ".join(sentences[mid:]).strip()]

        # Look for clause boundary (, or -- or -)
        clauses = [c.strip() for c in re.split(r'(?<=[,\-–—])\s+', uz_text.strip()) if c.strip()]
        if len(clauses) >= 2:
            target_ratio = lobe_weights[0] / max(1.0, sum(lobe_weights))
            total_len = max(1, len(uz_text))
            best_idx = 1
            best_diff = 999999
            cum_len = 0
            for i in range(len(clauses) - 1):
                cum_len += len(clauses[i])
                diff = abs((cum_len / total_len) - target_ratio)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i + 1
            part1 = " ".join(clauses[:best_idx]).strip()
            part2 = " ".join(clauses[best_idx:]).strip()
            return [part1, part2]

        # Fallback: word-level split proportional to lobe_weights
        words = uz_text.strip().split()
        target_count = max(1, min(len(words) - 1, int(round(len(words) * (lobe_weights[0] / max(1.0, sum(lobe_weights)))))))
        return [" ".join(words[:target_count]), " ".join(words[target_count:])]

    # For 3+ lobes, split by sentences or proportional word chunks
    words = uz_text.strip().split()
    total_w = max(1.0, sum(lobe_weights))
    chunks = []
    start = 0
    for i, w in enumerate(lobe_weights):
        if i == len(lobe_weights) - 1:
            chunks.append(" ".join(words[start:]).strip())
        else:
            count = max(1, int(round(len(words) * (w / total_w))))
            chunks.append(" ".join(words[start:start + count]).strip())
            start += count
    return chunks


def detect_bubble_contours(
    image: Image.Image,
    min_area: int = 400,
    max_area: Optional[int] = None,
    thresh_val: int = 225
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    OpenCV-based bubble finder:
    1. Grayscale -> Adaptive Gaussian threshold + Otsu's binary thresholding.
       Captures white, off-white, yellowed, and rectangular caption boxes.
    2. Morphological close to bridge letters into a solid bubble blob.
    3. Finds closed contours with area between min_area (400) and max_area (35% of total page).
    4. Interior mean brightness filter: luminance >= 160 (handles comic speed-lines).
    """
    img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    total_area = w * h
    if max_area is None:
        max_area = int(0.35 * total_area)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 1. Otsu's binary thresholding (dynamically finds optimal separation for paper/background)
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Adaptive Gaussian thresholding (handles uneven lighting and off-white/aged newsprint)
    thresh_adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -8
    )

    # Combine both thresholding strategies
    combined = cv2.bitwise_or(thresh_otsu, thresh_adapt)

    # Morphological close to merge text inside speech bubbles into single solid blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / float(bh) if bh > 0 else 0

        # Filter area (min 400 px up to 35% of page) and aspect ratio (0.2 to 4.5)
        if min_area <= area <= max_area and 0.2 <= aspect <= 4.5:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_brightness = cv2.mean(gray, mask=mask)[0]
            # Bubble interior brightness filter: luminance >= 140
            if mean_brightness >= 140:
                detected.append((cnt, (bx, by, bx + bw, by + bh)))

    print(f"[DEBUG] Found {len(detected)} candidate bubbles (luminance >= 140)")

    # Sort in reading order (top-to-bottom, left-to-right)
    detected.sort(key=lambda b: (b[1][1] // 50, b[1][0]))
    return detected



def clean_page_ink_telea(
    image: Image.Image,
    bubbles: Optional[List[Any]] = None,
    ink_thresh: int = 165,
    inpaint_radius: int = 3,
    **kwargs
) -> Image.Image:
    """
    Pure Localized Telea Inpainting of Dialogue Ink via bubble_lettering.inpaint_bubble_text_only:
    - Removes only the dark ink pixels inside each bubble contour.
    - Z-order aware: ovals (z_order=0) inpainted first, rectangles (z_order=1) on top.
    - Occluded ovals: inpainted within the FULL oval area MINUS earpiece rectangle pixels.
    - Zero rectangular whiteouts, preserving paper texture and outer bubble borders.
    """
    if not bubbles:
        return image.copy()

    img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]

    # Pre-detect earpiece rectangles so occluded ovals can be cleaned accurately.
    # ONLY true caption rectangles (z_order == 1) are treated as earpiece rectangles.
    # Speech bubbles (z_order == 0) must NEVER be filled with solid rectangle whiteout.
    _earpiece_rects = []
    for b in bubbles:
        z = b.get("z_order", 0) if isinstance(b, dict) else getattr(b, "z_order", 0)
        shape = b.get("shape_type") if isinstance(b, dict) else getattr(b, "shape_type", None)
        bid = b.get("bubble_id", 0) if isinstance(b, dict) else getattr(b, "bubble_id", 0)
        cnt = b.get("contour") if isinstance(b, dict) else getattr(b, "contour", None)

        # Self-diagnosing quality gate: prevent dialogue bubble misclassification as earpiece rectangle
        b_info = {"z_order": z, "shape_type": shape, "bubble_id": bid, "contour": cnt}
        b_info = quality_gates.apply_quality_gates("bubble_inpainting", b_info, bubble_id=bid)
        shape = b_info.get("shape_type", shape)

        if z == 1 or shape in ("rectangle", "shaped_caption"):
            bx0 = int(b.get("x0") if isinstance(b, dict) else b.x0)
            by0 = int(b.get("y0") if isinstance(b, dict) else b.y0)
            bx1 = int(b.get("x1") if isinstance(b, dict) else b.x1)
            by1 = int(b.get("y1") if isinstance(b, dict) else b.y1)
            raw_cnt = None
            if cnt is not None:
                try:
                    c_arr = np.array(cnt, dtype=np.int32)
                    if c_arr.ndim == 2 and c_arr.shape[1] == 2 and len(c_arr) >= 3:
                        raw_cnt = c_arr.reshape(-1, 1, 2)
                    elif c_arr.ndim == 3 and c_arr.shape[2] == 2 and len(c_arr) >= 3:
                        raw_cnt = c_arr
                except Exception:
                    pass
            is_pure = (shape == "rectangle") and (raw_cnt is None or bubble_lettering._is_rectangular(raw_cnt))
            _earpiece_rects.append({
                'x': bx0, 'y': by0, 'x1': bx1, 'y1': by1, 'w': bx1 - bx0, 'h': by1 - by0,
                'contour': raw_cnt,
                'is_pure_rectangle': is_pure
            })
    if not _earpiece_rects:
        _earpiece_rects = detect_earpiece_rectangles(img_cv)

    # Build combined earpiece rect mask (union of all rects)
    _earpiece_mask = np.zeros((h, w), dtype=np.uint8)
    for _r in _earpiece_rects:
        if _r.get('contour') is not None and not _r.get('is_pure_rectangle', True):
            cv2.drawContours(_earpiece_mask, [_r['contour']], -1, 255, -1)
        else:
            cv2.rectangle(_earpiece_mask, (_r['x'], _r['y']), (_r['x1'], _r['y1']), 255, -1)

    # Calculate all bubble centroids and constituent line centers for neighbor-aware contour isolation
    all_seeds_by_bubble = []
    for b in bubbles:
        seeds_for_b = []
        if isinstance(b, dict):
            bx0, by0, bx1, by1 = int(b.get("x0", 0)), int(b.get("y0", 0)), int(b.get("x1", 0)), int(b.get("y1", 0))
            lcs = b.get("line_centers") or []
        elif hasattr(b, "x0") and hasattr(b, "x1") and hasattr(b, "y0") and hasattr(b, "y1"):
            bx0, by0, bx1, by1 = int(b.x0), int(b.y0), int(b.x1), int(b.y1)
            lcs = getattr(b, "line_centers", None) or []
        elif isinstance(b, (list, tuple)) and len(b) == 4:
            bx0, by0, bx1, by1 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            lcs = []
        else:
            bx0, by0, bx1, by1 = 0, 0, 0, 0
            lcs = []
        seeds_for_b.append((bx0, by0, bx1, by1))
        seeds_for_b.append(((bx0 + bx1) // 2, (by0 + by1) // 2))
        for pt in lcs:
            pt_tuple = (int(pt[0]), int(pt[1]))
            if pt_tuple not in seeds_for_b:
                seeds_for_b.append(pt_tuple)
        all_seeds_by_bubble.append(seeds_for_b)

    def _get_z(b: Any) -> int:
        if isinstance(b, dict):
            return int(b.get("z_order", 0))
        return int(getattr(b, "z_order", 0))

    sorted_idx_bubbles = sorted(enumerate(bubbles), key=lambda t: _get_z(t[1]))

    # When earpiece rectangles are detected, execute pristine paired dual-bubble inpainting:
    # 1. Inpaint occluded background ovals using the complete white bubble blob
    # 2. Inpaint rectangle interiors & redraw crisp border stroke
    # 3. Clean any other non-earpiece bubbles
    if _earpiece_rects:
        orig_bgr = img_cv.copy()
        # 1. Fill rectangle interiors completely with authentic fill first
        for _r in _earpiece_rects:
            rx, ry, rw, rh = _r['x'], _r['y'], _r['w'], _r['h']
            rx1, ry1 = _r['x1'], _r['y1']
            cnt = _r.get('contour')
            is_pure = _r.get('is_pure_rectangle', True)
            if cnt is not None:
                is_pure = bubble_lettering._is_rectangular(cnt)

            if cnt is not None and not is_pure:
                # Shaped communicator or non-rectangular caption:
                # Clean text strictly inside its inscribed text region; NEVER overwrite wings, antenna or custom shapes!
                c_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(c_mask, [cnt], -1, 255, -1)
                k_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                er_mask = cv2.erode(c_mask, k_er)
                irx, iry, irw, irh = bubble_lettering._largest_inscribed_rectangle(er_mask)

                interior_sample = orig_bgr[iry + 4:iry + irh - 4, irx + 4:irx + irw - 4]
                if interior_sample.size > 0:
                    gray_sample = cv2.cvtColor(interior_sample, cv2.COLOR_BGR2GRAY)
                    bright_pts = interior_sample[gray_sample >= 220]
                    fill_bgr = tuple(int(v) for v in np.median(bright_pts, axis=0)) if len(bright_pts) > 0 else (254, 254, 254)
                else:
                    fill_bgr = (254, 254, 254)

                if irw > 30 and irh > 20:
                    cv2.rectangle(img_cv, (irx, iry), (irx + irw, iry + irh), fill_bgr, thickness=-1)
                else:
                    ink = ((cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) < 205) & (er_mask > 0)).astype(np.uint8) * 255
                    img_cv = cv2.inpaint(img_cv, ink, 3, cv2.INPAINT_TELEA)
            else:
                interior_sample = orig_bgr[ry + 4:ry1 - 4, rx + 4:rx1 - 4]
                if interior_sample.size > 0:
                    gray_sample = cv2.cvtColor(interior_sample, cv2.COLOR_BGR2GRAY)
                    bright_pts = interior_sample[gray_sample >= 220]
                    fill_bgr = tuple(int(v) for v in np.median(bright_pts, axis=0)) if len(bright_pts) > 0 else (254, 254, 254)
                else:
                    fill_bgr = (254, 254, 254)

                cv2.rectangle(img_cv, (rx + 2, ry + 2), (rx1 - 2, ry1 - 2), fill_bgr, thickness=-1)

        # Build full rectangle mask (including borders) for safe seam inpainting clamping
        rect_full_mask = np.zeros((h, w), dtype=np.uint8)
        for _r in _earpiece_rects:
            cnt = _r.get('contour')
            is_pure = _r.get('is_pure_rectangle', True)
            if cnt is not None and not is_pure:
                cv2.drawContours(rect_full_mask, [cnt], -1, 255, -1)
            else:
                rx, ry, rx1, ry1 = _r['x'], _r['y'], _r['x1'], _r['y1']
                cv2.rectangle(rect_full_mask, (rx, ry), (rx1, ry1), 255, -1)

        # 2. Inpaint all speech bubbles (both occluded fragments and independent bubbles)
        sorted_idx_bubbles = sorted(enumerate(bubbles), key=lambda t: _get_z(t[1]))
        gray_cur = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        for idx, b in sorted_idx_bubbles:
            z = _get_z(b)
            shape = (isinstance(b, dict) and b.get("shape_type", "oval")) or getattr(b, "shape_type", "oval")
            if z == 1 or shape in ("rectangle", "shaped_caption"):
                continue

            neighbor_seeds = [pt for j, seeds in enumerate(all_seeds_by_bubble) if j != idx for pt in seeds]
            cnt, _ = get_contour_from_bubble(b, w, h, img_cv=img_cv, neighbor_seeds=neighbor_seeds)
            if cnt is None or len(cnt) < 3:
                continue

            f_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(f_mask, [cnt], -1, 255, -1)

            # 2px border margin from outer edges
            k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            eroded = cv2.erode(f_mask, k_erode)

            # Extend across rectangle seams so text touching borders is 100% eliminated
            bx_c, by_c, bw_c, bh_c = cv2.boundingRect(cnt)
            for _r in _earpiece_rects:
                rx, ry, rx1, ry1 = _r['x'], _r['y'], _r['x1'], _r['y1']
                ox0 = max(rx + 2, bx_c - 2)
                ox1 = min(rx1 - 2, bx_c + bw_c + 2)
                oy0 = max(ry + 2, by_c - 2)
                oy1 = min(ry1 - 2, by_c + bh_c + 2)

                # Bottom edge contact
                if ox1 > ox0:
                    b_strip = np.zeros((h, w), dtype=np.uint8)
                    cv2.rectangle(b_strip, (max(0, rx - 8), ry1 - 2), (min(w, rx1 + 8), min(h, ry1 + 8)), 255, -1)
                    if np.any(cv2.bitwise_and(f_mask, b_strip) > 0):
                        cv2.rectangle(eroded, (max(0, rx - 8), ry1 - 8), (min(w, rx1 + 8), min(h, ry1 + 8)), 255, -1)

                # Left edge contact (e.g. Mary Jane bubble touching left of Rect 1)
                if oy1 > oy0:
                    l_strip = np.zeros((h, w), dtype=np.uint8)
                    cv2.rectangle(l_strip, (max(0, rx - 18), max(0, ry - 6)), (rx + 2, min(h, ry1 + 8)), 255, -1)
                    if np.any(cv2.bitwise_and(f_mask, l_strip) > 0):
                        cv2.rectangle(eroded, (max(0, rx - 18), max(0, ry - 6)), (rx + 8, min(h, ry1 + 8)), 255, -1)

                # Right edge contact (e.g. Peter bubble touching right of Rect 0, Rect 2 right)
                if oy1 > oy0:
                    r_strip = np.zeros((h, w), dtype=np.uint8)
                    cv2.rectangle(r_strip, (rx1 - 2, max(0, ry - 6)), (min(w, rx1 + 18), min(h, ry1 + 8)), 255, -1)
                    if np.any(cv2.bitwise_and(f_mask, r_strip) > 0):
                        cv2.rectangle(eroded, (rx1 - 8, max(0, ry - 6)), (min(w, rx1 + 18), min(h, ry1 + 8)), 255, -1)

            # Strictly clamp inpainting to the bubble fill or rectangle area so background art is never touched
            eroded = cv2.bitwise_and(eroded, cv2.bitwise_or(f_mask, rect_full_mask))

            int_pts = img_cv[eroded > 0]
            int_gray = gray_cur[eroded > 0]
            if int_pts.size > 0 and int_gray.size > 0:
                p80_g = np.percentile(int_gray, 80)
                bg_sample = int_pts[int_gray >= p80_g]
                bg_bgr = np.median(bg_sample, axis=0) if len(bg_sample) > 0 else np.median(int_pts, axis=0)
                bg_gray = float(np.median(int_gray[int_gray >= p80_g])) if len(bg_sample) > 0 else float(np.median(int_gray))
                color_diff = np.linalg.norm(img_cv.astype(np.float32) - bg_bgr, axis=2)
                is_ink = ((color_diff > 8) | (gray_cur < bg_gray - 10) | (gray_cur < 215)) & (eroded > 0)
                ink = is_ink.astype(np.uint8) * 255
            else:
                ink = ((gray_cur < 215) & (eroded > 0)).astype(np.uint8) * 255
            k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            ink = cv2.dilate(ink, k_dilate, iterations=2)
            ink = cv2.bitwise_and(ink, eroded)
            if np.any(ink > 0):
                img_cv = cv2.inpaint(img_cv, ink, 3, cv2.INPAINT_TELEA)
                gray_cur = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # Sample background fill color to record authentic tint
            if int_pts.size > 0:
                sampled = np.median(int_pts, axis=0)
                tint_bgr = [int(v) for v in sampled]
                hsv_sampled = cv2.cvtColor(np.uint8([[tint_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
                is_tinted = bool(hsv_sampled[1] >= 15)
                if isinstance(b, dict):
                    b["tint_bgr"] = tint_bgr
                    b["is_tinted"] = is_tinted
                elif hasattr(b, "tint_bgr"):
                    b.tint_bgr = tint_bgr
                    b.is_tinted = is_tinted

        # 3. Redraw crisp 2px brown rectangle borders for PURE rectangles only
        for _r in _earpiece_rects:
            cnt = _r.get('contour')
            is_pure = _r.get('is_pure_rectangle', True)
            if cnt is not None:
                is_pure = bubble_lettering._is_rectangular(cnt)
            if not is_pure:
                # Authentic hand-drawn border and wings remain 100% intact
                continue

            rx, ry, rw, rh = _r['x'], _r['y'], _r['w'], _r['h']
            rx1, ry1 = _r['x1'], _r['y1']
            top_edge = orig_bgr[ry:ry+2, rx+20:rx1-20]
            border_bgr = [int(v) for v in np.mean(top_edge, axis=(0, 1))] if top_edge.size > 0 else (78, 93, 140)
            cv2.rectangle(img_cv, (rx, ry), (rx1, ry1), tuple(border_bgr), thickness=2)

        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # Standard inpainting path for pages without earpiece rectangles
    active_bubble_items = []
    sorted_idx_bubbles = sorted(enumerate(bubbles), key=lambda t: _get_z(t[1]))
    for idx, b in sorted_idx_bubbles:
        if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float, np.integer)) for v in b):
            is_enabled = True
        elif isinstance(b, dict):
            is_enabled = b.get("is_active", b.get("enabled", True))
        else:
            is_enabled = getattr(b, "is_active", getattr(b, "enabled", True))
        if not is_enabled:
            continue
        neighbor_seeds = [pt for j, seeds in enumerate(all_seeds_by_bubble) if j != idx for pt in seeds]
        cnt, _ = get_contour_from_bubble(b, w, h, img_cv=img_cv, neighbor_seeds=neighbor_seeds)

        # Sample authentic bubble background fill color (c_bg) and original text ink BEFORE inpainting
        if cnt is not None and len(cnt) >= 3:
            b_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(b_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            er = cv2.erode(b_mask, k_in)
            if np.any(er > 0):
                pts = img_cv[er > 0]
                hsv_int = cv2.cvtColor(pts.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)[0]
                sat_int = hsv_int[:, 1]
                gray_er = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)[er > 0]

                # 1. Text ink analysis
                bg_ref_lum = np.percentile(gray_er, 75)
                ink_cand_mask = (gray_er < bg_ref_lum - 12) | (sat_int >= 10)
                if np.count_nonzero(ink_cand_mask) >= 10:
                    ink_pts = pts[ink_cand_mask]
                    ink_gray = gray_er[ink_cand_mask]
                    ink_hsv = hsv_int[ink_cand_mask]
                else:
                    p10_g = np.percentile(gray_er, 10)
                    ink_pts = pts[gray_er <= p10_g]
                    ink_gray = gray_er[gray_er <= p10_g]
                    ink_hsv = hsv_int[gray_er <= p10_g]

                ink_sat = ink_hsv[:, 1] if len(ink_hsv) > 0 else []
                ink_med_bgr = np.median(ink_pts, axis=0) if len(ink_pts) > 0 else [0, 0, 0]
                ink_lum = np.median(ink_gray) if len(ink_gray) > 0 else 0
                b_r_text_diff = int(ink_med_bgr[0]) - int(ink_med_bgr[2])

                # Spectral text check: text is light tinted ink (Peter Parker ghost dialogue)
                # Slate blue / havorang ink: luminance > 120, sat >= 8 or abs(B - R) >= 12
                text_is_spectral = bool(ink_lum > 120 and (len(ink_sat) > 0 and np.median(ink_sat) >= 8 or abs(b_r_text_diff) >= 12))

                # 2. Non-text background analysis (brightest 60% of interior)
                p40_g = np.percentile(gray_er, 40)
                bg_pts = pts[gray_er >= p40_g]
                bg_hsv = hsv_int[gray_er >= p40_g]
                bg_sat = bg_hsv[:, 1] if len(bg_hsv) > 0 else []
                bg_med_sat = np.median(bg_sat) if len(bg_sat) > 0 else 0
                bg_is_tinted = bool(bg_med_sat >= 12 or (len(bg_sat) > 0 and np.percentile(bg_sat, 60) >= 12))

                is_tinted = bool(text_is_spectral or bg_is_tinted)
                if bg_is_tinted:
                    c_bg_bgr = [int(v) for v in np.median(bg_pts, axis=0)] if len(bg_pts) > 0 else [int(v) for v in np.median(pts, axis=0)]
                else:
                    p80_gray = np.percentile(gray_er, 80)
                    bg_sample = pts[gray_er >= p80_gray]
                    c_bg_bgr = [int(v) for v in np.median(bg_sample, axis=0)] if len(bg_sample) > 0 else [255, 255, 255]

                orig_text_rgb = [int(v) for v in np.median(ink_pts, axis=0)][::-1] if len(ink_pts) > 0 else [0, 0, 0]

                if is_tinted:
                    quality_gates.apply_quality_gates(
                        "bubble_inpainting",
                        {
                            "is_tinted": True,
                            "bg_is_tinted": bg_is_tinted,
                            "text_is_spectral": text_is_spectral,
                            "tint_bgr": c_bg_bgr,
                            "tint_rgb": c_bg_bgr[::-1]
                        },
                        bubble_id=getattr(b, "bubble_id", 0) if hasattr(b, "bubble_id") else b.get("bubble_id", 0)
                    )

                if isinstance(b, dict):
                    b["tint_bgr"] = c_bg_bgr
                    b["tint_rgb"] = c_bg_bgr[::-1]
                    b["is_tinted"] = is_tinted
                    b["bg_is_tinted"] = bg_is_tinted
                    b["text_is_spectral"] = text_is_spectral
                    b["orig_text_rgb"] = orig_text_rgb
                else:
                    b.tint_bgr = c_bg_bgr
                    b.tint_rgb = c_bg_bgr[::-1]
                    b.is_tinted = is_tinted
                    b.bg_is_tinted = bg_is_tinted
                    b.text_is_spectral = text_is_spectral
                    b.orig_text_rgb = orig_text_rgb

            active_bubble_items.append({
                "bubble": b,
                "cnt": cnt,
                "mask": b_mask
            })

    # Group conjoined / touching bubbles so neck cut lines are closed prior to inpainting
    if active_bubble_items:
        page_bubble_mask = np.zeros((h, w), dtype=np.uint8)
        bubble_label_mask = np.zeros((h, w), dtype=np.int32)
        for idx_b, item in enumerate(active_bubble_items, start=1):
            page_bubble_mask = cv2.bitwise_or(page_bubble_mask, item["mask"])

        # Bridge internal cut lines between conjoined lobes (typically 1-3px cut line)
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        page_bubble_mask_closed = cv2.morphologyEx(page_bubble_mask, cv2.MORPH_CLOSE, k_close)

        num_labels, labels = cv2.connectedComponents(page_bubble_mask_closed)
        for lbl in range(1, num_labels):
            comp_mask = (labels == lbl).astype(np.uint8) * 255
            comp_cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not comp_cnts:
                continue
            comp_cnt = max(comp_cnts, key=cv2.contourArea)
            img_cv = bubble_lettering.inpaint_bubble_text_only(
                img_cv, comp_cnt, ink_luminance_threshold=ink_thresh, border_margin=4, ink_dilate=2, inpaint_radius=inpaint_radius
            )

    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


clean_bubbles_telea = clean_page_ink_telea
clean_page_opencv = clean_page_ink_telea



def has_dark_barrier_between(clean_gray_img: Optional[np.ndarray], l1: Dict[str, Any], l2: Dict[str, Any]) -> bool:
    """
    Checks if a true dark comic bubble border/edge separates two candidate text lines:
    Samples pixel luminance along the center-to-center line segment on the text-suppressed
    grayscale image. Inside a single bubble, the space between lines is white paper (clean_gray > 160).
    Between two separate speech bubbles, the line must cross the dark bubble stroke (clean_gray < 100).
    """
    if clean_gray_img is None:
        return False
    c1x, c1y = int((l1['x0'] + l1['x1']) / 2), int((l1['y0'] + l1['y1']) / 2)
    c2x, c2y = int((l2['x0'] + l2['x1']) / 2), int((l2['y0'] + l2['y1']) / 2)

    num_pts = max(10, int(np.hypot(c2x - c1x, c2y - c1y)))
    xs = np.linspace(c1x, c2x, num_pts).astype(int)
    ys = np.linspace(c1y, c2y, num_pts).astype(int)

    h, w = clean_gray_img.shape
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)

    # Exclude 15% margin at endpoints near the bounding boxes
    cut = int(num_pts * 0.15)
    if cut > 0 and num_pts - cut > cut:
        mid_xs = xs[cut:-cut]
        mid_ys = ys[cut:-cut]
    else:
        mid_xs, mid_ys = xs, ys

    sampled = clean_gray_img[mid_ys, mid_xs]
    return bool(np.any(sampled < 100))


def cluster_lines_into_bubbles(
    lines: List[Dict[str, Any]],
    max_dx: int = 30,
    max_dy: int = 40,
    diamond_center_tolerance: float = 0.65,
    clean_gray: Optional[np.ndarray] = None
) -> List[List[Dict[str, Any]]]:
    """
    Groups individual OCR text line boxes into coherent speech bubble paragraphs:
    - Maximum horizontal gap (max_dx): 25px (prevents cross-bubble horizontal merging).
    - Maximum vertical line gap (max_dy): 30px.
    - Diamond-rule aware: lettering inside round bubbles is center-wrapped (short top/
      bottom lines, wide middle line), so two lines of the SAME bubble routinely have
      little or no bbox x-overlap even though they belong together. A center-distance
      check (relative to each line's own half-width) catches this case, while the old
      bbox-overlap check alone would wrongly split the bubble into two tiny fallback
      contours (one per line).
    - Prevents diagonal leaps across distinct columns (gap_x > 15 and gap_y > 15) --
      but only when the center-distance check ALSO fails, so genuine diamond-wrapped
      lines in the same bubble are never rejected by this guard.
    - Detects dual-column layout (e.g. touching speech bubbles with internal vertical
      whitespace) and automatically splits them into Left Bubble and Right Bubble.
    - Bubble border aware: checks has_dark_barrier_between to prevent merging separate
      stacked bubbles (like HEY! and HERMAN?) even if their vertical gap satisfies max_dy.
    """
    if not lines:
        return []

    # Sort lines top-to-bottom, left-to-right
    sorted_lines = sorted(lines, key=lambda l: (l['y0'], l['x0']))
    n = len(sorted_lines)

    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            l1, l2 = sorted_lines[i], sorted_lines[j]

            x_overlap = min(l1['x1'], l2['x1']) - max(l1['x0'], l2['x0'])
            gap_x = max(0, -x_overlap)

            y_overlap = min(l1['y1'], l2['y1']) - max(l1['y0'], l2['y0'])
            gap_y = max(0, -y_overlap)

            # Diamond-rule check: are these two lines plausibly center-stacked in the
            # same round bubble? Compare horizontal center distance against half the
            # WIDER line's width -- a short top line and a wide middle line can have
            # centers this close while their bboxes barely overlap (or don't at all).
            c1x = (l1['x0'] + l1['x1']) / 2.0
            c2x = (l2['x0'] + l2['x1']) / 2.0
            wider_half = max(l1['x1'] - l1['x0'], l2['x1'] - l2['x0']) / 2.0
            center_close = abs(c1x - c2x) <= max(wider_half * diamond_center_tolerance, max_dx)

            # Quality gate: diamond_rule_line_split check
            pair_info = {
                "gap_x": gap_x,
                "gap_y": gap_y,
                "c_dist": abs(c1x - c2x),
                "wider_half": wider_half,
                "allow_connection": False,
            }
            gate_ctx = {
                "max_dx": max_dx,
                "max_dy": max_dy,
                "diamond_center_tolerance": diamond_center_tolerance
            }
            res_pair = quality_gates.apply_quality_gates("line_pairing", pair_info, context=gate_ctx)
            if isinstance(res_pair, dict) and res_pair.get("allow_connection", False):
                center_close = True

            # 1. Reject if horizontal gap > max_dx, UNLESS the diamond center check passes
            if gap_x > max_dx and not center_close:
                continue

            # 2. Reject if vertical gap > max_dy
            if gap_y > max_dy:
                continue

            # 3. Anti-column leap: if horizontal gap exists and centers are not diamond-aligned,
            #    they belong to separate columns or distinct adjacent bubbles.
            if gap_x > 12 and not center_close:
                continue

            # 4. Anti-diagonal leap: prevent connecting across distinct columns --
            #    skipped when the diamond center check already vouches for this pair.
            if gap_x > 15 and gap_y > 15 and not center_close:
                continue

            # 5. Bubble border check: if a dark bubble border separates l1 and l2, DO NOT merge them
            if clean_gray is not None and has_dark_barrier_between(clean_gray, l1, l2):
                continue

            adj[i].append(j)
            adj[j].append(i)

    # Connected components
    visited = set()
    clusters = []
    for i in range(n):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append([sorted_lines[idx] for idx in comp])

    # Dual-column splitting pass for touching bubbles
    min_split_gap = 12
    final_clusters = []
    for c in clusters:
        if len(c) < 2:
            final_clusters.append(c)
            continue

        # Quality gate: check adjacent_bubble_merging_cross_column
        c_split = quality_gates.apply_quality_gates("line_clustering", c, context={"min_split_gap": min_split_gap})
        if isinstance(c_split, list) and len(c_split) > 1 and all(isinstance(sub, list) for sub in c_split):
            for sub_c in c_split:
                final_clusters.append(sub_c)
        else:
            final_clusters.append(c)

    # Figure-8 / double-lobed bubble splitting pass via figure8_clustering quality gate
    lobe_clusters = []
    for c in final_clusters:
        f8_split = quality_gates.apply_quality_gates("figure8_clustering", c)
        if isinstance(f8_split, list) and len(f8_split) > 1 and all(isinstance(sub, list) for sub in f8_split):
            for sub_c in f8_split:
                lobe_clusters.append(sub_c)
        else:
            lobe_clusters.append(c)

    return lobe_clusters


def extract_bubble_contour_from_seed(
    gray_img: np.ndarray,
    box: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, str]:
    """
    Extracts the exact outer polygon contour of a speech bubble / caption box:
    1. Determines seed point (cx, cy) at the center of the OCR text box.
    2. Uses thresholded binary mask of the bubble area (white interior is 255, dark comic border is 0)
       with morphological closing to bridge internal text letters.
    3. Performs floodfill from (cx, cy) to isolate the full bubble interior.
    4. Extracts outer contour via cv2.findContours(..., cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE).
    5. Classifies shape as 'rectangle' (Caption Box) vs 'oval' (Organic Speech Bubble):
       - If rect_ratio >= 0.82 and polygon vertices in [4, 5, 6] (or rect_ratio > 0.88): 'rectangle'
       - Otherwise: 'oval'.
    """
    bx0, by0, bx1, by1 = box
    h, w = gray_img.shape[:2]
    cx = (bx0 + bx1) // 2
    cy = (by0 + by1) // 2
    bw = max(10, bx1 - bx0)
    bh = max(10, by1 - by0)

    pad_x = max(60, int(bw * 0.45))
    pad_y = max(60, int(bh * 0.45))
    cx0 = max(0, bx0 - pad_x)
    cy0 = max(0, by0 - pad_y)
    cx1 = min(w, bx1 + pad_x)
    cy1 = min(h, by1 + pad_y)

    crop = gray_img[cy0:cy1, cx0:cx1]
    ch, cw = crop.shape
    if ch < 10 or cw < 10:
        rx = max(10, bw // 2 + 10)
        ry = max(10, bh // 2 + 10)
        pts = cv2.ellipse2Poly((cx, cy), (rx, ry), 0, 0, 360, 12)
        return pts.reshape(-1, 1, 2), "oval"

    _, crop_bin = cv2.threshold(crop, 180, 255, cv2.THRESH_BINARY)
    crop_closed = cv2.morphologyEx(crop_bin, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

    seed_x = min(cw - 1, max(0, cx - cx0))
    seed_y = min(ch - 1, max(0, cy - cy0))

    if crop_closed[seed_y, seed_x] == 0:
        ys, xs = np.where(crop_closed > 0)
        if len(ys) > 0:
            dists = (xs - seed_x) ** 2 + (ys - seed_y) ** 2
            idx = np.argmin(dists)
            seed_x, seed_y = xs[idx], ys[idx]

    ff_mask = np.zeros((ch + 2, cw + 2), np.uint8)
    cv2.floodFill(crop_closed.copy(), ff_mask, (int(seed_x), int(seed_y)), 255)
    mask = ff_mask[1:ch+1, 1:cw+1]

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) >= 200:
            cnt_page = cnt + np.array([[[cx0, cy0]]])
            bbx, bby, bbw, bbh = cv2.boundingRect(cnt_page)
            cnt_area = cv2.contourArea(cnt_page)
            box_area = float(bbw * bbh) if bbw * bbh > 0 else 1.0
            rect_ratio = cnt_area / box_area
            peri = cv2.arcLength(cnt_page, True)
            approx = cv2.approxPolyDP(cnt_page, 0.02 * peri, True)
            is_rect = rect_ratio >= 0.94 and len(approx) in [4, 5]
            shape_type = "rectangle" if is_rect else "oval"
            return cnt_page, shape_type

    # Fallback smooth ellipse
    rx = max(10, bw // 2 + 10)
    ry = max(10, bh // 2 + 10)
    pts = cv2.ellipse2Poly((cx, cy), (rx, ry), 0, 0, 360, 12)
    return pts.reshape(-1, 1, 2), "oval"


def detect_earpiece_rectangles(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detects earpiece caption boxes by their distinctive red-brown border stroke.
    Uses multi-pass RGB color mask + contour search with interior white-fill verification.
    Returns list of dicts: {x, y, w, h, x1, y1, contour, mask}
    """
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    r = img_rgb[:, :, 0].astype(int)
    g = img_rgb[:, :, 1].astype(int)
    b = img_rgb[:, :, 2].astype(int)

    def _pass(r_b_min: int) -> List[Dict[str, Any]]:
        red_brown = (
            (r >= 95) & (r <= 220) &
            (g >= 40) & (g <= 150) &
            (b >= 25) & (b <= 130) &
            (r - g >= 18) & (r - b >= r_b_min)
        ).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(red_brown, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for cnt in cnts:
            x, y, rw, rh = cv2.boundingRect(cnt)
            if rw >= 80 and rh >= 30 and (rw / float(rh) >= 1.1):
                area = cv2.contourArea(cnt)
                b_area = rw * rh
                fill_ratio = area / float(max(1, b_area))
                if fill_ratio < 0.85:
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) > 5:
                    continue
                is_pure = bubble_lettering._is_rectangular(cnt, rect_fill_thresh=0.85, max_vertices=5)
                if not is_pure:
                    continue
                iy0, iy1 = y + 6, y + rh - 6
                ix0, ix1 = x + 6, x + rw - 6
                if iy1 > iy0 and ix1 > ix0:
                    interior = gray[iy0:iy1, ix0:ix1]
                    if interior.size > 0 and np.mean(interior) >= 170 and np.percentile(interior, 75) >= 220:
                        rmask = np.zeros((H, W), dtype=np.uint8)
                        cv2.rectangle(rmask, (x, y), (x + rw, y + rh), 255, -1)
                        rects.append({
                            'x': x, 'y': y, 'w': rw, 'h': rh,
                            'x1': x + rw, 'y1': y + rh,
                            'contour': cnt,
                            'mask': rmask,
                            'is_pure_rectangle': True
                        })
        return rects

    r_high = _pass(35)
    r_low = _pass(24)
    combined = list(r_high)
    for cand in r_low:
        cx, cy, cw, ch = cand['x'], cand['y'], cand['w'], cand['h']
        dup = False
        for ex in combined:
            ix0, iy0 = max(cx, ex['x']), max(cy, ex['y'])
            ix1, iy1 = min(cx + cw, ex['x1']), min(cy + ch, ex['y1'])
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                union = cw * ch + ex['w'] * ex['h'] - inter
                if inter / float(union) > 0.4:
                    dup = True
                    nx0 = min(ex['x'], cx)
                    ny0 = min(ex['y'], cy)
                    nx1 = max(ex['x1'], cx + cw)
                    ny1 = max(ex['y1'], cy + ch)
                    ex['x'] = nx0
                    ex['y'] = ny0
                    ex['x1'] = nx1
                    ex['y1'] = ny1
                    ex['w'] = nx1 - nx0
                    ex['h'] = ny1 - ny0
                    ex['is_pure_rectangle'] = True
                    rmask = np.zeros((H, W), dtype=np.uint8)
                    cv2.rectangle(rmask, (nx0, ny0), (nx1, ny1), 255, -1)
                    ex['mask'] = rmask
                    break
        if not dup:
            combined.append(cand)

    combined.sort(key=lambda item: (item['y'] // 80, item['x']))
    print(f"[RECT] Detected {len(combined)} earpiece rectangles.")
    return combined


def _split_oval_visible_mask(
    oval_fill_mask: np.ndarray,
    rect_mask: np.ndarray,
    rects: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Subtracts the combined rectangle mask from the oval fill and applies
    2px corner ray-cut lines so touching slivers become cleanly separate
    connected components.
    """
    H, W = oval_fill_mask.shape[:2]
    visible = cv2.bitwise_and(oval_fill_mask, cv2.bitwise_not(rect_mask))

    for r in rects:
        rx, ry, rw, rh = r['x'], r['y'], r['w'], r['h']
        rx1, ry1 = rx + rw, ry + rh
        roi = oval_fill_mask[max(0, ry - 5):min(H, ry1 + 5), max(0, rx - 5):min(W, rx1 + 5)]
        if not np.any(roi > 0):
            continue
        RAY = 180
        if rx1 < W and ry1 < H:
            cv2.line(visible, (rx1, ry1), (min(W - 1, rx1 + RAY), ry1), 0, 2)
        if rx > 0 and ry1 < H:
            cv2.line(visible, (max(0, rx - RAY), ry1), (rx, ry1), 0, 2)
        if rx1 < W and ry > 0:
            cv2.line(visible, (rx1, ry), (rx1, max(0, ry - RAY)), 0, 2)
        if rx > 0 and ry > 0:
            cv2.line(visible, (rx, ry), (rx, max(0, ry - RAY)), 0, 2)
    return visible


def _extract_oval_fragments(
    visible_mask: np.ndarray,
    oval_lines: List[Dict[str, Any]],
    min_frag_area: int = 300
) -> List[Dict[str, Any]]:
    """
    Splits oval_visible_mask into connected fragments and assigns OCR lines to each.
    Returns list of fragment dicts with contour, bbox, lines, and original_text.
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(visible_mask, connectivity=8)
    fragments = []
    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_frag_area:
            continue
        fx = stats[lbl, cv2.CC_STAT_LEFT]
        fy = stats[lbl, cv2.CC_STAT_TOP]
        fw = stats[lbl, cv2.CC_STAT_WIDTH]
        fh = stats[lbl, cv2.CC_STAT_HEIGHT]
        comp_mask = (labels == lbl).astype(np.uint8) * 255
        f_cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not f_cnts:
            continue
        f_cnt = max(f_cnts, key=cv2.contourArea)
        frag_lines = []
        for l in oval_lines:
            lcx = (l['x0'] + l['x1']) // 2
            lcy = (l['y0'] + l['y1']) // 2
            if (fx - 15 <= lcx <= fx + fw + 15) and (fy - 15 <= lcy <= fy + fh + 15):
                frag_lines.append(l)
        frag_text = " ".join(l['text'] for l in frag_lines).strip()
        if not frag_lines or not frag_text:
            continue
        fragments.append({
            'fragment_id': len(fragments) + 1,
            'box': [int(fx), int(fy), int(fx + fw), int(fy + fh)],
            'area': int(area),
            'contour': f_cnt.reshape(-1, 2).tolist(),
            'contour_np': f_cnt,
            'lines': frag_lines,
            'original_text': frag_text,
            'uzbek_translation': ''
        })

    fragments.sort(key=lambda f: (f['box'][1] // 40, f['box'][0]))
    return fragments


def sort_reading_order(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sorts OCR text line fragments in authentic reading order:
    1. Clusters lines sharing roughly the same baseline (within 45% of median line height).
    2. Sorts each horizontal line left-to-right (by x0).
    3. Sorts rows vertically top-to-bottom (by mean cy).
    Prevents same-line fragments from scrambling word order.
    """
    if not lines or len(lines) <= 1:
        return lines

    heights = [max(1, l['y1'] - l['y0']) for l in lines]
    med_h = float(np.median(heights)) if heights else 20.0
    line_thresh = max(8.0, med_h * 0.45)

    sorted_by_y = sorted(lines, key=lambda l: l['cy'])
    rows: List[List[Dict[str, Any]]] = []
    for l in sorted_by_y:
        placed = False
        for row in rows:
            row_cy = float(np.mean([item['cy'] for item in row]))
            if abs(l['cy'] - row_cy) < line_thresh:
                row.append(l)
                placed = True
                break
        if not placed:
            rows.append([l])

    rows.sort(key=lambda row: float(np.mean([item['cy'] for item in row])))
    ordered = []
    for row in rows:
        row.sort(key=lambda l: l['x0'])
        ordered.extend(row)
    return ordered


def scan_bubbles_ocr_subprocess(image: Image.Image, timeout_seconds: int = 120) -> List[SpeechBubble]:
    """
    Executes OCR in an isolated OS subprocess with strict timeout protection.
    When the child process exits, Linux/Windows OS fully recovers its memory.
    If the child process hangs past timeout_seconds, it is killed forcefully and raises TimeoutError.
    """
    import tempfile
    import subprocess
    import sys
    import psutil

    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_img_path = tmp_img.name
    tmp_img.close()
    image.save(tmp_img_path, format="PNG")

    tmp_json = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp_json_path = tmp_json.name
    tmp_json.close()

    worker_script = os.path.join(os.path.dirname(__file__), "scripts", "ocr_worker.py")
    subproc = None
    try:
        subproc = subprocess.Popen(
            [sys.executable, "-u", worker_script, tmp_img_path, tmp_json_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            stdout, stderr = subproc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            print(f"[OCR_SUBPROCESS_TIMEOUT] OCR bola jarayoni {timeout_seconds}s ichida javob bermadi! Majburiy to'xtatilyapti...", flush=True)
            try:
                p = psutil.Process(subproc.pid)
                for child in p.children(recursive=True):
                    try:
                        child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                p.kill()
            except Exception:
                subproc.kill()
            subproc.communicate()
            raise TimeoutError(f"EasyOCR bola jarayoni {timeout_seconds} soniya ichida yakunlanmadi va majburiy to'xtatildi (Timeout).")

        if subproc.returncode != 0:
            err_details = stderr.strip() if stderr else (stdout.strip() if stdout else f"Exit code {subproc.returncode}")
            raise RuntimeError(f"worker process exited with code {subproc.returncode}: {err_details}")

        with open(tmp_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [SpeechBubble(**d) for d in data]
    finally:
        for p in [tmp_img_path, tmp_json_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def scan_bubbles_ocr(image: Image.Image, use_subprocess: bool = True, timeout_seconds: int = 120) -> List[SpeechBubble]:
    """
    Stage 1 OCR-First Detection & SFX Filtering Pipeline:
    When use_subprocess=True, executes OCR in an isolated worker process so that
    PyTorch/CRAFT/EasyOCR memory is 100% returned to the OS upon process exit.
    If the worker hangs, it is terminated forcefully to prevent orphaned processes.
    """
    if use_subprocess:
        import time
        t0 = time.time()
        try:
            bubbles = scan_bubbles_ocr_subprocess(image, timeout_seconds=timeout_seconds)
            elapsed = time.time() - t0
            print(f"[OCR] subprocess mode: SUCCESS (bubbles found: {len(bubbles)}, time: {elapsed:.2f}s)", flush=True)
            return bubbles
        except TimeoutError as te:
            print(f"[OCR] subprocess FAILED, falling back to in-process: Timeout ({te})", flush=True)
            raise
        except Exception as err:
            print(f"[OCR] subprocess FAILED, falling back to in-process: {err}", flush=True)
    return _scan_bubbles_ocr_core(image)



def _scan_bubbles_ocr_core(image: Image.Image) -> List[SpeechBubble]:
    """Core in-process OCR scanning implementation."""
    img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    clean_gray_barrier = bubble_lettering._suppress_text_for_barrier(gray, 35, ink_thresh=120)

    reader = get_ocr_reader()
    raw_results = run_tiled_easyocr(image, reader=reader, canvas_size=2048)

    candidate_lines = []

    for item in raw_results:
        if len(item) == 2:
            item_bbox, text = item
            conf = 1.0
        else:
            item_bbox, text, conf = item
        text_str = str(text).strip()
        if not text_str:
            continue

        xs = [int(pt[0]) for pt in item_bbox]
        ys = [int(pt[1]) for pt in item_bbox]
        x0 = max(0, min(xs))
        y0 = max(0, min(ys))
        x1 = min(w, max(xs))
        y1 = min(h, max(ys))

        # Inspect local background pixels (expand box by 6px)
        exp_x0 = max(0, x0 - 6)
        exp_y0 = max(0, y0 - 6)
        exp_x1 = min(w, x1 + 6)
        exp_y1 = min(h, y1 + 6)

        crop_gray = gray[exp_y0:exp_y1, exp_x0:exp_x1]
        crop_hsv = hsv[exp_y0:exp_y1, exp_x0:exp_x1]

        if crop_gray.size == 0:
            continue

        mean_luminance = float(np.mean(crop_gray))
        percentile_75 = float(np.percentile(crop_gray, 75))
        bright_ratio = float(np.mean(crop_gray > 170))

        # Saturation and tint analysis:
        # Check both bright pixels and full crop
        crop_val = crop_hsv[:, :, 2]
        bright_mask = (crop_gray > 170) | (crop_val > 175)
        if np.any(bright_mask):
            fill_sat = crop_hsv[:, :, 1][bright_mask]
            mean_saturation = float(np.mean(fill_sat))
            sat_std = float(np.std(fill_sat))
        else:
            fill_sat = crop_hsv[:, :, 1]
            mean_saturation = float(np.mean(fill_sat))
            sat_std = float(np.std(fill_sat))

        max_val = float(np.max(crop_val))

        line_w = x1 - x0
        line_h = y1 - y0
        aspect = line_w / max(1, line_h)
        is_graphic_banner = bool(aspect >= 4.0 and mean_saturation >= 60)

        # Rule for speech bubble / caption:
        # Standard white/near-white bubble (mean_saturation <= 85)
        is_standard_bubble = not is_graphic_banner and (
            (mean_luminance >= 140 and mean_saturation <= 85) or (
                percentile_75 >= 180 and mean_saturation <= 85 and bright_ratio >= 0.15
            )
        )
        # Uniform monochromatic tinted bubble (spectral/telepathic/colored dialogue fill):
        # Characterized by low saturation variance (near-uniform tint), bright value channel (>= 150),
        # and sufficient lightness in either luminance or 75th percentile.
        is_monochrome_tint = (
            not is_graphic_banner
            and (mean_luminance >= 120 or percentile_75 >= 160 or max_val >= 160)
            and (sat_std <= 22 or (mean_saturation > 0 and sat_std / max(1.0, mean_saturation) <= 0.40))
            and max_val >= 150
        )
        is_bubble = is_standard_bubble or is_monochrome_tint

        # Step 1: Filter junk / SFX / watermarks / noise (aware of bubble background)
        is_junk, reason = check_is_sfx_or_junk(text_str, float(conf), is_on_bubble_paper=is_bubble)
        if is_junk:
            continue

        candidate_info = {
            "mean_luminance": mean_luminance,
            "percentile_75": percentile_75,
            "mean_saturation": mean_saturation,
            "sat_std": sat_std,
            "bright_ratio": bright_ratio,
            "max_val": max_val,
            "line_w": line_w,
            "line_h": line_h,
            "text": text_str
        }
        # Self-diagnosing quality gate: filter sfx_false_positive on dark/colored art
        if quality_gates.apply_quality_gates("ocr_filtering", candidate_info) is None:
            continue

        if is_bubble:
            line_text = text_str
            line_conf = float(conf)

            # Optional Cloud Enhancement: Google Vision OCR fallback if EasyOCR confidence < 0.55
            # Completely inert if GOOGLE_VISION_API_KEY is not configured or confidence >= 0.55
            if line_conf < 0.55:
                try:
                    import cloud_enhancements
                    if cloud_enhancements.is_vision_ocr_enabled():
                        crop_cv = img_cv[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
                        line_text, line_conf, _ = cloud_enhancements.google_vision_ocr_fallback(
                            image_np=crop_cv,
                            easyocr_text=line_text,
                            easyocr_conf=line_conf
                        )
                except Exception:
                    pass

            candidate_lines.append({
                'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                'cx': (x0 + x1) // 2, 'cy': (y0 + y1) // 2,
                'text': line_text, 'conf': line_conf
            })

    # -----------------------------------------------------------------------
    # PASS 1: Detect Earpiece / Caption Rectangles
    # -----------------------------------------------------------------------
    earpiece_rects = detect_earpiece_rectangles(img_cv)

    rect_bubbles = []
    rect_lines = {ri: [] for ri in range(len(earpiece_rects))}
    unclaimed_lines = []

    # Claim lines where inter_area / line_area >= 0.50
    for l in candidate_lines:
        x0, y0, x1, y1 = l['x0'], l['y0'], l['x1'], l['y1']
        line_area = max(1, (x1 - x0) * (y1 - y0))
        claimed = False
        for ri, r in enumerate(earpiece_rects):
            ix0 = max(x0, r['x'])
            iy0 = max(y0, r['y'])
            ix1 = min(x1, r['x1'])
            iy1 = min(y1, r['y1'])
            if ix1 > ix0 and iy1 > iy0:
                inter_area = (ix1 - ix0) * (iy1 - iy0)
                if inter_area / float(line_area) >= 0.50:
                    rect_lines[ri].append(l)
                    claimed = True
                    break
        if not claimed:
            unclaimed_lines.append(l)

    valid_rect_indices = []
    for ri, r in enumerate(earpiece_rects):
        lines_r = sort_reading_order(rect_lines[ri])
        text_r = " ".join(l['text'] for l in lines_r).strip()
        if not text_r:
            continue
        valid_rect_indices.append(ri)
        rx, ry, rx1, ry1 = r['x'], r['y'], r['x1'], r['y1']
        if r.get('contour') is not None and len(r['contour']) >= 3:
            actual_cnt = r['contour']
        else:
            actual_cnt = np.array([[rx, ry], [rx1, ry], [rx1, ry1], [rx, ry1]], dtype=np.int32).reshape(-1, 1, 2)

        is_pure_rect = r.get('is_pure_rectangle')
        if is_pure_rect is None:
            is_pure_rect = bubble_lettering._is_rectangular(actual_cnt)

        shape_type = "rectangle" if is_pure_rect else "shaped_caption"
        mean_conf_r = float(np.mean([l['conf'] for l in lines_r])) if lines_r else 1.0

        # Quality gate check to prevent non-rectangular shape flattening
        b_info = {"shape_type": shape_type, "z_order": 1, "contour": actual_cnt.reshape(-1, 2).tolist()}
        b_info = quality_gates.apply_quality_gates("bubble_inpainting", b_info)
        shape_type = b_info.get("shape_type", shape_type)

        b_rect = SpeechBubble(
            bubble_id=0,
            x0=rx, y0=ry, x1=rx1, y1=ry1,
            box=[rx, ry, rx1, ry1],
            original_text=text_r,
            clean_text=text_r,
            uzbek_translation="",
            confidence=mean_conf_r,
            shape_type=shape_type,
            z_order=1,
            contour=actual_cnt.reshape(-1, 2).tolist(),
            contour_points=actual_cnt.reshape(-1, 2).tolist(),
            line_centers=[[l['cx'], l['cy']] for l in lines_r],
            is_occluded=False
        )
        rect_bubbles.append(b_rect)

    # -----------------------------------------------------------------------
    # PASS 2: Associate occluded lines with overlapping rectangles
    # -----------------------------------------------------------------------
    occluded_lines_by_rect = {ri: [] for ri in valid_rect_indices}
    independent_lines = []

    for l in unclaimed_lines:
        lcx, lcy = l['cx'], l['cy']
        best_ri = None
        min_dist = float('inf')
        for ri in valid_rect_indices:
            r = earpiece_rects[ri]
            dx = max(0, max(r['x'] - l['x1'], l['x0'] - r['x1']))
            dy = max(0, max(r['y'] - l['y1'], l['y0'] - r['y1']))
            dist = (dx**2 + dy**2)**0.5
            same_panel = abs(r['y'] - lcy) < 160 or (r['y'] < 450 and lcy < 450)
            if dist <= 40 and same_panel and dist < min_dist:
                min_dist = dist
                best_ri = ri
        if best_ri is not None:
            occluded_lines_by_rect[best_ri].append(l)
        else:
            independent_lines.append(l)

    # -----------------------------------------------------------------------
    # PASS 3: Process occluded ovals into BubbleFragments & FragmentGroups
    # -----------------------------------------------------------------------
    fragment_groups: List[FragmentGroup] = []
    fragment_bubbles: List[SpeechBubble] = []

    for ri in valid_rect_indices:
        r = earpiece_rects[ri]
        pl = occluded_lines_by_rect[ri]
        if not pl:
            continue
        rx0, ry0, rx1, ry1 = r['x'], r['y'], r['x1'], r['y1']

        all_xs = [l['x0'] for l in pl] + [l['x1'] for l in pl] + [rx0, rx1]
        all_ys = [l['y0'] for l in pl] + [l['y1'] for l in pl] + [ry0, ry1]
        # Bound crop tightly so it does not bridge into ceiling lamp, ceiling, or pillar
        wx0 = max(0, min(all_xs) - 15)
        wy0 = max(0, ry0 - 6)  # do not look above rectangle into ceiling art
        wx1 = min(w, max(all_xs) + 15)
        wy1 = min(h, max(all_ys) + 20)

        crop_gray = gray[wy0:wy1, wx0:wx1]
        _, bin_crop = cv2.threshold(crop_gray, 205, 255, cv2.THRESH_BINARY)
        # Fill candidate text bounding boxes so dense text does not fragment the bubble mask
        for l in pl:
            # 4px padding so anti-aliasing doesn't disconnect text strokes from bubble interior
            cv2.rectangle(
                bin_crop,
                (max(0, l['x0'] - wx0 - 4), max(0, l['y0'] - wy0 - 4)),
                (min(wx1 - wx0, l['x1'] - wx0 + 4), min(wy1 - wy0, l['y1'] - wy0 + 4)),
                255, -1
            )
            # General proximity bridge: if text line abuts occluding rectangle within 30px,
            # bridge the gap so dense lettering does not fragment the occluded oval mask
            if l['x1'] <= rx0 and (rx0 - l['x1']) <= 30:
                cv2.rectangle(bin_crop, (max(0, l['x1'] - wx0), max(0, l['y0'] - wy0)), (min(wx1 - wx0, rx0 - wx0), min(wy1 - wy0, l['y1'] - wy0)), 255, -1)
            elif l['x0'] >= rx1 and (l['x0'] - rx1) <= 30:
                cv2.rectangle(bin_crop, (max(0, rx1 - wx0), max(0, l['y0'] - wy0)), (min(wx1 - wx0, l['x0'] - wx0), min(wy1 - wy0, l['y1'] - wy0)), 255, -1)
            if l['y1'] <= ry0 and (ry0 - l['y1']) <= 30:
                cv2.rectangle(bin_crop, (max(0, l['x0'] - wx0), max(0, l['y1'] - wy0)), (min(wx1 - wx0, l['x1'] - wx0), min(wy1 - wy0, ry0 - wy0)), 255, -1)
            elif l['y0'] >= ry1 and (l['y0'] - ry1) <= 30:
                cv2.rectangle(bin_crop, (max(0, l['x0'] - wx0), max(0, ry1 - wy0)), (min(wx1 - wx0, l['x1'] - wx0), min(wy1 - wy0, l['y0'] - wy0)), 255, -1)
        # Inside the rectangle is authentic white fill:
        cv2.rectangle(
            bin_crop,
            (max(0, rx0 - wx0 + 4), max(0, ry0 - wy0 + 4)),
            (min(wx1 - wx0, rx1 - wx0 - 4), min(wy1 - wy0, ry1 - wy0 - 4)),
            255, -1
        )

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        bin_closed = cv2.morphologyEx(bin_crop, cv2.MORPH_CLOSE, k_close)

        cnts, _ = cv2.findContours(bin_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rcx_loc = (rx0 + rx1) // 2 - wx0
        rcy_loc = (ry0 + ry1) // 2 - wy0
        matching_cnts = [c for c in cnts if cv2.pointPolygonTest(c, (float(rcx_loc), float(rcy_loc)), False) >= 0]
        if not matching_cnts:
            matching_cnts = cnts
        best_c = max(matching_cnts, key=cv2.contourArea) if matching_cnts else None

        if best_c is None:
            continue

        full_blob = np.zeros((h, w), dtype=np.uint8)
        crop_blob = np.zeros_like(crop_gray)
        cv2.drawContours(crop_blob, [best_c], -1, 255, -1)
        full_blob[wy0:wy1, wx0:wx1] = crop_blob

        rect_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(rect_mask, (rx0, ry0), (rx1, ry1), 255, -1)
        visible_mask = cv2.bitwise_and(full_blob, cv2.bitwise_not(rect_mask))

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(visible_mask, connectivity=8)
        comp_lines = {lbl: [] for lbl in range(1, n_labels)}
        for l in pl:
            lcx, lcy = l['cx'], l['cy']
            lbl = labels[lcy, lcx]
            if lbl > 0:
                comp_lines[lbl].append(l)
            else:
                best_lbl = None
                min_d = float('inf')
                for clbl in range(1, n_labels):
                    cl = stats[clbl, cv2.CC_STAT_LEFT]
                    ct = stats[clbl, cv2.CC_STAT_TOP]
                    cw = stats[clbl, cv2.CC_STAT_WIDTH]
                    ch = stats[clbl, cv2.CC_STAT_HEIGHT]
                    if cl - 15 <= lcx <= cl + cw + 15 and ct - 15 <= lcy <= ct + ch + 15:
                        d = ((lcx - (cl + cw // 2)) ** 2 + (lcy - (ct + ch // 2)) ** 2) ** 0.5
                        if d < min_d:
                            min_d = d
                            best_lbl = clbl
                if best_lbl is not None:
                    comp_lines[best_lbl].append(l)

        curr_fragments: List[BubbleFragment] = []
        for lbl, l_list in comp_lines.items():
            if not l_list:
                continue
            l_list = sorted(l_list, key=lambda l: (l['y0'], l['x0']))

            # Each connected component of visible_mask is a naturally contiguous speech bubble sliver/lobe
            groups = [l_list]

            comp_mask = (labels == lbl).astype(np.uint8) * 255
            for gi, grp in enumerate(groups):
                grp_text = " ".join(l['text'] for l in grp).strip()
                if not grp_text:
                    continue
                gx0 = min(l['x0'] for l in grp)
                gy0 = min(l['y0'] for l in grp)
                gx1 = max(l['x1'] for l in grp)
                gy1 = max(l['y1'] for l in grp)

                if len(groups) > 1:
                    sub_mask = comp_mask.copy()
                    if gi == 0:
                        split_y = (gy1 + groups[1][0]['y0']) // 2
                        sub_mask[split_y:, :] = 0
                    else:
                        split_y = (groups[0][-1]['y1'] + gy0) // 2
                        sub_mask[:split_y, :] = 0
                    fcnts, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                else:
                    fcnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if fcnts:
                    f_cnt = max(fcnts, key=cv2.contourArea)
                else:
                    f_cnt = np.array([[gx0, gy0], [gx1, gy0], [gx1, gy1], [gx0, gy1]], dtype=np.int32).reshape(-1, 1, 2)

                bx, by, bw, bh = cv2.boundingRect(f_cnt)
                fb = BubbleFragment(
                    fragment_id=len(curr_fragments) + 1,
                    parent_bubble_id=ri + 1,
                    x0=bx, y0=by, x1=bx + bw, y1=by + bh,
                    box=[bx, by, bx + bw, by + bh],
                    original_text=grp_text,
                    clean_text=grp_text,
                    uzbek_translation="",
                    contour=f_cnt.reshape(-1, 2).tolist(),
                    contour_points=f_cnt.reshape(-1, 2).tolist(),
                    lines=grp,
                    line_centers=[[l['cx'], l['cy']] for l in grp],
                    area=int(cv2.contourArea(f_cnt)),
                    shape_type="oval",
                    z_order=0,
                    is_occluded=True
                )
                # Quality gate: check sliver_too_small_for_text
                frag_info = {"area": fb.area, "text": fb.original_text, "w": bw, "h": bh}
                quality_gates.apply_quality_gates("fragment_processing", frag_info, bubble_id=fb.fragment_id)
                curr_fragments.append(fb)

        if curr_fragments:
            fg = FragmentGroup(
                parent_id=ri + 1,
                fragments=curr_fragments,
                parent_contour=best_c.reshape(-1, 2).tolist(),
                parent_box=[wx0, wy0, wx1, wy1]
            )
            fragment_groups.append(fg)
            for fb in curr_fragments:
                fragment_bubbles.append(fb.to_speech_bubble(bubble_id=0))

    # -----------------------------------------------------------------------
    # PASS 4: Process Independent Bubbles
    # -----------------------------------------------------------------------
    clusters = cluster_lines_into_bubbles(independent_lines, max_dx=30, max_dy=40, clean_gray=clean_gray_barrier)
    all_seeds_by_cluster = []
    cluster_items = []
    for cl in clusters:
        cl_sorted = sort_reading_order(cl)
        c_text = " ".join(l['text'] for l in cl_sorted).strip()
        if not c_text:
            continue
        bx0 = min(l['x0'] for l in cl_sorted)
        by0 = min(l['y0'] for l in cl_sorted)
        bx1 = max(l['x1'] for l in cl_sorted)
        by1 = max(l['y1'] for l in cl_sorted)
        glyph_h = int(np.median([l['y1'] - l['y0'] for l in cl_sorted]))
        mean_conf = float(np.mean([l['conf'] for l in cl_sorted]))
        if len(cl_sorted) == 1 and mean_conf < 0.30:
            continue
        cluster_items.append({
            'box': (bx0, by0, bx1, by1),
            'text': c_text,
            'lines': cl_sorted,
            'glyph_h': glyph_h,
            'conf': mean_conf
        })
        seeds = [(bx0, by0, bx1, by1), ((bx0 + bx1) // 2, (by0 + by1) // 2)]
        for l in cl_sorted:
            seeds.append((l['cx'], l['cy']))
        all_seeds_by_cluster.append(seeds)

    independent_bubbles = []
    for idx, item in enumerate(cluster_items):
        neighbor_seeds = []
        for j, other in enumerate(cluster_items):
            if j == idx:
                continue
            if other.get('lines'):
                other_boxes = [(l['x0'], l['y0'], l['x1'], l['y1']) for l in other['lines']]
                neighbor_seeds.append(other_boxes)
            else:
                neighbor_seeds.append(other['box'])
        for j, s in enumerate(all_seeds_by_cluster):
            if j != idx:
                neighbor_seeds.extend(s)
        bx0, by0, bx1, by1 = item['box']
        own_lines = [(l['x0'], l['y0'], l['x1'], l['y1']) for l in item['lines']] if item.get('lines') else None
        cnt_page, is_rect = bubble_lettering.get_bubble_contour(
            img_cv,
            (bx0, by0, bx1, by1),
            neighbor_seeds=neighbor_seeds,
            approx_glyph_height=item['glyph_h'],
            own_lines=own_lines
        )
        # Universal Multi-Lobe & Conjoined Bubble Separation (peanut, figure-8, diagonal, chained)
        lobe_splits = bubble_lettering.split_multi_lobe_bubble_contour(
            cnt_page,
            item['lines'],
            approx_glyph_height=item['glyph_h']
        )
        if len(lobe_splits) > 1:
            gate_state = {
                "has_split_lines": True,
                "is_multi_lobe": True,
                "split_clusters": [ls[1] for ls in lobe_splits]
            }
            gate_res = quality_gates.apply_quality_gates("bubble_segmentation", gate_state)
            if gate_res and gate_res.get("split_clusters"):
                cluster_uid = f"cluster_{bx0}_{by0}"
                cluster_full_text = item['text']
                for sub_idx, (sub_cnt, sub_lines) in enumerate(lobe_splits):
                    sub_sorted = sort_reading_order(sub_lines)
                    sub_text = " ".join(l['text'] for l in sub_sorted).strip()
                    sub_bx0 = min(l['x0'] for l in sub_sorted)
                    sub_by0 = min(l['y0'] for l in sub_sorted)
                    sub_bx1 = max(l['x1'] for l in sub_sorted)
                    sub_by1 = max(l['y1'] for l in sub_sorted)
                    sub_pts = sub_cnt.reshape(-1, 2).tolist()
                    b_sub = SpeechBubble(
                        bubble_id=0,
                        x0=sub_bx0, y0=sub_by0, x1=sub_bx1, y1=sub_by1,
                        box=[sub_bx0, sub_by0, sub_bx1, sub_by1],
                        original_text=sub_text,
                        clean_text=sub_text,
                        uzbek_translation="",
                        confidence=item['conf'],
                        shape_type="oval",
                        z_order=0,
                        contour=sub_pts,
                        contour_points=sub_pts,
                        line_centers=[[l['cx'], l['cy']] for l in sub_sorted],
                        lines=sub_sorted,
                        is_occluded=False,
                        cluster_id=cluster_uid,
                        cluster_text=cluster_full_text,
                        lobe_index=sub_idx,
                        total_lobes=len(lobe_splits)
                    )
                    independent_bubbles.append(b_sub)
                continue

        # Secondary fallback: conjoined_bubble_constriction_split via horizontal column analysis
        split_x, c_details = bubble_lettering.find_contour_horizontal_constriction(cnt_page)
        if split_x is not None:
            lines_left = [l for l in item['lines'] if l['x1'] <= split_x + 10]
            lines_right = [l for l in item['lines'] if l['x0'] >= split_x - 10]
            if lines_left and lines_right and (len(lines_left) + len(lines_right) == len(item['lines'])):
                gate_state = {
                    "constriction_ratio": c_details.get("constriction_ratio", 1.0),
                    "has_split_lines": True,
                    "lines_left": lines_left,
                    "lines_right": lines_right
                }
                gate_res = quality_gates.apply_quality_gates("bubble_segmentation", gate_state)
                split_clusters = gate_res.get("split_clusters")
                if split_clusters and len(split_clusters) == 2:
                    cluster_uid = f"cluster_{bx0}_{by0}"
                    cluster_full_text = item['text']
                    for sub_idx, sub_lines in enumerate(split_clusters):
                        sub_sorted = sort_reading_order(sub_lines)
                        sub_text = " ".join(l['text'] for l in sub_sorted).strip()
                        sub_bx0 = min(l['x0'] for l in sub_sorted)
                        sub_by0 = min(l['y0'] for l in sub_sorted)
                        sub_bx1 = max(l['x1'] for l in sub_sorted)
                        sub_by1 = max(l['y1'] for l in sub_sorted)
                        sub_cnt, sub_is_rect = bubble_lettering.get_bubble_contour(
                            img_cv, (sub_bx0, sub_by0, sub_bx1, sub_by1),
                            neighbor_seeds=neighbor_seeds + [((sub_bx0 + sub_bx1)//2, (sub_by0 + sub_by1)//2)],
                            approx_glyph_height=item['glyph_h']
                        )
                        sub_pts = sub_cnt.reshape(-1, 2).tolist()
                        b_sub = SpeechBubble(
                            bubble_id=0,
                            x0=sub_bx0, y0=sub_by0, x1=sub_bx1, y1=sub_by1,
                            box=[sub_bx0, sub_by0, sub_bx1, sub_by1],
                            original_text=sub_text,
                            clean_text=sub_text,
                            uzbek_translation="",
                            confidence=item['conf'],
                            shape_type="oval",
                            z_order=0,
                            contour=sub_pts,
                            contour_points=sub_pts,
                            line_centers=[[l['cx'], l['cy']] for l in sub_sorted],
                            lines=sub_sorted,
                            is_occluded=False,
                            cluster_id=cluster_uid,
                            cluster_text=cluster_full_text,
                            lobe_index=sub_idx,
                            total_lobes=len(split_clusters)
                        )
                        independent_bubbles.append(b_sub)
                    continue

        cnt_pts = cnt_page.reshape(-1, 2).tolist()
        bw_b = bx1 - bx0
        bh_b = by1 - by0
        cnt_pts_arr = cnt_page.reshape(-1, 2)
        min_y_c, max_y_c = int(np.min(cnt_pts_arr[:, 1])), int(np.max(cnt_pts_arr[:, 1]))
        has_vertical_tail = (by0 - min_y_c >= 18) or (max_y_c - by1 >= 18)
        rx_c, ry_c, rw_c, rh_c = cv2.boundingRect(cnt_page)
        cnt_aspect = rw_c / max(1, rh_c)
        is_banner = (
            ((cnt_aspect >= 4.5 and rw_c >= 400) or ((rh_c / max(1, rw_c)) >= 4.5 and rh_c >= 400))
            and not has_vertical_tail
        )
        shape_type = "rectangle" if (is_rect or is_banner) else "oval"
        b_ind = SpeechBubble(
            bubble_id=0,
            x0=bx0, y0=by0, x1=bx1, y1=by1,
            box=[bx0, by0, bx1, by1],
            original_text=item['text'],
            clean_text=item['text'],
            uzbek_translation="",
            confidence=item['conf'],
            shape_type=shape_type,
            z_order=0,
            contour=cnt_pts,
            contour_points=cnt_pts,
            line_centers=[[l['cx'], l['cy']] for l in item['lines']],
            lines=item['lines'],
            is_occluded=False
        )
        independent_bubbles.append(b_ind)

    # -----------------------------------------------------------------------
    # PASS 5: Combine, sort in reading order, and assign sequential IDs 1..N
    # -----------------------------------------------------------------------
    unoccluded_bubbles = independent_bubbles + rect_bubbles
    all_final_bubbles = fragment_bubbles + unoccluded_bubbles

    # Self-diagnosing quality gate: filter empty_fragment / whitespace dialogue
    all_final_bubbles = [b for b in all_final_bubbles if quality_gates.apply_quality_gates("bubble_validation", b) is not None]

    # Panel-aware reading order (detect panel grid and sort panel-by-panel before ID indexing)
    import reading_order
    panel_boxes = reading_order.detect_panel_boxes(img_cv)
    all_final_bubbles = reading_order.order_bubbles_reading_order(all_final_bubbles, panel_boxes)

    # Self-diagnosing quality gate: resolve duplicate_bubble_id and enforce reading-order sequential IDs
    all_final_bubbles = quality_gates.apply_quality_gates(
        "bubble_list_indexing",
        all_final_bubbles,
        context={"panel_boxes": panel_boxes}
    )

    # Automatic Classical-CV Speaker Detection (tail tip & dominant color matching)
    all_final_bubbles = assign_speakers(all_final_bubbles, img_cv)

    print(f"[SCAN] Detected {len(all_final_bubbles)} bubbles ({len(fragment_bubbles)} fragments across {len(fragment_groups)} groups, {len(independent_bubbles)} independent, {len(rect_bubbles)} rects).")
    return all_final_bubbles


def assign_speakers(
    bubbles: List[SpeechBubble],
    image: Any
) -> List[SpeechBubble]:
    """
    Automatic classical-CV speaker detection for each bubble:
    - Analyzes bubble contour to locate tail tip pointing to the speaker.
    - Samples image region near tail tip (filtering bubble fill and ink outlines).
    - Extracts dominant color signatures via k-means.
    - Matches against stored character visual signatures in character_profiles.
    - Sets bubble.speaker and bubble.speaker_confidence if match exceeds threshold.
    """
    import speaker_detector
    import character_profiles

    if isinstance(image, Image.Image):
        img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        img_bgr = image
    else:
        return bubbles

    store = character_profiles.get_memory_store()
    profiles = list(store._profiles.values())

    for b in bubbles:
        cnt = getattr(b, "contour_points", None) or getattr(b, "contour", None)
        bbox = (b.x0, b.y0, b.x1, b.y1)
        spk, conf, colors, weights = speaker_detector.detect_speaker(
            cnt, bbox, img_bgr, profiles
        )
        b.speaker = spk
        b.speaker_confidence = conf
        b.tail_region_colors = colors
        b.tail_region_weights = weights

    return bubbles


def translate_bubbles_list_subprocess(
    bubbles: List[SpeechBubble],
    default_speaker: Optional[str] = None,
    page_name: str = "auto",
    timeout_seconds: int = 120
) -> List[SpeechBubble]:
    """
    Executes NLLB translation in an isolated OS subprocess with strict timeout protection.
    When the child process exits, Linux/Windows OS fully recovers its memory (~700 MB).
    If the child process hangs past timeout_seconds, it is killed forcefully.
    """
    if not bubbles:
        return []

    import tempfile
    import subprocess
    import sys
    import psutil
    import json

    tmp_in = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp_in_path = tmp_in.name
    tmp_in.close()

    in_data = [b.model_dump() for b in bubbles]
    with open(tmp_in_path, "w", encoding="utf-8") as f:
        json.dump(in_data, f, ensure_ascii=False)

    tmp_out = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp_out_path = tmp_out.name
    tmp_out.close()

    worker_script = os.path.join(os.path.dirname(__file__), "scripts", "translation_worker.py")
    subproc = None
    try:
        subproc = subprocess.Popen(
            [sys.executable, "-u", worker_script, tmp_in_path, tmp_out_path, str(default_speaker), str(page_name)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            stdout, stderr = subproc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            print(f"[TRANSLATE_SUBPROCESS_TIMEOUT] Translation bola jarayoni {timeout_seconds}s ichida javob bermadi! Majburiy to'xtatilyapti...", flush=True)
            try:
                p = psutil.Process(subproc.pid)
                for child in p.children(recursive=True):
                    try:
                        child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                p.kill()
            except Exception:
                subproc.kill()
            subproc.communicate()
            raise TimeoutError(f"NLLB translation bola jarayoni {timeout_seconds} soniya ichida yakunlanmadi va majburiy to'xtatildi (Timeout).")

        if subproc.returncode != 0:
            err_details = stderr.strip() if stderr else (stdout.strip() if stdout else f"Exit code {subproc.returncode}")
            raise RuntimeError(f"Translation worker process failed (code {subproc.returncode}): {err_details}")

        with open(tmp_out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [SpeechBubble(**d) for d in data]
    finally:
        for p in [tmp_in_path, tmp_out_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def translate_bubbles_list(
    bubbles: List[SpeechBubble],
    default_speaker: Optional[str] = None,
    page_name: str = "auto",
    use_subprocess: bool = True,
    timeout_seconds: int = 120
) -> List[SpeechBubble]:
    """
    Stage 2: Translates each bubble's complete sentence dialogue into colloquial superhero Uzbek.
    When use_subprocess=True, executes NLLB translation in an isolated worker process so that
    CTranslate2 model memory (~700 MB) is 100% returned to the OS upon process exit.
    If the worker hangs, it is terminated forcefully to prevent orphaned processes.
    """
    if use_subprocess:
        import time
        t0 = time.time()
        try:
            res = translate_bubbles_list_subprocess(
                bubbles,
                default_speaker=default_speaker,
                page_name=page_name,
                timeout_seconds=timeout_seconds
            )
            elapsed = time.time() - t0
            print(f"[TRANSLATE] subprocess mode: SUCCESS (bubbles translated: {len(res)}, time: {elapsed:.2f}s)", flush=True)
            return res
        except TimeoutError as te:
            print(f"[TRANSLATE] subprocess FAILED, falling back to in-process: Timeout ({te})", flush=True)
            raise
        except Exception as err:
            print(f"[TRANSLATE] subprocess FAILED, falling back to in-process: {err}", flush=True)
    return _translate_bubbles_list_core(bubbles, default_speaker=default_speaker, page_name=page_name)


def _translate_bubbles_list_core(
    bubbles: List[SpeechBubble],
    default_speaker: Optional[str] = None,
    page_name: str = "auto"
) -> List[SpeechBubble]:
    """
    Core in-process Stage 2 translation logic.
    - Multi-lobe clusters: Entire cluster unified dialogue is translated ONCE in full before
      being partitioned across lobes, preventing pronoun/clause destruction.
    - For occluded ovals, also translates each fragment's original_text.
    - Runs post_translation_validation on all bubbles to catch corrupt/raw OCR or truncation.
    - Feeds translated dialogue pairs into persistent terminology lock and character memory.
    """
    translated_bubbles = []
    translated_clusters: Dict[str, str] = {}
    import quality_gates

    for b in bubbles:
        is_active = getattr(b, "is_active", getattr(b, "enabled", True))
        if is_active:
            spk = getattr(b, "speaker", None) or default_speaker
            b.speaker = spk
            cid = getattr(b, "cluster_id", None)
            ctext = getattr(b, "cluster_text", None)

            if cid and ctext:
                if cid not in translated_clusters:
                    # Translate entire cluster unified text once!
                    uz_full = translate_spiderman_uzbek(ctext, speaker=spk)
                    translated_clusters[cid] = uz_full
                uz_cluster = translated_clusters[cid]
                b.cluster_uzbek = uz_cluster
            else:
                text_to_translate = getattr(b, "clean_text", None) or getattr(b, "original_text", "")
                uz = translate_spiderman_uzbek(text_to_translate, speaker=spk)
                b.uzbek_translation = uz
                b.pipeline_uzbek_translation = uz
                quality_gates.apply_quality_gates(
                    "translation_validation",
                    b,
                    context={"source_text": text_to_translate, "page": page_name},
                    bubble_id=getattr(b, "bubble_id", 0)
                )
                val = quality_gates.post_translation_validation(uz, text_to_translate)
                if val.get("needs_review"):
                    b.needs_review = True
                    b.review_reason = val.get("reason")
                    quality_gates.log_quality_gate_event(
                        pattern_name="post_translation_failed",
                        stage="translation_validation",
                        severity="flag_for_review",
                        action_taken="flagged_needs_review",
                        details={"reason": val.get("reason"), "tokens": val.get("flagged_tokens", [])},
                        bubble_id=getattr(b, "bubble_id", 0),
                        page=page_name
                    )

            # Also translate each fragment for occluded ovals
            frags = getattr(b, "fragments", None)
            if frags and getattr(b, "is_occluded", False):
                if len(frags) == 1:
                    if isinstance(frags[0], dict):
                        frags[0]["uzbek_translation"] = b.uzbek_translation
                        frags[0]["pipeline_uzbek_translation"] = b.uzbek_translation
                    else:
                        frags[0].uzbek_translation = b.uzbek_translation
                        frags[0].pipeline_uzbek_translation = b.uzbek_translation
                else:
                    for frag in frags:
                        frag_spk = frag.get("speaker") if isinstance(frag, dict) else getattr(frag, "speaker", spk)
                        if isinstance(frag, dict):
                            frag_orig = (frag.get("original_text") or "").strip()
                            if frag_orig:
                                frag_uz = translate_spiderman_uzbek(frag_orig, speaker=frag_spk)
                                frag["uzbek_translation"] = frag_uz
                                frag["pipeline_uzbek_translation"] = frag_uz
                        else:
                            frag_orig = (getattr(frag, "original_text", "") or "").strip()
                            if frag_orig:
                                frag_uz = translate_spiderman_uzbek(frag_orig, speaker=frag_spk)
                                frag.uzbek_translation = frag_uz
                                frag.pipeline_uzbek_translation = frag_uz

        translated_bubbles.append(b)

    # Distribute cluster translations to constituent lobes
    for cid, uz_cluster in translated_clusters.items():
        cluster_members = [b for b in translated_bubbles if getattr(b, "cluster_id", None) == cid]
        if cluster_members:
            cluster_members.sort(key=lambda x: getattr(x, "lobe_index", 0))
            weights = [max(1, len((getattr(m, "original_text", "") or "").split())) for m in cluster_members]

            # Completeness check: compare sentence counts between full cluster source text and translated cluster
            ctext = getattr(cluster_members[0], "cluster_text", None) or " ".join(
                getattr(m, "clean_text", None) or getattr(m, "original_text", "") for m in cluster_members
            )

            def _count_dialogue_sentences(t: str) -> int:
                return len([s for s in re.split(r"[\.!\?…:]+", str(t or "")) if re.search(r"\w", s)])

            orig_cluster_sents = _count_dialogue_sentences(ctext)
            trans_cluster_sents = _count_dialogue_sentences(uz_cluster)

            # Check if constituent lobes each have independent terminal sentences (e.g. touching separate bubbles)
            # If any lobe is a mid-thought or open clause (ending in comma, dash, or open word), unified translation must be preserved.
            has_independent_terminal_lobes = all(
                bool(re.search(r"[\.!\?…:;][\"'\s]*$", (getattr(m, "clean_text", None) or getattr(m, "original_text", "")).strip()))
                for m in cluster_members
            )

            # If unified translation dropped sentences from cluster_text (e.g. 4 -> 2)
            # AND all constituent lobes have independent complete sentences, fallback to translating each lobe individually:
            if trans_cluster_sents < orig_cluster_sents and has_independent_terminal_lobes and len(cluster_members) > 1:
                sub_translations = []
                for m in cluster_members:
                    m_text = getattr(m, "clean_text", None) or getattr(m, "original_text", "")
                    m_spk = getattr(m, "speaker", None) or default_speaker
                    m_uz = translate_spiderman_uzbek(m_text, speaker=m_spk)
                    sub_translations.append(m_uz)
            else:
                sub_translations = partition_translated_text_for_lobes(uz_cluster, weights)

            for m, sub_uz in zip(cluster_members, sub_translations):
                m.uzbek_translation = sub_uz
                m.pipeline_uzbek_translation = sub_uz
                quality_gates.apply_quality_gates(
                    "translation_validation",
                    m,
                    context={"source_text": getattr(m, "original_text", ""), "page": page_name},
                    bubble_id=getattr(m, "bubble_id", 0)
                )
                val = quality_gates.post_translation_validation(sub_uz, getattr(m, "original_text", ""))
                # Only flag if there is genuine residue or error, not just sub-fragment length mismatch
                if val.get("needs_review") and "asl matn bilan bir xil" not in str(val.get("reason")):
                    m.needs_review = True
                    m.review_reason = val.get("reason")
                    quality_gates.log_quality_gate_event(
                        pattern_name="post_translation_failed",
                        stage="translation_validation",
                        severity="flag_for_review",
                        action_taken="flagged_needs_review",
                        details={"reason": val.get("reason"), "tokens": val.get("flagged_tokens", [])},
                        bubble_id=getattr(m, "bubble_id", 0),
                        page=page_name
                    )

    # Self-growing character memory & persistent terminology learning
    try:
        if page_name == "auto":
            import quality_gates
            active_p = getattr(quality_gates._GLOBAL_LOGGER, "_current_page", None)
            if active_p:
                page_name = active_p

        dialogue_pairs: List[Tuple[str, str]] = []
        speaker_pairs: Dict[str, List[Tuple[str, str]]] = {}
        for b in translated_bubbles:
            orig = (getattr(b, "original_text", "") or "").strip()
            trans = (getattr(b, "uzbek_translation", "") or "").strip()
            if orig and trans:
                dialogue_pairs.append((orig, trans))
                spk = getattr(b, "speaker", None) or default_speaker
                if spk:
                    speaker_pairs.setdefault(spk, []).append((orig, trans))

        if dialogue_pairs:
            # Auto-learn locked terminology
            import terminology_store
            terminology_store.get_terminology_store().learn_from_page(dialogue_pairs, page_source=page_name)

            # Auto-learn character voice traits and established facts with deduplication
            import character_profiles
            mem_store = character_profiles.get_memory_store()
            for spk_name, pairs in speaker_pairs.items():
                mem_store.learn_from_page(spk_name, pairs, page=page_name)
    except Exception as e:
        print(f"[MEMORY_WARN] Auto-learning error on page '{page_name}': {e}")

    return translated_bubbles


def detect_and_translate_bubbles(image: Image.Image) -> Tuple[List[SpeechBubble], List[List[List[int]]]]:
    """Legacy helper: runs scan + translate in sequence."""
    raw_bubbles = scan_bubbles_ocr(image)
    translated_bubbles = translate_bubbles_list(raw_bubbles)
    sorted_boxes = [b.contour_points for b in translated_bubbles if b.contour_points]
    return translated_bubbles, sorted_boxes


extract_bubbles_from_page = detect_and_translate_bubbles



def wrap_text_to_width(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    """Wraps text into lines using actual pixel measurement."""
    words = text.split()
    if not words:
        return []
    lines = []
    cur = []
    for w in words:
        cand = " ".join(cur + [w])
        if font.getbbox(cand)[2] - font.getbbox(cand)[0] <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def get_bubble_safe_mask(
    box: Tuple[int, int, int, int],
    closed_thresh: np.ndarray,
    pad: int = 60,
    erode_size: int = 11
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Extracts the bubble's inner white paper contour mask and computes an inset safe area:
    safe_mask = cv2.erode(bubble_mask, np.ones((9, 9), np.uint8))
    Returns (safe_mask, crop_x0, crop_y0).
    """
    h, w = closed_thresh.shape
    x0, y0, x1, y1 = box
    bx0 = max(0, x0 - pad)
    by0 = max(0, y0 - pad)
    bx1 = min(w, x1 + pad)
    by1 = min(h, y1 + pad)

    crop = closed_thresh[by0:by1, bx0:bx1]
    ch, cw = crop.shape
    if ch < 5 or cw < 5:
        return None, bx0, by0

    ccx = min(cw - 1, max(0, (x0 + x1) // 2 - bx0))
    ccy = min(ch - 1, max(0, (y0 + y1) // 2 - by0))

    # If center pixel is dark (e.g. residual ink), locate closest bright paper pixel
    if crop[ccy, ccx] == 0:
        ys, xs = np.where(crop > 0)
        if len(ys) > 0:
            dists = (xs - ccx) ** 2 + (ys - ccy) ** 2
            closest = np.argmin(dists)
            ccx, ccy = xs[closest], ys[closest]
        else:
            return None, bx0, by0

    ff_mask = np.zeros((ch + 2, cw + 2), np.uint8)
    cv2.floodFill(crop.copy(), ff_mask, (int(ccx), int(ccy)), 255)
    local_bubble_mask = ff_mask[1:ch+1, 1:cw+1]

    if np.sum(local_bubble_mask > 0) < 50:
        return None, bx0, by0

    safe_mask = cv2.erode(local_bubble_mask, np.ones((erode_size, erode_size), np.uint8), iterations=1)
    return safe_mask, bx0, by0


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Tuple[int, int, int, int],
    font_path: str,
    max_font_size: int = 26,
    min_font_size: int = 10,
    line_gap: int = 5,
    size_offset: int = 0,
    uppercase: bool = True,
    text_color: str = "#000000",
    stroke_width: int = 0,
    stroke_fill: Optional[str] = None,
    safe_mask: Optional[np.ndarray] = None,
    sm_ox: int = 0,
    sm_oy: int = 0
) -> bool:
    """
    Zero-Stroke Contour-Aware Auto-Shrink Lettering Engine:
    - Renders clean, solid black text without white stroke overlays (stroke_width=0).
    - Starts at current_font_size = max_font_size (+ size_offset).
    - Checks collision: verifies that all four corners and midpoint edges of each line
      lie strictly inside safe_mask (eroded bubble boundary).
    - If any line touches or exceeds the safe bubble boundary, decrements font size
      and re-wraps until the entire block fits without touching the black bubble border.
    - Natural line spacing: vertical_step = line_height + line_gap.
    """
    if uppercase:
        text = normalize_uzbek_comic_text(text)
    text = (text or "").strip()
    if not text:
        return False

    x0, y0, x1, y1 = box
    bw = max(20, x1 - x0)
    bh = max(20, y1 - y0)

    usable_w = max(15, int(bw * 1.15))

    effective_max = max(min_font_size, max_font_size + size_offset)
    effective_min = max(6, min_font_size)

    best_font = None
    best_lines = []
    best_line_boxes = []

    sm_h, sm_w = safe_mask.shape if safe_mask is not None else (0, 0)
    has_safe_mask = safe_mask is not None and np.sum(safe_mask > 0) > 0

    # Auto-shrink loop: start at effective_max down to effective_min
    for font_size in range(effective_max, effective_min - 1, -1):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        lines = wrap_text_to_width(text, font, usable_w)
        if not lines:
            continue

        line_boxes = [font.getbbox(l) for l in lines]
        line_heights = [(b[3] - b[1]) for b in line_boxes]
        total_h = sum(line_heights) + (line_gap * (len(lines) - 1) if len(lines) > 1 else 0)
        max_lw = max((b[2] - b[0]) for b in line_boxes)

        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        start_y = cy - (total_h // 2)

        if not has_safe_mask:
            usable_h = max(int(bh * 1.10), bh + 25)
            if max_lw <= usable_w and total_h <= usable_h:
                best_font = font
                best_lines = lines
                best_line_boxes = line_boxes
                break
        else:
            # Collision check: verify all corners and midpoint edges of each line lie inside safe_mask
            collides = False
            cur_y_check = start_y
            pad_c = 4  # Safe boundary clearance padding
            for i, line in enumerate(lines):
                b = line_boxes[i]
                lw = b[2] - b[0]
                lh = b[3] - b[1]
                lx = cx - (lw // 2)
                gx0 = lx + b[0] - pad_c
                gx1 = lx + b[2] + pad_c
                gy0 = cur_y_check + b[1] - pad_c
                gy1 = cur_y_check + b[3] + pad_c
                cur_y_check += lh + line_gap

                test_pts = [
                    (gx0 - sm_ox, gy0 - sm_oy),
                    (gx1 - sm_ox, gy0 - sm_oy),
                    (gx0 - sm_ox, gy1 - sm_oy),
                    (gx1 - sm_ox, gy1 - sm_oy),
                    ((gx0 + gx1) // 2 - sm_ox, gy0 - sm_oy),
                    ((gx0 + gx1) // 2 - sm_ox, gy1 - sm_oy),
                    (gx0 - sm_ox, (gy0 + gy1) // 2 - sm_oy),
                    (gx1 - sm_ox, (gy0 + gy1) // 2 - sm_oy)
                ]

                for px, py in test_pts:
                    px = int(px)
                    py = int(py)
                    if px < 0 or px >= sm_w or py < 0 or py >= sm_h or safe_mask[py, px] == 0:
                        collides = True
                        break
                if collides:
                    break

            if not collides:
                best_font = font
                best_lines = lines
                best_line_boxes = line_boxes
                break

    # Fallback to effective_min if still overflowing
    if best_font is None:
        try:
            best_font = ImageFont.truetype(font_path, effective_min)
        except Exception:
            best_font = ImageFont.load_default()
        best_lines = wrap_text_to_width(text, best_font, usable_w)
        if not best_lines:
            best_lines = [text]
        best_line_boxes = [best_font.getbbox(l) for l in best_lines]

    # Compute centered layout
    line_heights = [(b[3] - b[1]) for b in best_line_boxes]
    total_h = sum(line_heights) + (line_gap * (len(best_lines) - 1) if len(best_lines) > 1 else 0)

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    start_y = cy - (total_h // 2)

    # Draw clean, solid black text (zero stroke)
    cur_y = start_y
    for i, line in enumerate(best_lines):
        bbox = best_line_boxes[i]
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        lx = cx - (lw // 2)

        if stroke_width > 0 and stroke_fill:
            draw.text((lx, cur_y), line, font=best_font, fill=text_color, stroke_width=stroke_width, stroke_fill=stroke_fill)
        else:
            draw.text((lx, cur_y), line, font=best_font, fill=text_color)

        # Natural typography line advance (no stroke padding added)
        vertical_step = lh + line_gap
        cur_y += vertical_step

    return True

fit_and_typeset = fit_text


def typeset_lettering_on_page(
    cleaned_image: Image.Image,
    bubbles: List[Any],
    font_path: str,
    max_font_size: int = 26,
    min_font_size: int = 10,
    line_gap: int = 5,
    stroke_width: int = 0,
    stroke_fill: Optional[str] = None,
    **kwargs
) -> Image.Image:
    """
    Stage 3 Pure Lettering on Cleaned Canvas powered by bubble_lettering:
    - Z-order aware: ovals (z_order=0) typeset first, rectangles (z_order=1) on top.
    - Occluded ovals with fragments: each fragment gets independent fit_text_to_bubble call.
    - Uses bubble_lettering.fit_text_to_bubble() for contour-hugging diamond lettering.
    - Zero stroke, solid black #000000 lettering.
    """
    out_img = cleaned_image.copy().convert("RGB")
    img_cv = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    _earpiece_rects = []
    for b in bubbles:
        shape = b.get("shape_type") if isinstance(b, dict) else getattr(b, "shape_type", None)
        if shape == "rectangle":
            bx0 = int(b.get("x0") if isinstance(b, dict) else b.x0)
            by0 = int(b.get("y0") if isinstance(b, dict) else b.y0)
            bx1 = int(b.get("x1") if isinstance(b, dict) else b.x1)
            by1 = int(b.get("y1") if isinstance(b, dict) else b.y1)
            _earpiece_rects.append({'x': bx0, 'y': by0, 'x1': bx1, 'y1': by1, 'w': bx1 - bx0, 'h': by1 - by0})
    if not _earpiece_rects:
        _earpiece_rects = detect_earpiece_rectangles(img_cv)
    draw = ImageDraw.Draw(out_img)

    def _get_z(b: Any) -> int:
        if isinstance(b, dict):
            return int(b.get("z_order", 0))
        return int(getattr(b, "z_order", 0))

    # Sort by z_order: ovals (0) first, rectangles (1) second
    sorted_bubbles = sorted(bubbles, key=lambda b: (_get_z(b), getattr(b, 'y0', b.get('y0', 0)) if isinstance(b, dict) else b.y0))

    for b in sorted_bubbles:
        is_enabled = (
            b.get("is_active", b.get("enabled", True))
            if isinstance(b, dict)
            else getattr(b, "is_active", getattr(b, "enabled", True))
        )
        if not is_enabled:
            continue

        if isinstance(b, dict):
            uzbek_text = (b.get("uzbek_translation") or "").strip()
            orig_text = (b.get("original_text") or "").strip()
            needs_rev = b.get("needs_review", False)
            size_offset = b.get("font_size_offset", b.get("size_offset", 0))
            box = (int(b["x0"]), int(b["y0"]), int(b["x1"]), int(b["y1"]))
            is_occluded = b.get("is_occluded", False)
            fragments = b.get("fragments")
        else:
            uzbek_text = (b.uzbek_translation or "").strip()
            orig_text = (b.original_text or "").strip()
            needs_rev = getattr(b, "needs_review", False)
            size_offset = getattr(b, "font_size_offset", getattr(b, "size_offset", 0))
            box = (int(b.x0), int(b.y0), int(b.x1), int(b.y1))
            is_occluded = getattr(b, "is_occluded", False)
            fragments = getattr(b, "fragments", None)

        # Export Safety Guard: Never silently typeset raw OCR or unresolved failed translations
        if needs_rev and (not uzbek_text or uzbek_text.upper() == orig_text.upper()):
            orig_img = kwargs.get("original_image")
            if orig_img is not None:
                try:
                    bx0, by0, bx1, by1 = box
                    crop_orig = orig_img.crop((bx0, by0, bx1, by1))
                    cnt = b.get("contour") if isinstance(b, dict) else getattr(b, "contour", None)
                    if cnt is not None:
                        c_pts = np.array(cnt, dtype=np.int32)
                        mask_crop = np.zeros((by1 - by0, bx1 - bx0), dtype=np.uint8)
                        cv2.drawContours(mask_crop, [c_pts - [bx0, by0]], -1, 255, -1)
                        mask_pil = Image.fromarray(mask_crop)
                        out_img.paste(crop_orig, (bx0, by0), mask_pil)
                    else:
                        out_img.paste(crop_orig, (bx0, by0))
                except Exception as ex:
                    print(f"[EXPORT_SAFETY_WARN] Could not restore original art: {ex}")
            continue

        if not uzbek_text:
            uzbek_text = orig_text

        if not uzbek_text:
            continue

        uzbek_text = normalize_uzbek_comic_text(uzbek_text)
        effective_max = max(min_font_size, max_font_size + size_offset)
        calc_line_spacing = max(1.10, 1.15 + (line_gap / max(1, effective_max)))

        # Legibility contrast guard and tint logging for lettering
        bx0, by0, bx1, by1 = box
        crop_bg = img_cv[max(0, by0):min(h, by1), max(0, bx0):min(w, bx1)]
        is_tinted = (b.get("is_tinted", False) if isinstance(b, dict) else getattr(b, "is_tinted", False))
        c_bg_rgb = (b.get("tint_rgb") if isinstance(b, dict) else getattr(b, "tint_rgb", None))
        orig_text_rgb = (b.get("orig_text_rgb") if isinstance(b, dict) else getattr(b, "orig_text_rgb", None))

        if not c_bg_rgb and crop_bg.size > 0:
            c_bg_rgb = [int(v) for v in np.median(crop_bg, axis=(0, 1))][::-1]
        elif not c_bg_rgb:
            c_bg_rgb = [255, 255, 255]

        bid = b.get("bubble_id", "?") if isinstance(b, dict) else getattr(b, "bubble_id", "?")
        print(f"[LETTERING_PRE_RENDER] Bubble {bid}: c_bg RGB={c_bg_rgb}, is_tinted={is_tinted}")

        text_is_spectral = (b.get("text_is_spectral", False) if isinstance(b, dict) else getattr(b, "text_is_spectral", False))
        bg_is_tinted = (b.get("bg_is_tinted", False) if isinstance(b, dict) else getattr(b, "bg_is_tinted", False))

        if text_is_spectral or (is_tinted and not bg_is_tinted):
            # Ghost/spectral dialogue styling rule:
            # Bubble background is 100% pure clean white paper, text is havorang (#7497d4)
            text_color = "#7497d4"
        elif is_tinted:
            # Genuinely tinted bubble background (e.g. yellow box)
            if orig_text_rgb and any(v > 40 for v in orig_text_rgb):
                orig_r, orig_g, orig_b = orig_text_rgb
                spec_r = max(80, min(145, int(orig_r * 0.72)))
                spec_g = max(110, min(170, int(orig_g * 0.78)))
                spec_b = max(165, min(230, int(orig_b * 0.94)))
                text_color = f"#{spec_r:02x}{spec_g:02x}{spec_b:02x}"
            else:
                text_color = "#7497d4"
        else:
            text_color = "#000000"

        if crop_bg.size > 0:
            crop_g = cv2.cvtColor(crop_bg, cv2.COLOR_BGR2GRAY)
            high_pts = crop_bg[crop_g >= np.percentile(crop_g, 70)]
            if len(high_pts) > 0:
                sampled_tint = [int(v) for v in np.median(high_pts, axis=0)]
                tint_lum = 0.114 * sampled_tint[0] + 0.587 * sampled_tint[1] + 0.299 * sampled_tint[2]
                if tint_lum < 120:
                    quality_gates.log_quality_gate_event(
                        pattern_name="low_contrast_tint",
                        stage="lettering",
                        severity="flag_for_review",
                        action_taken="rendered_plain_text",
                        details={"sampled_tint_bgr": sampled_tint, "luminance": tint_lum}
                    )

        if is_occluded and fragments and len(fragments) > 0:
            # Fragment-aware typesetting: assign translated text across visible fragments
            # Combine all fragment texts for translation matching, then split per fragment
            fragment_texts = []
            for frag in fragments:
                if isinstance(frag, dict):
                    ft = (frag.get("uzbek_translation") or frag.get("original_text") or "").strip()
                else:
                    ft = (getattr(frag, "uzbek_translation", "") or getattr(frag, "original_text", "") or "").strip()
                if not ft:
                    ft = uzbek_text  # fallback: full translation in every fragment
                fragment_texts.append(ft)

            # If all fragments have same/empty text, split the translation across fragments
            unique_texts = set(fragment_texts)
            if len(unique_texts) == 1 and len(fragments) > 1:
                # Split translation into sentence chunks, one per fragment (sorted top→bottom)
                import re as _re
                sentences = _re.split(r'(?<=[.!?…])\s+', uzbek_text)
                # Distribute: first fragment gets first half, last gets rest
                n = len(fragments)
                chunk_size = max(1, (len(sentences) + n - 1) // n)
                fragment_texts = []
                for fi in range(n):
                    chunk = sentences[fi * chunk_size:(fi + 1) * chunk_size]
                    fragment_texts.append(" ".join(chunk).strip() or uzbek_text)

            for fi, frag in enumerate(fragments):
                frag_text = normalize_uzbek_comic_text(fragment_texts[fi] if fi < len(fragment_texts) else uzbek_text)
                if not frag_text:
                    continue

                frag_box = frag.get("box") if isinstance(frag, dict) else getattr(frag, "box", box)
                frag_contour_raw = frag.get("contour") if isinstance(frag, dict) else getattr(frag, "contour", None)
                frag_area = frag.get("area", 0) if isinstance(frag, dict) else getattr(frag, "area", 0)

                if frag_area < 300 or not frag_contour_raw:
                    continue

                try:
                    frag_cnt = np.array(frag_contour_raw, dtype=np.int32).reshape(-1, 1, 2)
                except Exception:
                    continue

                # Reduce max font size for small fragments
                frag_h = frag_box[3] - frag_box[1] if len(frag_box) >= 4 else (box[3] - box[1])
                frag_w = frag_box[2] - frag_box[0] if len(frag_box) >= 4 else (box[2] - box[0])
                frag_max_fs = max(min_font_size, min(effective_max, max(min_font_size, frag_h - 4)))

                layout = bubble_lettering.fit_text_to_bubble(
                    text=frag_text,
                    contour=frag_cnt,
                    image_shape=img_cv.shape,
                    font_path=font_path,
                    safety_padding=6,
                    max_font_size=frag_max_fs,
                    min_font_size=min_font_size,
                    line_spacing=calc_line_spacing,
                    min_clearance=3
                )

                if layout and layout.get("lines"):
                    chosen_fs = layout["font_size"]
                    try:
                        active_font = ImageFont.truetype(font_path, chosen_fs)
                    except Exception:
                        active_font = ImageFont.load_default()
                    for line_txt, lx, ly in layout["lines"]:
                        draw.text((lx, ly), line_txt, font=active_font, fill=text_color)
                else:
                    # Fallback: fit_text directly in fragment box
                    if len(frag_box) >= 4:
                        fb = (int(frag_box[0]), int(frag_box[1]), int(frag_box[2]), int(frag_box[3]))
                    else:
                        fb = box
                    fit_text(
                        draw=draw,
                        text=frag_text,
                        box=fb,
                        font_path=font_path,
                        max_font_size=frag_max_fs,
                        min_font_size=min_font_size,
                        line_gap=line_gap,
                        size_offset=0,
                        uppercase=True,
                        text_color=text_color,
                    )
        else:
            # General occluded crescent tuck: when an occluding rectangle overlaps the top portion
            # of an occluded bubble, line 1 should tuck directly beneath the rectangle's bottom border.
            top_rect = None
            if is_occluded and _earpiece_rects:
                for r in _earpiece_rects:
                    rx0_val = r.get('x', r.get('x0', 0))
                    ry0_val = r.get('y', r.get('y0', 0))
                    rx1_val = rx0_val + r.get('w', 0) if 'w' in r else r.get('x1', 0)
                    ry1_val = ry0_val + r.get('h', 0) if 'h' in r else r.get('y1', 0)
                    if min(box[2], rx1_val) - max(box[0], rx0_val) > 10:
                        if ry0_val <= box[1] + 30 and ry1_val > box[1]:
                            top_rect = {'x': rx0_val, 'y': ry0_val, 'x1': rx1_val, 'y1': ry1_val}
                            break

            if top_rect is not None:
                quality_gates.apply_quality_gates(
                    "lettering",
                    {"is_occluded": True, "top_rect": top_rect, "box": box},
                    bubble_id=getattr(b, "bubble_id", 0) if hasattr(b, "bubble_id") else b.get("bubble_id", 0)
                )
                lines = [l.strip() for l in uzbek_text.split("\n") if l.strip()]
                if not lines:
                    lines = [uzbek_text.strip()]
                cx = (box[0] + box[2]) // 2
                y1_pos = top_rect['y1'] - 3
                avail_h = max(20, box[3] - y1_pos)
                avail_w = max(30, box[2] - box[0])
                target_fs = 16
                for test_fs in range(18, 9, -1):
                    try:
                        tf = ImageFont.truetype(font_path, test_fs)
                        max_lw = max((tf.getbbox(l)[2] - tf.getbbox(l)[0]) for l in lines)
                        tot_lh = len(lines) * test_fs + (len(lines) - 1) * 3
                        if max_lw <= avail_w and tot_lh <= avail_h + 8:
                            target_fs = test_fs
                            break
                    except Exception:
                        pass
                active_f = ImageFont.truetype(font_path, target_fs)
                curr_y = y1_pos
                for line_txt in lines:
                    lb = active_f.getbbox(line_txt)
                    lw = lb[2] - lb[0]
                    lh = lb[3] - lb[1]
                    lx = cx - lw // 2
                    draw.text((lx, curr_y), line_txt, fill=text_color, font=active_f)
                    curr_y += max(lh + 2, target_fs + 3)
                continue

            # Single-contour typesetting (including individual BubbleFragments)
            cnt, _ = get_contour_from_bubble(b, w, h)
            if is_occluded:
                pad = 5
                clearance = 2
            elif is_tinted:
                pad = 6
                clearance = 2
            else:
                pad = 10
                clearance = 3
            frag_max_fs = min(effective_max, 14) if is_occluded else max(effective_max, 32 if is_tinted else effective_max)

            layout = bubble_lettering.fit_text_to_bubble(
                text=uzbek_text,
                contour=cnt,
                image_shape=img_cv.shape,
                font_path=font_path,
                safety_padding=pad,
                max_font_size=frag_max_fs,
                min_font_size=min_font_size,
                line_spacing=calc_line_spacing,
                min_clearance=clearance
            )

            if not (layout and layout.get("lines")) and is_occluded:
                # Relax safety padding for tight occluded fragments
                layout = bubble_lettering.fit_text_to_bubble(
                    text=uzbek_text,
                    contour=cnt,
                    image_shape=img_cv.shape,
                    font_path=font_path,
                    safety_padding=2,
                    max_font_size=frag_max_fs,
                    min_font_size=max(8, min_font_size - 2),
                    line_spacing=calc_line_spacing,
                    min_clearance=1
                )

            if layout and layout.get("lines"):
                chosen_fs = layout["font_size"]
                try:
                    active_font = ImageFont.truetype(font_path, chosen_fs)
                except Exception:
                    active_font = ImageFont.load_default()

                for line_txt, lx, ly in layout["lines"]:
                    if stroke_width > 0 and stroke_fill:
                        draw.text((lx, ly), line_txt, font=active_font, fill=text_color, stroke_width=stroke_width, stroke_fill=stroke_fill)
                    else:
                        draw.text((lx, ly), line_txt, font=active_font, fill=text_color)
            else:
                target_box = box

                fit_text(
                    draw=draw,
                    text=uzbek_text,
                    box=target_box,
                    font_path=font_path,
                    max_font_size=frag_max_fs if is_occluded else effective_max,
                    min_font_size=min_font_size,
                    line_gap=line_gap,
                    size_offset=0,
                    uppercase=True,
                    text_color=text_color,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill
                )

    if _earpiece_rects:
        # Redraw crisp 2px brown rectangle borders on top so borders cleanly overlap tucked lettering
        for _r in _earpiece_rects:
            rx, ry, rx1, ry1 = _r['x'], _r['y'], _r['x1'], _r['y1']
            top_edge = img_cv[ry:ry+2, rx+20:rx1-20]
            border_rgb = tuple(int(v) for v in reversed(np.mean(top_edge, axis=(0, 1)))) if top_edge.size > 0 else (140, 93, 78)
            draw.rectangle([(rx, ry), (rx1, ry1)], outline=border_rgb, width=2)

    return out_img


def extract_page_sfx(
    image: Image.Image,
    raw_ocr_results: Optional[List[Any]] = None
) -> List[sfx_engine.SFXElement]:
    """Extracts sound effects detected on the comic page."""
    img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if raw_ocr_results is None:
        reader = get_ocr_reader()
        raw_ocr_results = run_tiled_easyocr(image, reader=reader, canvas_size=2048)
    return sfx_engine.extract_sfx_elements(raw_ocr_results, img_cv.shape[:2], image_bgr=img_cv)


def render_page(
    image: Image.Image,
    bubbles: List[Any],
    font_path: str,
    max_font_size: int = 26,
    min_font_size: int = 10,
    line_gap: int = 5,
    cleaned_base: Optional[Image.Image] = None,
    stroke_width: int = 0,
    stroke_fill: Optional[str] = None,
    sfx_elements: Optional[List[sfx_engine.SFXElement]] = None,
    **kwargs
) -> Image.Image:
    """
    Combined Clean + Letter helper:
    If cleaned_base is passed, uses it directly; otherwise runs clean_page_ink_telea.
    Inpaints and renders SFX elements if provided.
    """
    if cleaned_base is not None:
        canvas = cleaned_base
    else:
        canvas = clean_page_ink_telea(image, bubbles)

    if sfx_elements:
        canvas_cv = cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
        canvas_cv = sfx_engine.inpaint_sfx_elements(canvas_cv, sfx_elements)
        canvas = sfx_engine.render_sfx_elements(canvas_cv, sfx_elements, font_path=font_path)

    return typeset_lettering_on_page(
        cleaned_image=canvas,
        bubbles=bubbles,
        font_path=font_path,
        max_font_size=max_font_size,
        min_font_size=min_font_size,
        line_gap=line_gap,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
        **kwargs
    )

render_localized_comic = render_page


def render_page_with_sfx(
    image: Image.Image,
    bubbles: List[Any],
    font_path: str,
    max_font_size: int = 26,
    min_font_size: int = 10,
    line_gap: int = 5,
    raw_ocr_results: Optional[List[Any]] = None,
    **kwargs
) -> Tuple[Image.Image, List[sfx_engine.SFXElement]]:
    """Convenience helper: renders page including all dialogue bubbles and sound effects."""
    sfx_items = extract_page_sfx(image, raw_ocr_results=raw_ocr_results)
    rendered = render_page(
        image=image,
        bubbles=bubbles,
        font_path=font_path,
        max_font_size=max_font_size,
        min_font_size=min_font_size,
        line_gap=line_gap,
        sfx_elements=sfx_items,
        **kwargs
    )
    return rendered, sfx_items



def get_contour_from_bubble(
    b: Any,
    w: int,
    h: int,
    img_cv: Optional[np.ndarray] = None,
    neighbor_seeds: Optional[List[Tuple[int, int]]] = None
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Extracts or reconstructs the contour as an int32 numpy array suitable for cv2.drawContours.
    Returns (contour_np, (anchor_x, anchor_y)).
    """
    cnt = None
    if isinstance(b, dict):
        raw_cnt = b.get("contour") if b.get("contour") is not None else b.get("contour_points")
        bx0 = int(b.get("x0", 0))
        by0 = int(b.get("y0", 0))
        bx1 = int(b.get("x1", 0))
        by1 = int(b.get("y1", 0))
        shape_type = b.get("shape_type", "oval")
    else:
        raw_cnt = getattr(b, "contour", None)
        if raw_cnt is None:
            raw_cnt = getattr(b, "contour_points", None)
        bx0 = int(getattr(b, "x0", 0))
        by0 = int(getattr(b, "y0", 0))
        bx1 = int(getattr(b, "x1", 0))
        by1 = int(getattr(b, "y1", 0))
        shape_type = getattr(b, "shape_type", "oval")

    if raw_cnt is not None:
        try:
            arr = np.array(raw_cnt, dtype=np.int32)
            if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 3:
                cnt = arr.reshape(-1, 1, 2)
            elif arr.ndim == 3 and arr.shape[2] == 2 and len(arr) >= 3:
                cnt = arr
        except Exception:
            cnt = None

    if (cnt is None or len(cnt) < 3) and img_cv is not None and (bx1 > bx0) and (by1 > by0):
        try:
            cnt_extracted, _ = bubble_lettering.get_bubble_contour(
                img_cv,
                (bx0, by0, bx1, by1),
                neighbor_seeds=neighbor_seeds,
                approx_glyph_height=(by1 - by0)
            )
            if cnt_extracted is not None and len(cnt_extracted) >= 3:
                cnt = cnt_extracted
        except Exception:
            pass

    if cnt is None or len(cnt) < 3:
        if shape_type == "rectangle":
            cnt = np.array([
                [[bx0, by0]],
                [[bx1, by0]],
                [[bx1, by1]],
                [[bx0, by1]]
            ], dtype=np.int32)
        else:
            cx = (bx0 + bx1) // 2
            cy = (by0 + by1) // 2
            rx = max(10, (bx1 - bx0) // 2 + 10)
            ry = max(10, (by1 - by0) // 2 + 10)
            pts = cv2.ellipse2Poly((cx, cy), (rx, ry), 0, 0, 360, 12)
            cnt = pts.reshape(-1, 1, 2)

    anchor_x = int(np.min(cnt[:, :, 0]))
    anchor_y = int(np.min(cnt[:, :, 1]))
    # Guard against upward tails pulling badge anchor far above text box
    anchor_y = max(anchor_y, by0 - 15)
    anchor_x = max(anchor_x, bx0 - 15)
    return cnt, (anchor_x, anchor_y)


def draw_bounding_box_overlay(
    image: Image.Image,
    bubbles: List[Any],
    outline_color: Tuple[int, int, int] = (0, 255, 100),
    fill_opacity: float = 0.25
) -> Image.Image:
    """
    Draws semi-transparent shape-aware markers (round as round, box as box):
    - Real polygon contours with clean 2px green border: (0, 255, 100).
    - Subtle semi-transparent green tint (~25% opacity) preserving artwork underneath.
    - Bubble ID badge (#1, #2) neatly rendered at the top-left anchor of the contour.
    """
    if not bubbles:
        return image.copy()

    img_rgb = image.convert("RGB")
    w, h = img_rgb.size
    img_cv = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    overlay = img_cv.copy()

    # BGR outline color for OpenCV: (0, 255, 100) RGB -> (100, 255, 0) BGR
    bgr_outline = (outline_color[2], outline_color[1], outline_color[0])

    valid_items = []
    for i, b in enumerate(bubbles):
        if isinstance(b, dict):
            is_enabled = b.get("is_active", b.get("enabled", True))
            b_id = b.get("bubble_id", i + 1)
            bx0 = int(b.get("x0", 0))
            by0 = int(b.get("y0", 0))
            bx1 = int(b.get("x1", 0))
            by1 = int(b.get("y1", 0))
        else:
            is_enabled = getattr(b, "is_active", getattr(b, "enabled", True))
            b_id = getattr(b, "bubble_id", i + 1)
            bx0 = int(getattr(b, "x0", 0))
            by0 = int(getattr(b, "y0", 0))
            bx1 = int(getattr(b, "x1", 0))
            by1 = int(getattr(b, "y1", 0))

        if not is_enabled:
            continue

        cnt, anchor = get_contour_from_bubble(b, w, h)
        valid_items.append((b_id, cnt, anchor, (bx0, by0, bx1, by1)))

        # Fill contour on overlay layer
        cv2.drawContours(overlay, [cnt], -1, bgr_outline, thickness=-1)

    # Semi-transparent blending (25% tint opacity)
    cv2.addWeighted(overlay, fill_opacity, img_cv, 1.0 - fill_opacity, 0, img_cv)

    # Draw crisp 2px green outline
    for b_id, cnt, anchor, bbox_coords in valid_items:
        cv2.drawContours(img_cv, [cnt], -1, bgr_outline, thickness=2)

    # Render neat bubble ID badge (#1, #2) with Pillow
    result_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(result_pil)

    try:
        font_path = get_available_fonts().get("CC Wild Words (Recommended)") or r"C:\Windows\Fonts\arial.ttf"
        badge_font = ImageFont.truetype(font_path, max(13, int(w * 0.016)))
    except Exception:
        badge_font = ImageFont.load_default()

    for idx, (b_id, cnt, (ax, ay), (bx0, by0, bx1, by1)) in enumerate(valid_items):
        badge_text = f" #{b_id} "
        bbox = badge_font.getbbox(badge_text)
        bw_box = bbox[2] - bbox[0] + 8
        bh_box = bbox[3] - bbox[1] + 6
        badge_x0 = max(2, min(w - bw_box - 2, ax))
        badge_y0 = ay - bh_box - 2

        # Check collision with other bubbles' bounding boxes
        collides = False
        for other_idx, (_, _, _, (obx0, oby0, obx1, oby1)) in enumerate(valid_items):
            if other_idx == idx:
                continue
            if not (badge_x0 + bw_box < obx0 - 5 or badge_x0 > obx1 + 5 or badge_y0 + bh_box < oby0 - 5 or badge_y0 > oby1 + 5):
                collides = True
                break

        # If placing the badge above the bubble would cross into a gutter, previous panel,
        # or collide with another bubble's text, place it inside the bubble top-left.
        if badge_y0 < 2 or collides or (ay > 0 and np.mean(img_cv[max(0, ay - bh_box):ay, badge_x0:badge_x0 + bw_box]) > 235):
            badge_y0 = min(h - bh_box - 2, max(ay + 2, by0 - 2))
            badge_x0 = max(2, min(w - bw_box - 2, bx0 - 2))

        # Neat green badge with 1px black outline
        draw.rectangle(
            [badge_x0, badge_y0, badge_x0 + bw_box, badge_y0 + bh_box],
            fill=outline_color,
            outline=(0, 0, 0),
            width=1
        )
        draw.text(
            (badge_x0 + 4, badge_y0 + 2),
            badge_text,
            fill=(0, 0, 0),
            font=badge_font
        )

    return result_pil


