"""
retranslate_page3.py - Full end-to-end re-translation of 3.png with side-by-side comparison.
"""

import os
import sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine

REFERENCE_GEMINI = {
    1: 'ISHDAGI IKKINCHI KUN...',
    2: '...VA ULARDAN HECH BIRI MENING BOSHQACHA SPIDER-MAN EKANIMNI BILMAYDI.',
    3: 'BIROQ TEZ ORADA BARCHALARI MENING ANCHA USTUN SPIDER-MAN EKANIMNI ANGLAB YETISHADI!',
    4: 'ANA, KELYAPTI!',
    5: 'IKKI KUNDAN BERI SPIDEY BOSH MAVZU:',
    6: "NIMA? U BIR SAFAR BO'LSA HAM SHUNGA ARZIMAYDIMI?",
    7: "O, AGAR REJALASHTIRGAN BO'LSA HAM, BUNDAN ORTIQ LOYIQ BO'LA OLMASDI.",
    8: 'SPIDER-MAN, MEN RUT GOLDMAN, TALABALAR DEKANI.',
    9: '--VA EMPIRE STATE UNIVERSITETIDAGI BARCHA NOMIDAN...',
    10: "...O'G'IRLANGAN ILMIY USKUNALARIMIZNI SHU QADAR TEZ TOPGANINGIZ UCHUN SIZGA MINNATDORCHILIK BILDIRMOQCHIMAN.",
    11: 'ARZIMAYDI, XONIM.',
    12: 'VA SHUNCHA KAMSTARIN. UMID QILAMANK, QARSHI EMASSIZ...',
    13: "...LEKIN SIZGA SHAXSAN MINNATDORCHILIK BILDIRMOQCHI BO'LGAN MAXSUS MEHMONIMIZ BOR.",
    14: '--BUNI MARLA JEYMSON XOTIRA BINOSIGA QAYTARGANIM UCHUNMI?',
    15: 'QIZIQ, BU KIM EKAN-A?'
}

OLD_APP_OUTPUT = {
    1: 'ISHDAGI IKKINCHI KUN...',
    2: "VA ULARDAN HECH BIRI BOSHQACHA O'RINNI BILMAYDI",
    3: 'BIROQ TEZ ORADA BARCHALARI MENING ANCHA USTUN SPIDER-MAN EKANIMNI ANGLAB YETISHADI!',
    4: 'ANA, KELYAPTI!',
    5: 'IKKI KUN KETMA-KET VA SPIDEYNING ASOSIY HIKOYASI:',
    6: "NIMA? U BIR MARTA BO'LSA HAM ARZIMAYDIMI?",
    7: "O, AGAR U BUNI REJALASHTIRGAN BO'LSA, BUNDAN ORTIQ ARZIMASDI:",
    8: "O'G'LIM RUTA GOLDAMON, TALABALAR DEKANI",
    9: 'EMPIRE STATE UNIVERSITETIDAGI HAMMA HAQIDA',
    10: "...O'G'IRLANGAN ILMIY USKUNALARIMIZNI SHU QADAR TEZ TOPGANINGIZ UCHUN SIZGA RAHMAT AYTMOQCHIMAN.",
    11: 'BU HAMMAGA ARZIMAYDI EDI, XONIM',
    12: "VA SHUNCHA KAMSTARIN. UMID QILAMAN, XAFA BO'LMAYSIZ...",
    13: "...LEKIN SIZGA SHAXSAN RAHMAT AYTMOQCHI BO'LGAN MAXSUS MEHMONIMIZ BOR:",
    14: '...MARLA JAMESON XOTIRALAR QANOTASIGA QAYTARGANINGIZ UCHUNMI?',
    15: 'QIZIQ, BU KIM EKAN-A?'
}

def main():
    img_path = 'C:/Users/Ozod/Desktop/3.png'
    img = Image.open(img_path)
    print("1. Scanning 3.png with new OCR reading-order and line assembly...")
    bubbles = engine.scan_bubbles_ocr(img)
    print(f"Detected {len(bubbles)} bubbles.")

    print("\n2. Translating bubbles with default speaker 'Superior Spider-Man'...")
    translated = engine.translate_bubbles_list(bubbles, default_speaker="Superior Spider-Man")

    print("\n" + "=" * 100)
    print(f"{'#':<3} | {'GROUND TRUTH OCR':<45} | {'NEW APP OUTPUT'}")
    print("=" * 100)
    for b in translated:
        print(f"#{b.bubble_id:<2} | {b.original_text[:43]:<45} | {b.uzbek_translation}")

    print("\n" + "=" * 100)
    print("DETAILED 3-WAY COMPARISON (OLD APP vs GEMINI REF vs NEW APP)")
    print("=" * 100)
    for b in translated:
        bid = b.bubble_id
        old_val = OLD_APP_OUTPUT.get(bid, "N/A")
        gemini_ref = REFERENCE_GEMINI.get(bid, "N/A")
        new_val = b.uzbek_translation
        print(f"\n--- BUBBLE #{bid} ---")
        print(f"  ORIGINAL EN: {b.original_text}")
        print(f"  OLD APP:     {old_val}")
        print(f"  GEMINI REF:  {gemini_ref}")
        print(f"  NEW APP:     {new_val}")

if __name__ == '__main__':
    main()
