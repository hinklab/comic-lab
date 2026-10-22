========================================================================
COMIC-LAB STUDIO - O'RNATISH VA ISHLATISH QO'LLANMASI
========================================================================

Ushbu dastur mustaqil (standalone) ilova bo'lib, kompyuteringizga Python yoki
boshqa qo'shimcha dasturlar o'rnatishni talab QILMAYDI.

------------------------------------------------------------------------
1. DASTURNI ISHGA TUSHIRISH
------------------------------------------------------------------------
- Papka ichidagi "ComicLab.exe" faylini ikki marta bosing.
- Dastur bir necha soniyada ochiladi va to'liq ekranda ishga tushadi.
- Xohlasangiz, "ComicLab.exe" faylini sichqonchaning o'ng tugmasi bilan bosib:
  "Yorliq yaratish" (Create shortcut) qilib, ish stolingizga (Desktop) chiqarib qo'yishingiz mumkin.

------------------------------------------------------------------------
2. GEMINI AI SOZLAMASI (TARJIMA UCHUN)
------------------------------------------------------------------------
Dastur tarjima va pufakchalarni aniqlashda Google Gemini Vision modelidan
foydalanadi. Buning uchun:
- Ushbu papka ichida ".env" fayli mavjud bo'lishi kerak.
- .env faylini bloknot (Notepad) orqali oching va o'z Gemini API kalitingizni kiriting:

GEMINI_API_KEY=AIzaSy...sizning_kalitingiz...

(Agar bir nechta kalit bo'lsa, vergul bilan ajratib yozishingiz mumkin:
GEMINI_API_KEY=kalit1,kalit2,kalit3)

------------------------------------------------------------------------
3. TIZIM TALABLARI
------------------------------------------------------------------------
- Operatsion tizim: Windows 10 yoki Windows 11 (64-bit)
- Operativ xotira (RAM): 4 GB yoki undan yuqori (Dastur atigi ~65 MB ishlatadi)
- Internet: Gemini AI tarjimasi uchun internet ulanishi talab etiladi.
- Shriftlar: Barcha 27 ta professional komiks shriftlari dastur ichiga o'rnatilgan.
