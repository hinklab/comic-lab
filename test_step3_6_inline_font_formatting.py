"""
Unit tests for Step 3.6: Partial / Inline Text Font Selection in Speech Bubbles.
Tests token creation, font tag regex parsing, multi-line paragraph state preservation,
word measurement with custom font tokens, hyphenation font preservation, and HTML markup.
"""
import re
import unittest
from pathlib import Path

INDEX_HTML_PATH = Path(__file__).resolve().parent / "web" / "index.html"


class TestInlineFontFormatting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            cls.html_content = f.read()

    def test_create_word_token_has_font_family(self):
        """createWordToken must accept fontFamily as 5th argument and store it."""
        self.assertIn("function createWordToken(text, isBold = null, isItalic = null, sizeDelta = 0, fontFamily = null)", self.html_content)
        self.assertIn("fontFamily: fontFamily || null,", self.html_content)

    def test_font_regex_in_parse_formatted_line(self):
        """parseFormattedLineToWordTokens must support [font=...] and [/font] and HTML <font face=...>."""
        self.assertIn("\\[font=[^\\]]+\\]", self.html_content)
        self.assertIn("\\[\\/font\\]", self.html_content)
        self.assertIn(".replace(/<font\\b[^>]*face=[\"']?([^\"'>]+)[\"']?[^>]*>/gi, '[font=$1]')", self.html_content)

    def test_parse_state_carried_over_paragraphs(self):
        """parseTextToParagraphs must pass parseState across lines."""
        self.assertIn("const parseState = { isBold: false, isItalic: false, sizeDelta: 0, fontFamily: null };", self.html_content)
        self.assertIn("parseFormattedLineToWordTokens(line, parseState);", self.html_content)

    def test_get_token_font_resolves_token_font_family(self):
        """getTokenFont must check (token && token.fontFamily) first."""
        self.assertIn("(token && token.fontFamily) ? token.fontFamily :", self.html_content)
        self.assertIn("resolveFontFamilyCase", self.html_content)

    def test_hyphenation_preserves_token_font_family(self):
        """fitWordWithHyphenUzbek must pass tokenOrStr.fontFamily to prefix and suffix tokens."""
        self.assertIn("createWordToken(pt.prefix, tokenOrStr.isBold, tokenOrStr.isItalic, tokenOrStr.sizeDelta, tokenOrStr.fontFamily)", self.html_content)
        self.assertIn("createWordToken(pt.suffix, tokenOrStr.isBold, tokenOrStr.isItalic, tokenOrStr.sizeDelta, tokenOrStr.fontFamily)", self.html_content)

    def test_format_selected_text_font_action(self):
        """formatSelectedText must handle action === 'font' and wrap or toggle font tags."""
        self.assertIn("action === 'font'", self.html_content)
        self.assertIn("[font=${fontName}]", self.html_content)
        self.assertIn("[/font]", self.html_content)

    def test_format_selected_text_clear_strips_font_tags(self):
        """formatSelectedText action === 'clear' must strip [font=...] and [/font] tags."""
        self.assertIn(".replace(/\\[\\/?font(?:=[^\\]]+)?\\]/gi, '')", self.html_content)

    def test_on_font_family_change_checks_text_selection(self):
        """onFontFamilyChange must detect highlighted text in insp-uz and route to formatSelectedText('font', fontName)."""
        self.assertIn("hasTextSelection = true;", self.html_content)
        self.assertIn("formatSelectedText('font', fontName);", self.html_content)

    def test_toolbar_has_inline_font_picker(self):
        """Rich-text toolbar must have dedicated Shrift button and inline-font-menu."""
        self.assertIn('id="btn-inline-font-picker"', self.html_content)
        self.assertIn('id="inline-font-menu"', self.html_content)
        self.assertIn('toggleInlineFontMenu', self.html_content)

    def test_textarea_has_selection_tracking_listeners(self):
        """insp-uz textarea must trigger trackUzbekTextSelection on select, mouseup, keyup."""
        self.assertIn('onselect="trackUzbekTextSelection()"', self.html_content)
        self.assertIn('onmouseup="trackUzbekTextSelection()"', self.html_content)
        self.assertIn('onkeyup="trackUzbekTextSelection()"', self.html_content)

    def test_python_simulated_token_parser(self):
        """Simulate JS token regex parsing in Python to verify mixed fonts and formatting."""
        text = "SALOM [font=Gandur New]DOOM![/font] SEN BUNI BILASAN."
        token_regex = re.compile(r'(\[font=[^\]]+\]|\[\/font\]|\[\/?\+?\-?\d+\]|\[\/?size[^\]]*\]|\*{1,3}|[^\s\*\[\]]+|\s+|\[|\])', re.IGNORECASE)
        tokens = token_regex.findall(text)

        current_font = None
        words = []
        for part in tokens:
            m = re.match(r'^\[font=([^\]]+)\]$', part, re.IGNORECASE)
            if m:
                current_font = m.group(1).strip()
            elif part.lower() == '[/font]':
                current_font = None
            elif part.strip():
                words.append((part, current_font))

        self.assertEqual(words, [
            ("SALOM", None),
            ("DOOM!", "Gandur New"),
            ("SEN", None),
            ("BUNI", None),
            ("BILASAN.", None)
        ])


if __name__ == "__main__":
    unittest.main()
