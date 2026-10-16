import unittest
import os
import urllib.request
import json
from fontTools.ttLib import TTFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class TestMasonFontIntegration(unittest.TestCase):
    def test_mason_font_files_exist(self):
        fonts_dir = os.path.join(BASE_DIR, "fonts")
        regular_path = os.path.join(fonts_dir, "Mason-Regular.ttf")
        bold_path = os.path.join(fonts_dir, "Mason-Bold.ttf")
        demi_path = os.path.join(fonts_dir, "Mason-Demi.ttf")

        self.assertTrue(os.path.exists(regular_path), f"Missing {regular_path}")
        self.assertTrue(os.path.exists(bold_path), f"Missing {bold_path}")
        self.assertTrue(os.path.exists(demi_path), f"Missing {demi_path}")

        # Check font header tables
        font_reg = TTFont(regular_path)
        self.assertIn("cmap", font_reg)
        self.assertIn("name", font_reg)
        names = {r.nameID: r.toUnicode() for r in font_reg['name'].names if r.nameID in [1, 2, 4]}
        self.assertEqual(names.get(1), "Mason")

    def test_server_font_map_resolution(self):
        import server
        reg_font = server.get_font_file("Mason", is_bold=False)
        bold_font = server.get_font_file("Mason", is_bold=True)
        serif_font = server.get_font_file("Mason Serif", is_bold=False)

        self.assertTrue(os.path.exists(reg_font))
        self.assertTrue(os.path.exists(bold_font))
        self.assertTrue(os.path.exists(serif_font))
        self.assertIn("Mason-Regular.ttf", reg_font)
        self.assertIn("Mason-Bold.ttf", bold_font)

    def test_web_index_declarations(self):
        index_path = os.path.join(BASE_DIR, "web", "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. @font-face
        self.assertIn("font-family: 'Mason'", content)
        self.assertIn("/fonts/Mason-Regular.ttf", content)
        self.assertIn("/fonts/Mason-Bold.ttf", content)

        # 2. Preview class
        self.assertIn(".ff-mason", content)

        # 3. Dropdown list
        self.assertIn('data-font="Mason"', content)
        self.assertIn('Mason (Doctor Doom)', content)

        # 4. Fallback and Case mapping
        self.assertIn("'Mason': 'ff-mason'", content)
        self.assertIn("document.fonts.load(\"16px 'Mason'\")", content)

    def test_server_http_font_endpoint(self):
        url = "http://localhost:8000/fonts/Mason-Regular.ttf"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = resp.read()
            self.assertGreater(len(data), 10000)

if __name__ == "__main__":
    unittest.main()
