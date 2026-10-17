import unittest
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class TestPageContextMenuDownload(unittest.TestCase):
    def test_context_menu_contains_download_action(self):
        for path in [os.path.join(BASE_DIR, "web", "index.html"), os.path.join(BASE_DIR, "index.html")]:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("pageContextActionDownload()", content)
            self.assertIn("Sahifani yuklab olish", content)
            self.assertIn("function pageContextActionDownload", content)
            self.assertIn("function downloadPageHD", content)
            self.assertIn("const menuH = 185;", content)

if __name__ == "__main__":
    unittest.main()
