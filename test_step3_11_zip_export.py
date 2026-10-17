import unittest
import io
import zipfile
import base64
from PIL import Image
from starlette.testclient import TestClient
import server

class TestChapterZipExport(unittest.TestCase):
    def test_export_zip_with_client_rendered_b64_empty_cache(self):
        # Create a tiny 100x100 dummy test image in memory
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Explicitly make sure page is NOT in PAGES_CACHE
        server.PAGES_CACHE.clear()

        client = TestClient(server.app)
        payload = {
            "chapter_name": "test_chapter",
            "font_name": "CC Wild Words",
            "pages": [
                {
                    "page_id": "random_test_id_1",
                    "page_num": 1,
                    "filename": "01_page.png",
                    "status": "ready",
                    "rendered_image_b64": f"data:image/png;base64,{b64_str}"
                },
                {
                    "page_id": "random_test_id_2",
                    "page_num": 2,
                    "filename": "02_page.png",
                    "status": "skipped",
                    "original_image_b64": f"data:image/png;base64,{b64_str}"
                }
            ]
        }

        resp = client.post("/api/export_zip", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("content-type"), "application/zip")

        # Verify ZIP contains the 2 files and is NOT 22 bytes!
        zip_bytes = resp.content
        self.assertGreater(len(zip_bytes), 200)

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            namelist = zf.namelist()
            self.assertIn("01_page.png", namelist)
            self.assertIn("02_page.png", namelist)
            self.assertGreater(len(zf.read("01_page.png")), 50)
            self.assertGreater(len(zf.read("02_page.png")), 50)

if __name__ == "__main__":
    unittest.main()
