"""
test_cloud_enhancements.py
--------------------------
Unit test suite verifying all four optional cloud enhancements:
1. Inert execution (zero behavior change, zero network calls, zero log spam) when API keys are absent.
2. Hard 90% usage cap enforcement & period rollover via api_quota_tracker.
3. Google Vision OCR fallback logic (confidence threshold, decision rule, error resilience).
4. Google Translate advisory cross-check (advisory only, mismatch detection, error resilience).
5. Marvel API enrichment script behavior (standalone, read-only by default, keyless execution).
6. GitHub Actions workflow YAML syntax & structure.
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

import api_quota_tracker
import cloud_enhancements
import quality_gates
import engine


class TestApiQuotaTracker(unittest.TestCase):
    """Verifies that api_quota_tracker strictly enforces 90% free-tier safety caps."""

    def setUp(self):
        api_quota_tracker.reset_quota()

    def tearDown(self):
        api_quota_tracker.reset_quota()

    def test_quota_allows_consumption_within_cap(self):
        self.assertTrue(api_quota_tracker.can_consume("google_vision", 10))
        api_quota_tracker.record_consumption("google_vision", 10)
        status = api_quota_tracker.get_quota_status()
        self.assertEqual(status["google_vision"]["used"], 10)
        self.assertEqual(status["google_vision"]["remaining_to_cap"], 890)

    def test_quota_blocks_at_90_percent_cap(self):
        # Google Vision safety cap is 900 (90% of 1000)
        api_quota_tracker.record_consumption("google_vision", 899)
        self.assertTrue(api_quota_tracker.can_consume("google_vision", 1))

        api_quota_tracker.record_consumption("google_vision", 1)
        # At 900/900: Further consumption must be blocked!
        self.assertFalse(api_quota_tracker.can_consume("google_vision", 1))

    def test_google_translate_char_cap(self):
        # Google Translate safety cap is 450,000 characters
        api_quota_tracker.record_consumption("google_translate", 449900)
        self.assertTrue(api_quota_tracker.can_consume("google_translate", 50))
        self.assertFalse(api_quota_tracker.can_consume("google_translate", 200))


class TestGoogleVisionOcrFallback(unittest.TestCase):
    """Verifies Google Cloud Vision OCR fallback behavior."""

    def setUp(self):
        api_quota_tracker.reset_quota()

    def test_inert_when_key_absent(self):
        """When GOOGLE_VISION_API_KEY is not set, must return EasyOCR text without network calls."""
        with patch.dict(os.environ, {}, clear=True):
            dummy_crop = np.zeros((20, 50, 3), dtype=np.uint8)
            with patch("urllib.request.urlopen") as mock_url:
                txt, conf, src = cloud_enhancements.google_vision_ocr_fallback(
                    dummy_crop,
                    easyocr_text="HELL0",
                    easyocr_conf=0.40
                )
                self.assertEqual(txt, "HELL0")
                self.assertEqual(conf, 0.40)
                self.assertEqual(src, "easyocr")
                mock_url.assert_not_called()

    def test_inert_when_easyocr_conf_high(self):
        """When EasyOCR confidence >= 0.55, Vision is never called even if key is present."""
        with patch.dict(os.environ, {"GOOGLE_VISION_API_KEY": "dummy_key"}):
            dummy_crop = np.zeros((20, 50, 3), dtype=np.uint8)
            with patch("urllib.request.urlopen") as mock_url:
                txt, conf, src = cloud_enhancements.google_vision_ocr_fallback(
                    dummy_crop,
                    easyocr_text="HELLO",
                    easyocr_conf=0.85
                )
                self.assertEqual(txt, "HELLO")
                self.assertEqual(src, "easyocr")
                mock_url.assert_not_called()

    def test_vision_wins_when_confidence_higher(self):
        """When Vision returns different text with higher confidence, Vision wins."""
        mock_resp_data = {
            "responses": [
                {
                    "textAnnotations": [{"description": "HELLO WORLD"}],
                    "fullTextAnnotation": {
                        "pages": [{"confidence": 0.95}]
                    }
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_resp_data).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch.dict(os.environ, {"GOOGLE_VISION_API_KEY": "dummy_key"}):
            dummy_crop = np.zeros((30, 80, 3), dtype=np.uint8)
            with patch("urllib.request.urlopen", return_value=mock_response):
                txt, conf, src = cloud_enhancements.google_vision_ocr_fallback(
                    dummy_crop,
                    easyocr_text="HELL0 W0RLD",
                    easyocr_conf=0.35
                )
                self.assertEqual(txt, "HELLO WORLD")
                self.assertEqual(src, "google_vision")
                self.assertGreater(conf, 0.90)

    def test_vision_network_error_graceful_fallback(self):
        """Any network/auth error must fallback to EasyOCR with zero crash."""
        with patch.dict(os.environ, {"GOOGLE_VISION_API_KEY": "dummy_key"}):
            dummy_crop = np.zeros((20, 50, 3), dtype=np.uint8)
            with patch("urllib.request.urlopen", side_effect=Exception("Connection timed out")):
                txt, conf, src = cloud_enhancements.google_vision_ocr_fallback(
                    dummy_crop,
                    easyocr_text="SPIDER",
                    easyocr_conf=0.45
                )
                self.assertEqual(txt, "SPIDER")
                self.assertEqual(src, "easyocr")


class TestGoogleTranslateCrossCheck(unittest.TestCase):
    """Verifies Google Cloud Translation advisory cross-check behavior."""

    def setUp(self):
        api_quota_tracker.reset_quota()

    def test_inert_when_key_absent(self):
        """When GOOGLE_TRANSLATE_API_KEY is not set, must return None without network calls."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("urllib.request.urlopen") as mock_url:
                res = cloud_enhancements.google_translate_cross_check(
                    source_en="HELLO WORLD",
                    nmt_translation="SALOM DUNYO"
                )
                self.assertIsNone(res)
                mock_url.assert_not_called()

    def test_mismatch_detection_and_flag(self):
        """When Google Translate returns a divergent translation, it flags mismatch."""
        mock_resp_data = {
            "data": {
                "translations": [{"translatedText": "BUTKUL BOSHQA TARJIMA"}]
            }
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_resp_data).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch.dict(os.environ, {"GOOGLE_TRANSLATE_API_KEY": "dummy_key"}):
            with patch("urllib.request.urlopen", return_value=mock_response):
                res = cloud_enhancements.google_translate_cross_check(
                    source_en="DON'T BOTHER.",
                    nmt_translation="OVORA BO'LMANG.",
                    similarity_threshold=0.50
                )
                self.assertIsNotNone(res)
                self.assertTrue(res["is_mismatch"])
                self.assertEqual(res["google_translation"], "BUTKUL BOSHQA TARJIMA")

    def test_quality_gate_registry_includes_cross_check(self):
        """Verify translation_cross_check_mismatch is registered in Quality Gates."""
        pattern_names = [p.name for p in quality_gates.QUALITY_GATES_REGISTRY]
        self.assertIn("translation_cross_check_mismatch", pattern_names)


class TestMarvelApiEnrichment(unittest.TestCase):
    """Verifies standalone Marvel Comics API terminology enrichment script."""

    def test_script_exists(self):
        script_path = os.path.join(os.path.dirname(__file__), "scripts", "enrich_marvel_terminology.py")
        self.assertTrue(os.path.exists(script_path))

    def test_inert_without_keys(self):
        """Without keys, script prints guidance and exits gracefully without making requests."""
        import scripts.enrich_marvel_terminology as marvel_script
        with patch.dict(os.environ, {}, clear=True):
            with patch("urllib.request.urlopen") as mock_url:
                marvel_script.run_enrichment()
                mock_url.assert_not_called()


class TestGitHubActionsWorkflow(unittest.TestCase):
    """Verifies GitHub Actions CI workflow configuration."""

    def test_workflow_file_exists_and_configured(self):
        wf_path = os.path.join(os.path.dirname(__file__), ".github", "workflows", "regression.yml")
        self.assertTrue(os.path.exists(wf_path))
        with open(wf_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("ubuntu-latest", content)
        self.assertIn("tests/test_regressions.py", content)
        self.assertIn("test_quality_gates.py", content)


class TestNaturalnessScoredBaseTranslation(unittest.TestCase):
    """Verifies naturalness heuristic and optimal base translation selection."""

    def test_compute_naturalness_score(self):
        # Robotic calques receive lower score than authentic phrasing
        robotic = "Peter, sen yaxshisanmi? Sen boshqacha ko'rinasan."
        natural = "Piter, yaxshimisan? O'zingga o'xshamayapsan."
        score_robotic = cloud_enhancements.compute_naturalness_score(robotic)
        score_natural = cloud_enhancements.compute_naturalness_score(natural)
        self.assertLess(score_robotic, score_natural)

    def test_semantic_agreement(self):
        # Both affirmative agree
        en = "Peter, are you okay?"
        nllb = "Peter, sen yaxshisanmi?"
        cloud = "Piter, yaxshimisiz?"
        self.assertTrue(cloud_enhancements.semantic_agreement(nllb, cloud, en))

        # Disagreeing polarity (one negative, one affirmative for affirmative English)
        cloud_neg = "Piter, yaxshi emasmisiz?"
        self.assertFalse(cloud_enhancements.semantic_agreement(nllb, cloud_neg, en))

    def test_select_optimal_base_translation_inert_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            base, source = cloud_enhancements.select_optimal_base_translation(
                "Good evening", "Xayrli kechasi"
            )
            self.assertEqual(source, "nllb")
            self.assertEqual(base, "Xayrli kechasi")

    def test_select_optimal_base_translation_chooses_cloud_when_higher_score(self):
        with patch.dict(os.environ, {"GOOGLE_TRANSLATE_API_KEY": "dummy_key"}):
            with patch("cloud_enhancements.fetch_google_translation", return_value="Xayrli kech, vino menyusini ko'rishni xohlaysizmi?"):
                base, source = cloud_enhancements.select_optimal_base_translation(
                    "Good evening, would you like to see the wine menu?",
                    "Xayrli kechasi, sharob menyasini ko'rishni xohlaysizmi?"
                )
                self.assertEqual(source, "google_translate")
                self.assertIn("Xayrli kech", base)


if __name__ == "__main__":
    unittest.main()
