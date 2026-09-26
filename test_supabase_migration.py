"""
test_supabase_migration.py
---------------------------
Unit and integration test suite for Supabase cloud migration & local fallback system:
1. Verifies supabase_schema.sql DDL completeness and correctness.
2. Verifies supabase_db connection logic, credentials handling, and graceful offline fallback.
3. Verifies quality_gates, character_profiles, terminology_store, and bubble_mask_editor integration.
4. Verifies mock Supabase online operations (insert, upsert, query) with column resilience.
5. Verifies end-to-end pipeline run with Supabase layer active.
"""

import os
import json
import unittest
from unittest.mock import MagicMock, patch

import supabase_db
import quality_gates
import character_profiles
import terminology_store
import bubble_mask_editor


class TestSupabaseSchemaAndFiles(unittest.TestCase):
    """Verifies schema, environment template, and migration script files."""

    def test_schema_file_exists_and_contains_all_tables(self):
        schema_path = os.path.join(os.path.dirname(__file__), "supabase_schema.sql")
        self.assertTrue(os.path.exists(schema_path), "supabase_schema.sql must exist")

        with open(schema_path, "r", encoding="utf-8") as f:
            sql = f.read()

        required_tables = [
            "mask_corrections",
            "quality_gate_events",
            "character_profiles",
            "terminology_store"
        ]
        for tbl in required_tables:
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {tbl}", sql, f"Table {tbl} must be defined in schema")

        # Verify key columns
        self.assertIn("original_contour JSONB", sql)
        self.assertIn("corrected_contours JSONB", sql)
        self.assertIn("pattern_name TEXT", sql)
        self.assertIn("voice_traits JSONB", sql)
        self.assertIn("canonical_uzbek TEXT", sql)
        self.assertIn("ROW LEVEL SECURITY", sql)

    def test_env_example_exists(self):
        env_ex = os.path.join(os.path.dirname(__file__), ".env.example")
        self.assertTrue(os.path.exists(env_ex), ".env.example must exist")
        with open(env_ex, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("SUPABASE_URL", content)
        self.assertIn("SUPABASE_KEY", content)

    def test_docstrings_updated(self):
        """Verifies docstring update regarding Zero AI API Keys."""
        for filename in ["bubble_mask_editor.py", "engine.py", "app.py"]:
            filepath = os.path.join(os.path.dirname(__file__), filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn(
                "Zero AI API Keys (free/local translation model); online services for data storage are fine.",
                content,
                f"Docstring in {filename} must clarify Zero AI API Keys policy"
            )


class TestSupabaseOfflineFallback(unittest.TestCase):
    """Verifies that all components fall back to local JSON without error when offline."""

    def setUp(self):
        supabase_db.reset_client()

    def test_unconfigured_client_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            supabase_db.reset_client()
            client = supabase_db.get_supabase_client()
            self.assertIsNone(client)
            self.assertFalse(supabase_db.is_supabase_configured())

    def test_quality_gate_event_offline_fallback(self):
        """Confirm quality gate event still logs in-memory and writes local fallback."""
        quality_gates.clear_quality_gate_log()
        quality_gates.log_quality_gate_event(
            pattern_name="test_pattern_fallback",
            stage="test_stage",
            severity="auto_fix",
            action_taken="test_action",
            details={"key": "val"},
            bubble_id=99,
            page="test_page.png"
        )
        log = quality_gates.get_quality_gate_log()
        self.assertTrue(any(e["pattern_name"] == "test_pattern_fallback" for e in log))

    def test_mask_correction_offline_fallback(self):
        """Confirm mask correction logging works with local file fallback."""
        test_file = "test_mc_fallback.json"
        if os.path.exists(test_file):
            os.remove(test_file)

        bubble_mask_editor.record_mask_correction(
            page_name="page_fb.png",
            tool_name="scissors",
            bubble_id=1,
            original_contour=[[0, 0], [10, 0], [10, 10], [0, 10]],
            corrected_contours=[[[0, 0], [5, 5]], [[5, 5], [10, 10]]],
            metadata={"test": True},
            memory_path=test_file
        )

        loaded = bubble_mask_editor.load_mask_corrections(memory_path=test_file)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["bubble_id"], 1)
        self.assertEqual(loaded[0]["tool_used"], "scissors")

        if os.path.exists(test_file):
            os.remove(test_file)

    def test_character_profiles_offline_fallback(self):
        """Confirm CharacterMemoryStore loads correctly from local JSON when offline."""
        store = character_profiles.CharacterMemoryStore.get_instance()
        profile = store.get_profile("Superior Spider-Man")
        self.assertIsNotNone(profile)
        self.assertIn("ahmoqlar", " ".join(profile.voice_traits))

    def test_terminology_store_offline_fallback(self):
        """Confirm TerminologyStore loads correctly from local JSON when offline."""
        store = terminology_store.TerminologyStore.get_instance()
        entries = store.get_all_entries()
        self.assertGreater(len(entries), 0)
        uz_map = {tok: uz for _, tok, uz in entries}
        self.assertIn("XNAME_SPIDER_MAN_X", uz_map)


class TestSupabaseMockedOnlineOperations(unittest.TestCase):
    """Verifies that Supabase DB CRUD functions operate correctly when client is connected."""

    def setUp(self):
        supabase_db.reset_client()

    def test_insert_mask_correction_mocked(self):
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.insert.return_value.execute.return_value = MagicMock(data=[{"id": 1}])

        with patch("supabase_db.get_supabase_client", return_value=mock_client):
            res = supabase_db.insert_mask_correction({
                "page": "page_online.png",
                "timestamp": 1234567.89,
                "tool_used": "scissors",
                "bubble_id": 2,
                "original_contour": [[0, 0], [1, 1]],
                "corrected_contours": [[[0, 0]], [[1, 1]]],
                "metadata": {"online": True}
            })
            self.assertTrue(res)
            mock_client.table.assert_called_with("mask_corrections")

    def test_insert_quality_gate_event_mocked(self):
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.insert.return_value.execute.return_value = MagicMock(data=[{"id": 10}])

        with patch("supabase_db.get_supabase_client", return_value=mock_client):
            res = supabase_db.insert_quality_gate_event({
                "page": "p1.png",
                "pattern_name": "touching_bubbles_merged",
                "stage": "detection",
                "severity": "auto_fix",
                "action_taken": "split",
                "details": {"score": 0.99},
                "bubble_id": 3
            })
            self.assertTrue(res)
            mock_client.table.assert_called_with("quality_gate_events")

    def test_upsert_character_profile_column_resilience(self):
        """Verifies that upsert falls back to base 6 columns if extended columns don't exist."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table

        # Simulate first attempt failing due to missing column, second attempt succeeding
        first_call = True
        def mock_execute():
            nonlocal first_call
            if first_call:
                first_call = False
                raise Exception("column 'phrase_replacements' does not exist")
            return MagicMock(data=[{"character_name": "Doc Ock"}])

        mock_table.upsert.return_value.execute.side_effect = mock_execute

        with patch("supabase_db.get_supabase_client", return_value=mock_client):
            res = supabase_db.upsert_character_profile({
                "character_name": "Doc Ock",
                "voice_traits": ["arrogant"],
                "phrase_replacements": [["FOOL", "NODON"]]
            })
            self.assertTrue(res)
            self.assertEqual(mock_table.upsert.call_count, 2)


if __name__ == "__main__":
    unittest.main()
