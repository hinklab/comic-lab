"""
test_memory_and_terminology.py - Unit tests for persistent terminology lock & self-growing character memory.
"""

import json
import os
import tempfile
import unittest
import character_profiles
import terminology_store
import engine


class TestPersistentTerminologyStore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_json = os.path.join(self.temp_dir.name, "test_terms.json")
        self.store = terminology_store.TerminologyStore(store_path=self.test_json)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_term_locking_and_masking(self):
        """Test that registered terms are masked and unmasked into canonical form."""
        self.store.register_term(
            source_term="Spider-Man",
            canonical_uzbek="Spider-Man",
            first_seen_page="page_1",
            locked=True,
            aliases=["Spidey"]
        )

        masked, unmask_map = self.store.mask_terms("Spider-Man swung past the building.")
        self.assertIn("XNAME_SPIDER_MAN_X", masked)

        unmasked = self.store.unmask_terms(masked, unmask_map)
        self.assertEqual(unmasked, "Spider-Man swung past the building.")

    def test_locked_term_never_overwritten_by_auto_learn(self):
        """Test that manual or locked terms cannot be overwritten by subsequent translations."""
        self.store.register_term(
            source_term="Empire State University",
            canonical_uzbek="Empire State Universiteti",
            locked=True
        )

        # Attempt to overwrite with lower-quality translation
        success = self.store.register_term(
            source_term="Empire State University",
            canonical_uzbek="Imperiya Davlat Kolleji",
            locked=False
        )
        self.assertFalse(success)

        # Confirm canonical remains unchanged
        entries = self.store.get_all_entries()
        canonical_map = {pat: uz for pat, tok, uz in entries}
        self.assertTrue(any("Empire State Universiteti" in uz for uz in canonical_map.values()))

    def test_auto_learning_from_page(self):
        """Test auto-learning of new multi-word entities from dialogue."""
        pairs = [
            ("I received an award from Oscorp Industries.", "Men Oscorp Industries tashkilotidan mukofot oldim.")
        ]
        self.store.learn_from_page(pairs, page_source="page_test")
        
        # Verify it was learned and persisted
        self.assertIn("oscorp industries", self.store._entries)
        self.assertTrue(self.store._entries["oscorp industries"]["locked"])

    def test_disk_persistence_reload(self):
        """Test that reloading from disk restores all entries exactly."""
        self.store.register_term("Doctor Octopus", "Doktor Oktavius", locked=True)
        self.store.save()

        # Reload new store instance from same disk file
        new_store = terminology_store.TerminologyStore(store_path=self.test_json)
        self.assertIn("doctor octopus", new_store._entries)
        self.assertEqual(new_store._entries["doctor octopus"]["canonical_uzbek"], "Doktor Oktavius")


class TestSelfGrowingCharacterMemory(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_json = os.path.join(self.temp_dir.name, "test_char_mem.json")
        self.mem_store = character_profiles.CharacterMemoryStore(memory_path=self.test_json)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_deduplication_exact_and_fuzzy(self):
        """Test that similar facts (sim >= 0.65) are rejected with skipped log."""
        char = "Superior Spider-Man"
        # Novel base fact not in seed
        added1 = self.mem_store.add_fact(
            char,
            "Calibrates seismic dampeners within web-shooters to neutralize shockwaves",
            page="page_1"
        )
        self.assertTrue(added1)
        initial_count = len(self.mem_store.get_established_facts(char))

        # Near-duplicate fact with slight wording change
        added2 = self.mem_store.add_fact(
            char,
            "Calibrates seismic dampeners in web-shooters to neutralize vibration shockwaves",
            page="page_2"
        )
        self.assertFalse(added2)
        self.assertEqual(len(self.mem_store.get_established_facts(char)), initial_count)

    def test_novel_fact_accepted(self):
        """Test that genuinely novel facts are accepted and grow the memory."""
        char = "Superior Spider-Man"
        initial_count = len(self.mem_store.get_established_facts(char))

        novel_fact = "Considers Anna Maria Marconi to be an intellectual equal in biochemical engineering"
        added = self.mem_store.add_fact(char, novel_fact, page="page_10")
        self.assertTrue(added)
        self.assertEqual(len(self.mem_store.get_established_facts(char)), initial_count + 1)

    def test_learn_from_page_dialogue_extraction(self):
        """Test extracting facts from comic dialogue pairs."""
        char = "Superior Spider-Man"
        pairs = [
            ("PLAN: EPSILON FIVE!", "REJA: EPSILON-BESH!"),
            ("INSTANT FRICTIONLESS SURFACE!", "ISHQALANISHSIZ TEZKOR SIRT!")
        ]
        
        # First pass should add or skip based on seeds
        self.mem_store.learn_from_page(char, pairs, page="page_test")
        count_after_first = len(self.mem_store.get_established_facts(char))

        # Re-running the exact same page MUST skip all duplicates (count does not increase)
        added_on_rerun = self.mem_store.learn_from_page(char, pairs, page="page_test_rerun")
        self.assertEqual(added_on_rerun, 0)
        self.assertEqual(len(self.mem_store.get_established_facts(char)), count_after_first)

    def test_established_facts_extracted_strictly_from_english(self):
        """Test that established_facts are derived exclusively from English source text."""
        char = "Superior Spider-Man"
        initial_facts_count = len(self.mem_store.get_established_facts(char))

        # English contains 'FRICTIONLESS', Uzbek contains random text
        pairs = [("USE INSTANT FRICTIONLESS COATING!", "MUTLAQO ODDIY MATN.")]
        self.mem_store.learn_from_page(char, pairs, page="page_en_fact")
        
        facts = self.mem_store.get_established_facts(char)
        self.assertTrue(any("frictionless" in f.lower() for f in facts))

        # Inverse: English contains generic text, Uzbek contains triggering keywords
        # Must NOT extract fact because English does not contain the concept!
        pairs_inv = [("JUST A NORMAL STREET SCENE.", "BU YERDA TRAXEYA VA HIQILDOQ HAMDA SPIDER-BOTLAR BOR.")]
        facts_before = len(self.mem_store.get_established_facts(char))
        self.mem_store.learn_from_page(char, pairs_inv, page="page_uz_fake")
        facts_after = len(self.mem_store.get_established_facts(char))
        self.assertEqual(facts_before, facts_after)

    def test_pre_edit_mt_never_pollutes_sample_lines(self):
        """Test that raw MT output during initial translation never enters sample_lines."""
        char = "Superior Spider-Man"
        profile = self.mem_store.get_profile(char)
        initial_samples_count = len(profile.sample_lines)

        pairs = [("I'LL FLATTEN YOU LIKE A PANCAKE!", "MEN SENI BLINCHIKDEK TEKISLAYMAN!")]
        self.mem_store.learn_from_page(char, pairs, page="page_raw_mt")

        # Confirm sample_lines count did not grow from raw MT
        self.assertEqual(len(profile.sample_lines), initial_samples_count)
        self.assertFalse(any("BLINCHIKDEK" in s.get("voice_uz", "") for s in profile.sample_lines))

    def test_manual_user_edit_captured_as_sample_line(self):
        """
        Test user manual edit capture:
        1. Pre-edit raw MT does not linger in sample_lines.
        2. User's approved Uzbek text is captured as sample_line and exact_lines.
        3. Updating the edit replaces it in-place without stale duplicates.
        4. Trivial edits are skipped.
        5. Double-counting is prevented.
        """
        char = "Superior Spider-Man"
        en = "I'LL FLATTEN YOU LIKE A PANCAKE!"
        raw_mt = "MEN SENI BLINCHIKDEK TEKISLAYMAN!"
        user_edit = "SIZNI BUTKUL MAYDALAB TASHLAYMAN, NODON!"

        profile = self.mem_store.get_profile(char)
        initial_sample_count = len(profile.sample_lines)

        # 1. Confirm pre-edit raw MT is NOT in sample_lines
        self.assertFalse(any(s.get("en") == en for s in profile.sample_lines))

        # 2. User saves/confirms manual edit
        recorded, reason = self.mem_store.record_user_edited_sample_line(
            character_name=char,
            en_text=en,
            edited_uzbek=user_edit,
            pipeline_uzbek=raw_mt,
            page="test_page"
        )
        self.assertTrue(recorded)
        self.assertEqual(reason, "added")
        self.assertEqual(len(profile.sample_lines), initial_sample_count + 1)

        # Confirm new sample_line properties
        new_entry = [s for s in profile.sample_lines if s.get("en") == en][0]
        self.assertEqual(new_entry["voice_uz"], user_edit)
        self.assertEqual(new_entry["literal_mt"], raw_mt)
        # Confirm raw MT does NOT linger as voice_uz
        self.assertNotEqual(new_entry["voice_uz"], raw_mt)

        # 3. User makes a second edit on the same dialogue line
        second_edit = "SIZNI BUTKUL KULGA AYLANTIRAMAN, ARZIMAS AHMOQ!"
        updated, update_reason = self.mem_store.record_user_edited_sample_line(
            character_name=char,
            en_text=en,
            edited_uzbek=second_edit,
            pipeline_uzbek=raw_mt,
            page="test_page"
        )
        self.assertTrue(updated)
        self.assertEqual(update_reason, "updated")
        # Ensure count DID NOT grow (no duplicate entries!)
        self.assertEqual(len(profile.sample_lines), initial_sample_count + 1)
        updated_entry = [s for s in profile.sample_lines if s.get("en") == en][0]
        self.assertEqual(updated_entry["voice_uz"], second_edit)

        # 4. Trivial edit test (punctuation only)
        trivial_edit = "SIZNI BUTKUL KULGA AYLANTIRAMAN, ARZIMAS AHMOQ."
        rec_triv, reason_triv = self.mem_store.record_user_edited_sample_line(
            character_name=char,
            en_text=en,
            edited_uzbek=trivial_edit,
            pipeline_uzbek=second_edit,
            page="test_page"
        )
        self.assertFalse(rec_triv)
        self.assertEqual(reason_triv, "trivial_or_identical")

        # 5. Duplicate call test (confirm no double-counting)
        rec_dup, reason_dup = self.mem_store.record_user_edited_sample_line(
            character_name=char,
            en_text=en,
            edited_uzbek=second_edit,
            pipeline_uzbek=raw_mt,
            page="test_page"
        )
        self.assertFalse(rec_dup)
        self.assertEqual(reason_dup, "duplicate")
        self.assertEqual(len(profile.sample_lines), initial_sample_count + 1)

    def test_persistence_across_reloads(self):
        """Test that character memory survives restart from disk."""
        char = "Superior Spider-Man"
        novel_fact = "Maintains a secret lab sub-level at Horizon Labs"
        self.mem_store.add_fact(char, novel_fact, page="page_secret")

        # Reload store from same file
        reloaded = character_profiles.CharacterMemoryStore(memory_path=self.test_json)
        facts = reloaded.get_established_facts(char)
        self.assertIn(novel_fact, facts)

    def test_growing_memory_influences_naturalization(self):
        """Test that profile naturalization dynamically responds to established facts."""
        profile = self.mem_store.get_profile("Superior Spider-Man")
        self.assertIsNotNone(profile)
        
        # Ensure robotlarim -> Spider-botlarim because Spider-bots fact is established
        dialogue = "Barcha robotlarim tayyor."
        polished = profile.naturalize(dialogue, en_orig="")
        self.assertIn("SPIDER-BOTLAR", polished.upper())


if __name__ == "__main__":
    unittest.main()

