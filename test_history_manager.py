"""
test_history_manager.py
-----------------------
Unit tests for Comic-Lab History Manager (Undo / Redo engine).
"""

import unittest
import streamlit as st
import history_manager


class TestHistoryManager(unittest.TestCase):
    def setUp(self):
        # Reset session state for history
        st.session_state.undo_stack = []
        st.session_state.redo_stack = []
        st.session_state.bubbles = [
            {"bubble_id": 1, "original_text": "HELLO", "uzbek_translation": "SALOM", "box": [10, 10, 50, 50]},
            {"bubble_id": 2, "original_text": "WORLD", "uzbek_translation": "DUNYO", "box": [60, 60, 100, 100]}
        ]
        st.session_state.analysis_version = 0

    def test_initial_state(self):
        self.assertFalse(history_manager.can_undo())
        self.assertFalse(history_manager.can_redo())
        success, msg = history_manager.undo_action()
        self.assertFalse(success)
        success_r, msg_r = history_manager.redo_action()
        self.assertFalse(success_r)

    def test_push_and_undo(self):
        # Snapshot before edit
        history_manager.push_undo_snapshot("Birinchi pufakni o'zgartirish")
        self.assertTrue(history_manager.can_undo())
        self.assertFalse(history_manager.can_redo())

        # Modify bubbles
        st.session_state.bubbles[0]["uzbek_translation"] = "YANGI SALOM"
        st.session_state.bubbles.append({"bubble_id": 3, "original_text": "NEW"})

        # Undo
        success, desc = history_manager.undo_action()
        self.assertTrue(success)
        self.assertIn("Birinchi pufakni o'zgartirish", desc)

        # Verify state restored
        self.assertEqual(len(st.session_state.bubbles), 2)
        self.assertEqual(st.session_state.bubbles[0]["uzbek_translation"], "SALOM")
        self.assertEqual(st.session_state.analysis_version, 1)

        # Now can redo
        self.assertFalse(history_manager.can_undo())
        self.assertTrue(history_manager.can_redo())

    def test_redo(self):
        history_manager.push_undo_snapshot("Tahrir 1")
        st.session_state.bubbles[0]["box"] = [20, 20, 80, 80]

        # Undo
        history_manager.undo_action()
        self.assertEqual(st.session_state.bubbles[0]["box"], [10, 10, 50, 50])

        # Redo
        success, desc = history_manager.redo_action()
        self.assertTrue(success)
        self.assertEqual(st.session_state.bubbles[0]["box"], [20, 20, 80, 80])

        # Now can undo again
        self.assertTrue(history_manager.can_undo())
        self.assertFalse(history_manager.can_redo())

    def test_redo_stack_invalidation_on_new_action(self):
        history_manager.push_undo_snapshot("Action 1")
        st.session_state.bubbles[0]["original_text"] = "MODIFIED 1"

        # Undo
        history_manager.undo_action()
        self.assertTrue(history_manager.can_redo())

        # User performs a NEW action instead of redoing
        history_manager.push_undo_snapshot("Action 2")
        st.session_state.bubbles[1]["original_text"] = "MODIFIED 2"

        # Redo stack must be cleared
        self.assertFalse(history_manager.can_redo())
        self.assertTrue(history_manager.can_undo())

    def test_deep_copy_independence(self):
        history_manager.push_undo_snapshot("Snapshot")
        # Mutate nested dictionary in-place
        st.session_state.bubbles[0]["box"][0] = 999
        st.session_state.bubbles[0]["box"][1] = 888

        history_manager.undo_action()
        # Restored values must not have been corrupted by in-place mutation
        self.assertEqual(st.session_state.bubbles[0]["box"], [10, 10, 50, 50])

    def test_finalize_split_undo_restores_original_bubble(self):
        import bubble_mask_editor
        orig_bubble = {
            "bubble_id": 1,
            "original_text": "HEY PETER PARKER",
            "uzbek_translation": "SALOM PITER PARKER",
            "contour": [[10, 10], [50, 10], [50, 50], [10, 50]],
            "lines": []
        }
        sub1 = {
            "bubble_id": 1,
            "original_text": "HEY PETER",
            "uzbek_translation": "SALOM PITER",
            "contour": [[10, 10], [30, 10], [30, 50], [10, 50]],
            "lines": []
        }
        sub2 = {
            "bubble_id": 2,
            "original_text": "PARKER",
            "uzbek_translation": "PARKER",
            "contour": [[30, 10], [50, 10], [50, 50], [30, 50]],
            "lines": []
        }
        st.session_state.bubbles = [orig_bubble]

        # Finalize split (which internally pushes an undo snapshot)
        try:
            bubble_mask_editor._finalize_split(0, sub1, sub2, "page_test.png")
        except Exception:
            # st.rerun() raises a Streamlit RerunException in test mode
            pass

        self.assertEqual(len(st.session_state.bubbles), 2)
        self.assertTrue(history_manager.can_undo())

        # Undo the split
        succ, desc = history_manager.undo_action()
        self.assertTrue(succ)
        self.assertEqual(len(st.session_state.bubbles), 1)
        self.assertEqual(st.session_state.bubbles[0]["original_text"], "HEY PETER PARKER")
        self.assertEqual(st.session_state.bubbles[0]["uzbek_translation"], "SALOM PITER PARKER")


if __name__ == "__main__":
    unittest.main()
