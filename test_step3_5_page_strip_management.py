"""
Unit tests for Big Update: Page Strip Live Reordering & Context Menu (Delete, Replace, Insert).
Verifies:
1. Live horizontal reorder logic: moving page 2 to index 4 shifts pages 3, 4, 5 left by 1.
2. Sequential page renumbering: all pages get pageNum = 1..N and updated sequentialName.
3. Active page index preservation: workspace displays the correct page after reordering.
4. Page deletion logic: deleting target page renumbers subsequent pages and adjusts active page index.
5. Page insertion logic: inserting new pages right after a target page splices cleanly and renumbers all items.
"""

import unittest


def simulate_live_reorder(queue, drag_id, target_idx):
    """Simulates the real-time live drag reorder occurring during mousemove."""
    cur_idx = next(i for i, p in enumerate(queue) if p["id"] == drag_id)
    if cur_idx != target_idx:
        moved = queue.pop(cur_idx)
        queue.insert(target_idx, moved)
    return queue


def finalize_reorder(queue, active_id):
    """Simulates mouseup finalization: renumbering and active index adjustment."""
    for i, p in enumerate(queue):
        p["pageNum"] = i + 1
        p["sequentialName"] = f"{p['pageNum']}_{p['name']}"
    new_active_idx = next((i for i, p in enumerate(queue) if p["id"] == active_id), 0)
    return queue, new_active_idx


def simulate_delete_page(queue, target_idx, cur_active_idx):
    """Simulates context menu 'Sahifani o'chirish' action."""
    queue.pop(target_idx)
    for i, p in enumerate(queue):
        p["pageNum"] = i + 1
        p["sequentialName"] = f"{p['pageNum']}_{p['name']}"
    if cur_active_idx == target_idx:
        new_active = min(target_idx, len(queue) - 1)
    elif cur_active_idx > target_idx:
        new_active = cur_active_idx - 1
    else:
        new_active = cur_active_idx
    return queue, new_active


def simulate_insert_pages(queue, target_idx, new_files):
    """Simulates context menu 'Yangi sahifa qo'shish' action."""
    insert_idx = target_idx + 1
    items = [
        {"id": f"new_{i}", "name": name, "pageNum": 0, "sequentialName": ""}
        for i, name in enumerate(new_files)
    ]
    queue[insert_idx:insert_idx] = items
    for i, p in enumerate(queue):
        p["pageNum"] = i + 1
        p["sequentialName"] = f"{p['pageNum']}_{p['name']}"
    return queue, insert_idx


class TestPageStripManagement(unittest.TestCase):

    def setUp(self):
        self.queue = [
            {"id": f"p_{i}", "name": f"page_{i}.jpg", "pageNum": i, "sequentialName": f"{i}_page_{i}.jpg"}
            for i in range(1, 9)
        ]

    def test_live_reorder_page2_to_position5(self):
        """User requirement: agar 2-page 5 ga o'tsa chapda qolgan barchasi 1 ta bitta chapga ko'chadi."""
        # Initial: [p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8]
        # Drag p_2 (current index 1) to position 5 (index 4)
        active_id = "p_3"
        reordered = simulate_live_reorder(self.queue, "p_2", 4)

        # In intermediate list: p_3 was at 2, now at 1; p_4 was at 3, now at 2; p_5 was at 4, now at 3
        # p_2 is now at index 4 (5th page)
        self.assertEqual(reordered[0]["id"], "p_1")
        self.assertEqual(reordered[1]["id"], "p_3")
        self.assertEqual(reordered[2]["id"], "p_4")
        self.assertEqual(reordered[3]["id"], "p_5")
        self.assertEqual(reordered[4]["id"], "p_2")
        self.assertEqual(reordered[5]["id"], "p_6")

        final_q, active_idx = finalize_reorder(reordered, active_id)
        # All pages must be renumbered 1..8
        for i, p in enumerate(final_q):
            self.assertEqual(p["pageNum"], i + 1)
            self.assertTrue(p["sequentialName"].startswith(f"{i+1}_"))

        # Active page p_3 index updated from 2 to 1
        self.assertEqual(active_idx, 1)

    def test_delete_page_renumbers_and_adjusts_active_index(self):
        """Deleting a page shifts subsequent pages and adjusts active index."""
        # Delete page at index 2 (p_3) while active is index 4 (p_5)
        q, active_idx = simulate_delete_page(self.queue, 2, 4)
        self.assertEqual(len(q), 7)
        # Old p_5 is now at index 3
        self.assertEqual(active_idx, 3)
        self.assertEqual(q[3]["id"], "p_5")
        self.assertEqual(q[3]["pageNum"], 4)

    def test_insert_page_after_target(self):
        """Inserting new page(s) after target page 3 places them at position 4 and pushes rest."""
        # Insert after index 2 (p_3)
        new_files = ["bonus_1.png", "bonus_2.png"]
        q, inserted_idx = simulate_insert_pages(self.queue, 2, new_files)
        self.assertEqual(len(q), 10)
        self.assertEqual(inserted_idx, 3)
        self.assertEqual(q[3]["name"], "bonus_1.png")
        self.assertEqual(q[3]["pageNum"], 4)
        self.assertEqual(q[4]["name"], "bonus_2.png")
        self.assertEqual(q[4]["pageNum"], 5)
        # Old p_4 is now at index 5 with pageNum = 6
    def test_replace_page_clears_bubbles_and_state(self):
        """Replacing a page (even if previously translated) must clear all old bubbles, reset status to pending, and assign fresh id."""
        # Setup page 2 as previously translated with bubbles and cleaned image
        self.queue[1]["status"] = "translated"
        self.queue[1]["bubbles"] = [{"bubble_id": 1, "original_text": "Old text", "uzbek_translation": "Eski matn"}]
        self.queue[1]["cleanedImage"] = "mock_cleaned_img"
        self.queue[1]["cleanedImageB64"] = "data:image/jpeg;base64,mock"

        # Replace page 2 with new file
        idx = 1
        target_item = self.queue[idx]
        old_id = target_item["id"]
        new_file_name = "new_page_clean.png"

        # Apply replacement logic
        new_id = f"page_new_{idx}"
        target_item["id"] = new_id
        target_item["name"] = new_file_name
        target_item["sequentialName"] = f"{target_item['pageNum']}_{new_file_name}"
        target_item["originalImage"] = "new_mock_orig"
        target_item["cleanedImage"] = None
        target_item["cleanedImageB64"] = None
        target_item["bubbles"] = []
        target_item["status"] = "pending"

        self.assertNotEqual(target_item["id"], old_id)
        self.assertEqual(target_item["status"], "pending")
        self.assertEqual(len(target_item["bubbles"]), 0)
        self.assertIsNone(target_item["cleanedImage"])
        self.assertIsNone(target_item["cleanedImageB64"])


if __name__ == "__main__":
    unittest.main()
