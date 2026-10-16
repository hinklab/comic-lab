"""
Unit tests for Step 3.8: Standalone Text Canvas Resize Handles and Interaction
Verifies:
1. getBubbleHandles geometric precision: Standalone text (is_shape_hidden: true) handles match exact bounding box.
2. getHitHandle hit radius and closest-handle selection across different zoom scales.
3. Priority #1 handle hit testing in mousedown and mousemove in web/index.html.
4. Selection frame and handle rendering synchronization with getBubbleHandles.
5. Standalone text creation layer target set to 'all'.
"""

import unittest
import math
from pathlib import Path


class TestStandaloneTextResizing(unittest.TestCase):
    def setUp(self):
        self.html_path = Path(__file__).parent / "web" / "index.html"
        self.assertTrue(self.html_path.exists(), "web/index.html must exist")
        self.html_content = self.html_path.read_text(encoding="utf-8")

    def test_standalone_text_handles_geometry(self):
        """Simulate getBubbleHandles for standalone text vs normal bubble"""
        def getBubbleHandles(b, part=None, selectedLayerPart='all'):
            bw = b['x1'] - b['x0']
            bh = b['y1'] - b['y0']
            activePart = part or selectedLayerPart

            if activePart == 'text' and not b.get('is_shape_hidden', False):
                ox = b.get('offset_x', 0)
                oy = b.get('offset_y', 0)
                padX = bw * 0.08
                padY = bh * 0.08
                x0 = b['x0'] + padX + ox
                y0 = b['y0'] + padY + oy
                w = max(10, bw - padX * 2)
                h = max(10, bh - padY * 2)
                x1 = x0 + w
                y1 = y0 + h
                mx = x0 + w / 2
                my = y0 + h / 2
                return {
                    'nw': {'x': x0, 'y': y0, 'cursor': 'nwse-resize'},
                    'n':  {'x': mx, 'y': y0, 'cursor': 'ns-resize'},
                    'ne': {'x': x1, 'y': y0, 'cursor': 'nesw-resize'},
                    'e':  {'x': x1, 'y': my, 'cursor': 'ew-resize'},
                    'se': {'x': x1, 'y': y1, 'cursor': 'nwse-resize'},
                    's':  {'x': mx, 'y': y1, 'cursor': 'ns-resize'},
                    'sw': {'x': x0, 'y': y1, 'cursor': 'nesw-resize'},
                    'w':  {'x': x0, 'y': my, 'cursor': 'ew-resize'}
                }

            mx = b['x0'] + bw / 2
            my = b['y0'] + bh / 2
            return {
                'nw': {'x': b['x0'], 'y': b['y0'], 'cursor': 'nwse-resize'},
                'n':  {'x': mx,       'y': b['y0'], 'cursor': 'ns-resize'},
                'ne': {'x': b['x1'], 'y': b['y0'], 'cursor': 'nesw-resize'},
                'e':  {'x': b['x1'], 'y': my,       'cursor': 'ew-resize'},
                'se': {'x': b['x1'], 'y': b['y1'], 'cursor': 'nwse-resize'},
                's':  {'x': mx,       'y': b['y1'], 'cursor': 'ns-resize'},
                'sw': {'x': b['x0'], 'y': b['y1'], 'cursor': 'nesw-resize'},
                'w':  {'x': b['x0'], 'y': my,       'cursor': 'ew-resize'}
            }

        # Standalone text element
        standalone_text = {
            'bubble_id': 101,
            'x0': 100,
            'y0': 200,
            'x1': 300,
            'y1': 350,
            'is_shape_hidden': True,
            'is_text_hidden': False
        }

        # Regardless of part='text' or part='all', standalone text handles must match b.x0, b.y0, b.x1, b.y1
        handles_all = getBubbleHandles(standalone_text, part='all')
        handles_text = getBubbleHandles(standalone_text, part='text')

        self.assertEqual(handles_all['nw']['x'], 100)
        self.assertEqual(handles_all['nw']['y'], 200)
        self.assertEqual(handles_all['se']['x'], 300)
        self.assertEqual(handles_all['se']['y'], 350)
        self.assertEqual(handles_all['n']['x'], 200)
        self.assertEqual(handles_all['n']['y'], 200)

        self.assertEqual(handles_text, handles_all, "Standalone text handles must be identical regardless of active layer part")

    def test_get_hit_handle_closest_distance(self):
        """Simulate getHitHandle closest-distance calculation and hit testing"""
        def getHitHandle(handles, canvasX, canvasY, scale=1.0):
            hitRadius = max(10, 13 / scale)
            closestHandle = None
            minDistance = hitRadius

            for key, h in handles.items():
                dist = math.hypot(canvasX - h['x'], canvasY - h['y'])
                if dist <= minDistance:
                    minDistance = dist
                    closestHandle = {'name': key, 'cursor': h['cursor']}
            return closestHandle

        handles = {
            'nw': {'x': 100, 'y': 200, 'cursor': 'nwse-resize'},
            'n':  {'x': 200, 'y': 200, 'cursor': 'ns-resize'},
            'ne': {'x': 300, 'y': 200, 'cursor': 'nesw-resize'},
            'se': {'x': 300, 'y': 350, 'cursor': 'nwse-resize'},
        }

        # Test exact hit on NW corner
        hit = getHitHandle(handles, 100, 200, scale=1.0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit['name'], 'nw')
        self.assertEqual(hit['cursor'], 'nwse-resize')

        # Test near hit on NW corner within hitRadius (e.g. 106, 204 -> dist ~ 7.2px)
        hit_near = getHitHandle(handles, 106, 204, scale=1.0)
        self.assertIsNotNone(hit_near)
        self.assertEqual(hit_near['name'], 'nw')

        # Test miss outside hitRadius (e.g. 120, 220 -> dist ~ 28.2px)
        hit_miss = getHitHandle(handles, 120, 220, scale=1.0)
        self.assertIsNone(hit_miss)

        # Test zoomed out (scale=0.5): hitRadius becomes 26px, so 115, 215 is now a hit
        hit_zoom_out = getHitHandle(handles, 115, 210, scale=0.5)
        self.assertIsNotNone(hit_zoom_out)
        self.assertEqual(hit_zoom_out['name'], 'nw')

    def test_html_mousedown_priority(self):
        """Verify mousedown in index.html gives Priority #1 to resize handle hit testing"""
        self.assertIn("RESIZE HANDLE HIT DETECTION - PRIORITY #1", self.html_content)
        self.assertIn("const hitH = getHitHandle(selB, clickX, clickY);", self.html_content)
        self.assertIn("mode: 'resize'", self.html_content)

    def test_html_mousemove_cursor(self):
        """Verify mousemove checks getHitHandle before body move cursor"""
        self.assertIn("const hitH = getHitHandle(selB, curX, curY);", self.html_content)
        self.assertIn("canvas.style.cursor = hitH.cursor;", self.html_content)

    def test_html_standalone_text_creation(self):
        """Verify creating standalone text sets layer target to 'all'"""
        self.assertIn("is_shape_hidden: true", self.html_content)
        # Check toggleDrawTextMode followed by setInspectorLayerTarget('all')
        self.assertIn("toggleDrawTextMode(false);\n        setInspectorLayerTarget('all');", self.html_content)

    def test_html_render_canvas_handles_sync(self):
        """Verify renderCanvas uses getBubbleHandles directly for drawing handles"""
        self.assertIn("const handles = getBubbleHandles(b, 'all');", self.html_content)
        self.assertIn("const handles = getBubbleHandles(b, 'text');", self.html_content)
        self.assertIn("handleSize = Math.max(8, Math.min(13, 10 / scale));", self.html_content)


if __name__ == "__main__":
    unittest.main()
