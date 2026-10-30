import unittest
import numpy as np
from modules.pipeline import (
    pluralize_object,
    get_position,
    collect_scene_details,
    describe_scene_with_context,
)


class PipelineTests(unittest.TestCase):
    def test_pluralize(self):
        self.assertEqual(pluralize_object("person", 1), "person")
        self.assertEqual(pluralize_object("person", 2), "people")
        self.assertEqual(pluralize_object("car", 3), "cars")
        self.assertEqual(pluralize_object("bus", 2), "buses")

    def test_get_position(self):
        self.assertEqual(get_position(100, 1000), "left")
        self.assertEqual(get_position(500, 1000), "center")
        self.assertEqual(get_position(900, 1000), "right")

    def test_empty_scene_description(self):
        caption = describe_scene_with_context([], {}, 640)
        self.assertEqual(caption, "No objects are currently detected in the frame.")

    def test_scene_description_with_tracks(self):
        tracks = [
            {"class_name": "car", "track_id": 1, "bbox": [100, 100, 200, 200]},
            {"class_name": "person", "track_id": 2, "bbox": [300, 100, 350, 300]},
        ]
        caption = describe_scene_with_context(tracks, {}, 640)
        self.assertIn("car", caption)
        self.assertIn("person", caption)


if __name__ == "__main__":
    unittest.main()

