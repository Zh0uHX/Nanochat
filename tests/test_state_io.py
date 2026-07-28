import json
import os
import tempfile
import unittest

from nanochat.state_io import load_json, save_json_atomic


class TestStateIO(unittest.TestCase):
    def test_atomic_json_round_trip(self):
        expected = {"rank": 3, "buffer": [[1, 2], [3]], "nested": {"step": 9}}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rank_state.json")
            save_json_atomic(path, expected)
            self.assertEqual(load_json(path), expected)
            self.assertEqual(
                [name for name in os.listdir(directory) if name.endswith(".tmp")],
                [],
            )

    def test_load_requires_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "invalid.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump([1, 2, 3], handle)
            with self.assertRaisesRegex(TypeError, "expected a JSON object"):
                load_json(path)


if __name__ == "__main__":
    unittest.main()
