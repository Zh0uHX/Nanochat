import unittest
from pathlib import Path
import tempfile

from nanochat.provenance import _content_manifest_sha256, configuration_hash


class ProvenanceTests(unittest.TestCase):
    def test_configuration_hash_is_order_independent(self):
        left = {"batch_size": 8, "nested": {"seed": 7, "dtype": "bfloat16"}}
        right = {"nested": {"dtype": "bfloat16", "seed": 7}, "batch_size": 8}
        self.assertEqual(configuration_hash(left), configuration_hash(right))

    def test_configuration_hash_changes_with_semantics(self):
        base = {"batch_size": 8, "seed": 7}
        changed = {"batch_size": 16, "seed": 7}
        self.assertNotEqual(configuration_hash(base), configuration_hash(changed))

    def test_content_manifest_covers_untracked_file_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "new_source.py")
            path.write_text("value = 1\n", encoding="utf-8")
            before = _content_manifest_sha256(directory, ["new_source.py"])
            path.write_text("value = 2\n", encoding="utf-8")
            after = _content_manifest_sha256(directory, ["new_source.py"])
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
