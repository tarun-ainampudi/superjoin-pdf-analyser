import tempfile
import unittest
from pathlib import Path

from src.knowledge_layer.cache import clear_session_cache, get_session_cache_dir, get_dataset_cache_dir


class CacheBehaviorTests(unittest.TestCase):
    def test_session_cache_directory_is_unique_and_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = get_session_cache_dir(root, "session-a")
            second = get_session_cache_dir(root, "session-b")

            self.assertNotEqual(str(first), str(second))
            first.mkdir(parents=True, exist_ok=True)
            (first / "marker.txt").write_text("cached", encoding="utf-8")

            clear_session_cache(root)

            self.assertFalse((root / ".cache" / "session").exists())

    def test_dataset_cache_directory_is_created_per_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = get_dataset_cache_dir(root)
            self.assertTrue(cache_dir.name == "dataset")
            self.assertTrue(cache_dir.parent.name == ".cache")
            self.assertTrue(cache_dir.parent.parent == root)


if __name__ == "__main__":
    unittest.main()
