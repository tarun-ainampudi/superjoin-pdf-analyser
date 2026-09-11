import tempfile
import hashlib
import unittest
from pathlib import Path

from src.knowledge_layer.cache import (
    clear_session_cache,
    get_session_cache_dir,
    get_dataset_cache_dir,
    load_cached_facts,
    save_cached_facts,
)
from src.knowledge_layer.fact_extraction import extract_facts_from_pdf
from src.knowledge_layer import llm


class CacheBehaviorTests(unittest.TestCase):
    def test_session_cache_directory_is_unique_and_cleared(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            first = get_session_cache_dir(root, "session-a")
            second = get_session_cache_dir(root, "session-b")

            self.assertNotEqual(str(first), str(second))
            first.mkdir(parents=True, exist_ok=True)
            (first / "marker.txt").write_text("cached", encoding="utf-8")

            clear_session_cache(root)

            self.assertFalse((root / ".cache" / "session").exists())

    def test_dataset_cache_directory_is_created_per_repo(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            cache_dir = get_dataset_cache_dir(root)
            self.assertTrue(cache_dir.name == "dataset")
            self.assertTrue(cache_dir.parent.name == ".cache")
            self.assertTrue(cache_dir.parent.parent == root)

    def test_cache_uses_content_not_temporary_path(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            cache_dir = get_dataset_cache_dir(root)
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first = first_dir / "upload.pdf"
            second = second_dir / "upload.pdf"
            first.write_bytes(b"same pdf bytes")
            second.write_bytes(b"same pdf bytes")

            save_cached_facts(str(first), [], cache_dir)

            self.assertEqual(load_cached_facts(str(second), cache_dir), [])

    def test_corrupt_cache_is_ignored_and_removed(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            cache_dir = get_dataset_cache_dir(root)
            pdf = root / "report.pdf"
            pdf.write_bytes(b"content")
            save_cached_facts(str(pdf), [], cache_dir)
            cache_file = next(cache_dir.glob("*.json"))
            cache_file.write_text("not json", encoding="utf-8")

            self.assertIsNone(load_cached_facts(str(pdf), cache_dir))
            self.assertFalse(cache_file.exists())

    def test_legacy_filename_is_loaded_and_migrated(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            cache_dir = get_dataset_cache_dir(root)
            pdf = root / "report.pdf"
            pdf.write_bytes(b"legacy content")
            stat = pdf.stat()
            legacy_payload = (
                f"{pdf.name}|{stat.st_size}|{int(stat.st_mtime_ns)}|{pdf.resolve()}"
            ).encode("utf-8")
            legacy_file = cache_dir / f"{hashlib.sha256(legacy_payload).hexdigest()}.json"
            legacy_file.write_text("[]", encoding="utf-8")

            self.assertEqual(load_cached_facts(str(pdf), cache_dir), [])
            self.assertEqual(len(list(cache_dir.glob("*.json"))), 2)

    def test_cache_hit_is_reported_as_active_backend(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            cache_dir = get_dataset_cache_dir(root)
            pdf = root / "report.pdf"
            pdf.write_bytes(b"cached PDF")
            save_cached_facts(str(pdf), [], cache_dir)
            llm.set_used_backend("ollama")

            self.assertEqual(extract_facts_from_pdf(str(pdf), cache_dir=cache_dir), [])
            self.assertEqual(llm.get_used_backend(), "cache")


if __name__ == "__main__":
    unittest.main()
