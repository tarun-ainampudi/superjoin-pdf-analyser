"""Persistent cache helpers for dataset PDFs and session-scoped uploads."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

from .models import Fact


def _cache_root(root: Optional[Path] = None) -> Path:
    base = root or Path(__file__).resolve().parents[2]
    return base / ".cache"


def get_dataset_cache_dir(root: Optional[Path] = None) -> Path:
    cache_dir = _cache_root(root) / "dataset"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_session_cache_dir(root: Optional[Path] = None, session_id: Optional[str] = None) -> Path:
    cache_dir = _cache_root(root) / "session"
    if session_id:
        cache_dir = cache_dir / session_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def clear_session_cache(root: Optional[Path] = None) -> Path:
    session_dir = _cache_root(root) / "session"
    if session_dir.exists():
        shutil.rmtree(session_dir)
    return session_dir


def _file_signature(path: str) -> str:
    file_path = Path(path)
    stat = file_path.stat()
    payload = f"{file_path.name}|{stat.st_size}|{int(stat.st_mtime_ns)}|{file_path.resolve()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_cached_facts(path: str, cache_dir: Optional[Path] = None) -> Optional[List[Fact]]:
    base_dir = Path(cache_dir) if cache_dir else get_dataset_cache_dir()
    cache_file = base_dir / f"{_file_signature(path)}.json"
    if not cache_file.exists():
        return None
    with cache_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [Fact(**item) for item in data]


def save_cached_facts(path: str, facts: Iterable[Fact], cache_dir: Optional[Path] = None) -> Path:
    base_dir = Path(cache_dir) if cache_dir else get_dataset_cache_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_file = base_dir / f"{_file_signature(path)}.json"
    payload = [fact.__dict__ for fact in facts]
    with cache_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return cache_file
