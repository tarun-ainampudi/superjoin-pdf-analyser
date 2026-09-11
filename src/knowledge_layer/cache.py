"""Persistent cache helpers for dataset PDFs and session-scoped uploads."""

import hashlib
import json
import os
import shutil
import tempfile
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
    # Hash content, not the temporary upload path or mtime. Streamlit uploads
    # are written to a fresh temp directory on every rerun, so metadata-based
    # signatures made a perfectly valid session cache impossible to reuse.
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy_file_signature(path: str) -> str:
    """Return the cache filename scheme used before content-addressed cache.

    Kept only for reading old cache entries. It includes the file path and
    mtime, so it is unsuitable for newly written entries (especially uploads
    recreated in a temporary directory on each Streamlit rerun).
    """
    file_path = Path(path)
    stat = file_path.stat()
    payload = (
        f"{file_path.name}|{stat.st_size}|{int(stat.st_mtime_ns)}|{file_path.resolve()}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_cache_file(cache_file: Path) -> Optional[List[Fact]]:
    try:
        with cache_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("cache payload is not a list")
        return [Fact(**item) for item in data]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # A partially written or old-schema entry must never prevent analysis.
        # Remove only this invalid entry; the next extraction recreates it.
        cache_file.unlink(missing_ok=True)
        return None


def load_cached_facts(path: str, cache_dir: Optional[Path] = None) -> Optional[List[Fact]]:
    base_dir = Path(cache_dir) if cache_dir else get_dataset_cache_dir()
    current_file = base_dir / f"{_file_signature(path)}.json"
    if current_file.exists():
        return _read_cache_file(current_file)

    # Backward compatibility: use an entry made by the old filename/mtime/path
    # key when the original PDF is still available at that path. Successful
    # reads are migrated to the new stable name for all future loads.
    legacy_file = base_dir / f"{_legacy_file_signature(path)}.json"
    if not legacy_file.exists():
        return None
    facts = _read_cache_file(legacy_file)
    if facts is not None:
        save_cached_facts(path, facts, base_dir)
    return facts


def save_cached_facts(path: str, facts: Iterable[Fact], cache_dir: Optional[Path] = None) -> Path:
    base_dir = Path(cache_dir) if cache_dir else get_dataset_cache_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_file = base_dir / f"{_file_signature(path)}.json"
    payload = [fact.__dict__ for fact in facts]
    # An atomic replace prevents a browser refresh or process interruption from
    # leaving a corrupt JSON cache that breaks future loads.
    fd, temp_name = tempfile.mkstemp(prefix=f".{cache_file.stem}-", suffix=".tmp", dir=base_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, cache_file)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return cache_file
