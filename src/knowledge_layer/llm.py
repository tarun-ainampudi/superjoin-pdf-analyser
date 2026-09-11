"""Local Ollama client, Gemini fallback, and robust JSON parsing.

The small local model (granite4.1:3b) often returns concatenated JSON objects
instead of a proper array, so we repair the response defensively. If the local
model fails or is unavailable, the system can fall back to Gemini.
"""

import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

ROOT_DIR = Path(__file__).resolve().parents[2]
_ENV_FILE = ROOT_DIR / ".env"
if load_dotenv is not None:
    load_dotenv(_ENV_FILE, override=False)
else:
    try:
        if _ENV_FILE.exists():
            for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except Exception as exc:  # pragma: no cover
        logging.getLogger("superjoin.ollama").warning("Failed to parse .env fallback: %s", exc)

from .config import (
    GEMINI_API_KEY,
    GEMINI_API_URL,
    GEMINI_MODEL,
    GEMINI_TIMEOUT,
    OLLAMA_CALL_TIMEOUT,
    OLLAMA_MODEL,
    OLLAMA_RETRIES,
    OLLAMA_URL,
)

logger = logging.getLogger("superjoin.ollama")


class ModelTimeoutError(TimeoutError):
    """Raised when a model response exceeds the allowed wall-clock budget, so
    the caller can fall back to the standard (heuristic) approach."""


class ModelUnavailableError(RuntimeError):
    """Raised after all configured model backends have been exhausted."""

# Availability probes are cached per model so each backend's reachability is
# checked only once per process (e.g. a single Ollama /api/tags round-trip),
# rather than repeated on every call. This matters because call_model falls
# back between backends and the UI queries availability multiple times.
_availability: Dict[str, Optional[bool]] = {"ollama": None, "gemini": None}
# The backend that most recently served a model call (updated by call_model).
_used_backend: str = "cache"


def _ollama_reachable() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as r:
            status = getattr(r, "status", 200)
            logger.info("Ollama availability check status=%s", status)
            return status == 200
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ollama availability check failed: %s", exc)
        return False


def ollama_available(refresh: bool = False) -> bool:
    """Return whether a local Ollama instance is reachable.

    The network probe is performed only once and cached; subsequent calls in
    the same process return the remembered result unless ``refresh`` is True.
    """
    if refresh or _availability["ollama"] is None:
        _availability["ollama"] = _ollama_reachable()
    return bool(_availability["ollama"])


def gemini_available(refresh: bool = False) -> bool:
    """Return True if a Gemini API key is configured.

    Cached so the availability of each model is determined only once.
    """
    if refresh or _availability["gemini"] is None:
        _availability["gemini"] = bool(GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "").strip())
        logger.info("Gemini availability: %s (model=%s)", _availability["gemini"], GEMINI_MODEL)
    return bool(_availability["gemini"])


def get_active_backend() -> str:
    """Return the preferred backend name: gemini, ollama, or none.

    Uses the cached per-model availability checks so each model is probed at
    most once.
    """
    if gemini_available():
        return "gemini"
    if ollama_available():
        return "ollama"
    return "none"


def get_used_backend() -> str:
    """Return the backend that most recently served a model call ('' if none)."""
    return _used_backend


def set_used_backend(backend: str) -> None:
    """Record the path that supplied the current result (including cache)."""
    global _used_backend
    _used_backend = backend


def reset_availability() -> None:
    """Forget cached availability results so the next query re-probes each model.

    This lets the UI detect backend changes between runs (e.g. Ollama being
    started or stopped) while still probing each model only once per run.
    """
    _availability["ollama"] = None
    _availability["gemini"] = None


def _request_ollama(endpoint: str, payload: Dict[str, Any]) -> Any:
    req = urllib.request.Request(
        f"{OLLAMA_URL}{endpoint}",
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    logger.debug("Ollama request -> %s payload=%s", endpoint, payload)
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_CALL_TIMEOUT) as r:
            return json.load(r)
    except socket.timeout as exc:
        raise ModelTimeoutError(
            f"Ollama response exceeded {OLLAMA_CALL_TIMEOUT}s timeout"
        ) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise ModelTimeoutError(
                f"Ollama response exceeded {OLLAMA_CALL_TIMEOUT}s timeout"
            ) from exc
        raise


def _gemini_request(system: str, user: str) -> str:
    api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    url = f"{GEMINI_API_URL}/models/{GEMINI_MODEL}:generateContent"
    body = {
        "contents": [{"parts": [{"text": f"{system}\n\n{user}"}]}],
        "generationConfig": {"temperature": 0},
    }
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key,
    }
    logger.info("Calling Gemini backend: model=%s, url=%s", GEMINI_MODEL, url)

    for attempt in range(1, 3):
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=GEMINI_TIMEOUT) as r:
                payload = json.load(r)
            break
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            logger.warning("Gemini HTTP %s on attempt %s: %s", exc.code, attempt, error_body[:400])
            if exc.code == 429:
                # Quota/rate-limit exhausted: hand straight off to the fallback
                # backend instead of waiting inside a retry loop.
                logger.warning("Gemini rate-limit (429); switching to fallback backend")
                raise
            if exc.code != 200:
                logger.warning("Gemini request failed with HTTP %s: %s", exc.code, error_body[:400])
                raise
            if attempt < 2:
                continue
            raise
    else:  # pragma: no cover - loop always breaks or raises
        payload = {}

    try:
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        logger.info("Gemini model responded successfully")
        return text
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini response parsing failed: %s | payload=%s", exc, payload)
        raise RuntimeError("Gemini returned an unexpected payload") from exc


def ollama_chat(system: str, user: str, use_json_format: bool = False) -> str:
    """Send a chat request to local Ollama and return the raw response text."""
    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "temperature": 0,
    }
    if use_json_format:
        payload["format"] = "json"

    for attempt in range(1, OLLAMA_RETRIES + 1):
        try:
            logger.info(
                "Attempting local Ollama chat: model=%s, attempt=%s, prompt_chars=%s",
                OLLAMA_MODEL,
                attempt,
                len(system) + len(user),
            )
            data = _request_ollama("/api/chat", payload)
            content = data.get("message", {}).get("content", "")
            if not content:
                raise ValueError("Empty response from Ollama chat endpoint")
            logger.info("Local Ollama chat succeeded on attempt %s", attempt)
            return content
        except urllib.error.HTTPError as exc:
            response = exc.read().decode("utf-8", errors="ignore")
            logger.warning("Ollama HTTP %s on chat attempt %s: %s", exc.code, attempt, response[:500])
            if exc.code in (500, 502, 503, 504) and attempt < OLLAMA_RETRIES:
                time.sleep(2 ** (attempt - 1))
                continue
            if exc.code == 500 and attempt == 1:
                logger.warning("Retrying with the simpler /api/generate endpoint because /api/chat returned 500")
                try:
                    generate_payload = {
                        "model": OLLAMA_MODEL,
                        "prompt": f"{system}\n\n{user}",
                        "stream": False,
                        "temperature": 0,
                    }
                    data = _request_ollama("/api/generate", generate_payload)
                    content = data.get("response", "")
                    if content:
                        logger.info("Local Ollama generate fallback succeeded after /api/chat 500")
                        return content
                except Exception as generate_exc:  # noqa: BLE001
                    logger.warning("Ollama generate fallback failed: %s", generate_exc)
            raise
        except ModelTimeoutError:
            # A time-out already used the whole response budget. Do not retry;
            # propagate so the caller can fall back to the heuristic path.
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama chat failed on attempt %s: %s", attempt, exc)
            if attempt < OLLAMA_RETRIES:
                time.sleep(2 ** (attempt - 1))
                continue
            raise

    raise RuntimeError(f"Ollama chat failed for model {OLLAMA_MODEL} after {OLLAMA_RETRIES} attempts")


def call_model(system: str, user: str, use_json_format: bool = False) -> str:
    """Use the active backend (Gemini first, then local Ollama as a fallback)
    and record which backend actually served the call.

    Availability is read from the per-model cache, so each backend is probed
    only once; on failure we try the next backend without re-probing it.

    A Gemini 429 (quota) is treated as a failure and we switch to Ollama. If
    the Ollama response takes longer than ``OLLAMA_CALL_TIMEOUT`` (1 minute by
    default), ``ModelTimeoutError`` is raised so the caller can fall back to
    the standard (heuristic) approach.
    """
    global _used_backend

    _used_backend = "heuristic"

    if gemini_available():
        try:
            result = _gemini_request(system, user)
            _used_backend = "gemini"
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini backend failed; trying Ollama fallback. Error: %s", exc)
            # Auth, quota, and malformed-request failures will not succeed on
            # subsequent chunks, so avoid repeatedly calling Gemini. Network
            # and server failures remain eligible for a later retry.
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 401, 403, 404, 429):
                _availability["gemini"] = False

    if ollama_available():
        try:
            result = ollama_chat(system, user, use_json_format=use_json_format)
            _used_backend = "ollama"
            return result
        except ModelTimeoutError:
            logger.warning("Ollama response exceeded %ss; falling back to standard approach", OLLAMA_CALL_TIMEOUT)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama fallback also failed: %s", exc)
            # ollama_chat has already applied its configured retries. Do not
            # pay that cost for every remaining document chunk in this run.
            _availability["ollama"] = False
            raise ModelUnavailableError("Ollama failed after its retry budget") from exc

    raise ModelUnavailableError("No usable AI backend is available. Start Ollama or set GEMINI_API_KEY.")


def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Parse a model response that may be a JSON array, a single object, or
    several concatenated objects (the common failure mode of small models)."""
    results: List[Dict[str, Any]] = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [o for o in parsed if isinstance(o, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        pass

    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            j = i
            while j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[i : j + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                results.append(obj)
                        except Exception:
                            pass
                        i = j + 1
                        break
                j += 1
            else:
                i += 1
        else:
            i += 1
    return results
