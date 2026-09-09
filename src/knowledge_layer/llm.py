"""Local Ollama client and robust JSON parsing.

The small local model (granite4.1:3b) frequently returns concatenated JSON
objects instead of a proper array, so we repair the response defensively.
"""

import json
import urllib.request
from typing import Any, Dict, List

from .config import OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL


def ollama_available() -> bool:
    """Check quickly that a local Ollama instance is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


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
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as r:
        data = json.load(r)
    return data["message"]["content"]


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