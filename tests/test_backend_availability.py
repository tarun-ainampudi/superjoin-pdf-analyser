import unittest
import urllib.error
from unittest.mock import patch

from src.knowledge_layer import llm


class BackendAvailabilityTests(unittest.TestCase):
    def setUp(self):
        llm.reset_availability()

    def test_availability_is_probed_only_once(self):
        calls = {"ollama": 0}

        orig = llm._ollama_reachable

        def counting_probe():
            calls["ollama"] += 1
            return orig()

        llm._ollama_reachable = counting_probe
        try:
            first = llm.ollama_available()
            second = llm.ollama_available()
            third = llm.ollama_available()
        finally:
            llm._ollama_reachable = orig

        self.assertEqual(first, second)
        self.assertEqual(second, third)
        self.assertEqual(calls["ollama"], 1, "Ollama should be probed only once")

    def test_reset_availability_forces_reprobe(self):
        calls = {"ollama": 0}

        def counting_probe():
            calls["ollama"] += 1
            return True

        orig = llm._ollama_reachable
        llm._ollama_reachable = counting_probe
        try:
            llm.ollama_available()
            self.assertEqual(calls["ollama"], 1)
            # Without reset the cached result is reused (no re-probe).
            llm.ollama_available()
            self.assertEqual(calls["ollama"], 1)
            # After reset the next call re-probes.
            llm.reset_availability()
            llm.ollama_available()
            self.assertEqual(calls["ollama"], 2)
        finally:
            llm._ollama_reachable = orig
            llm.reset_availability()

    def test_gemini_probe_uses_model_response_status(self):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        original_key = llm.GEMINI_API_KEY
        llm.GEMINI_API_KEY = "test-key"
        try:
            with patch("src.knowledge_layer.llm.urllib.request.urlopen", return_value=Response()) as urlopen:
                self.assertTrue(llm._gemini_reachable())
                self.assertEqual(urlopen.call_args.args[0].get_method(), "POST")

            with patch(
                "src.knowledge_layer.llm.urllib.request.urlopen",
                side_effect=urllib.error.HTTPError("https://gemini.test", 429, "quota", {}, None),
            ):
                self.assertFalse(llm._gemini_reachable())
        finally:
            llm.GEMINI_API_KEY = original_key

    def test_ollama_probe_uses_model_response_status(self):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with patch("src.knowledge_layer.llm.urllib.request.urlopen", return_value=Response()) as urlopen:
            self.assertTrue(llm._ollama_reachable())
            request = urlopen.call_args.args[0]
            self.assertEqual(request.get_method(), "POST")
            self.assertTrue(request.full_url.endswith("/api/chat"))

    def test_gemini_429_switches_to_ollama(self):
        import urllib.error

        llm._availability["gemini"] = True
        llm._availability["ollama"] = None
        llm._used_backend = ""

        orig_gemini = llm._gemini_request
        orig_chat = llm.ollama_chat
        orig_probe = llm._ollama_reachable

        def raise_429(system, user):
            raise urllib.error.HTTPError(
                "http://gemini.test/models/x:generateContent", 429, "Too Many Requests", {}, None
            )

        llm._gemini_request = raise_429
        llm._ollama_reachable = lambda: True
        llm.ollama_chat = lambda system, user, use_json_format=False: "pong"
        try:
            result = llm.call_model("system", "user")
        finally:
            llm._gemini_request = orig_gemini
            llm.ollama_chat = orig_chat
            llm._ollama_reachable = orig_probe
            llm.reset_availability()

        self.assertEqual(result, "pong")
        self.assertEqual(llm.get_used_backend(), "ollama")
        self.assertFalse(llm._availability["gemini"])

    def test_ollama_timeout_raises_model_timeout_error(self):
        llm._availability["gemini"] = False
        llm._availability["ollama"] = None
        llm._used_backend = ""

        orig_request = llm._request_ollama
        orig_probe = llm._ollama_reachable
        llm._ollama_reachable = lambda: True

        orig_request = llm._request_ollama

        def timeout_request(endpoint, payload):
            # Same shape of error the real _request_ollama raises on a socket timeout.
            raise llm.ModelTimeoutError(
                f"Ollama response exceeded {llm.OLLAMA_CALL_TIMEOUT}s timeout"
            )

        llm._request_ollama = timeout_request
        try:
            with self.assertRaises(llm.ModelTimeoutError):
                llm.call_model("system", "user")
        finally:
            llm._request_ollama = orig_request
            llm._ollama_reachable = orig_probe
            llm.reset_availability()

    def test_failed_ollama_is_disabled_for_remaining_chunks(self):
        llm._availability["gemini"] = False
        llm._availability["ollama"] = None
        orig_probe = llm._ollama_reachable
        orig_chat = llm.ollama_chat
        llm._ollama_reachable = lambda: True
        llm.ollama_chat = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline"))
        try:
            with self.assertRaises(llm.ModelUnavailableError):
                llm.call_model("system", "user")
            self.assertFalse(llm._availability["ollama"])
        finally:
            llm._ollama_reachable = orig_probe
            llm.ollama_chat = orig_chat
            llm.reset_availability()


if __name__ == "__main__":
    unittest.main()
