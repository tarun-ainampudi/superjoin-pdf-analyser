import unittest

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

    def test_gemini_429_switches_to_ollama(self):
        import urllib.error

        llm._availability["gemini"] = None
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


if __name__ == "__main__":
    unittest.main()
