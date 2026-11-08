"""Unit tests for the live harness HTTP API.

Run:  python3 -m unittest tests.test_api -v
"""
import json
import os
import urllib.request
import urllib.error
import unittest


BASE_URL = os.environ.get("LLM_HARNESS_URL", "http://127.0.0.1:8090")
TIMEOUT = 5


def _get(path):
    """Helper: GET a URL path and return parsed JSON."""
    url = BASE_URL + path
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read().decode()
            return json.loads(data)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        raise unittest.SkipTest(f"Server unreachable at {BASE_URL}")


class TestHealthz(unittest.TestCase):
    """Test /healthz endpoint."""

    def test_healthz_returns_ok(self):
        data = _get("/healthz")
        self.assertEqual(data.get("ok"), True)
        self.assertIn("phase", data)
        self.assertIsInstance(data["phase"], str)


class TestApiState(unittest.TestCase):
    """Test /api/state endpoint."""

    def test_state_returns_ok(self):
        data = _get("/api/state")
        self.assertEqual(data.get("ok"), True)
        self.assertIn("models", data)
        self.assertIsInstance(data["models"], list)
        self.assertIn("active", data)
        self.assertIn("stats", data)
        self.assertIsInstance(data["stats"], dict)


class TestApiVersion(unittest.TestCase):
    """Test /api/version endpoint."""

    def test_version_returns_ok(self):
        data = _get("/api/version")
        self.assertEqual(data.get("name"), "local-llm-harness")
        self.assertIn("version", data)
        self.assertIsInstance(data["version"], str)
        self.assertIn("features", data)
        self.assertIsInstance(data["features"], list)


class TestApiModels(unittest.TestCase):
    """Test /api/models endpoint."""

    def test_models_returns_list(self):
        data = _get("/api/models")
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        first = data[0]
        self.assertIn("key", first)
        self.assertIn("id", first)
        self.assertIn("ctx", first)
        self.assertIn("api_port", first)
        self.assertIn("external", first)


if __name__ == "__main__":
    unittest.main()
