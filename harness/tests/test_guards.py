"""Unit tests for the agent guards (no model needed).

Run:  python3 -m unittest tests.test_guards -v
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

import harness  # noqa: E402


class BashGuards(unittest.TestCase):
    def test_self_restart_is_blocked(self):
        out = harness.run_tool("bash", {"command": "systemctl --user restart local-llm-harness"})
        self.assertIn("blocked", out.lower())
        self.assertIn("die immediately", out.lower())

    def test_self_stop_is_blocked(self):
        out = harness.run_tool("bash", {"command": "systemctl --user stop local-llm-harness"})
        self.assertIn("blocked", out.lower())

    def test_kill_targets_pid_helper(self):
        self.assertTrue(harness.kill_targets_pid("kill -HUP 12345", 12345))
        self.assertTrue(harness.kill_targets_pid("kill -9 12345", 12345))
        self.assertFalse(harness.kill_targets_pid("kill -HUP 999", 12345))
        self.assertFalse(harness.kill_targets_pid("echo hello 12345", 12345))

    def test_destructive_is_blocked(self):
        out = harness.run_tool("bash", {"command": "rm -rf /"})
        self.assertIn("blocked", out.lower())

    def test_normal_command_runs(self):
        out = harness.run_tool("bash", {"command": "echo guard-ok"})
        self.assertIn("guard-ok", out)


class GitHubToolGuard(unittest.TestCase):
    def test_rejects_bad_repo_and_empty_query(self):
        self.assertIn("error", harness.run_tool("github_tree", {"repo": "../evil"}).lower())
        self.assertIn("error", harness.run_tool("github_search", {"query": ""}).lower())


class RestoreFileTest(unittest.TestCase):
    def test_restore_previous_version(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.txt")
            harness.run_tool("write_file", {"path": p, "content": "one"})
            harness.run_tool("write_file", {"path": p, "content": "two"})
            out = harness.run_tool("restore_file", {"path": p})
            self.assertIn("restored", out)
            self.assertEqual(open(p).read(), "one")


class WebFetchGuard(unittest.TestCase):
    def test_blocks_local_and_private(self):
        self.assertIn("blocked", harness.run_tool("web_fetch", {"url": "http://127.0.0.1:8090/healthz"}).lower())
        self.assertIn("blocked", harness.run_tool("web_fetch", {"url": "http://169.254.169.254/"}).lower())

    def test_rejects_bad_scheme(self):
        self.assertIn("error", harness.run_tool("web_fetch", {"url": "file:///etc/passwd"}).lower())


class WriteFileGuard(unittest.TestCase):
    def test_identical_write_is_a_noop(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.txt")
            self.assertIn("wrote", harness.run_tool("write_file", {"path": p, "content": "hello"}))
            again = harness.run_tool("write_file", {"path": p, "content": "hello"})
            self.assertIn("no change needed", again)
            self.assertEqual(open(p).read(), "hello")

    def test_append_file_works(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.txt")
            harness.run_tool("write_file", {"path": p, "content": "a"})
            harness.run_tool("append_file", {"path": p, "content": "b"})
            self.assertEqual(open(p).read(), "ab")

    def test_changed_write_lands(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.txt")
            harness.run_tool("write_file", {"path": p, "content": "a"})
            harness.run_tool("write_file", {"path": p, "content": "b"})
            self.assertEqual(open(p).read(), "b")


class Watchdog(unittest.TestCase):
    def test_reloads_when_swapped_idle_and_not_busy(self):
        self.assertTrue(harness.watchdog_should_reload(2_000_000, 300, 0))

    def test_never_reloads_while_request_in_flight(self):
        self.assertFalse(harness.watchdog_should_reload(9_000_000, 999, 1))

    def test_severe_swap_needs_only_five_seconds_idle(self):
        self.assertTrue(harness.watchdog_should_reload(5_000_000, 6, 0))
        self.assertFalse(harness.watchdog_should_reload(5_000_000, 2, 0))
        self.assertFalse(harness.watchdog_should_reload(5_000_000, 6, 1))

    def test_never_reloads_below_threshold_or_recent_use(self):
        self.assertFalse(harness.watchdog_should_reload(500_000, 999, 0))
        self.assertFalse(harness.watchdog_should_reload(2_000_000, 10, 0))


class FinalizeFailureNote(unittest.TestCase):
    def test_mentions_command_and_output(self):
        msg = harness.finalize_failure_note("python3 -m unittest discover", "FAILED (failures=2)")
        self.assertIn("python3 -m unittest discover", msg)
        self.assertIn("FAILED (failures=2)", msg)
        self.assertIn("Do NOT claim the tests pass", msg)


class VerifyNudge(unittest.TestCase):
    def test_includes_failing_output_tail(self):
        msg = harness.verify_nudge_text("python3 -m unittest tests.test_x", "line1\nAssertionError: 1 != 2")
        self.assertIn("tests.test_x", msg)
        self.assertIn("AssertionError: 1 != 2", msg)
        self.assertIn("Fix the code or the test", msg)

    def test_missing_output_is_handled(self):
        self.assertIn("(no output)", harness.verify_nudge_text("py_compile x.py", ""))


class PlanDetection(unittest.TestCase):
    def test_plans_are_detected(self):
        self.assertTrue(harness.plan_like("Let me check the network first."))
        self.assertTrue(harness.plan_like("I'll look up the files now."))

    def test_results_are_not_plans(self):
        self.assertFalse(harness.plan_like("The file contains three lines."))
        self.assertFalse(harness.plan_like("Your phone is at 192.168.4.50."))


if __name__ == "__main__":
    unittest.main()
