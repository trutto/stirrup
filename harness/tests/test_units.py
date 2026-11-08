"""Unit tests for pure helper functions in scripts/harness.py.

Run:  python3 -m unittest tests.test_units -v
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

import harness  # noqa: E402


class EstimateTokensTest(unittest.TestCase):
    def test_non_empty_returns_at_least_one(self):
        self.assertGreaterEqual(harness.estimate_tokens([{"role": "user", "content": "hi"}]), 1)

    def test_longer_string_gives_more_tokens(self):
        short = harness.estimate_tokens([{"role": "user", "content": "hi"}])
        long_ = harness.estimate_tokens([{"role": "user", "content": "a" * 100}])
        self.assertGreater(long_, short)

    def test_list_of_parts_counts_json(self):
        parts = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        msg = {"role": "user", "content": parts}
        self.assertGreater(harness.estimate_tokens([msg]), 0)


class StripThinkTest(unittest.TestCase):
    def test_strips_think_tags(self):
        self.assertEqual(harness.strip_think("<think>secret</think>Hello"), "Hello")

    def test_no_tags_unchanged(self):
        self.assertEqual(harness.strip_think("just text"), "just text")

    def test_leading_trailing_whitespace_stripped(self):
        result = harness.strip_think("  \n<think>secret</think>  Hello  \n  ")
        self.assertEqual(result, "Hello")


class YouTubeParseTest(unittest.TestCase):
    def test_parses_and_sorts(self):
        page = ('x"videoRenderer":{"videoId":"aaa","title":{"runs":[{"text":"Old video"}]},'
                '"ownerText":{"runs":[{"text":"Chan"}]},"publishedTimeText":{"simpleText":"1 month ago"}}'
                'x"videoRenderer":{"videoId":"bbb","title":{"runs":[{"text":"New video"}]},'
                '"ownerText":{"runs":[{"text":"Chan2"}]},"publishedTimeText":{"simpleText":"2 days ago"}}')
        items = harness.parse_youtube_results(page, 5)
        self.assertEqual([it["id"] for it in items], ["bbb", "aaa"])


class FailureCountTest(unittest.TestCase):
    def test_unittest_and_pytest(self):
        self.assertEqual(harness.test_failure_count("FAILED (failures=1, errors=3)"), 4)
        self.assertEqual(harness.test_failure_count("FAILED (failures=2)"), 2)
        self.assertEqual(harness.test_failure_count("OK"), 0)
        self.assertEqual(harness.test_failure_count("2 failed, 5 passed"), 2)
        self.assertIsNone(harness.test_failure_count("hello"))


class ToolArgsParseTest(unittest.TestCase):
    def test_tolerates_literal_newlines(self):
        a = harness.parse_tool_args('{"command": "echo one\ntwo"}')
        self.assertEqual(a, {"command": "echo one\ntwo"})

    def test_returns_none_for_garbage(self):
        self.assertIsNone(harness.parse_tool_args("not json at all"))


class MachineFactsTest(unittest.TestCase):
    def test_omarchy_is_real_in_prompts(self):
        self.assertIn("Omarchy", harness.MACHINE_FACTS)
        self.assertIn("not a parody", harness.MACHINE_FACTS)
        self.assertIn("Hyprland", harness.MACHINE_FACTS)
        self.assertIn("Omarchy", harness.CHAT_SYSTEM)


class PlanLikeTest(unittest.TestCase):
    def test_plan_detected(self):
        self.assertTrue(harness.plan_like("Let me check the logs."))

    def test_plan_detected_future_intent(self):
        self.assertTrue(harness.plan_like("I will look that up."))

    def test_result_not_plan(self):
        self.assertFalse(harness.plan_like("The file has three lines."))

    def test_long_text_not_plan(self):
        self.assertFalse(harness.plan_like(" ".join(["word"] * 65)))


class AgentTrimTest(unittest.TestCase):
    def test_system_kept_and_oldest_dropped(self):
        convo = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "old1"},
            {"role": "assistant", "content": "old2"},
            {"role": "user", "content": "mid"},
            {"role": "assistant", "content": "mid2"},
            {"role": "user", "content": "new"},
            {"role": "assistant", "content": "new2"},
        ]
        result = harness.agent_trim(convo, budget=50)
        # system message must be present
        self.assertEqual(result[0]["role"], "system")
        # oldest messages should be dropped (rest is shorter)
        self.assertLess(len(result), len(convo))

    def test_no_orphan_tool_at_start(self):
        convo = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "tool", "content": "tool1"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": "tool2"},
        ]
        result = harness.agent_trim(convo, budget=50)
        # if a tool message is at the start of the rest, its assistant predecessor must be present
        if result and len(result) > 1 and result[1].get("role") == "tool":
            self.assertEqual(result[2].get("role"), "assistant")


class ReadTailTest(unittest.TestCase):
    def test_returns_last_n_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            for i in range(5):
                f.write(f"line{i}\n")
            path = f.name
        result = harness._read_tail(path, lines=2)
        self.assertEqual(result, ["line3", "line4"])


if __name__ == "__main__":
    unittest.main()
