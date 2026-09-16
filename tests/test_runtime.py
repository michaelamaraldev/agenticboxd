import json
import unittest

from src.agents.recommendation import RequestBudget, RuntimeBudgetExceeded


class BudgetTests(unittest.TestCase):
    def test_scope_counts_model_and_tools_independently(self):
        budget = RequestBudget()
        scope = budget.scope("recommendation", model_limit=8, tool_limit=8)
        for _ in range(8):
            scope.consume_model()
            scope.consume_tool()
        with self.assertRaises(RuntimeBudgetExceeded):
            scope.consume_model()
        with self.assertRaises(RuntimeBudgetExceeded):
            scope.consume_tool()

    def test_each_specialist_invocation_has_its_own_limit(self):
        budget = RequestBudget()
        for _ in range(2):
            specialist = budget.scope("taste", model_limit=4, tool_limit=4)
            for _ in range(4):
                specialist.consume_model()
                specialist.consume_tool()
            with self.assertRaises(RuntimeBudgetExceeded):
                specialist.consume_model()

    def test_nested_and_initial_profile_calls_share_global_budget(self):
        budget = RequestBudget()
        for name, count in (("taste", 4), ("recommendation", 8), ("history", 4)):
            scope = budget.scope(name, model_limit=count, tool_limit=count)
            for _ in range(count):
                scope.consume_model()
                scope.consume_tool()
        extra = budget.scope("taste", model_limit=4, tool_limit=4)
        with self.assertRaises(RuntimeBudgetExceeded):
            extra.consume_model()
        with self.assertRaises(RuntimeBudgetExceeded):
            extra.consume_tool()

    def test_metrics_contain_counts_without_messages(self):
        budget = RequestBudget()
        scope = budget.scope("taste", model_limit=4, tool_limit=4)
        scope.consume_model()
        scope.consume_tool()
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["model_calls"], 1)
        self.assertEqual(snapshot["tool_calls"], 1)
        serialized = json.dumps(snapshot)
        self.assertNotIn("messages", serialized)
        self.assertNotIn("prompt", serialized)
