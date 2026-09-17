import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from test_local_data import fixture_data
from test_orchestration import ScriptedModel, base_script, call, choice

from app import Recommender
from src.local_data import (
    LocalCatalog,
    create_search_candidates_tool,
    request_constraints,
)


class EvidenceFlowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "taste.json"
        self.data, self.cache = fixture_data()

    def recommend(self, script, request="Uma viagem tranquila"):
        model = ScriptedModel(script=script)
        result = Recommender(
            self.data, model, self.cache, taste_cache_path=self.path
        ).recommend(request)
        return model, result

    def profile_script(self):
        return [
            call("search_taste_evidence", {"source": "ratings", "page_size": 1}, "rating"),
            call("TasteProfile", {
                "likes": [{"text": "Gostou de Memória", "evidence_ids": ["ratings:1"]}],
                "dislikes": [],
            }, "profile"),
        ]

    def test_nonempty_profile_evidence_survives_new_instance(self):
        final = choice(personal_refs=["ratings:1"])
        cold = self.profile_script() + base_script(cold=False, final=final)
        _, first = self.recommend(cold)
        model, second = self.recommend(base_script(cold=False, final=final))
        self.assertTrue(second.metrics["taste_cache_hit"])
        self.assertEqual(model.cursor, 3)
        self.assertEqual(first.recommendations[0].evidence, second.recommendations[0].evidence)
        self.assertIn("ratings:1", second.recommendations[0].evidence)

    def test_history_agent_can_investigate_and_supply_personal_evidence(self):
        self.recommend(base_script())
        script = [
            call("analyze_history", {"question": "Verifique as sessões recentes."}, "history"),
            call("search_history_evidence", {"date_from": "2026-01-01"}, "diary"),
            call("HistoryContext", {
                "relevant_patterns": [{"text": "Reviu Memória", "evidence_ids": ["diary:1"]}],
                "rewatch_signals": [],
            }, "history-result"),
            *base_script(cold=False, final=choice(personal_refs=["diary:1"])),
        ]
        model, result = self.recommend(
            script, "Quero algo diferente do que assisti recentemente"
        )
        self.assertEqual(model.cursor, 6)
        self.assertIn("diary:1", result.recommendations[0].evidence)
        self.assertEqual(result.metrics["tool_calls"], 6)

    def test_history_cannot_cite_evidence_seen_only_by_taste(self):
        script = [
            *self.profile_script(),
            call("analyze_history", {}, "history"),
            call("HistoryContext", {
                "relevant_patterns": [{"text": "Suposição", "evidence_ids": ["ratings:1"]}],
                "rewatch_signals": [],
            }, "history-result"),
            *base_script(cold=False),
        ]
        with self.assertRaises(ValueError):
            self.recommend(script)

    def test_final_cannot_cite_personal_record_omitted_from_specialist_result(self):
        script = [
            call("search_taste_evidence", {"source": "ratings"}, "ratings"),
            call("TasteProfile", {
                "likes": [{"text": "Gostou de Memória", "evidence_ids": ["ratings:1"]}],
                "dislikes": [],
            }, "profile"),
            *base_script(cold=False, final=choice(personal_refs=["ratings:2"])),
        ]
        with self.assertRaises(ValueError):
            self.recommend(script)

    def test_no_verified_candidates_needs_no_inference(self):
        model = ScriptedModel(script=[])
        data = replace(self.data, watchlist=())
        result = Recommender(data, model, self.cache, taste_cache_path=self.path).recommend("Uma viagem")
        self.assertEqual(result.recommendations, ())
        self.assertEqual(model.cursor, 0)

    def test_empty_search_can_finish_without_confirming_invented_candidates(self):
        _, result = self.recommend([
            call("TasteProfile", {"likes": [], "dislikes": []}, "profile"),
            call("search_candidates", {"query": "inexistente"}, "search"),
            call("RecommendationChoices", {"recommendations": []}, "final"),
        ])
        self.assertEqual(result.recommendations, ())

    def test_missing_runtime_never_satisfies_numeric_filter(self):
        facts = dict(self.cache.movies)
        facts["film:0"] = facts["film:0"].model_copy(update={"runtime_minutes": None})
        catalog = LocalCatalog(self.data, self.cache.model_copy(update={"movies": facts}))
        for filters in ({"runtime_min": 1}, {"runtime_max": 999}):
            with self.subTest(filters=filters):
                rows = catalog.search(page_size=20, **filters)["records"]
                self.assertNotIn("film:0", [row["watchlist_id"] for row in rows])

    def test_explicit_request_constraint_is_enforced_by_every_search(self):
        constraints = request_constraints("Quero algo de até 90 minutos")
        self.assertEqual(constraints.runtime_max, 90)
        tool = create_search_candidates_tool(
            LocalCatalog(self.data, self.cache), constraints
        )
        result = json.loads(tool.invoke({"runtime_max": 200, "page_size": 20}))
        self.assertTrue(result["records"])
        self.assertTrue(
            all(record["runtime_minutes"] <= 90 for record in result["records"])
        )

    def test_portuguese_hour_constraints_preserve_inclusive_and_strict_meaning(self):
        cases = (
            ("menos de duas horas", None, 119),
            ("até duas horas", None, 120),
            ("mais de uma hora", 61, None),
        )
        for request, runtime_min, runtime_max in cases:
            with self.subTest(request=request):
                constraints = request_constraints(request)
                self.assertEqual(constraints.runtime_min, runtime_min)
                self.assertEqual(constraints.runtime_max, runtime_max)

    def test_numeric_minute_constraints_preserve_inclusive_and_strict_meaning(self):
        cases = (
            ("até 90 minutos", None, 90),
            ("menos de 90 min", None, 89),
            ("pelo menos 80 minutos", 80, None),
            ("mais de 80 min", 81, None),
        )
        for request, runtime_min, runtime_max in cases:
            with self.subTest(request=request):
                constraints = request_constraints(request)
                self.assertEqual(constraints.runtime_min, runtime_min)
                self.assertEqual(constraints.runtime_max, runtime_max)

    def test_hour_constraint_cannot_be_relaxed_by_search_arguments(self):
        constraints = request_constraints("Quero um filme com menos de duas horas")
        facts = dict(self.cache.movies)
        facts["film:0"] = facts["film:0"].model_copy(
            update={"runtime_minutes": 120}
        )
        cache = self.cache.model_copy(update={"movies": facts})
        tool = create_search_candidates_tool(
            LocalCatalog(self.data, cache), constraints
        )
        result = json.loads(tool.invoke({"runtime_max": 300, "page_size": 20}))
        self.assertTrue(result["records"])
        self.assertNotIn(1, [record["candidate_id"] for record in result["records"]])
        self.assertTrue(
            all(record["runtime_minutes"] <= 119 for record in result["records"])
        )
