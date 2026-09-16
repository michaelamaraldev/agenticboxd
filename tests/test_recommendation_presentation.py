import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError
from test_local_data import fixture_data
from test_orchestration import ScriptedModel, call, choice

from app import Recommender, _render_result, run_cli
from src.agents.recommendation import RECOMMENDATION_SYSTEM_MESSAGE
from src.models import CineResult, RecommendationChoices


def two_choices(primary_candidate_id=2):
    return {
        "primary_candidate_id": primary_candidate_id,
        "recommendations": [
            {
                "candidate_id": 1,
                "fit_reason": "É uma viagem contemplativa. O drama combina com seu pedido.",
                "personal_evidence_ids": ["request"],
                "movie_evidence_ids": ["movie:1:overview"],
            },
            {
                "candidate_id": 2,
                "fit_reason": (
                    "Este é o destaque porque acompanha o clima pedido. "
                    "A sinopse confirmada apresenta uma viagem. "
                    "O gênero dramático reforça essa escolha."
                ),
                "personal_evidence_ids": ["request"],
                "movie_evidence_ids": ["movie:2:overview", "movie:2:genres"],
            },
        ],
    }


class RecommendationPrimaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache_path = Path(self.directory.name) / "taste.json"
        self.data, self.cache = fixture_data()

    def recommender(self, script):
        return Recommender(
            self.data,
            ScriptedModel(script=script),
            self.cache,
            taste_cache_path=self.cache_path,
        )

    def test_prompt_prioritizes_current_context_without_rain_heuristic(self):
        prompt = str(RECOMMENDATION_SYSTEM_MESSAGE.content).casefold()
        for concept in ("current request", "occasion", "feeling", "tie-breaker"):
            self.assertIn(concept, prompt)
        self.assertIn(
            "occasion or feeling alone is not a reason to call analyze_history",
            prompt,
        )
        self.assertNotIn("rainy movie", prompt)
        self.assertNotIn("rain =", prompt)

    def test_nonempty_choices_require_a_valid_primary_candidate(self):
        valid = RecommendationChoices.model_validate(two_choices())
        self.assertEqual(valid.primary_candidate_id, 2)

        missing = two_choices()
        del missing["primary_candidate_id"]
        with self.assertRaises(ValidationError):
            RecommendationChoices.model_validate(missing)

        with self.assertRaises(ValidationError):
            RecommendationChoices.model_validate(two_choices(primary_candidate_id=3))

    def test_empty_choices_allow_no_primary_candidate(self):
        choices = RecommendationChoices.model_validate(
            {"primary_candidate_id": None, "recommendations": []}
        )
        self.assertIsNone(choices.primary_candidate_id)

    def test_primary_is_marked_first_and_all_facts_remain_tmdb_confirmed(self):
        script = [
            call("TasteProfile", {"likes": [], "dislikes": []}, "profile"),
            call("search_candidates", {"page_size": 10}, "search"),
            call("get_tmdb_details", {"candidate_ids": [1, 2]}, "details"),
            call("RecommendationChoices", two_choices(), "final"),
        ]
        result = self.recommender(script).recommend("Quero uma viagem dramática")

        self.assertEqual([item.title for item in result.recommendations], ["Filme 1", "Filme 0"])
        self.assertTrue(result.recommendations[0].is_primary)
        self.assertFalse(result.recommendations[1].is_primary)
        self.assertEqual(result.recommendations[0].fit_reason.count("."), 3)
        self.assertEqual(
            set(result.recommendations[0].evidence),
            {"request", "movie:2:overview", "movie:2:genres"},
        )
        expected = [self.cache.movies["film:1"], self.cache.movies["film:0"]]
        for recommendation, fact in zip(result.recommendations, expected):
            for field in (
                "title",
                "year",
                "overview",
                "genres",
                "directors",
                "runtime_minutes",
                "cast",
            ):
                self.assertEqual(getattr(recommendation, field), getattr(fact, field))

    def test_renderer_separates_primary_and_alternatives_without_metrics_or_flow(self):
        script = [
            call("TasteProfile", {"likes": [], "dislikes": []}, "profile"),
            call("search_candidates", {}, "search"),
            call("get_tmdb_details", {"candidate_ids": [1, 2]}, "details"),
            call("RecommendationChoices", two_choices(), "final"),
        ]
        result = self.recommender(script).recommend("Quero uma viagem dramática")
        output = []
        _render_result(result, output.append)
        rendered = "\n".join(output)

        self.assertIn("RECOMENDAÇÃO PRINCIPAL", rendered)
        self.assertIn("Por que recomendo", rendered)
        self.assertIn("EVIDÊNCIAS", rendered)
        self.assertIn("FICHA DO FILME", rendered)
        self.assertIn("OUTRAS OPÇÕES", rendered)
        self.assertLess(rendered.index("Filme 1"), rendered.index("OUTRAS OPÇÕES"))
        self.assertGreater(rendered.rindex("Filme 0"), rendered.index("OUTRAS OPÇÕES"))
        self.assertNotIn("Métricas locais", rendered)
        self.assertNotIn("Fluxo LangChain", rendered)
        self.assertNotIn("AIMessage", rendered)


class RealTimeFlowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache_path = Path(self.directory.name) / "taste.json"
        self.data, self.cache = fixture_data()

    def recommender(self, script):
        return Recommender(
            self.data,
            ScriptedModel(script=script),
            self.cache,
            taste_cache_path=self.cache_path,
        )

    def test_callback_emits_nested_agents_in_chronological_natural_language(self):
        script = [
            call(
                "search_taste_evidence",
                {"source": "ratings", "page_size": 1},
                "taste-search",
            ),
            call(
                "TasteProfile",
                {
                    "likes": [
                        {"text": "Aprecia drama", "evidence_ids": ["ratings:1"]}
                    ],
                    "dislikes": [],
                },
                "profile",
            ),
            call("analyze_history", {"question": "Verifique sessões recentes"}, "history"),
            call(
                "search_history_evidence",
                {"date_from": "2026-01-01"},
                "history-search",
            ),
            call(
                "HistoryContext",
                {
                    "relevant_patterns": [
                        {"text": "Reassistiu drama", "evidence_ids": ["diary:1"]}
                    ],
                    "rewatch_signals": [],
                },
                "history-result",
            ),
            call("search_candidates", {}, "candidate-search"),
            call("get_tmdb_details", {"candidate_ids": [1]}, "details"),
            call(
                "RecommendationChoices",
                choice(
                    personal_refs=["diary:1"],
                    primary_candidate_id=1,
                ),
                "final",
            ),
        ]
        events = []
        result = self.recommender(script).recommend(
            "Quero rever uma viagem tranquila", flow_callback=events.append
        )
        flow = "\n".join(events)

        expected_order = (
            "agente de gosto chamou a tool search_taste_evidence",
            "tool search_taste_evidence devolveu",
            "agente de gosto concluiu",
            "agente recomendador chamou a tool analyze_history",
            "agente de histórico chamou a tool search_history_evidence",
            "tool search_history_evidence devolveu",
            "agente de histórico concluiu",
            "tool analyze_history devolveu",
            "agente recomendador chamou a tool search_candidates",
            "tool search_candidates devolveu",
            "agente recomendador chamou a tool get_tmdb_details",
            "tool get_tmdb_details confirmou",
            "agente recomendador concluiu",
        )
        positions = [flow.index(fragment) for fragment in expected_order]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(tuple(events), result.flow)
        self.assertTrue(events[0].startswith("1. "))
        for forbidden in ("↳", "->", "<-", "`", '"', "AIMessage", "ToolMessage", "tool_call"):
            self.assertNotIn(forbidden, flow)
        for private_text in (
            "Verifique sessões recentes",
            "Quero rever uma viagem tranquila",
            "Aprecia drama",
            "Reassistiu drama",
            "Filme 0",
        ):
            self.assertNotIn(private_text, flow)

    def test_callback_receives_call_before_tool_return(self):
        events = []
        observations = []

        class ObservingModel(ScriptedModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                observations.append(tuple(events))
                return super()._generate(messages, stop, run_manager, **kwargs)

        model = ObservingModel(
            script=[
                call("TasteProfile", {"likes": [], "dislikes": []}, "profile"),
                call("search_candidates", {}, "search"),
                call("get_tmdb_details", {"candidate_ids": [1]}, "details"),
                call("RecommendationChoices", choice(), "final"),
            ]
        )
        recommender = Recommender(
            self.data, model, self.cache, taste_cache_path=self.cache_path
        )
        recommender.recommend("Uma viagem", flow_callback=events.append)

        before_model_after_search = "\n".join(observations[2])
        self.assertIn("chamou a tool search_candidates", before_model_after_search)
        self.assertIn("tool search_candidates devolveu", before_model_after_search)
        self.assertNotIn("get_tmdb_details confirmou", before_model_after_search)

    def test_tool_failure_emits_safe_event_and_propagates_error(self):
        script = [
            call("TasteProfile", {"likes": [], "dislikes": []}, "profile"),
            call("search_candidates", {}, "search"),
            call("get_tmdb_details", {"candidate_ids": [1]}, "details"),
            call("get_tmdb_details", {"candidate_ids": [1]}, "details-again"),
        ]
        events = []
        with self.assertRaises(ValueError):
            self.recommender(script).recommend("Uma viagem", flow_callback=events.append)

        flow = "\n".join(events)
        self.assertIn("A tool get_tmdb_details falhou.", flow)
        self.assertNotIn("only", flow)
        self.assertNotIn("candidate_ids", flow)
        self.assertNotIn("Filme 0", flow)

    def test_cli_streams_flow_once_and_keeps_metrics_programmatic(self):
        timeline = []

        class StreamingRecommender:
            def recommend(self, request, flow_callback=None):
                self.assert_callback(flow_callback)
                flow_callback("1. O agente recomendador chamou a tool search_candidates.")
                timeline.append("recommendation-finished")
                return CineResult(
                    flow=("1. O agente recomendador chamou a tool search_candidates.",),
                    metrics={"model_calls": 3, "taste_cache_hit": True},
                )

            def assert_callback(self, callback):
                if callback is None:
                    raise AssertionError("run_cli must provide a real-time flow callback")

        run_cli(
            StreamingRecommender(),
            input_fn=lambda _: "Uma viagem",
            output_fn=timeline.append,
        )
        rendered = "\n".join(timeline)

        self.assertEqual(rendered.count("FLUXO DOS AGENTES"), 1)
        self.assertEqual(rendered.count("chamou a tool search_candidates"), 1)
        self.assertLess(
            timeline.index("1. O agente recomendador chamou a tool search_candidates."),
            timeline.index("recommendation-finished"),
        )
        self.assertNotIn("Métricas locais", rendered)


if __name__ == "__main__":
    unittest.main()
