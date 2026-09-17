import tempfile
import unittest
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from test_local_data import fixture_data

from app import Recommender, _needs_history, _render_result
from src.agents.recommendation import RuntimeBudgetExceeded


class ScriptedModel(BaseChatModel):
    script: list[AIMessage]
    cursor: int = 0
    seen: list[Any] = Field(default_factory=list, exclude=True)
    bound_tool_names: list[tuple[str, ...]] = Field(default_factory=list, exclude=True)

    @property
    def _llm_type(self):
        return "scripted-local-model"

    @property
    def _identifying_params(self):
        return {"model": "scripted-local-model", "temperature": 0}

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names.append(tuple(tool.name for tool in tools))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        if self.cursor >= len(self.script):
            raise AssertionError("Unexpected model call after scripted responses")
        response = self.script[self.cursor]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def call(name, args, identifier):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": str(identifier), "type": "tool_call"}],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


def choice(candidate_id=1, personal_refs=None, movie_refs=None, primary_candidate_id=None):
    return {
        "primary_candidate_id": (
            candidate_id if primary_candidate_id is None else primary_candidate_id
        ),
        "recommendations": [{
            "candidate_id": candidate_id,
            "fit_reason": "Pode combinar com o clima que você pediu.",
            "personal_evidence_ids": ["request"] if personal_refs is None else personal_refs,
            "movie_evidence_ids": [f"movie:{candidate_id}:overview"] if movie_refs is None else movie_refs,
        }]
    }


def base_script(cold=True, final=None):
    messages = [call("TasteProfile", {"likes": [], "dislikes": []}, "profile")] if cold else []
    messages.extend([
        call("search_candidates", {}, "search"),
        call("get_tmdb_details", {"candidate_ids": [1]}, "details"),
        call("RecommendationChoices", choice() if final is None else final, "final"),
    ])
    return messages


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache_path = Path(self.directory.name) / "taste.json"
        self.data, self.cache = fixture_data()

    def recommender(self, model):
        return Recommender(self.data, model, self.cache, taste_cache_path=self.cache_path)

    def test_natural_history_signals_do_not_match_unrelated_requests(self):
        for request in (
            "Quero rever algum filme",
            "Quero algo diferente do que já assisti",
        ):
            with self.subTest(request=request):
                self.assertTrue(_needs_history(request))

        for request in (
            "Quero ver uma viagem tranquila",
            "Quero um filme recente",
        ):
            with self.subTest(request=request):
                self.assertFalse(_needs_history(request))

    def test_real_agent_graph_can_skip_specialists_and_preserve_verified_facts(self):
        model = ScriptedModel(script=base_script())
        result = self.recommender(model).recommend("Uma viagem tranquila")
        self.assertEqual(model.cursor, 4)
        self.assertNotIn("analyze_history", " ".join(result.flow))
        self.assertNotIn("analyze_taste", " ".join(result.flow))
        recommendation = result.recommendations[0]
        fact = self.cache.movies["film:0"]
        for field in ("title", "year", "overview", "genres", "directors", "runtime_minutes", "cast"):
            self.assertEqual(getattr(recommendation, field), getattr(fact, field))
        self.assertEqual(result.metrics["model_calls"], 4)
        self.assertEqual(result.metrics["tool_calls"], 4)
        rendered = []
        _render_result(result, rendered.append)
        self.assertIn("Uma viagem", "\n".join(rendered))
        self.assertIn("Uma viagem tranquila", "\n".join(rendered))

    def test_warm_new_instance_does_not_recompute_profile(self):
        self.recommender(ScriptedModel(script=base_script())).recommend("Uma viagem tranquila")
        warm = ScriptedModel(script=base_script(cold=False))
        result = self.recommender(warm).recommend("Uma viagem tranquila")
        self.assertEqual(warm.cursor, 3)
        self.assertEqual(result.metrics["model_calls"], 3)
        self.assertEqual(result.metrics["tool_calls"], 3)
        self.assertTrue(result.metrics["taste_cache_hit"])
        recommendation_tools = next(
            names for names in warm.bound_tool_names if "search_candidates" in names
        )
        self.assertNotIn("analyze_taste", recommendation_tools)
        self.assertNotIn("analyze_history", recommendation_tools)

    def test_warm_common_request_cannot_exceed_three_model_and_tool_calls(self):
        self.recommender(ScriptedModel(script=base_script())).recommend("Uma viagem tranquila")
        script = [
            call("search_candidates", {}, "search-1"),
            call("search_candidates", {"query": "viagem"}, "search-2"),
            call("get_tmdb_details", {"candidate_ids": [1]}, "details"),
            call("RecommendationChoices", choice(), "final"),
        ]
        with self.assertRaises(RuntimeBudgetExceeded):
            self.recommender(ScriptedModel(script=script)).recommend(
                "Uma viagem tranquila"
            )

    def test_cold_taste_specialist_cannot_exceed_two_model_and_tool_calls(self):
        script = [
            call("search_taste_evidence", {"source": "ratings"}, "taste-1"),
            call("search_taste_evidence", {"source": "reviews"}, "taste-2"),
            call("TasteProfile", {"likes": [], "dislikes": []}, "profile"),
        ]
        with self.assertRaises(RuntimeBudgetExceeded):
            self.recommender(ScriptedModel(script=script)).recommend(
                "Uma viagem tranquila"
            )

    def test_temporal_request_routes_only_to_history_with_six_total_calls(self):
        self.recommender(ScriptedModel(script=base_script())).recommend(
            "Uma viagem tranquila"
        )
        model = ScriptedModel(script=[
            call(
                "analyze_history",
                {"question": "Verifique o que assisti recentemente"},
                "history",
            ),
            call(
                "search_history_evidence",
                {"date_from": "2026-01-01"},
                "history-search",
            ),
            call(
                "HistoryContext",
                {
                    "relevant_patterns": [
                        {
                            "text": "Reviu Mem\u00f3ria recentemente",
                            "evidence_ids": ["diary:1"],
                        }
                    ],
                    "rewatch_signals": [],
                },
                "history-result",
            ),
            call("search_candidates", {}, "search"),
            call("get_tmdb_details", {"candidate_ids": [1]}, "details"),
            call(
                "RecommendationChoices",
                choice(personal_refs=["diary:1"]),
                "final",
            ),
        ])

        result = self.recommender(model).recommend(
            "Quero algo diferente do que assisti recentemente"
        )

        self.assertLessEqual(result.metrics["model_calls"], 6)
        self.assertLessEqual(result.metrics["tool_calls"], 6)
        recommendation_tools = next(
            names for names in model.bound_tool_names if "search_candidates" in names
        )
        self.assertIn("analyze_history", recommendation_tools)
        self.assertNotIn("analyze_taste", recommendation_tools)

    def test_history_specialist_rejects_a_second_internal_search(self):
        self.recommender(ScriptedModel(script=base_script())).recommend(
            "Uma viagem tranquila"
        )
        script = [
            call("analyze_history", {}, "history"),
            call("search_history_evidence", {}, "history-search-1"),
            call(
                "search_history_evidence",
                {"date_from": "2026-01-01"},
                "history-search-2",
            ),
            call(
                "HistoryContext",
                {"relevant_patterns": [], "rewatch_signals": []},
                "history-result",
            ),
        ]

        with self.assertRaises(RuntimeBudgetExceeded):
            self.recommender(ScriptedModel(script=script)).recommend(
                "Quero algo diferente do que assisti recentemente"
            )

    def test_invalid_primary_is_discarded_and_confirmed_choice_is_promoted(self):
        self.recommender(ScriptedModel(script=base_script())).recommend(
            "Uma viagem tranquila"
        )
        final = {
            "primary_candidate_id": 2,
            "recommendations": [
                choice(candidate_id=1)["recommendations"][0],
                {
                    "candidate_id": 2,
                    "fit_reason": "ESCOLHA INVÁLIDA NÃO PODE APARECER",
                    "personal_evidence_ids": ["request"],
                    "movie_evidence_ids": ["movie:2:overview"],
                },
            ],
        }
        model = ScriptedModel(script=base_script(cold=False, final=final))

        result = self.recommender(model).recommend("Uma viagem tranquila")

        self.assertEqual(len(result.recommendations), 1)
        recommendation = result.recommendations[0]
        self.assertEqual(recommendation.title, self.cache.movies["film:0"].title)
        self.assertTrue(recommendation.is_primary)
        self.assertNotIn("ESCOLHA INVÁLIDA", recommendation.fit_reason)
        self.assertGreaterEqual(len(result.warnings), 2)
        self.assertEqual(result.metrics["model_calls"], 3)
        self.assertEqual(result.metrics["tool_calls"], 3)

    def test_all_invalid_choices_fall_back_to_first_confirmed_movie(self):
        self.recommender(ScriptedModel(script=base_script())).recommend(
            "Uma viagem tranquila"
        )
        invalid_reason = "TEXTO INVÁLIDO NÃO DEVE SER REUTILIZADO"
        final = choice(candidate_id=2)
        final["recommendations"][0]["fit_reason"] = invalid_reason
        model = ScriptedModel(script=base_script(cold=False, final=final))

        result = self.recommender(model).recommend("Uma viagem tranquila")

        self.assertEqual(len(result.recommendations), 1)
        recommendation = result.recommendations[0]
        fact = self.cache.movies["film:0"]
        self.assertTrue(recommendation.is_primary)
        self.assertNotEqual(recommendation.fit_reason, invalid_reason)
        for field in (
            "watchlist_id",
            "title",
            "year",
            "overview",
            "genres",
            "directors",
            "runtime_minutes",
            "cast",
        ):
            self.assertEqual(getattr(recommendation, field), getattr(fact, field))
        self.assertIn("request", recommendation.evidence)
        movie_evidence = [
            evidence_id
            for evidence_id in recommendation.evidence
            if evidence_id.startswith("movie:1:")
        ]
        self.assertEqual(len(movie_evidence), 1)
        self.assertIn(
            movie_evidence[0].rsplit(":", 1)[-1],
            {"title", "year", "overview", "genres", "directors", "runtime_minutes"},
        )
        self.assertGreaterEqual(len(result.warnings), 2)
        self.assertEqual(result.metrics["model_calls"], 3)
        self.assertEqual(result.metrics["tool_calls"], 3)

    def test_missing_primary_promotes_first_confirmed_choice_without_retry(self):
        self.recommender(ScriptedModel(script=base_script())).recommend(
            "Uma viagem tranquila"
        )
        final = choice(candidate_id=1)
        del final["primary_candidate_id"]
        model = ScriptedModel(script=base_script(cold=False, final=final))

        result = self.recommender(model).recommend("Uma viagem tranquila")

        self.assertEqual(model.cursor, 3)
        self.assertEqual(result.metrics["model_calls"], 3)
        self.assertEqual(result.metrics["tool_calls"], 3)
        self.assertEqual(len(result.recommendations), 1)
        recommendation = result.recommendations[0]
        self.assertTrue(recommendation.is_primary)
        fact = self.cache.movies["film:0"]
        for field in (
            "watchlist_id",
            "title",
            "year",
            "overview",
            "genres",
            "directors",
            "runtime_minutes",
            "cast",
        ):
            self.assertEqual(getattr(recommendation, field), getattr(fact, field))

    def test_final_validation_rejects_movie_above_strict_hour_limit(self):
        facts = dict(self.cache.movies)
        facts["film:0"] = facts["film:0"].model_copy(
            update={"runtime_minutes": 120}
        )
        cache = self.cache.model_copy(update={"movies": facts})
        model = ScriptedModel(script=base_script())
        recommender = Recommender(
            self.data,
            model,
            cache,
            taste_cache_path=self.cache_path,
        )
        with self.assertRaisesRegex(ValueError, "maximum runtime"):
            recommender.recommend("Quero um filme com menos de duas horas")

    def test_foreign_movie_field_or_unsupplied_personal_evidence_is_rejected(self):
        for final in (
            choice(movie_refs=["movie:2:overview"]),
            choice(movie_refs=["movie:1:invented"]),
            choice(personal_refs=["ratings:99999"]),
            choice(personal_refs=["ratings:1"]),
        ):
            with self.subTest(final=final):
                model = ScriptedModel(script=base_script(final=final))
                path = Path(self.directory.name) / f"unused-{id(model)}.json"
                with self.assertRaises(ValueError):
                    Recommender(self.data, model, self.cache, taste_cache_path=path).recommend("Uma viagem tranquila")

    def test_details_cannot_be_called_twice_even_for_identical_selection(self):
        script = base_script()
        script.insert(-1, call("get_tmdb_details", {"candidate_ids": [1]}, "details-again"))
        with self.assertRaises(ValueError):
            self.recommender(ScriptedModel(script=script)).recommend("Uma viagem tranquila")

    def test_structured_output_tool_counts_toward_recommender_limit(self):
        script = [call("TasteProfile", {"likes": [], "dislikes": []}, "profile")]
        script.extend(call("search_candidates", {}, f"search-{i}") for i in range(7))
        script.append(AIMessage(content="", tool_calls=[
            {"name": "get_tmdb_details", "args": {"candidate_ids": [1]}, "id": "details", "type": "tool_call"},
            {"name": "RecommendationChoices", "args": choice(), "id": "final", "type": "tool_call"},
        ]))
        with self.assertRaises(RuntimeBudgetExceeded):
            self.recommender(ScriptedModel(script=script)).recommend("Uma viagem tranquille")

    def test_model_loop_without_tools_is_bounded(self):
        script = [call("TasteProfile", {"likes": [], "dislikes": []}, "profile")]
        script.extend(AIMessage(content="Continuando a pensar") for _ in range(9))
        with self.assertRaises((RuntimeBudgetExceeded, ValueError)):
            self.recommender(ScriptedModel(script=script)).recommend("Uma viagem tranquila")
