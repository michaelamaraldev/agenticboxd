from __future__ import annotations

import csv
import io
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field

import app
from app import Recommender, load_letterboxd, run_cli
from src.agents.recommendation import final_prompt, selection_prompt
from src.agents.taste import taste_prompt
from src.models import CineResult, LetterboxdData, TmdbCache, TmdbFact


class RecordingFakeModel(FakeMessagesListChatModel):
    seen_messages: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_messages.append(messages)
        return super()._generate(messages, stop, run_manager, **kwargs)


def fact(watchlist_id: str, title: str = "Film", year: int = 2000) -> TmdbFact:
    return TmdbFact(
        watchlist_id=watchlist_id,
        tmdb_id=year,
        title=title,
        year=year,
        overview="Sinopse confirmada.",
        genres=("Romance",),
        directors=("Director",),
        runtime_minutes=100,
        cast=("Pessoa A",),
        release_date=None,
    )


def cache_for(data: LetterboxdData) -> TmdbCache:
    return TmdbCache(
        movies={
            item.watchlist_id: fact(item.watchlist_id, item.title, item.year)
            for item in data.watchlist
        }
    )


def tool_call(name: str, call_id: str, args: dict[str, Any] | None = None) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {}, "id": call_id, "type": "tool_call"}],
    )


def scripted_responses(watchlist_id: str) -> list[BaseMessage]:
    return [
        tool_call(
            "TasteProfile",
            "taste-profile",
            {"likes": ["natural dialogue"], "dislikes": ["graphic violence"]},
        ),
        tool_call(
            "HistoryContext",
            "history-context",
            {"relevant_patterns": ["recent romances"], "rewatch_signals": []},
        ),
        tool_call(
            "WatchlistSelection",
            "selection",
            {"candidate_ids": [1]},
        ),
        tool_call(
            "RecommendationChoices",
            "final-choice",
            {
                "recommendations": [
                    {
                        "selection_position": 1,
                        "fit_reason": "It matches the requested mood and taste.",
                    }
                ]
            },
        ),
    ]


def test_agents_use_taste_and_verified_tmdb_details(letterboxd_fixture_dir: Path) -> None:
    data = load_letterboxd(letterboxd_fixture_dir)
    target = data.watchlist[0]
    model = RecordingFakeModel(responses=scripted_responses(target.watchlist_id))

    result = Recommender(data, model, cache_for(data)).recommend("Something warm")

    assert result.recommendations[0].watchlist_id == target.watchlist_id
    assert result.recommendations[0].overview == "Sinopse confirmada."
    assert result.recommendations[0].directors == ("Director",)
    prompts = [str(message.content) for call in model.seen_messages for message in call]
    assert any("RATINGS_TSV" in prompt and "REVIEWS_TSV" in prompt for prompt in prompts)
    assert any("DIARY_TSV" in prompt and "Something warm" in prompt for prompt in prompts)
    assert any("Sinopse confirmada." in prompt for prompt in prompts)
    user_prompts = [
        str(call[-1].content) for call in model.seen_messages if call and call[-1].type == "human"
    ]
    assert "RATINGS_TSV" in user_prompts[0]
    assert "DIARY_TSV" in user_prompts[1]
    assert "VERIFIED_WATCHLIST_TSV" in user_prompts[2]
    assert "SELECTED_TMDB_FACTS_JSON" in user_prompts[3]


def test_prompt_contains_arbitrary_utterance_and_every_confirmed_movie_in_order(
    real_data_dir: Path,
) -> None:
    data = load_letterboxd(real_data_dir)
    target = data.watchlist[0]
    model = RecordingFakeModel(responses=scripted_responses(target.watchlist_id))
    utterance = "Surprise me with something formally playful and emotionally restrained."

    Recommender(data, model, cache_for(data)).recommend(utterance)

    prompt = next(
        str(message.content)
        for call in model.seen_messages
        for message in call
        if "VERIFIED_WATCHLIST_TSV" in str(message.content)
    )
    catalog = prompt.split("VERIFIED_WATCHLIST_TSV:\n", 1)[1].split(
        "END_VERIFIED_WATCHLIST", 1
    )[0]
    rows = list(csv.reader(io.StringIO(catalog), dialect="excel-tab"))
    assert utterance in prompt
    assert [row[0] for row in rows[1:]] == [str(index) for index in range(1, 1_175)]
    assert rows[0] == ["candidate_id", "title", "year", "genres", "directors"]
    assert all(row[3:] == ["Romance", "Director"] for row in rows[1:])
    assert prompt.endswith("Never list every matching movie.")


def test_unmatched_movies_are_not_shown_to_the_agent(letterboxd_fixture_dir: Path) -> None:
    data = load_letterboxd(letterboxd_fixture_dir)
    target = data.watchlist[0]
    missing = data.watchlist[1]
    cache = cache_for(data)
    cache = cache.model_copy(
        update={
            "movies": {target.watchlist_id: cache.movies[target.watchlist_id]},
            "unmatched": (missing.watchlist_id,),
        }
    )
    model = RecordingFakeModel(responses=scripted_responses(target.watchlist_id))

    result = Recommender(data, model, cache).recommend("Anything")

    prompt = next(
        str(message.content)
        for call in model.seen_messages
        for message in call
        if "VERIFIED_WATCHLIST_TSV" in str(message.content)
    )
    assert target.title in prompt
    assert missing.title not in prompt
    assert any("TMDb" in warning for warning in result.warnings)


def test_taste_profile_is_cached_between_requests(letterboxd_fixture_dir: Path) -> None:
    data = load_letterboxd(letterboxd_fixture_dir)
    target = data.watchlist[0]
    first = scripted_responses(target.watchlist_id)
    second = first[1:]
    model = RecordingFakeModel(responses=first + second)
    recommender = Recommender(data, model, cache_for(data))

    recommender.recommend("First")
    recommender.recommend("Second")

    prompts = [str(message.content) for call in model.seen_messages for message in call]
    assert sum("RATINGS_TSV" in prompt for prompt in prompts) == 1
    assert sum("DIARY_TSV" in prompt for prompt in prompts) == 2


def test_unknown_selection_is_rejected(letterboxd_fixture_dir: Path) -> None:
    data = load_letterboxd(letterboxd_fixture_dir)
    model = RecordingFakeModel(
        responses=[
            tool_call("TasteProfile", "profile", {"likes": [], "dislikes": []}),
            tool_call(
                "HistoryContext",
                "history",
                {"relevant_patterns": ["recent viewing"], "rewatch_signals": []},
            ),
            tool_call(
                "WatchlistSelection",
                "selection",
                {"candidate_ids": [9999]},
            ),
        ]
    )

    with pytest.raises(ValueError, match="unknown candidate_id"):
        Recommender(data, model, cache_for(data)).recommend("Anything")


def test_invalid_taste_output_stops_without_retry(letterboxd_fixture_dir: Path) -> None:
    data = load_letterboxd(letterboxd_fixture_dir)
    model = RecordingFakeModel(
        responses=[
            tool_call("TasteProfile", "invalid-profile", {"likes": []}),
            AIMessage(content="retry should not happen"),
        ]
    )

    with pytest.raises(StructuredOutputValidationError):
        Recommender(data, model, cache_for(data)).recommend("Anything")

    assert model.i == 1


def test_empty_verified_catalog_returns_before_agents() -> None:
    data = LetterboxdData(watchlist=(), ratings_tsv="", reviews_tsv="")
    model = RecordingFakeModel(responses=[AIMessage(content="unused")])

    result = Recommender(data, model, TmdbCache()).recommend("Anything")

    assert result.recommendations == ()
    assert model.seen_messages == []


def test_llm_facing_docstrings_are_english() -> None:
    from src.tools.analyze_taste import create_analyze_taste_tool
    from src.tools.get_tmdb_details import create_get_tmdb_details_tool

    assert taste_prompt.__doc__ is not None and taste_prompt.__doc__.isascii()
    assert selection_prompt.__doc__ is not None and selection_prompt.__doc__.isascii()
    assert final_prompt.__doc__ is not None and final_prompt.__doc__.isascii()
    assert "taste" in (create_analyze_taste_tool.__doc__ or "").casefold()
    assert "tmdb" in (create_get_tmdb_details_tool.__doc__ or "").casefold()


def test_system_prompts_define_flexible_but_safe_recommendation_rules() -> None:
    taste = taste_prompt()
    selection = selection_prompt()
    final = final_prompt()

    assert "recurring patterns" in taste
    assert "do not invent preferences" in taste
    assert "request as primary" in selection
    assert "taste profile as personalization and a tiebreaker" in selection
    assert "general knowledge" in selection
    assert "entire verified watchlist catalog" in selection
    assert "between 1 and 10 unique candidate_ids" in selection
    assert "Never return every match" in selection
    assert "up to 3" in final
    assert "Brazilian Portuguese" in final
    assert "only from the supplied TMDb data" in final
    assert "Copy selection_position exactly" in final
    assert all(
        "structured response immediately" in prompt
        for prompt in (taste, selection, final)
    )


def test_cli_uses_argument_without_input() -> None:
    class FakeRecommender:
        def __init__(self) -> None:
            self.requests: list[str] = []

        def recommend(self, request: str) -> CineResult:
            self.requests.append(request)
            return CineResult()

    recommender = FakeRecommender()

    run_cli(
        recommender,
        utterance="A romance similar to Rohmer",
        input_fn=lambda _: pytest.fail("input must not be called"),
        output_fn=lambda _: None,
    )

    assert recommender.requests == ["A romance similar to Rohmer"]


def test_cli_without_argument_asks_once() -> None:
    class FakeRecommender:
        def __init__(self) -> None:
            self.requests: list[str] = []

        def recommend(self, request: str) -> CineResult:
            self.requests.append(request)
            return CineResult()

    prompts: list[str] = []
    recommender = FakeRecommender()

    def input_once(prompt: str) -> str:
        prompts.append(prompt)
        return "Anything"

    run_cli(recommender, input_fn=input_once, output_fn=lambda _: None)

    assert recommender.requests == ["Anything"]
    assert prompts == ["O que você quer assistir? "]


def test_main_parses_utterance(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[str] = []

    class FakeRecommender:
        def recommend(self, request: str) -> CineResult:
            requests.append(request)
            return CineResult()

    monkeypatch.setattr(app, "create_recommender", FakeRecommender)

    app.main(["--utterance", "A quiet comedy"])

    assert requests == ["A quiet comedy"]


def test_main_help_explains_the_program(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        app.main(["--help"])

    help_text = capsys.readouterr().out
    assert "watchlist do Letterboxd" in help_text
    assert "--utterance" in help_text
    assert "Rohmer" in help_text


def test_tmdb_token_comes_from_dotenv_or_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TMDB_API_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("TMDB_API_TOKEN=from-dotenv\n", encoding="utf-8")
    assert app.tmdb_token() == "from-dotenv"
    monkeypatch.setenv("TMDB_API_TOKEN", "override")
    assert app.tmdb_token() == "override"
