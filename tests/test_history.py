from __future__ import annotations

from pathlib import Path

from app import load_letterboxd
from src.agents.history import history_prompt, history_request
from src.agents.recommendation import final_request, selection_request
from src.models import HistoryContext, LetterboxdData, TasteProfile


def test_letterboxd_context_uses_likes_favorites_rankings_and_diary(
    real_data_dir: Path,
) -> None:
    data = load_letterboxd(real_data_dir)

    assert len(data.liked_films_tsv.splitlines()) == 127
    assert "Ran\t1985" in data.favorite_films_tsv
    assert "Boyfriends and Girlfriends\t1987" in data.favorite_films_tsv
    assert "Éric Rohmer - ranking\t1\tBoyfriends and Girlfriends\t1987" in data.rankings_tsv
    assert len(data.diary_tsv.splitlines()) == 167
    assert data.diary_tsv.index("Inception") < data.diary_tsv.index("GoodFellas")


def test_taste_context_never_contains_profile_personal_fields() -> None:
    data = LetterboxdData(
        watchlist=(),
        ratings_tsv="ratings",
        reviews_tsv="reviews",
        liked_films_tsv="likes",
        favorite_films_tsv="favorites",
        rankings_tsv="rankings",
        diary_tsv="diary",
    )
    from src.agents.taste import taste_request

    prompt = taste_request(data)

    assert "LIKED_FILMS_TSV:\nlikes" in prompt
    assert "FAVORITE_FILMS_TSV:\nfavorites" in prompt
    assert "RANKINGS_TSV:\nrankings" in prompt
    assert "DIARY" not in prompt


def test_history_prompt_relates_the_full_diary_to_the_request() -> None:
    request = "Quero algo diferente do que tenho visto recentemente."
    data = LetterboxdData(watchlist=(), ratings_tsv="", reviews_tsv="", diary_tsv="diary")

    prompt = history_request(request, data)

    assert request in prompt
    assert "DIARY_TSV:\ndiary" in prompt
    assert "Do not recommend" in history_prompt()


def test_recommendation_requests_include_history_context() -> None:
    profile = TasteProfile(likes=("dialogue",), dislikes=())
    history = HistoryContext(relevant_patterns=("recent dramas",), rewatch_signals=())

    selection = selection_request(
        "pedido inteiro", profile.model_dump_json(), history.model_dump_json(), "catalog"
    )
    final = final_request(
        "pedido inteiro", profile.model_dump_json(), history.model_dump_json(), "details"
    )

    assert "pedido inteiro" in selection and "recent dramas" in selection
    assert "pedido inteiro" in final and "recent dramas" in final
