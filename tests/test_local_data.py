import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.local_data import EvidenceStore, LocalCatalog, TasteCache
from src.models import (
    Film,
    HistoryContext,
    LetterboxdData,
    TasteProfile,
    TmdbCache,
    TmdbFact,
)


def fixture_data(count=25):
    films = tuple(Film(watchlist_id=f"film:{i}", title=f"Filme {i}", year=2000+i) for i in range(count))
    data = LetterboxdData(
        watchlist=films,
        ratings_tsv="title\tyear\trating\nMemória\t2000\t5\nOutro\t2001\t2\n",
        reviews_tsv="title\tyear\trating\treview\ttags\nMemória\t2000\t5\tGosto de viagens introspectivas\tdrama\n",
        diary_tsv="watched_date\ttitle\tyear\trating\trewatch\ttags\n2026-01-01\tMemória\t2000\t5\tYes\tdrama\n2025-01-01\tOutro\t2001\t2\tNo\tcomedia\n",
    )
    facts = {
        film.watchlist_id: TmdbFact(
            watchlist_id=film.watchlist_id, tmdb_id=i+100, title=film.title,
            year=film.year, overview="Memória e ação" if i == 12 else "Uma viagem",
            genres=("Drama",), directors=("Diretora Única",), cast=("Atriz Ágata",),
            runtime_minutes=80+i, release_date=None,
        )
        for i, film in enumerate(films)
    }
    facts["outside"] = facts["film:0"].model_copy(update={"watchlist_id": "outside"})
    if count > 24:
        del facts["film:24"]
    return data, TmdbCache(movies=facts)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.data, self.cache = fixture_data()
        self.catalog = LocalCatalog(self.data, self.cache)

    def test_pages_are_bounded_disjoint_and_stable(self):
        first = self.catalog.search()
        second = self.catalog.search(page=2)
        self.assertEqual(first["total"], 24)
        self.assertEqual(len(first["records"]), 10)
        self.assertEqual(len(second["records"]), 10)
        ids = {row["candidate_id"] for row in first["records"]}
        self.assertFalse(ids.intersection(row["candidate_id"] for row in second["records"]))
        self.assertLessEqual(len(self.catalog.search(page_size=100)["records"]), 20)
        searched = self.catalog.search(query="MEMORIA ACAO")["records"]
        self.assertEqual(len(searched), 1)
        self.assertEqual(searched[0]["candidate_id"], second["records"][2]["candidate_id"])

    def test_search_covers_verified_fields_and_filters(self):
        self.assertEqual(self.catalog.search(query="agata")["total"], 24)
        self.assertEqual(self.catalog.search(query="unica")["total"], 24)
        rows = self.catalog.search(year_min=2005, year_max=2010, runtime_max=87)["records"]
        self.assertEqual([r["year"] for r in rows], [2005, 2006, 2007])
        self.assertEqual(self.catalog.search(runtime_min=100)["total"], 4)
        self.assertEqual(self.catalog.search(query="inexistente")["records"], [])
        self.assertEqual(self.catalog.search(page=999)["records"], [])

    def test_only_current_watchlist_and_confirmed_cache_are_searchable(self):
        catalog = LocalCatalog(replace(self.data, watchlist=self.data.watchlist[12:]), self.cache)
        rows = catalog.search(page_size=20)["records"]
        self.assertEqual(len(rows), 12)
        self.assertEqual({r["watchlist_id"] for r in rows}, {f"film:{i}" for i in range(12, 24)})


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.data, _ = fixture_data()

    def test_queries_expose_ids_and_coverage(self):
        store = EvidenceStore(self.data)
        page = store.search("ratings", page_size=1)
        self.assertEqual(page["total"], 2)
        self.assertEqual(len(page["records"]), 1)
        hit = store.search("ratings", query="MEMORIA")["records"][0]
        self.assertEqual(hit["evidence_id"], page["records"][0]["evidence_id"])
        self.assertEqual(store.search("ratings", query="absent")["records"], [])

    def test_history_date_and_rewatch_queries(self):
        page = EvidenceStore(self.data).search("diary", date_from="2026-01-01", date_to="2026-12-31", rewatch=True)
        self.assertEqual(page["total"], 1)
        self.assertEqual(page["records"][0]["title"], "Memória")

    def test_existing_but_unseen_evidence_is_rejected(self):
        store = EvidenceStore(self.data)
        first = store.search("ratings", page_size=1)["records"][0]["evidence_id"]
        second = EvidenceStore(self.data).search("ratings", page=2, page_size=1)["records"][0]["evidence_id"]
        store.validate_refs([first])
        with self.assertRaises(ValueError):
            store.validate_refs([second])
        with self.assertRaises(ValueError):
            store.validate_refs(["invented:1"])

    def test_empty_profiles_and_history_are_valid(self):
        self.assertEqual(TasteProfile(likes=(), dislikes=()).likes, ())
        self.assertEqual(HistoryContext(relevant_patterns=()).relevant_patterns, ())


class TasteCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "profile.json"
        self.cache = TasteCache(self.path)
        self.data, _ = fixture_data()
        self.config = {"model": "qwen3.5:4b", "temperature": 0, "num_ctx": 32768}
        self.key = self.cache.fingerprint(self.data, self.config, "v1")

    def test_persistence_across_instances_and_invalid_keys(self):
        profile = TasteProfile(likes=(), dislikes=())
        self.assertIsNone(self.cache.save(self.key, profile))
        self.assertEqual(TasteCache(self.path).load(self.key), profile)
        self.assertIsNone(self.cache.load("different"))
        self.assertIsInstance(json.loads(self.path.read_text(encoding="utf-8")), dict)

    def test_fingerprint_includes_all_taste_sources_model_and_version(self):
        for source in ("ratings_tsv", "reviews_tsv", "liked_films_tsv", "favorite_films_tsv", "rankings_tsv"):
            with self.subTest(source=source):
                changed = replace(self.data, **{source: getattr(self.data, source) + "changed"})
                self.assertNotEqual(self.key, self.cache.fingerprint(changed, self.config, "v1"))
        for config in ({**self.config, "model": "other"}, {**self.config, "num_ctx": 8192}):
            self.assertNotEqual(self.key, self.cache.fingerprint(self.data, config, "v1"))
        self.assertNotEqual(self.key, self.cache.fingerprint(self.data, self.config, "v2"))
        changed = replace(self.data, watchlist=(), diary_tsv="changed")
        self.assertEqual(self.key, self.cache.fingerprint(changed, self.config, "v1"))

    def test_corruption_is_a_cache_miss(self):
        for text in ("not json", "[]", '{"profile": {"likes": 2}}'):
            self.path.write_text(text, encoding="utf-8")
            self.assertIsNone(self.cache.load(self.key))

    def test_atomic_replace_failure_preserves_previous_file_and_returns_warning(self):
        profile = TasteProfile(likes=(), dislikes=())
        self.cache.save(self.key, profile)
        original = self.path.read_bytes()
        with patch("os.replace", side_effect=PermissionError("denied")):
            warning = self.cache.save("new-key", profile)
        self.assertTrue(warning)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.cache.load(self.key), profile)
