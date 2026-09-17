import json
import unittest

from pydantic import ValidationError
from test_local_data import fixture_data

from src.local_data import (
    LocalCatalog,
    create_search_candidates_tool,
    request_constraints,
)
from src.models import WatchlistSelection
from src.tools.get_tmdb_details import TmdbToolState, create_get_tmdb_details_tool


class EfficientToolContractTests(unittest.TestCase):
    def setUp(self):
        self.data, self.cache = fixture_data()

    def test_candidate_search_falls_back_without_relaxing_objective_filters(self):
        constraints = request_constraints("Quero algo de at\u00e9 90 minutos")
        tool = create_search_candidates_tool(
            LocalCatalog(self.data, self.cache), constraints
        )

        result = json.loads(
            tool.invoke({"query": "termo que nao existe", "page_size": 99})
        )

        self.assertTrue(result["records"])
        self.assertLessEqual(len(result["records"]), 5)
        self.assertTrue(
            all(record["runtime_minutes"] <= 90 for record in result["records"])
        )

    def test_candidate_search_returns_only_compact_selection_fields(self):
        tool = create_search_candidates_tool(LocalCatalog(self.data, self.cache))

        result = json.loads(tool.invoke({"page_size": 99}))

        self.assertEqual(len(result["records"]), 5)
        expected = {
            "candidate_id",
            "title",
            "year",
            "overview",
            "genres",
            "directors",
            "runtime_minutes",
            "evidence_ids",
        }
        self.assertTrue(result["records"])
        for record in result["records"]:
            self.assertEqual(set(record), expected)
            available_fields = set(record) - {"candidate_id", "evidence_ids"}
            referenced_fields = {
                evidence_id.rsplit(":", 1)[-1]
                for evidence_id in record["evidence_ids"]
            }
            self.assertLessEqual(referenced_fields, available_fields)

    def test_watchlist_selection_schema_accepts_at_most_three_candidates(self):
        with self.assertRaises(ValidationError):
            WatchlistSelection(candidate_ids=(1, 2, 3, 4))

    def test_tmdb_details_accepts_at_most_three_candidates(self):
        catalog = LocalCatalog(self.data, self.cache)
        tool = create_get_tmdb_details_tool(
            self.cache,
            catalog.candidates,
            TmdbToolState(),
        )

        with self.assertRaises(ValueError):
            tool.invoke({"candidate_ids": [1, 2, 3, 4]})


if __name__ == "__main__":
    unittest.main()
