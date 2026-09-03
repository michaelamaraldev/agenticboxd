from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def letterboxd_fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "letterboxd"


@pytest.fixture
def real_data_dir() -> Path:
    return Path(__file__).parents[1] / "data"
