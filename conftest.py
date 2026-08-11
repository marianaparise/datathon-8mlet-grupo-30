"""Shared pytest fixtures.

This file also fixes collection: pytest's ``prepend`` import mode puts the test
file's basedir on ``sys.path``, not the repo root, so ``from src import ...``
fails under ``pytest tests/``. A conftest at the root puts the root there.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config, data


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    """The untouched log. Skips when the base has not been downloaded."""
    if not config.RAW_CSV.exists():
        pytest.skip("base ausente — rode `make data`")
    return data.load_raw()


@pytest.fixture(scope="session")
def prepared(raw: pd.DataFrame) -> pd.DataFrame:
    return data.prepare(raw)


@pytest.fixture
def toy() -> pd.DataFrame:
    """Small hand-built frame so most tests do not need the real base."""
    rows = [
        ("cellular", "mon", 999, "nonexistent", "admin.", 1, "no"),
        ("cellular", "tue", 3, "success", "unknown", 2, "yes"),
        ("cellular", "wed", 999, "failure", "technician", 1, "no"),
        ("cellular", "thu", 6, "success", "admin.", 1, "yes"),
        ("telephone", "fri", 999, "nonexistent", "unknown", 3, "no"),
        ("telephone", "mon", 999, "nonexistent", "services", 1, "no"),
        ("telephone", "fri", 12, "failure", "admin.", 2, "yes"),
        ("telephone", "wed", 999, "nonexistent", "unknown", 1, "no"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["contact", "day_of_week", "pdays", "poutcome", "job", "campaign", "y"],
    )
    frame["duration"] = 100
    frame["euribor3m"] = 4.0
    return frame
