"""Tests for raw data loading."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.data import SchemaError, binarize_target, drop_forbidden, load_raw

pytestmark = pytest.mark.skipif(
    not config.RAW_CSV.exists(), reason="base ausente — rode `make data`"
)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_raw()


def test_load_raw_shape(raw: pd.DataFrame) -> None:
    assert raw.shape == (config.RAW_N_ROWS, len(config.RAW_COLUMNS))


def test_load_raw_columns(raw: pd.DataFrame) -> None:
    assert tuple(raw.columns) == config.RAW_COLUMNS


def test_column_groups_partition_the_schema() -> None:
    """Context, action, forbidden and target must tile RAW_COLUMNS exactly, no overlap."""
    groups = (
        config.CONTEXT_COLUMNS,
        config.ACTION_COLUMNS,
        config.FORBIDDEN_COLUMNS,
        (config.TARGET,),
    )
    flat = [col for group in groups for col in group]

    assert len(flat) == len(set(flat)), "coluna repetida entre os grupos"
    assert set(flat) == set(config.RAW_COLUMNS)


def test_wrong_separator_is_rejected(tmp_path) -> None:
    """A comma-separated read collapses to one column and must not pass silently."""
    bad = tmp_path / "comma.csv"
    bad.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    with pytest.raises(SchemaError, match="separador errado"):
        load_raw(bad)


def test_missing_file_points_to_make_data(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        load_raw(tmp_path / "ausente.csv")


def test_drop_forbidden_removes_duration(raw: pd.DataFrame) -> None:
    cleaned = drop_forbidden(raw)

    assert "duration" not in cleaned.columns
    assert len(cleaned.columns) == len(raw.columns) - 1
    assert "duration" in raw.columns, "o original não pode ser mutado"


def test_drop_forbidden_is_idempotent(raw: pd.DataFrame) -> None:
    once = drop_forbidden(raw)
    assert list(drop_forbidden(once).columns) == list(once.columns)


def test_binarize_target(raw: pd.DataFrame) -> None:
    y = binarize_target(raw)

    assert set(y.unique()) == {0, 1}
    assert y.sum() == 4_640
    assert y.mean() == pytest.approx(0.1127, abs=1e-4)
