"""Tests for raw data loading and the preparation pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.data import (
    SchemaError,
    add_first_contact,
    add_week_window,
    binarize_target,
    build_arm,
    drop_forbidden,
    load_raw,
    prepare,
    split_train_test,
)


def test_load_raw_shape(raw: pd.DataFrame) -> None:
    assert raw.shape == (config.RAW_N_ROWS, len(config.RAW_COLUMNS))


def test_load_raw_columns(raw: pd.DataFrame) -> None:
    assert tuple(raw.columns) == config.RAW_COLUMNS


def test_column_groups_partition_the_schema() -> None:
    """Context, action, forbidden and target tile RAW_COLUMNS exactly, no overlap."""
    groups = (
        config.CONTEXT_COLUMNS,
        config.ACTION_COLUMNS,
        config.FORBIDDEN_COLUMNS,
        (config.TARGET,),
    )
    flat = [col for group in groups for col in group]

    assert len(flat) == len(set(flat)), "coluna repetida entre os grupos"
    assert set(flat) == set(config.RAW_COLUMNS)


def test_client_and_macro_partition_the_context() -> None:
    assert set(config.CLIENT_COLUMNS) | set(config.MACRO_COLUMNS) == set(
        config.CONTEXT_COLUMNS
    )
    assert not set(config.CLIENT_COLUMNS) & set(config.MACRO_COLUMNS)


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


def test_week_windows_cover_every_day(raw: pd.DataFrame) -> None:
    assert set(raw["day_of_week"].unique()) == set(config.WEEK_WINDOWS)


def test_add_week_window_maps_every_row(toy: pd.DataFrame) -> None:
    out = add_week_window(toy)

    assert out[config.WEEK_WINDOW_COLUMN].notna().all()
    assert set(out[config.WEEK_WINDOW_COLUMN]) <= set(config.WEEK_WINDOWS.values())
    assert config.WEEK_WINDOW_COLUMN not in toy.columns, "o original foi mutado"


def test_add_week_window_rejects_unknown_day(toy: pd.DataFrame) -> None:
    broken = toy.assign(day_of_week="sat")

    with pytest.raises(SchemaError, match="sat"):
        add_week_window(broken)


def test_add_first_contact_flags_the_sentinel(toy: pd.DataFrame) -> None:
    out = add_first_contact(toy)

    expected = (toy["pdays"] == config.PDAYS_SENTINEL).astype(int)
    assert out[config.FIRST_CONTACT_COLUMN].equals(expected)


def test_build_arm_stays_inside_the_cartesian_product(toy: pd.DataFrame) -> None:
    out = add_week_window(toy)
    arm = build_arm(out)

    possible = {
        f"{c}|{w}"
        for c in out["contact"].unique()
        for w in config.WEEK_WINDOWS.values()
    }
    assert set(arm) <= possible
    assert arm.notna().all()


def test_build_arm_rejects_missing_dimension(toy: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="week_window"):
        build_arm(toy)


def test_prepare_derives_the_expected_columns(toy: pd.DataFrame) -> None:
    out = prepare(toy)

    assert "duration" not in out.columns
    for column in (
        config.WEEK_WINDOW_COLUMN,
        config.FIRST_CONTACT_COLUMN,
        config.ARM_COLUMN,
        config.TARGET_BINARY,
    ):
        assert column in out.columns


def test_prepare_does_not_mutate_the_input(toy: pd.DataFrame) -> None:
    before = toy.copy()
    prepare(toy)

    pd.testing.assert_frame_equal(toy, before)


def test_prepare_keeps_unknown_as_a_level(prepared: pd.DataFrame) -> None:
    """`unknown` is a recorded answer here, not a missing value — never imputed."""
    assert (prepared["default"] == config.UNKNOWN_TOKEN).sum() == 8_597
    assert prepared.notna().all().all()


def test_split_is_a_partition(prepared: pd.DataFrame) -> None:
    train, test = split_train_test(prepared)

    assert len(train) + len(test) == len(prepared)
    assert not set(train.index) & set(test.index)
    assert len(test) == pytest.approx(len(prepared) * config.TEST_SIZE, rel=0.01)


def test_split_is_reproducible(prepared: pd.DataFrame) -> None:
    first, _ = split_train_test(prepared)
    second, _ = split_train_test(prepared)
    other, _ = split_train_test(prepared, seed=config.SEED + 1)

    assert first.index.equals(second.index)
    assert not first.index.equals(other.index)


def test_split_preserves_target_and_arm_shares(prepared: pd.DataFrame) -> None:
    train, test = split_train_test(prepared)

    assert train[config.TARGET_BINARY].mean() == pytest.approx(
        test[config.TARGET_BINARY].mean(), abs=0.005
    )
    train_share = train[config.ARM_COLUMN].value_counts(normalize=True)
    test_share = test[config.ARM_COLUMN].value_counts(normalize=True)
    assert (train_share - test_share).abs().max() < 0.005


def test_every_arm_has_support_in_both_folds(prepared: pd.DataFrame) -> None:
    """The calibrated environment of Phase 2 estimates P(y | x, arm) per arm."""
    train, test = split_train_test(prepared)

    for fold in (train, test):
        counts = fold.groupby(config.ARM_COLUMN, observed=True)[config.TARGET_BINARY]
        assert counts.size().min() > 0
        assert counts.sum().min() > 0, "braço sem nenhuma conversão no fold"
