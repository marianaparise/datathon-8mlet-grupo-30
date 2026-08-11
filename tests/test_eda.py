"""Tests for the descriptive analysis helpers."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from src import config, eda
from src.data import prepare


def test_wilson_interval_brackets_the_estimate() -> None:
    low, high = eda.wilson_interval(50, 100)

    assert low < 0.5 < high
    assert (low, high) == pytest.approx((0.4038, 0.5962), abs=1e-4)


def test_wilson_interval_stays_inside_the_unit_range() -> None:
    """The normal approximation would run past 0 here; Wilson must not."""
    low, high = eda.wilson_interval(0, 40)

    assert low == pytest.approx(0.0, abs=1e-9)
    assert 0 < high < 1


def test_wilson_interval_shrinks_with_sample_size() -> None:
    small = eda.wilson_interval(10, 100)
    large = eda.wilson_interval(1_000, 10_000)

    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_interval_is_vectorised() -> None:
    low, high = eda.wilson_interval(np.array([1, 50]), np.array([10, 100]))

    assert low.shape == high.shape == (2,)
    assert (low < high).all()


def test_conversion_by_counts_add_up(toy: pd.DataFrame) -> None:
    table = eda.conversion_by(prepare(toy), "contact")

    assert table["n"].sum() == len(toy)
    assert (table["cvr"] == table["conversions"] / table["n"]).all()
    assert (table["cvr_low"] <= table["cvr"]).all()
    assert (table["cvr"] <= table["cvr_high"]).all()


def test_unknown_report_only_lists_columns_that_have_it(toy: pd.DataFrame) -> None:
    report = eda.unknown_report(toy)

    assert report.loc["job", "n_unknown"] == 3
    assert "contact" not in report.index


def test_screen_arm_spaces_rejects_a_thin_cell(toy: pd.DataFrame) -> None:
    screened = eda.screen_arm_spaces(
        prepare(toy), [("contact",)], min_events=1_000, min_conversions=100
    )

    assert not screened.loc[0, "passa"]
    assert screened.loc[0, "n_bracos"] == 2


def test_month_run_count_detects_shuffling(raw: pd.DataFrame) -> None:
    """File order is chronological — that is what makes a temporal split thinkable."""
    shuffled = raw.sample(frac=1, random_state=config.SEED)

    assert eda.month_run_count(raw) == 26
    assert eda.month_run_count(shuffled) > 30_000


def test_period_index_is_monotonic(raw: pd.DataFrame) -> None:
    periodo = eda.period_index(raw)

    assert periodo.is_monotonic_increasing
    assert periodo.iloc[0] == 0
    assert periodo.max() == eda.month_run_count(raw) - 1


def test_macro_indicators_are_calendar_stamps(prepared: pd.DataFrame) -> None:
    """Every macro column is constant within a period — it cannot personalise."""
    report = eda.macro_calendar_report(prepared).set_index("indicador")

    assert (report["r2_periodo"] > 0.99).all()


def test_arm_space_clears_the_support_floor(prepared: pd.DataFrame) -> None:
    """Guards the central decision of Phase 1 against regression."""
    table = eda.arm_support(prepared, list(config.ARM_COLUMNS))

    assert len(table) == 6
    assert table["n"].min() >= config.MIN_EVENTS_PER_ARM
    assert table["conversions"].min() >= config.MIN_CONVERSIONS_PER_ARM


def test_cardinality_report_covers_every_column(raw: pd.DataFrame) -> None:
    report = eda.cardinality_report(raw)

    assert list(report.index) == list(raw.columns)
    assert report.loc["contact", "n_unique"] == 2
    assert report.loc["nr.employed", "n_unique"] == 11
    assert (report["n_missing"] == 0).all(), "a base não tem nulos"


def test_duration_alone_beats_the_legitimate_features(raw: pd.DataFrame) -> None:
    """The number that justifies dropping the column, and that the README publishes."""
    table = eda.duration_leakage(raw).set_index("conjunto")

    assert set(table.index) == {"com duration", "sem duration", "só duration"}
    assert table.loc["só duration", "n_features"] == 1
    assert table.loc["com duration", "auc"] > table.loc["sem duration", "auc"]
    assert table.loc["só duration", "auc"] > table.loc["sem duration", "auc"]
    assert table.loc["com duration", "auc"] > 0.9


def test_duration_leakage_is_reproducible(raw: pd.DataFrame) -> None:
    first = eda.duration_leakage(raw, seed=config.SEED)
    second = eda.duration_leakage(raw, seed=config.SEED)

    pd.testing.assert_frame_equal(first, second)


def test_contact_by_month_shares_add_up(prepared: pd.DataFrame) -> None:
    table = eda.contact_by_month(prepared)

    shares = table.drop(columns="n")
    assert shares.sum(axis=1).round(6).eq(1.0).all()
    assert table["n"].sum() == len(prepared)


def test_channel_mix_drifts_across_the_campaign(prepared: pd.DataFrame) -> None:
    """The confounding the README declares: the log opens 100% telephone."""
    table = eda.contact_by_month(prepared)

    assert table["telephone"].iloc[0] == pytest.approx(1.0)
    assert table["cellular"].max() > 0.9


def test_set_style_applies_without_touching_import_time() -> None:
    eda.set_style()

    assert matplotlib.rcParams["figure.dpi"] == 110


def test_plots_return_figures(prepared: pd.DataFrame, raw: pd.DataFrame) -> None:
    table = eda.conversion_by(prepared, config.ARM_COLUMN)

    assert isinstance(eda.plot_conversion(table, title="t"), Figure)
    assert isinstance(eda.plot_macro_over_time(prepared), Figure)
    assert isinstance(eda.plot_contact_over_time(prepared), Figure)
    assert isinstance(eda.plot_duration_leakage(raw), Figure)


def test_save_figure_writes_a_file(tmp_path) -> None:
    figure = Figure()
    figure.add_subplot().plot([0, 1], [0, 1])

    written = eda.save_figure(figure, "teste.png", directory=tmp_path)

    assert written.exists()
    assert written.stat().st_size > 0
