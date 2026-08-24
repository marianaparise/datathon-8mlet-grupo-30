"""Tests for the temporal sensitivity analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.arms import ArmSpace
from src.eda import period_index
from src.scenarios import (
    arms_present,
    channel_confounding_report,
    split_by_period,
)


@pytest.fixture(scope="module")
def periods(prepared: pd.DataFrame) -> pd.Series:
    return period_index(prepared)


def test_split_keeps_every_row_exactly_once(prepared, periods) -> None:
    early, late = split_by_period(prepared, periods)

    assert len(early) + len(late) == len(prepared)
    assert not set(early.index) & set(late.index)


def test_split_is_chronological(prepared, periods) -> None:
    """Whole periods land on one side; the cut never splits a month."""
    early, late = split_by_period(prepared, periods)

    assert periods[early.index].max() < periods[late.index].min()


@pytest.mark.parametrize("cut", [0.0, 1.0, -0.1, 2.0])
def test_invalid_cut_is_rejected(prepared, periods, cut) -> None:
    with pytest.raises(ValueError, match="cut precisa"):
        split_by_period(prepared, periods, cut=cut)


def test_mobile_is_absent_from_the_opening_window(prepared, periods) -> None:
    """The fact the whole sensitivity analysis rests on."""
    early, late = split_by_period(prepared, periods)

    assert arms_present(early, ArmSpace.from_frame(prepared)) == (
        "telephone|early",
        "telephone|late",
        "telephone|mid",
    )
    assert (early["contact"] == "cellular").sum() == 0
    assert (late["contact"] == "cellular").mean() > 0.85


def test_confounding_report_covers_every_arm(prepared, periods) -> None:
    space = ArmSpace.from_frame(prepared)
    report = channel_confounding_report(prepared, periods, space)

    assert list(report.by_arm["arm"]) == list(space.labels)
    assert report.coexistence_rows == 29_051


def test_mobile_arms_are_untouched_by_the_restriction(prepared, periods) -> None:
    """Mobile only ever ran in the late window, so restricting cannot move it."""
    space = ArmSpace.from_frame(prepared)
    report = channel_confounding_report(prepared, periods, space)
    mobile = report.by_arm[report.by_arm["arm"].str.startswith("cellular")]

    assert mobile["delta"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_landline_arms_improve_once_the_calendar_is_removed(prepared, periods) -> None:
    """The confound, isolated: landline looks far worse than it was."""
    space = ArmSpace.from_frame(prepared)
    report = channel_confounding_report(prepared, periods, space)
    landline = report.by_arm[report.by_arm["arm"].str.startswith("telephone")]

    assert (landline["delta"] > 0.04).all(), "todo braço de fixo deveria subir"


def test_pooled_gap_is_several_times_the_honest_one(prepared, periods) -> None:
    """The headline of the limitation section, asserted so it cannot rot."""
    space = ArmSpace.from_frame(prepared)
    report = channel_confounding_report(prepared, periods, space)

    assert report.pooled_gap > 1.5
    assert 0.10 < report.coexistence_gap < 0.30
    assert report.inflation > 5.0


def test_report_needs_both_channels_to_compute_a_gap(prepared, periods) -> None:
    early, _ = split_by_period(prepared, periods)
    space = ArmSpace.from_frame(prepared)

    only_landline = channel_confounding_report(
        early, period_index(early), space, cut=0.5
    )
    assert pd.isna(only_landline.pooled_gap) or only_landline.pooled_gap >= 0


def test_target_column_is_the_binary_one(prepared, periods) -> None:
    """Guards against silently reporting on the raw 'yes'/'no' column."""
    space = ArmSpace.from_frame(prepared)
    report = channel_confounding_report(prepared, periods, space)

    assert config.TARGET_BINARY in prepared.columns
    assert (report.by_arm["cvr_pooled"].dropna() <= 1.0).all()
