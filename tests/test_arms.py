"""Tests for the arm space."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.arms import ArmSpace, UnknownArmError, arm_distribution, best_historical_arm


@pytest.fixture
def space() -> ArmSpace:
    return ArmSpace(("cellular|mid", "telephone|early"))


@pytest.fixture
def log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            config.ARM_COLUMN: [
                "cellular|mid",
                "cellular|mid",
                "cellular|mid",
                "telephone|early",
            ],
            config.TARGET_BINARY: [1, 1, 0, 0],
        }
    )


def test_round_trip_between_label_and_index(space) -> None:
    for index, label in enumerate(space.labels):
        assert space.index(label) == index
        assert space.label(index) == label


def test_ordering_is_stable_and_sorted(prepared) -> None:
    """Index order is baked into artefacts; re-deriving it would mislabel."""
    first = ArmSpace.from_frame(prepared)
    second = ArmSpace.from_frame(prepared.sample(frac=1.0, random_state=1))

    assert first.labels == second.labels
    assert list(first.labels) == sorted(first.labels)


def test_space_matches_the_phase_one_decision(prepared) -> None:
    space = ArmSpace.from_frame(prepared)

    assert space.n_arms == 6
    assert len(space) == 6
    assert set(space.labels) == {
        f"{contact}|{window}"
        for contact in ("cellular", "telephone")
        for window in ("early", "mid", "late")
    }


def test_encode_maps_a_series(space, log) -> None:
    encoded = space.encode(log[config.ARM_COLUMN])

    assert encoded.tolist() == [0, 0, 0, 1]
    assert encoded.dtype == int


def test_unknown_labels_are_rejected(space, log) -> None:
    with pytest.raises(UnknownArmError, match="cellular\\|late"):
        space.encode(pd.Series(["cellular|late"]))
    with pytest.raises(UnknownArmError, match="fora do espaço"):
        space.index("inexistente")
    with pytest.raises(UnknownArmError, match="fora de"):
        space.label(99)


def test_missing_arm_column_is_rejected() -> None:
    with pytest.raises(UnknownArmError, match="ausente"):
        ArmSpace.from_frame(pd.DataFrame({"outra": [1]}))


def test_arm_distribution_is_a_distribution(space, log) -> None:
    shares = arm_distribution(log, space)

    assert shares.sum() == pytest.approx(1.0)
    assert shares.tolist() == [0.75, 0.25]


def test_arm_distribution_follows_index_order(prepared) -> None:
    space = ArmSpace.from_frame(prepared)
    shares = arm_distribution(prepared, space)

    assert len(shares) == space.n_arms
    assert shares.sum() == pytest.approx(1.0)
    assert shares[space.index("cellular|mid")] == pytest.approx(0.3876, abs=1e-3)


def test_best_historical_arm(space, log) -> None:
    assert best_historical_arm(log, space) == 0


def test_best_historical_arm_on_the_real_log(prepared) -> None:
    """Phase 1 found the modal arm is also the best — the baseline trap."""
    space = ArmSpace.from_frame(prepared)
    best = best_historical_arm(prepared, space)
    shares = arm_distribution(prepared, space)

    assert space.label(best) == "cellular|mid"
    assert int(np.argmax(shares)) == best, "braço modal e melhor braço coincidem"
