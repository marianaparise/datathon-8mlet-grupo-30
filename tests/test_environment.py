"""Tests for the calibrated environment.

The quality gates live here: an environment that is not calibrated, or that
extrapolates where no arm was ever played, would poison every number that comes
after it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config, data
from src.arms import ArmSpace
from src.environment import (
    CalibratedEnvironment,
    EnvironmentDiagnostics,
    EnvironmentError,
    build_environment,
    context_frame,
    contextual_ceiling,
)
from src.evaluation import Environment, Observation


@pytest.fixture(scope="module")
def fitted(prepared) -> tuple[CalibratedEnvironment, EnvironmentDiagnostics, ArmSpace]:
    train, test = data.split_train_test(prepared)
    space = ArmSpace.from_frame(prepared)
    env, diagnostics = build_environment(train, test, space)
    return env, diagnostics, space


def test_context_excludes_the_forbidden_column(prepared) -> None:
    frame = context_frame(prepared)

    assert "duration" not in frame.columns
    assert config.TARGET not in frame.columns
    assert config.TARGET_BINARY not in frame.columns


def test_context_excludes_the_arm_itself(prepared) -> None:
    """The policy chooses the arm; letting it leak into the context is circular."""
    frame = context_frame(prepared)

    for column in (*config.ACTION_COLUMNS, config.ARM_COLUMN):
        assert column not in frame.columns


def test_macro_columns_are_out_by_default(prepared) -> None:
    """Phase 1: they are calendar stamps, constant within a period."""
    default = context_frame(prepared)
    ablation = context_frame(prepared, include_macro=True)

    for column in config.MACRO_COLUMNS:
        assert column not in default.columns
        assert column in ablation.columns


def test_pdays_sentinel_is_neutralised(prepared) -> None:
    frame = context_frame(prepared)

    assert (frame["pdays"] == config.PDAYS_SENTINEL).sum() == 0
    assert frame["pdays"].max() < 100


def test_missing_context_column_is_rejected(prepared) -> None:
    with pytest.raises(EnvironmentError, match="ausentes"):
        context_frame(prepared.drop(columns=["job"]))


def test_environment_satisfies_the_protocol(fitted) -> None:
    env, _, _ = fitted
    assert isinstance(env, Environment)


def test_sample_returns_a_well_formed_observation(fitted) -> None:
    env, _, space = fitted
    observation = env.sample(np.random.default_rng(0))

    assert isinstance(observation, Observation)
    assert observation.features.shape == (env.n_features,)
    assert observation.expected_rewards.shape == (space.n_arms,)
    rewards = observation.expected_rewards
    assert ((rewards >= 0) & (rewards <= 1)).all()


def test_sampling_is_reproducible(fitted) -> None:
    env, _, _ = fitted
    first = env.sample(np.random.default_rng(11))
    second = env.sample(np.random.default_rng(11))

    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.expected_rewards, second.expected_rewards)


def test_environment_is_calibrated(fitted) -> None:
    """Gate one: probabilities have to mean what they say."""
    _, diagnostics, _ = fitted

    assert diagnostics.brier <= config.MAX_BRIER_SCORE
    total = diagnostics.calibration.query("arm == 'TOTAL'").iloc[0]
    assert total["gap"] < 0.01


def test_every_arm_is_calibrated_not_just_the_total(fitted) -> None:
    """Gate one, sharpened: a good total can hide a bad low-volume arm."""
    _, diagnostics, space = fitted
    per_arm = diagnostics.calibration.query("arm != 'TOTAL'")

    assert len(per_arm) == space.n_arms
    assert per_arm["gap"].max() < 0.02, "algum braço está mal calibrado"


def test_model_beats_the_logistic_sanity_check(fitted) -> None:
    """Gate two: if the boosting does not beat a linear baseline, the gain is noise."""
    _, diagnostics, _ = fitted

    assert diagnostics.auc > diagnostics.baseline_auc
    assert diagnostics.auc > 0.70


def test_overlap_is_reported_for_every_arm(fitted) -> None:
    """Gate three: positivity. Where an arm was never played, we extrapolate."""
    _, diagnostics, space = fitted
    overlap = diagnostics.overlap

    assert len(overlap) == space.n_arms
    assert (overlap["median"] > config.MIN_ARM_PROPENSITY).all()
    assert (overlap["share_below_floor"] < 0.10).all()


def test_uncalibrated_environment_refuses_to_build(prepared, monkeypatch) -> None:
    """The gate must actually stop the pipeline, not just warn."""
    monkeypatch.setattr(config, "MAX_BRIER_SCORE", 0.0001)
    train, test = data.split_train_test(prepared)

    with pytest.raises(EnvironmentError, match="descalibrado"):
        build_environment(train, test, ArmSpace.from_frame(prepared))


def test_contextual_ceiling_is_zero_when_arms_are_identical() -> None:
    matrix = np.tile(np.array([0.2, 0.2, 0.2]), (100, 1))
    ceiling = contextual_ceiling(matrix)

    assert ceiling.absolute_gain == pytest.approx(0.0)
    assert ceiling.switch_share == 0.0


def test_contextual_ceiling_detects_a_flip() -> None:
    matrix = np.array([[0.9, 0.1]] * 50 + [[0.1, 0.9]] * 50)
    ceiling = contextual_ceiling(matrix)

    assert ceiling.fixed_cvr == pytest.approx(0.5)
    assert ceiling.oracle_cvr == pytest.approx(0.9)
    assert ceiling.switch_share == pytest.approx(0.5)


def test_real_ceiling_is_positive_but_modest(fitted) -> None:
    """The number that decides whether LinTS has anything to win."""
    _, diagnostics, space = fitted
    ceiling = diagnostics.ceiling

    assert space.label(ceiling.best_global_arm) == "cellular|mid"
    assert ceiling.absolute_gain > 0
    assert 0.0 < ceiling.relative_gain < 0.20


def test_mismatched_shapes_are_rejected() -> None:
    space = ArmSpace(("a", "b"))

    with pytest.raises(EnvironmentError, match="tamanhos diferentes"):
        CalibratedEnvironment(space, np.zeros((3, 2)), np.zeros((4, 2)))
    with pytest.raises(EnvironmentError, match="uma coluna por braço"):
        CalibratedEnvironment(space, np.zeros((3, 2)), np.zeros((3, 5)))
