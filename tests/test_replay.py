"""Tests for the replay track."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.policies import EpsilonGreedy, FixedArm, LoggingPolicy
from src.replay import (
    ReplayResult,
    compare_tracks,
    inverse_propensity_weights,
    rank_agreement,
    replay_episode,
    replay_experiment,
    summarize_replay,
)


@pytest.fixture
def log() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A tiny synthetic log: arm 0 converts often, arm 1 rarely."""
    rng = np.random.default_rng(0)
    n = 2_000
    arms = rng.integers(0, 2, size=n)
    rewards = (rng.random(n) < np.where(arms == 0, 0.30, 0.05)).astype(float)
    features = np.ones((n, 1))
    weights = np.ones(n)
    return features, arms, rewards, weights


def test_only_matching_arms_are_accepted(log) -> None:
    """The defining rule: an event counts only when the policy agrees with it."""
    features, arms, rewards, weights = log
    rng = np.random.default_rng(0)

    result = replay_episode(
        FixedArm(0, 2, rng=rng), features, arms, rewards, weights, rng=rng
    )

    assert result.n_accepted == int((arms == 0).sum())
    assert result.n_events == len(arms)


def test_rewards_are_the_observed_ones(log) -> None:
    """Nothing is estimated here — that is the whole point of track C."""
    features, arms, rewards, weights = log
    rng = np.random.default_rng(0)

    result = replay_episode(
        FixedArm(1, 2, rng=rng), features, arms, rewards, weights, rng=rng
    )

    assert set(np.unique(result.rewards)) <= {0.0, 1.0}
    assert result.cvr == pytest.approx(rewards[arms == 1].mean(), abs=1e-12)


def test_acceptance_rate_tracks_the_arm_share(log) -> None:
    features, arms, rewards, weights = log
    rng = np.random.default_rng(0)

    result = replay_episode(
        FixedArm(0, 2, rng=rng), features, arms, rewards, weights, rng=rng
    )

    assert result.acceptance_rate == pytest.approx((arms == 0).mean())


def test_a_learning_policy_finds_the_good_arm(log) -> None:
    features, arms, rewards, weights = log

    good = replay_experiment(
        lambda rng: FixedArm(0, 2, rng=rng), features, arms, rewards, weights,
        seeds=range(3),
    )
    bad = replay_experiment(
        lambda rng: FixedArm(1, 2, rng=rng), features, arms, rewards, weights,
        seeds=range(3),
    )
    learner = replay_experiment(
        lambda rng: EpsilonGreedy(2, rng=rng, epsilon=0.05), features, arms, rewards,
        weights, seeds=range(3),
    )

    assert good.cvrs.mean() > learner.cvrs.mean() > bad.cvrs.mean()


def test_replay_is_reproducible(log) -> None:
    features, arms, rewards, weights = log

    def build() -> np.ndarray:
        return replay_experiment(
            lambda rng: EpsilonGreedy(2, rng=rng), features, arms, rewards, weights,
            seeds=range(3),
        ).cvrs

    assert np.array_equal(build(), build())


def test_shuffle_order_changes_which_events_are_kept(log) -> None:
    """A single pass is one sample of the replay, not the answer."""
    features, arms, rewards, weights = log

    experiment = replay_experiment(
        lambda rng: EpsilonGreedy(2, rng=rng), features, arms, rewards, weights,
        seeds=range(5),
    )
    accepted = [r.n_accepted for r in experiment.runs]

    assert len(set(accepted)) > 1


def test_inverse_propensity_weights_reward_rare_arms() -> None:
    propensity = np.array([[0.9, 0.1], [0.5, 0.5]])
    arms = np.array([1, 0])

    weights = inverse_propensity_weights(propensity, arms)

    assert weights[0] == pytest.approx(10.0)
    assert weights[1] == pytest.approx(2.0)


def test_propensity_floor_caps_the_weight() -> None:
    """Without the floor one row with propensity 1e-9 would own the estimate."""
    propensity = np.array([[1e-9, 1.0]])
    arms = np.array([0])

    weights = inverse_propensity_weights(propensity, arms, floor=0.01)

    assert weights[0] == pytest.approx(100.0)


def test_ips_equals_plain_estimate_under_uniform_weights(log) -> None:
    features, arms, rewards, weights = log
    rng = np.random.default_rng(0)

    result = replay_episode(
        FixedArm(0, 2, rng=rng), features, arms, rewards, weights, rng=rng
    )

    assert result.cvr_ips == pytest.approx(result.cvr)


def test_ips_reweights_towards_the_underplayed_arm() -> None:
    """Two arms, same reward pattern, but one is over-represented in the log."""
    rewards = np.array([1.0, 1.0, 1.0, 0.0])
    weights = np.array([1.0, 1.0, 1.0, 9.0])
    result = ReplayResult("x", 0, n_events=4, rewards=rewards, weights=weights)

    assert result.cvr == pytest.approx(0.75)
    assert result.cvr_ips == pytest.approx(3 / 12)


def test_effective_sample_size_penalises_uneven_weights() -> None:
    even = ReplayResult("x", 0, 4, np.ones(4), np.ones(4))
    uneven = ReplayResult("x", 0, 4, np.ones(4), np.array([1.0, 1.0, 1.0, 100.0]))

    assert even.effective_sample_size == pytest.approx(4.0)
    assert uneven.effective_sample_size < 2.0


def test_empty_replay_does_not_divide_by_zero() -> None:
    empty = ReplayResult("x", 0, 100, np.array([]), np.array([]))

    assert empty.n_accepted == 0
    assert empty.cvr == 0.0
    assert empty.cvr_ips == 0.0
    assert empty.effective_sample_size == 0.0
    assert empty.acceptance_rate == 0.0


def test_summarize_replay_sorts_by_the_ips_estimate(log) -> None:
    features, arms, rewards, weights = log
    results = [
        replay_experiment(
            lambda rng, arm=arm: FixedArm(arm, 2, rng=rng),
            features, arms, rewards, weights, seeds=range(3),
        )
        for arm in (1, 0)
    ]

    table = summarize_replay(results)

    assert table["cvr_ips"].is_monotonic_decreasing
    assert table.loc[0, "policy"] == "FixedArm[0]"


def test_logging_policy_accepts_its_own_mixture(log) -> None:
    """Sanity: replaying the logging policy should accept roughly sum(p^2)."""
    features, arms, rewards, weights = log
    shares = np.array([0.5, 0.5])

    experiment = replay_experiment(
        lambda rng: LoggingPolicy(shares, rng=rng), features, arms, rewards, weights,
        seeds=range(5),
    )

    expected = float((shares**2).sum())
    assert experiment.mean_acceptance == pytest.approx(expected, abs=0.03)


def test_compare_tracks_aligns_and_ranks() -> None:
    simulated = pd.DataFrame({"policy": ["a", "b", "c"], "cvr": [0.3, 0.2, 0.1]})
    replayed = pd.DataFrame({"policy": ["a", "b", "c"], "cvr_ips": [0.35, 0.15, 0.25]})

    comparison = compare_tracks(simulated, replayed)

    assert list(comparison["rank_ambiente"]) == [1, 2, 3]
    assert list(comparison["rank_replay"]) == [1, 3, 2]
    assert list(comparison["delta_rank"]) == [0, 1, -1]


def test_rank_agreement_is_one_when_orders_match() -> None:
    simulated = pd.DataFrame({"policy": ["a", "b", "c"], "cvr": [0.3, 0.2, 0.1]})
    replayed = pd.DataFrame({"policy": ["a", "b", "c"], "cvr_ips": [0.5, 0.4, 0.3]})

    assert rank_agreement(compare_tracks(simulated, replayed)) == pytest.approx(1.0)


def test_rank_agreement_is_minus_one_when_orders_invert() -> None:
    simulated = pd.DataFrame({"policy": ["a", "b", "c"], "cvr": [0.3, 0.2, 0.1]})
    replayed = pd.DataFrame({"policy": ["a", "b", "c"], "cvr_ips": [0.1, 0.2, 0.3]})

    assert rank_agreement(compare_tracks(simulated, replayed)) == pytest.approx(-1.0)
