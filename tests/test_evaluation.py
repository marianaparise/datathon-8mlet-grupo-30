"""Tests for the runner, the metrics and the MLflow wiring."""

from __future__ import annotations

import mlflow
import numpy as np
import pytest
from bandit_testbed import StaticBernoulliTestbed

from src.evaluation import (
    Environment,
    ExperimentResult,
    confidence_interval,
    log_experiment,
    run_episode,
    run_experiment,
    summarize,
)
from src.policies import EpsilonGreedy, FixedArm, ThompsonSampling


@pytest.fixture
def env() -> StaticBernoulliTestbed:
    return StaticBernoulliTestbed([0.10, 0.40, 0.05])


def test_testbed_satisfies_the_environment_protocol(env) -> None:
    """Structural typing is the whole point: no inheritance, no import cycle."""
    assert isinstance(env, Environment)


def test_episode_shapes_line_up(env) -> None:
    rng = np.random.default_rng(0)
    result = run_episode(ThompsonSampling(3, rng=rng), env, n_rounds=200, rng=rng)

    assert result.n_rounds == 200
    for series in (result.arms, result.rewards, result.expected, result.best_expected):
        assert len(series) == 200
    assert result.pull_counts.sum() == 200
    assert set(np.unique(result.rewards)) <= {0, 1}


def test_regret_is_non_negative_and_monotonic(env) -> None:
    rng = np.random.default_rng(0)
    result = run_episode(EpsilonGreedy(3, rng=rng), env, n_rounds=500, rng=rng)
    curve = result.cumulative_regret

    assert (curve >= -1e-12).all()
    assert (np.diff(curve) >= -1e-12).all()
    assert result.total_regret == pytest.approx(curve[-1])


def test_optimal_policy_has_zero_regret(env) -> None:
    rng = np.random.default_rng(0)
    result = run_episode(FixedArm(1, 3, rng=rng), env, n_rounds=300, rng=rng)

    assert result.total_regret == pytest.approx(0.0)


def test_worst_policy_accumulates_the_most_regret(env) -> None:
    rng = np.random.default_rng(0)
    best = run_episode(FixedArm(1, 3, rng=rng), env, n_rounds=300, rng=rng)
    worst = run_episode(FixedArm(2, 3, rng=rng), env, n_rounds=300, rng=rng)

    assert worst.total_regret > best.total_regret


def test_zero_rounds_is_handled(env) -> None:
    rng = np.random.default_rng(0)
    result = run_episode(ThompsonSampling(3, rng=rng), env, n_rounds=0, rng=rng)

    assert result.n_rounds == 0
    assert result.cvr == 0.0
    assert result.exploration_rate == 0.0


def test_experiment_runs_every_seed_with_a_fresh_policy(env) -> None:
    result = run_experiment(
        lambda rng: ThompsonSampling(3, rng=rng), env, n_rounds=200, seeds=range(4)
    )

    assert len(result.episodes) == 4
    assert [e.seed for e in result.episodes] == [0, 1, 2, 3]
    assert len(set(result.cvrs)) > 1, "seeds diferentes deveriam divergir"


def test_experiment_is_reproducible(env) -> None:
    def build() -> ExperimentResult:
        return run_experiment(
            lambda rng: ThompsonSampling(3, rng=rng), env, n_rounds=200, seeds=range(3)
        )

    assert np.array_equal(build().cvrs, build().cvrs)


def test_confidence_interval_brackets_the_mean() -> None:
    values = np.array([0.10, 0.12, 0.11, 0.13, 0.09])
    low, high = confidence_interval(values)

    assert low < values.mean() < high


def test_confidence_interval_degenerates_gracefully() -> None:
    assert confidence_interval(np.array([0.2])) == (0.2, 0.2)
    assert confidence_interval(np.array([])) == (0.0, 0.0)


def test_summarize_ranks_and_computes_uplift(env) -> None:
    good = run_experiment(
        lambda rng: FixedArm(1, 3, rng=rng), env, n_rounds=400, seeds=range(3)
    )
    bad = run_experiment(
        lambda rng: FixedArm(2, 3, rng=rng), env, n_rounds=400, seeds=range(3)
    )

    table = summarize([bad, good], baseline=bad.policy)

    assert list(table["policy"])[0] == good.policy, "melhor política deveria vir antes"
    assert table.loc[table["policy"] == bad.policy, "uplift_vs_baseline"].iloc[0] == 0.0
    assert table.loc[table["policy"] == good.policy, "uplift_vs_baseline"].iloc[0] > 0


def test_summarize_rejects_an_unknown_baseline(env) -> None:
    result = run_experiment(
        lambda rng: FixedArm(0, 3, rng=rng), env, n_rounds=50, seeds=range(2)
    )
    with pytest.raises(KeyError, match="ausente"):
        summarize([result], baseline="ausente")


def test_mlflow_records_params_and_metrics(env, tmp_path) -> None:
    """Experiment not logged in MLflow did not happen — project rule."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test-run")

    result = run_experiment(
        lambda rng: ThompsonSampling(3, rng=rng), env, n_rounds=100, seeds=range(3)
    )
    log_experiment(result, params={"epsilon": 0.1}, baseline_cvr=0.10, n_rounds=100)

    runs = mlflow.search_runs(experiment_names=["test-run"])
    assert len(runs) == 4, "esperado 1 run pai + 3 filhos"

    parent = runs[runs["tags.mlflow.runName"] == result.policy].iloc[0]
    assert parent["params.n_seeds"] == "3"
    assert parent["params.epsilon"] == "0.1"
    assert parent["metrics.cvr_final"] == pytest.approx(result.cvrs.mean())
    assert parent["metrics.cvr_ci_low"] < parent["metrics.cvr_ci_high"]
    assert not np.isnan(parent["metrics.uplift_vs_baseline"])
