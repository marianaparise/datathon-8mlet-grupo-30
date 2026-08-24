"""The measuring rig: runs policies against an environment and scores them.

Deliberately knows nothing about banks or arms. The environment is consumed
through a structural :class:`Environment` protocol, so the calibrated
environment of Phase 2 and the synthetic testbeds used in the tests are
interchangeable here without either importing the other.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from src import config
from src.policies import Policy


@dataclass(frozen=True)
class Observation:
    """One eligible client offered to a policy.

    Attributes:
        features: Numeric context vector the policy sees, shape ``(n_features,)``.
        expected_rewards: True conversion probability of every arm for this
            client, shape ``(n_arms,)``. The policy never sees this — it is the
            oracle the runner uses to draw rewards and to score regret.
    """

    features: np.ndarray
    expected_rewards: np.ndarray


@runtime_checkable
class Environment(Protocol):
    """What the runner needs from an environment. Structural, not inherited."""

    @property
    def n_arms(self) -> int:
        """Number of arms on offer."""
        ...

    @property
    def n_features(self) -> int:
        """Width of the context vector."""
        ...

    def sample(self, rng: np.random.Generator) -> Observation:
        """Draw one client."""
        ...


@dataclass
class EpisodeResult:
    """Per-round trace of a single policy run.

    Attributes:
        policy: Name of the policy that produced the trace.
        seed: Seed the episode ran under.
        arms: Arm chosen at each round.
        rewards: Realised 0/1 reward at each round.
        expected: Expected reward of the chosen arm at each round.
        best_expected: Expected reward of the best arm at each round.
        n_arms: Size of the arm space, kept for the pull-count histogram.
    """

    policy: str
    seed: int
    arms: np.ndarray
    rewards: np.ndarray
    expected: np.ndarray
    best_expected: np.ndarray
    n_arms: int

    @property
    def n_rounds(self) -> int:
        """Number of rounds played."""
        return len(self.arms)

    @property
    def cvr(self) -> float:
        """Realised conversion rate over the episode."""
        return float(self.rewards.mean()) if self.n_rounds else 0.0

    @property
    def cumulative_regret(self) -> np.ndarray:
        """Running sum of ``best_expected - expected``, never decreasing."""
        return np.cumsum(self.best_expected - self.expected)

    @property
    def total_regret(self) -> float:
        """Regret accumulated over the whole episode."""
        return float((self.best_expected - self.expected).sum())

    @property
    def cumulative_cvr(self) -> np.ndarray:
        """Conversion rate as it evolves round by round."""
        rounds = np.arange(1, self.n_rounds + 1)
        return np.cumsum(self.rewards) / rounds

    @property
    def pull_counts(self) -> np.ndarray:
        """How many times each arm was chosen, shape ``(n_arms,)``."""
        return np.bincount(self.arms, minlength=self.n_arms)

    @property
    def exploration_rate(self) -> float:
        """Share of rounds spent off the arm the policy pulled most.

        A coarse but honest read of how much traffic went to exploring, and the
        "análise de exploração" the challenge statement asks for.
        """
        if not self.n_rounds:
            return 0.0
        return 1.0 - float(self.pull_counts.max() / self.n_rounds)


@dataclass
class ExperimentResult:
    """Every seed of one policy.

    Attributes:
        policy: Name of the policy.
        episodes: One :class:`EpisodeResult` per seed.
    """

    policy: str
    episodes: list[EpisodeResult] = field(default_factory=list)

    @property
    def cvrs(self) -> np.ndarray:
        """Final conversion rate of each seed."""
        return np.array([e.cvr for e in self.episodes])

    @property
    def regrets(self) -> np.ndarray:
        """Total regret of each seed."""
        return np.array([e.total_regret for e in self.episodes])

    @property
    def mean_cumulative_cvr(self) -> np.ndarray:
        """Conversion curve averaged across seeds."""
        return np.mean([e.cumulative_cvr for e in self.episodes], axis=0)

    @property
    def mean_cumulative_regret(self) -> np.ndarray:
        """Regret curve averaged across seeds."""
        return np.mean([e.cumulative_regret for e in self.episodes], axis=0)

    @property
    def mean_pull_counts(self) -> np.ndarray:
        """Pull histogram averaged across seeds."""
        return np.mean([e.pull_counts for e in self.episodes], axis=0)


def confidence_interval(
    values: np.ndarray, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Student-t interval for the mean across seeds.

    Not a Wilson interval: the quantity here is the spread of per-seed means,
    not a binomial proportion.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        point = float(values[0]) if n else 0.0
        return point, point

    half = student_t.ppf(0.5 + confidence / 2, df=n - 1) * float(
        values.std(ddof=1) / np.sqrt(n)
    )
    mean = float(values.mean())
    return mean - half, mean + half


def run_episode(
    policy: Policy,
    env: Environment,
    *,
    n_rounds: int = config.N_ROUNDS,
    rng: np.random.Generator,
    seed: int = -1,
) -> EpisodeResult:
    """Play ``n_rounds`` decisions and record every one.

    The reward draw lives here rather than in the environment so that the same
    stream of clients yields the same rewards for every policy under a given
    seed — otherwise part of the measured difference would be luck.
    """
    arms = np.empty(n_rounds, dtype=int)
    rewards = np.empty(n_rounds, dtype=int)
    expected = np.empty(n_rounds, dtype=float)
    best_expected = np.empty(n_rounds, dtype=float)

    for step in range(n_rounds):
        observation = env.sample(rng)
        arm = policy.select(observation.features)

        arm_probability = float(observation.expected_rewards[arm])
        reward = int(rng.random() < arm_probability)
        policy.update(observation.features, arm, reward)

        arms[step] = arm
        rewards[step] = reward
        expected[step] = arm_probability
        best_expected[step] = float(observation.expected_rewards.max())

    return EpisodeResult(
        policy=policy.name,
        seed=seed,
        arms=arms,
        rewards=rewards,
        expected=expected,
        best_expected=best_expected,
        n_arms=env.n_arms,
    )


def run_experiment(
    factory: Callable[[np.random.Generator], Policy],
    env: Environment,
    *,
    n_rounds: int = config.N_ROUNDS,
    seeds: Sequence[int] | None = None,
) -> ExperimentResult:
    """Run one policy across several seeds.

    A fresh policy is built per seed — carrying learned state between seeds
    would make later runs look better for no reason. A single-seed curve is not
    a result.
    """
    seed_list = list(range(config.N_SEEDS) if seeds is None else seeds)

    episodes = []
    name = ""
    for seed in seed_list:
        rng = np.random.default_rng(config.SEED + seed)
        policy = factory(rng)
        name = policy.name
        episodes.append(
            run_episode(policy, env, n_rounds=n_rounds, rng=rng, seed=seed)
        )

    return ExperimentResult(policy=name, episodes=episodes)


def summarize(
    results: Sequence[ExperimentResult], *, baseline: str | None = None
) -> pd.DataFrame:
    """Comparison table: CVR and regret with intervals, plus uplift.

    Args:
        results: One entry per policy.
        baseline: Policy name the uplift column is measured against. Defaults to
            the first result.

    Returns:
        One row per policy, sorted by mean conversion rate.
    """
    if not results:
        return pd.DataFrame()

    reference = baseline or results[0].policy
    by_name = {r.policy: r for r in results}
    if reference not in by_name:
        raise KeyError(f"Baseline {reference!r} não está entre os resultados.")
    baseline_cvr = float(by_name[reference].cvrs.mean())

    rows = []
    for result in results:
        cvr_low, cvr_high = confidence_interval(result.cvrs)
        regret_low, regret_high = confidence_interval(result.regrets)
        mean_cvr = float(result.cvrs.mean())
        rows.append(
            {
                "policy": result.policy,
                "cvr": mean_cvr,
                "cvr_low": cvr_low,
                "cvr_high": cvr_high,
                "regret": float(result.regrets.mean()),
                "regret_low": regret_low,
                "regret_high": regret_high,
                "uplift_vs_baseline": (
                    mean_cvr / baseline_cvr - 1.0 if baseline_cvr else float("nan")
                ),
                "exploration_rate": float(
                    np.mean([e.exploration_rate for e in result.episodes])
                ),
                "n_seeds": len(result.episodes),
            }
        )

    return pd.DataFrame(rows).sort_values("cvr", ascending=False).reset_index(drop=True)


def start_tracking(
    *,
    tracking_uri: str = config.MLFLOW_TRACKING_URI,
    experiment: str = config.MLFLOW_EXPERIMENT,
) -> None:
    """Point MLflow at the local store. Called by entrypoints, never on import.

    MLflow is imported here, not at module level, and that is deliberate. This
    module sits on the import path of the serving API — ``app`` imports
    ``golden_set``, which imports ``environment``, which imports the
    :class:`Observation` defined here. A top-level ``import mlflow`` would drag
    the whole tracking stack into the container for code that only ever runs
    during an experiment.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)


def log_experiment(
    result: ExperimentResult,
    *,
    params: Mapping[str, Any] | None = None,
    baseline_cvr: float | None = None,
    n_rounds: int = config.N_ROUNDS,
) -> None:
    """Record one policy as a parent run with a child run per seed.

    The parent carries the mean and the interval across seeds — the number that
    can actually be quoted. The children keep every seed auditable, so nobody
    has to take the average on faith.
    """
    import mlflow  # ver a nota em `start_tracking`

    cvr_low, cvr_high = confidence_interval(result.cvrs)
    regret_low, regret_high = confidence_interval(result.regrets)
    mean_cvr = float(result.cvrs.mean())

    with mlflow.start_run(run_name=result.policy):
        mlflow.log_params(
            {
                "policy": result.policy,
                "n_rounds": n_rounds,
                "n_seeds": len(result.episodes),
                "n_arms": result.episodes[0].n_arms if result.episodes else 0,
                **dict(params or {}),
            }
        )
        mlflow.log_metrics(
            {
                "cvr_final": mean_cvr,
                "cvr_ci_low": cvr_low,
                "cvr_ci_high": cvr_high,
                "regret_final": float(result.regrets.mean()),
                "regret_ci_low": regret_low,
                "regret_ci_high": regret_high,
                "exploration_rate": float(
                    np.mean([e.exploration_rate for e in result.episodes])
                ),
            }
        )
        if baseline_cvr:
            mlflow.log_metric("uplift_vs_baseline", mean_cvr / baseline_cvr - 1.0)

        for episode in result.episodes:
            child_name = f"{result.policy}#{episode.seed}"
            with mlflow.start_run(run_name=child_name, nested=True):
                mlflow.log_params({"policy": result.policy, "seed": episode.seed})
                mlflow.log_metrics(
                    {
                        "cvr_final": episode.cvr,
                        "regret_final": episode.total_regret,
                        "exploration_rate": episode.exploration_rate,
                    }
                )
