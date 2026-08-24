"""Track C: replaying policies against the real log, with no model in the way.

Rejection sampling, following Li, Chu, Langford & Wang (WSDM 2011). The rule is
blunt: walk the log event by event, and count an event only when the policy
happens to pick the arm that was actually played. Then the reward is the ``y``
that was really observed — never an estimate.

That is the whole point of this module. The calibrated environment of Phase 2
answers "what would each arm have yielded?" with a model, and inherits whatever
that model gets wrong. Here nothing is modelled, so the two tracks fail in
different ways. Agreement between them is evidence; disagreement is a finding.

The catch, stated plainly: the unbiasedness proof in the paper assumes the
logging policy chose arms uniformly at random. The bank did not — it picked
channel and timing on operational grounds. :func:`replay_experiment` therefore
reports the plain estimate *and* a self-normalised inverse-propensity estimate,
which corrects for how unevenly the log covers the arms.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config
from src.evaluation import confidence_interval
from src.policies import Policy


@dataclass
class ReplayResult:
    """One pass over the shuffled log for a single policy.

    Attributes:
        policy: Name of the policy replayed.
        seed: Seed that fixed the shuffle and the policy's randomness.
        n_events: Events offered.
        rewards: Observed ``y`` of the accepted events, in acceptance order.
        weights: Inverse-propensity weight of each accepted event.
    """

    policy: str
    seed: int
    n_events: int
    rewards: np.ndarray
    weights: np.ndarray

    @property
    def n_accepted(self) -> int:
        """Events where the policy's arm matched the logged arm."""
        return len(self.rewards)

    @property
    def acceptance_rate(self) -> float:
        """Share of the log that survived rejection sampling."""
        return self.n_accepted / self.n_events if self.n_events else 0.0

    @property
    def cvr(self) -> float:
        """Plain replay estimate: the mean observed reward of accepted events."""
        return float(self.rewards.mean()) if self.n_accepted else 0.0

    @property
    def cvr_ips(self) -> float:
        """Self-normalised inverse-propensity estimate (Hájek).

        Divides by the sum of weights rather than by the count, which trades a
        little bias for a large drop in variance — the plain IPS ratio explodes
        whenever some arm was rarely played for a given kind of client, which is
        exactly this dataset's situation on the landline arms.
        """
        if not self.n_accepted:
            return 0.0
        total = float(self.weights.sum())
        return float((self.weights * self.rewards).sum() / total) if total else 0.0

    @property
    def effective_sample_size(self) -> float:
        """Kish effective sample size of the weighted estimate.

        Weighting buys unbiasedness with precision. This says how many
        unweighted events the weighted sample is actually worth.
        """
        if not self.n_accepted:
            return 0.0
        squared = float((self.weights**2).sum())
        return float(self.weights.sum() ** 2 / squared) if squared else 0.0


@dataclass
class ReplayExperiment:
    """Every seed of one policy replayed.

    Attributes:
        policy: Name of the policy.
        runs: One :class:`ReplayResult` per seed.
    """

    policy: str
    runs: list[ReplayResult] = field(default_factory=list)

    @property
    def cvrs(self) -> np.ndarray:
        """Plain estimate per seed."""
        return np.array([r.cvr for r in self.runs])

    @property
    def cvrs_ips(self) -> np.ndarray:
        """Inverse-propensity estimate per seed."""
        return np.array([r.cvr_ips for r in self.runs])

    @property
    def mean_acceptance(self) -> float:
        """Average share of the log that survived."""
        return float(np.mean([r.acceptance_rate for r in self.runs]))

    @property
    def mean_accepted(self) -> float:
        """Average number of accepted events."""
        return float(np.mean([r.n_accepted for r in self.runs]))

    @property
    def mean_ess(self) -> float:
        """Average effective sample size after weighting."""
        return float(np.mean([r.effective_sample_size for r in self.runs]))


def inverse_propensity_weights(
    propensity: np.ndarray,
    arms: np.ndarray,
    *,
    floor: float = config.MIN_ARM_PROPENSITY,
) -> np.ndarray:
    """Weight each logged event by the inverse of how likely its arm was.

    An event whose arm the bank almost never played for that kind of client
    stands for many similar clients who never got it, so it counts for more.

    The floor is not cosmetic: without it a propensity of 1e-6 would hand a
    single row a weight of a million and let it decide the whole estimate.
    """
    chosen = propensity[np.arange(len(arms)), arms]
    return 1.0 / np.maximum(chosen, floor)


def replay_episode(
    policy: Policy,
    features: np.ndarray,
    arms: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
    *,
    rng: np.random.Generator,
    seed: int = -1,
) -> ReplayResult:
    """Walk the shuffled log once, keeping only the events the policy agrees with.

    Rejected events are dropped whole: the policy does not learn from them. That
    is what keeps the surviving stream consistent with a world in which the
    policy had been the one making the calls.
    """
    order = rng.permutation(len(features))

    kept_rewards: list[float] = []
    kept_weights: list[float] = []

    for index in order:
        context = features[index]
        arm = policy.select(context)
        if arm != arms[index]:
            continue

        reward = float(rewards[index])
        policy.update(context, arm, reward)
        kept_rewards.append(reward)
        kept_weights.append(float(weights[index]))

    return ReplayResult(
        policy=policy.name,
        seed=seed,
        n_events=len(features),
        rewards=np.array(kept_rewards, dtype=float),
        weights=np.array(kept_weights, dtype=float),
    )


def replay_experiment(
    factory: Callable[[np.random.Generator], Policy],
    features: np.ndarray,
    arms: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
    *,
    seeds: Sequence[int] | None = None,
) -> ReplayExperiment:
    """Replay one policy over several shuffles of the log.

    The shuffle matters more here than in the simulator: which events a policy
    accepts depends on the order it meets them, so a single pass is one sample,
    not the answer.
    """
    seed_list = list(range(config.N_SEEDS) if seeds is None else seeds)

    runs = []
    name = ""
    for seed in seed_list:
        rng = np.random.default_rng(config.SEED + seed)
        policy = factory(rng)
        name = policy.name
        runs.append(
            replay_episode(
                policy, features, arms, rewards, weights, rng=rng, seed=seed
            )
        )

    return ReplayExperiment(policy=name, runs=runs)


def summarize_replay(results: Sequence[ReplayExperiment]) -> pd.DataFrame:
    """Comparison table for the replay track, sorted by the IPS estimate."""
    rows = []
    for result in results:
        plain_low, plain_high = confidence_interval(result.cvrs)
        ips_low, ips_high = confidence_interval(result.cvrs_ips)
        rows.append(
            {
                "policy": result.policy,
                "cvr_replay": float(result.cvrs.mean()),
                "cvr_replay_low": plain_low,
                "cvr_replay_high": plain_high,
                "cvr_ips": float(result.cvrs_ips.mean()),
                "cvr_ips_low": ips_low,
                "cvr_ips_high": ips_high,
                "acceptance_rate": result.mean_acceptance,
                "n_accepted": result.mean_accepted,
                "effective_n": result.mean_ess,
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values("cvr_ips", ascending=False).reset_index(drop=True)


def compare_tracks(
    simulated: pd.DataFrame,
    replayed: pd.DataFrame,
    *,
    simulated_column: str = "cvr",
    replayed_column: str = "cvr_ips",
) -> pd.DataFrame:
    """Put the two evaluation tracks side by side, ranked.

    The question this answers is not "do the numbers match" — they cannot, since
    one is a simulation over 20.000 rounds and the other a thinned pass over a
    real log. It is "do the two agree on the ordering", which is what a decision
    would actually rest on.
    """
    left = simulated[["policy", simulated_column]].rename(
        columns={simulated_column: "cvr_ambiente"}
    )
    right = replayed[["policy", replayed_column]].rename(
        columns={replayed_column: "cvr_replay"}
    )

    merged = left.merge(right, on="policy", how="inner")
    merged["rank_ambiente"] = merged["cvr_ambiente"].rank(ascending=False).astype(int)
    merged["rank_replay"] = merged["cvr_replay"].rank(ascending=False).astype(int)
    merged["delta_rank"] = merged["rank_replay"] - merged["rank_ambiente"]
    return merged.sort_values("rank_ambiente").reset_index(drop=True)


def rank_agreement(comparison: pd.DataFrame) -> float:
    """Spearman correlation between the two tracks' rankings.

    1.0 means the tracks order the policies identically. Anything well below
    that is a finding for the README, not a number to bury.
    """
    if len(comparison) < 2:
        return float("nan")
    return float(
        comparison["rank_ambiente"].corr(comparison["rank_replay"], method="spearman")
    )
