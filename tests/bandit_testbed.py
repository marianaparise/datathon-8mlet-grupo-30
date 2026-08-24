"""Synthetic environments used only to verify that the algorithms are correct.

Deliberately confined to ``tests/``: nothing here may ever reach a reported
number. Checking that UCB1 converges on arms with known means is verification of
an implementation, not a simulated result — the rule against synthetic reward in
``CLAUDE.md`` is about what gets published, not about unit tests.
"""

from __future__ import annotations

import numpy as np

from src.evaluation import Observation


class StaticBernoulliTestbed:
    """Fixed conversion rates per arm; the context carries no information."""

    def __init__(self, probabilities: np.ndarray | list[float]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=float)
        if self.probabilities.ndim != 1 or not len(self.probabilities):
            raise ValueError("probabilities precisa ser 1-D e não vazio.")

    @property
    def n_arms(self) -> int:
        return len(self.probabilities)

    @property
    def n_features(self) -> int:
        return 1

    @property
    def best_arm(self) -> int:
        return int(np.argmax(self.probabilities))

    def sample(self, rng: np.random.Generator) -> Observation:
        return Observation(
            features=np.ones(1, dtype=float),
            expected_rewards=self.probabilities,
        )


class ContextualTestbed:
    """Two client types with opposite best arms.

    The context is ``[1, z]`` with ``z`` in {-1, +1}. Arm 0 wins for one type and
    arm 1 for the other, while arm 2 is uniformly mediocre. No context-free
    policy can do better than the best *average* arm here, so a contextual
    policy that works must pull ahead — that is what the comparison test asserts.
    """

    HIGH = 0.30
    LOW = 0.05
    MIDDLE = 0.16

    @property
    def n_arms(self) -> int:
        return 3

    @property
    def n_features(self) -> int:
        return 2

    @property
    def best_average_rate(self) -> float:
        """Rate of the best arm for a policy that cannot read the context."""
        return self.MIDDLE

    @property
    def oracle_rate(self) -> float:
        """Rate of a policy that always picks the right arm for the type."""
        return self.HIGH

    def sample(self, rng: np.random.Generator) -> Observation:
        z = 1.0 if rng.random() < 0.5 else -1.0
        if z > 0:
            rewards = np.array([self.HIGH, self.LOW, self.MIDDLE])
        else:
            rewards = np.array([self.LOW, self.HIGH, self.MIDDLE])
        return Observation(features=np.array([1.0, z]), expected_rewards=rewards)
