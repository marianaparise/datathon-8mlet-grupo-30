"""Bandit policies: the decision makers.

Every policy answers the same two questions — ``select`` picks an arm for a
client, ``update`` learns from what happened — so the runner and the replay can
swap one for another without knowing which is which.

The randomness generator is always injected. Nothing here touches the global
``np.random`` state, because a shared global would make seeds meaningless.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src import config


class Policy(ABC):
    """Common interface for every decision rule.

    Attributes:
        n_arms: Size of the arm space.
        rng: Injected generator; the sole source of randomness.
    """

    def __init__(self, n_arms: int, *, rng: np.random.Generator) -> None:
        """Store the arm count and the generator, then reset the state."""
        if n_arms < 1:
            raise ValueError(f"n_arms precisa ser >= 1, recebido {n_arms}.")
        self.n_arms = n_arms
        self.rng = rng
        self.reset()

    @property
    def name(self) -> str:
        """Short label used in results tables and MLflow runs."""
        return type(self).__name__

    def reset(self) -> None:  # noqa: B027
        """Clear learned state. Called on construction.

        Intentionally concrete and empty: the stateless policies (``FixedArm``,
        ``LoggingPolicy``) have nothing to clear, so forcing them to implement
        an abstract hook would be noise.
        """

    @abstractmethod
    def select(self, context: np.ndarray) -> int:
        """Choose an arm for this client."""

    @abstractmethod
    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Learn from the observed outcome of ``arm``."""

    def _break_ties(self, scores: np.ndarray) -> int:
        """Argmax that splits ties at random.

        Deterministic argmax would hand every early tie to arm 0, which biases
        exploration towards whatever happens to sort first.
        """
        winners = np.flatnonzero(scores == scores.max())
        return int(winners[0] if len(winners) == 1 else self.rng.choice(winners))


class LoggingPolicy(Policy):
    """Replays the historical arm mixture. The project's main baseline.

    This is what the campaign actually did: not the best arm, but a blend. It
    learns nothing, which is exactly the point of a control.
    """

    def __init__(
        self, arm_probabilities: np.ndarray, *, rng: np.random.Generator
    ) -> None:
        """Store the historical mixture and validate that it is a distribution."""
        probabilities = np.asarray(arm_probabilities, dtype=float)
        if probabilities.ndim != 1:
            raise ValueError("arm_probabilities precisa ser 1-D.")
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError(
                f"arm_probabilities deve somar 1, soma {probabilities.sum():.4f}."
            )
        self.arm_probabilities = probabilities
        super().__init__(len(probabilities), rng=rng)

    def select(self, context: np.ndarray) -> int:
        """Draw an arm from the historical mixture, ignoring the context."""
        return int(self.rng.choice(self.n_arms, p=self.arm_probabilities))

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """No-op: the logging policy does not learn."""


class FixedArm(Policy):
    """Always plays the same arm. The hard comparator.

    Pointed at the best historical arm, this is where any non-contextual bandit
    converges, so it is the bar the contextual policy has to clear.
    """

    def __init__(self, arm: int, n_arms: int, *, rng: np.random.Generator) -> None:
        """Store the arm to always play."""
        if not 0 <= arm < n_arms:
            raise ValueError(f"arm {arm} fora de [0, {n_arms}).")
        self.arm = arm
        super().__init__(n_arms, rng=rng)

    @property
    def name(self) -> str:
        """Label carrying the pinned arm index."""
        return f"FixedArm[{self.arm}]"

    def select(self, context: np.ndarray) -> int:
        """Return the pinned arm."""
        return self.arm

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """No-op: a fixed rule does not learn."""


class EpsilonGreedy(Policy):
    """Exploits the best empirical mean, explores uniformly at rate ``epsilon``.

    The exploration never anneals: at round ten thousand it still spends the
    same share of traffic on arms already known to be poor. That flaw is the
    reason UCB1 and Thompson Sampling exist, and the comparison shows it.
    """

    def __init__(
        self,
        n_arms: int,
        *,
        rng: np.random.Generator,
        epsilon: float = config.EPSILON,
    ) -> None:
        """Store the exploration rate."""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon precisa estar em [0, 1], recebido {epsilon}.")
        self.epsilon = epsilon
        super().__init__(n_arms, rng=rng)

    def reset(self) -> None:
        """Zero the per-arm pull counts and reward sums."""
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.sums = np.zeros(self.n_arms, dtype=float)

    @property
    def means(self) -> np.ndarray:
        """Empirical mean reward per arm; unplayed arms score zero."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(
                self.counts > 0, self.sums / np.maximum(self.counts, 1), 0.0
            )

    def select(self, context: np.ndarray) -> int:
        """Explore with probability ``epsilon``, otherwise take the best mean."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        return self._break_ties(self.means)

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Fold the observed reward into the running mean of ``arm``."""
        self.counts[arm] += 1
        self.sums[arm] += reward


class UCB1(Policy):
    """Optimism in the face of uncertainty (Auer, Cesa-Bianchi & Fischer, 2002).

    Scores each arm by its mean plus a bonus that shrinks as evidence piles up,
    so exploration is aimed at what is genuinely unknown rather than sprayed at
    random. Every arm is played once before the formula takes over.
    """

    def __init__(
        self, n_arms: int, *, rng: np.random.Generator, c: float = config.UCB_C
    ) -> None:
        """Store the exploration scaling."""
        if c < 0:
            raise ValueError(f"c precisa ser >= 0, recebido {c}.")
        self.c = c
        super().__init__(n_arms, rng=rng)

    def reset(self) -> None:
        """Zero the counts, the sums and the round clock."""
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.sums = np.zeros(self.n_arms, dtype=float)
        self.t = 0

    def select(self, context: np.ndarray) -> int:
        """Play any untried arm first, then the highest upper bound."""
        untried = np.flatnonzero(self.counts == 0)
        if len(untried):
            return int(untried[0])

        means = self.sums / self.counts
        bonus = self.c * np.sqrt(2.0 * np.log(max(self.t, 1)) / self.counts)
        return self._break_ties(means + bonus)

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Fold in the reward and advance the clock."""
        self.counts[arm] += 1
        self.sums[arm] += reward
        self.t += 1


class ThompsonSampling(Policy):
    """Bayesian exploration over a Beta posterior per arm.

    Draws one sample from each arm's belief and plays the winner. Exploration is
    not a rule bolted on top: an arm with little evidence has a wide posterior,
    so its draws scatter and it gets tried. As evidence accumulates the
    posterior narrows and the policy settles by itself.

    The prior is a modelling choice, not a default: ``Beta(1, 1)`` is uniform
    over [0, 1] and assumes nothing, while the informed variant in
    ``config.TS_ALPHA_PRIOR_INFORMED`` centres on the 11.27% base rate with the
    weight of ten observations.
    """

    def __init__(
        self,
        n_arms: int,
        *,
        rng: np.random.Generator,
        alpha_prior: float = config.TS_ALPHA_PRIOR,
        beta_prior: float = config.TS_BETA_PRIOR,
    ) -> None:
        """Store the Beta prior shared by every arm."""
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("Os parâmetros da Beta precisam ser positivos.")
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        super().__init__(n_arms, rng=rng)

    @property
    def name(self) -> str:
        """Label carrying the prior, since it changes the behaviour."""
        return f"ThompsonSampling[{self.alpha_prior:g},{self.beta_prior:g}]"

    def reset(self) -> None:
        """Set every arm's posterior back to the prior."""
        self.alpha = np.full(self.n_arms, self.alpha_prior, dtype=float)
        self.beta = np.full(self.n_arms, self.beta_prior, dtype=float)

    def select(self, context: np.ndarray) -> int:
        """Sample a conversion rate per arm and play the highest draw."""
        return self._break_ties(self.rng.beta(self.alpha, self.beta))

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Add the outcome to the arm's success or failure count."""
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward


class LinTS(Policy):
    """Contextual Thompson Sampling with linear payoffs (Agrawal & Goyal, 2013).

    The only policy here that reads the client. It keeps a Gaussian posterior
    over a coefficient vector per arm, samples one, and plays whichever arm
    scores highest for *this* context — so it can learn that one arm suits
    retirees and another suits students, instead of crowning a single winner.

    The precision inverse is carried by Sherman-Morrison rank-1 updates, and the
    Cholesky factor is cached per arm and refreshed only for the arm that moved.
    Recomputing all of it every round costs more than the whole experiment.
    """

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        *,
        rng: np.random.Generator,
        v: float = config.LINTS_V,
        lambda_: float = config.LINTS_LAMBDA,
    ) -> None:
        """Store the posterior width and the ridge term."""
        if n_features < 1:
            raise ValueError(f"n_features precisa ser >= 1, recebido {n_features}.")
        if v <= 0 or lambda_ <= 0:
            raise ValueError("v e lambda_ precisam ser positivos.")
        self.n_features = n_features
        self.v = v
        self.lambda_ = lambda_
        super().__init__(n_arms, rng=rng)

    @property
    def name(self) -> str:
        """Label carrying the exploration width."""
        return f"LinTS[v={self.v:g}]"

    def reset(self) -> None:
        """Reset every arm to the ridge prior, with no observations folded in."""
        eye = np.eye(self.n_features)
        self.precision_inv = np.array(
            [eye / self.lambda_ for _ in range(self.n_arms)], dtype=float
        )
        self.b = np.zeros((self.n_arms, self.n_features), dtype=float)
        self._chol: list[np.ndarray | None] = [None] * self.n_arms

    def _factor(self, arm: int) -> np.ndarray:
        """Cholesky factor of the arm's covariance, computed at most once per update."""
        cached = self._chol[arm]
        if cached is not None:
            return cached

        covariance = self.precision_inv[arm]
        try:
            factor = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            # Rank-1 updates can nudge the matrix just outside positive definite.
            # Nudging the diagonal is cheaper and stabler than refactorising.
            jitter = 1e-9 * np.eye(self.n_features)
            factor = np.linalg.cholesky(covariance + jitter)

        self._chol[arm] = factor
        return factor

    def select(self, context: np.ndarray) -> int:
        """Sample a coefficient vector per arm; play the best score for this client."""
        x = np.asarray(context, dtype=float).ravel()
        scores = np.empty(self.n_arms, dtype=float)

        for arm in range(self.n_arms):
            mean = self.precision_inv[arm] @ self.b[arm]
            noise = self._factor(arm) @ self.rng.standard_normal(self.n_features)
            scores[arm] = x @ (mean + self.v * noise)

        return self._break_ties(scores)

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Fold the observation into the arm's posterior via Sherman-Morrison."""
        x = np.asarray(context, dtype=float).ravel()
        inv = self.precision_inv[arm]

        inv_x = inv @ x
        denominator = 1.0 + float(x @ inv_x)
        inv -= np.outer(inv_x, inv_x) / denominator

        # Rank-1 updates erode symmetry over tens of thousands of rounds, and an
        # asymmetric matrix breaks the Cholesky the sampler depends on.
        self.precision_inv[arm] = (inv + inv.T) / 2.0
        self.b[arm] += reward * x
        self._chol[arm] = None
