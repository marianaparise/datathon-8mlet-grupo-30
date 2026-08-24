"""Tests for the bandit policies."""

from __future__ import annotations

import numpy as np
import pytest
from bandit_testbed import ContextualTestbed, StaticBernoulliTestbed

from src import config
from src.evaluation import run_episode
from src.policies import (
    UCB1,
    EpsilonGreedy,
    FixedArm,
    LinTS,
    LoggingPolicy,
    Policy,
    ThompsonSampling,
)

CONTEXT = np.ones(1)


def make_all(n_arms: int, rng: np.random.Generator) -> list[Policy]:
    """Every policy, built for the same arm space."""
    return [
        LoggingPolicy(np.full(n_arms, 1 / n_arms), rng=rng),
        FixedArm(0, n_arms, rng=rng),
        EpsilonGreedy(n_arms, rng=rng),
        UCB1(n_arms, rng=rng),
        ThompsonSampling(n_arms, rng=rng),
        LinTS(n_arms, 1, rng=rng),
    ]


@pytest.mark.parametrize("index", range(6))
def test_interface_conformance(index) -> None:
    policy = make_all(4, np.random.default_rng(0))[index]

    assert isinstance(policy, Policy)
    assert isinstance(policy.name, str) and policy.name
    arm = policy.select(CONTEXT)
    assert isinstance(arm, int)
    assert 0 <= arm < 4
    assert policy.update(CONTEXT, arm, 1) is None


@pytest.mark.parametrize("index", range(6))
def test_same_seed_gives_the_same_sequence(index) -> None:
    """Reproducibility is a project rule, not a nicety."""
    first = make_all(4, np.random.default_rng(7))[index]
    second = make_all(4, np.random.default_rng(7))[index]

    for _ in range(50):
        arm_a, arm_b = first.select(CONTEXT), second.select(CONTEXT)
        assert arm_a == arm_b
        first.update(CONTEXT, arm_a, 1)
        second.update(CONTEXT, arm_b, 1)


def test_different_seeds_diverge() -> None:
    first = EpsilonGreedy(6, rng=np.random.default_rng(1), epsilon=1.0)
    second = EpsilonGreedy(6, rng=np.random.default_rng(2), epsilon=1.0)

    choices_a = [first.select(CONTEXT) for _ in range(60)]
    choices_b = [second.select(CONTEXT) for _ in range(60)]
    assert choices_a != choices_b


def test_no_policy_touches_global_numpy_state() -> None:
    """RNG is injected; a policy reaching for the global state would break seeding."""
    np.random.seed(0)
    before = np.random.get_state()[1][0]

    policy = ThompsonSampling(4, rng=np.random.default_rng(3))
    for _ in range(100):
        policy.update(CONTEXT, policy.select(CONTEXT), 1)

    assert np.random.get_state()[1][0] == before


@pytest.mark.parametrize(
    "factory",
    [
        lambda n, rng: EpsilonGreedy(n, rng=rng, epsilon=0.1),
        lambda n, rng: UCB1(n, rng=rng),
        lambda n, rng: ThompsonSampling(n, rng=rng),
    ],
)
def test_learning_policies_converge_on_the_best_arm(factory) -> None:
    """Each learner should concentrate its pulls where the reward is."""
    env = StaticBernoulliTestbed([0.05, 0.10, 0.40, 0.08])
    rng = np.random.default_rng(config.SEED)

    result = run_episode(factory(env.n_arms, rng), env, n_rounds=3_000, rng=rng)
    share = result.pull_counts[env.best_arm] / result.n_rounds

    assert share > 0.7, f"só {share:.1%} das puxadas no melhor braço"


def test_fixed_and_logging_policies_do_not_learn() -> None:
    env = StaticBernoulliTestbed([0.05, 0.40])
    rng = np.random.default_rng(config.SEED)

    fixed = run_episode(FixedArm(0, 2, rng=rng), env, n_rounds=500, rng=rng)
    assert fixed.pull_counts[0] == 500

    logging = run_episode(
        LoggingPolicy(np.array([0.5, 0.5]), rng=rng), env, n_rounds=2_000, rng=rng
    )
    assert 0.4 < logging.pull_counts[0] / 2_000 < 0.6


def test_epsilon_greedy_never_stops_exploring() -> None:
    """The known flaw, asserted: exploration stays at epsilon forever."""
    env = StaticBernoulliTestbed([0.01, 0.90])
    rng = np.random.default_rng(config.SEED)

    result = run_episode(
        EpsilonGreedy(2, rng=rng, epsilon=0.2), env, n_rounds=4_000, rng=rng
    )
    tail = result.arms[-1_000:]
    bad_share = float((tail == 0).mean())

    assert 0.05 < bad_share < 0.18, "esperado ~epsilon/2 de tráfego no braço ruim"


def test_ucb1_tries_every_arm_before_committing() -> None:
    policy = UCB1(5, rng=np.random.default_rng(0))

    first_five = []
    for _ in range(5):
        arm = policy.select(CONTEXT)
        first_five.append(arm)
        policy.update(CONTEXT, arm, 0)

    assert sorted(first_five) == [0, 1, 2, 3, 4]


def test_thompson_posterior_tracks_the_evidence() -> None:
    policy = ThompsonSampling(2, rng=np.random.default_rng(0))
    for _ in range(30):
        policy.update(CONTEXT, 0, 1)
        policy.update(CONTEXT, 1, 0)

    assert policy.alpha[0] == pytest.approx(config.TS_ALPHA_PRIOR + 30)
    assert policy.beta[1] == pytest.approx(config.TS_BETA_PRIOR + 30)


def test_informed_prior_changes_the_name_and_the_state() -> None:
    uniform = ThompsonSampling(3, rng=np.random.default_rng(0))
    informed = ThompsonSampling(
        3,
        rng=np.random.default_rng(0),
        alpha_prior=config.TS_ALPHA_PRIOR_INFORMED,
        beta_prior=config.TS_BETA_PRIOR_INFORMED,
    )

    assert uniform.name != informed.name
    expected = config.TS_ALPHA_PRIOR_INFORMED / (
        config.TS_ALPHA_PRIOR_INFORMED + config.TS_BETA_PRIOR_INFORMED
    )
    assert expected == pytest.approx(0.113, abs=0.005)


def test_lints_beats_context_free_thompson_when_context_matters() -> None:
    """The load-bearing test of the contextual machinery.

    On a testbed where the best arm flips with the client type, a context-free
    policy is capped at the best average arm. LinTS has to clear that cap — if
    it does not, the contextual implementation is broken, and any null result on
    the real data would be uninterpretable.
    """
    env = ContextualTestbed()

    rng_lin = np.random.default_rng(config.SEED)
    contextual = run_episode(
        LinTS(env.n_arms, env.n_features, rng=rng_lin),
        env,
        n_rounds=6_000,
        rng=rng_lin,
    )

    rng_ts = np.random.default_rng(config.SEED)
    context_free = run_episode(
        ThompsonSampling(env.n_arms, rng=rng_ts), env, n_rounds=6_000, rng=rng_ts
    )

    assert contextual.cvr > context_free.cvr
    assert contextual.cvr > env.best_average_rate


def test_lints_keeps_its_covariance_symmetric() -> None:
    """Rank-1 updates erode symmetry, and the sampler's Cholesky depends on it."""
    policy = LinTS(2, 4, rng=np.random.default_rng(0))
    rng = np.random.default_rng(1)

    for _ in range(500):
        context = rng.standard_normal(4)
        policy.update(context, policy.select(context), float(rng.random() < 0.3))

    for arm in range(2):
        matrix = policy.precision_inv[arm]
        assert np.allclose(matrix, matrix.T, atol=1e-10)
        assert np.all(np.linalg.eigvalsh(matrix) > 0), "covariância não é positiva"


def test_single_arm_space_is_degenerate_but_valid() -> None:
    for policy in make_all(1, np.random.default_rng(0)):
        assert policy.select(CONTEXT) == 0


def test_all_zero_rewards_do_not_break_anything() -> None:
    env = StaticBernoulliTestbed([0.0, 0.0, 0.0])
    rng = np.random.default_rng(config.SEED)

    result = run_episode(UCB1(3, rng=rng), env, n_rounds=300, rng=rng)
    assert result.rewards.sum() == 0
    assert result.total_regret == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: EpsilonGreedy(3, rng=np.random.default_rng(0), epsilon=1.5),
            "epsilon",
        ),
        (lambda: UCB1(3, rng=np.random.default_rng(0), c=-1.0), "c precisa"),
        (
            lambda: ThompsonSampling(3, rng=np.random.default_rng(0), alpha_prior=0.0),
            "positivos",
        ),
        (lambda: FixedArm(9, 3, rng=np.random.default_rng(0)), "fora de"),
        (
            lambda: LoggingPolicy(np.array([0.5, 0.2]), rng=np.random.default_rng(0)),
            "somar 1",
        ),
        (lambda: LinTS(3, 0, rng=np.random.default_rng(0)), "n_features"),
    ],
)
def test_invalid_hyperparameters_are_rejected(factory, message) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
