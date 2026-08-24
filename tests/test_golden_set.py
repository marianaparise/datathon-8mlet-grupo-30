"""Tests for the Golden Set."""

from __future__ import annotations

import numpy as np
import pytest

from src import config, data
from src.arms import ArmSpace
from src.environment import build_environment
from src.golden_set import (
    TIE_THRESHOLD,
    GoldenCase,
    explain,
    golden_set_table,
    select_golden_set,
)


@pytest.fixture(scope="module")
def golden(prepared) -> tuple[list[GoldenCase], ArmSpace, object]:
    train, test = data.split_train_test(prepared)
    space = ArmSpace.from_frame(prepared)
    environment, _ = build_environment(train, test, space)
    return select_golden_set(test, environment, space), space, test


def make_case(
    scores: list[float], *, global_best: int = 0, **profile: object
) -> GoldenCase:
    array = np.array(scores)
    return GoldenCase(
        criterion="teste",
        row_label=0,
        profile=profile,
        scores=array,
        recommended=int(array.argmax()),
        global_best=global_best,
    )


def test_the_challenge_asks_for_five(golden) -> None:
    cases, _, _ = golden
    assert len(cases) == 5


def test_every_case_states_its_criterion(golden) -> None:
    """Chosen by reason, not sampled — otherwise the table proves nothing."""
    cases, _, _ = golden
    criteria = [c.criterion for c in cases]

    assert len(set(criteria)) == 5
    assert all(c.strip() for c in criteria)


def test_cases_are_distinct_clients(golden) -> None:
    cases, _, _ = golden
    assert len({c.row_label for c in cases}) == 5


def test_selection_is_reproducible(prepared) -> None:
    train, test = data.split_train_test(prepared)
    space = ArmSpace.from_frame(prepared)
    environment, _ = build_environment(train, test, space)

    first = [c.row_label for c in select_golden_set(test, environment, space)]
    second = [c.row_label for c in select_golden_set(test, environment, space)]
    assert first == second


def test_every_arm_is_scored_for_every_case(golden) -> None:
    """The set shows the whole ranking, not just the winner."""
    cases, space, _ = golden

    for case in cases:
        assert case.scores.shape == (space.n_arms,)
        assert ((case.scores >= 0) & (case.scores <= 1)).all()


def test_recommendation_is_the_argmax(golden) -> None:
    cases, _, _ = golden
    for case in cases:
        assert case.recommended == int(case.scores.argmax())
        assert case.best_score == pytest.approx(case.scores.max())


def test_profiles_never_leak_the_forbidden_column(golden) -> None:
    cases, _, _ = golden
    for case in cases:
        assert "duration" not in case.profile
        assert config.TARGET not in case.profile
        assert config.TARGET_BINARY not in case.profile


def test_a_flat_ranking_is_reported_as_a_tie() -> None:
    """Isotonic calibration plateaus; an argmax over ties is not a preference."""
    case = make_case([0.7111, 0.7111, 0.7111, 0.5143, 0.30, 0.20])

    assert case.is_tie
    assert not case.switches
    assert case.margin == pytest.approx(0.0)


def test_a_real_gap_is_not_a_tie() -> None:
    case = make_case([0.50, 0.44, 0.20, 0.10, 0.10, 0.10], global_best=2)

    assert not case.is_tie
    assert case.switches
    assert case.gain_over_global == pytest.approx(0.30)


def test_tie_threshold_is_the_boundary() -> None:
    just_under = make_case([0.10 + TIE_THRESHOLD * 0.9, 0.10, 0.05])
    just_over = make_case([0.10 + TIE_THRESHOLD * 1.1, 0.10, 0.05])

    assert just_under.is_tie
    assert not just_over.is_tie


def test_switching_needs_both_a_different_arm_and_a_real_gap() -> None:
    """A different argmax inside the noise floor is not personalisation."""
    noisy = make_case([0.1001, 0.1000, 0.05], global_best=1)

    assert noisy.recommended != noisy.global_best
    assert noisy.is_tie
    assert not noisy.switches, "empate não pode ser vendido como troca de braço"


def test_explanation_of_a_tie_says_so(golden) -> None:
    _, space, _ = golden
    case = make_case([0.20, 0.20, 0.10])

    assert "empatad" in explain(case, space).lower()


def test_explanation_of_a_tie_that_still_beats_the_global_arm(golden) -> None:
    """Top two tied but both above the average-best: the call is what to avoid."""
    _, space, _ = golden
    case = make_case([0.0751, 0.0750, 0.0644, 0.05, 0.04, 0.03], global_best=2)

    text = explain(case, space)
    assert case.is_tie
    assert "empatad" in text.lower()
    assert "não" in text.lower()


def test_explanation_reads_prior_success_first(golden) -> None:
    _, space, _ = golden
    case = make_case([0.5, 0.2, 0.1], poutcome="success", campaign=7, first_contact=1)

    assert "campanha anterior" in explain(case, space)


def test_explanation_does_not_let_first_contact_swallow_fatigue(golden) -> None:
    """96% of rows are first_contact; ordering it first would hide everything else."""
    _, space, _ = golden
    case = make_case(
        [0.5, 0.2, 0.1], poutcome="nonexistent", campaign=6, first_contact=1
    )

    assert "6 ligações" in explain(case, space)


def test_table_has_one_column_per_arm(golden) -> None:
    cases, space, _ = golden
    table = golden_set_table(cases, space)

    assert len(table) == len(cases)
    for label in space.labels:
        assert label in table.columns
    assert "recomendado" in table.columns


def test_at_least_one_case_genuinely_switches(golden) -> None:
    """The set has to contain the interesting case, or it demonstrates nothing."""
    cases, _, _ = golden
    assert any(c.switches for c in cases)


def test_the_switching_case_is_worth_something(golden) -> None:
    cases, _, _ = golden
    switchers = [c for c in cases if c.switches]

    assert max(c.gain_over_global for c in switchers) > 0.02
