"""Temporal sensitivity: how much of the channel effect is really the calendar?

The campaign did not use both channels at once. Landline covers the first
quarter of the log almost exclusively, mobile takes over from the second — and
the base conversion rate climbs from 3.5% to 47% along the way. Pooling all of
it into one environment hands the channel credit for whatever the calendar did,
because period is deliberately absent from the context.

This module measures that. :func:`channel_confounding_report` puts the pooled
gap next to the gap measured *inside* the window where both channels ran, and
the difference between the two is the size of the confound.

It also holds the new-arm experiment, kept even though it came out negative:
freezing a rule on the pre-mobile evidence costs almost nothing afterwards,
because within the coexistence window the channels are close. That null result
is the reason the README does not claim adaptive decisioning would have rescued
a stale rule here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.arms import ArmSpace, arm_distribution, best_historical_arm
from src.environment import CalibratedEnvironment, build_environment
from src.evaluation import run_experiment, summarize
from src.policies import FixedArm, Policy

PolicyFactory = Callable[[np.random.Generator], Policy]


@dataclass
class NewArmScenario:
    """Outcome of the new-arm experiment.

    Attributes:
        table: Comparison of every policy under the post-introduction world.
        stale_arm: Index of the arm a fixed rule would have been pinned to.
        oracle_arm: Index of the arm that is actually best afterwards.
        available_before: Arm labels that existed in the early slice.
        environment: The calibrated environment of the later slice.
    """

    table: pd.DataFrame
    stale_arm: int
    oracle_arm: int
    available_before: tuple[str, ...]
    environment: CalibratedEnvironment

    @property
    def stale_cvr(self) -> float:
        """Conversion of the frozen rule."""
        return self._cvr_of(f"FixedArm[{self.stale_arm}]")

    @property
    def oracle_cvr(self) -> float:
        """Conversion of a rule that already knows the new arm is better."""
        return self._cvr_of(f"FixedArm[{self.oracle_arm}]")

    def _cvr_of(self, policy: str) -> float:
        row = self.table.loc[self.table["policy"] == policy, "cvr"]
        return float(row.iloc[0]) if len(row) else float("nan")


def arms_present(df: pd.DataFrame, space: ArmSpace) -> tuple[str, ...]:
    """Arm labels that actually occur in a slice of the log."""
    seen = set(df[config.ARM_COLUMN].unique())
    return tuple(label for label in space.labels if label in seen)


def split_by_period(
    df: pd.DataFrame, periods: pd.Series, *, cut: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut the log in two along the campaign timeline.

    Args:
        df: The prepared frame.
        periods: Period index aligned with ``df``.
        cut: Share of the *periods* — not of the rows — that go to the early
            slice. Cutting on periods keeps whole months on one side.

    Returns:
        ``(early, late)``.
    """
    if not 0 < cut < 1:
        raise ValueError(f"cut precisa estar em (0, 1), recebido {cut}.")

    threshold = periods.quantile(cut)
    return df[periods <= threshold].copy(), df[periods > threshold].copy()


def run_new_arm_scenario(
    early: pd.DataFrame,
    late: pd.DataFrame,
    space: ArmSpace,
    extra_policies: dict[str, PolicyFactory],
    *,
    n_rounds: int = config.N_ROUNDS,
    seeds: Sequence[int] | None = None,
) -> NewArmScenario:
    """Freeze a rule on the early evidence, then let the world move on.

    The environment is calibrated on the *late* slice, where every arm exists.
    The fixed rule, however, is chosen using only what the early slice could
    have shown — which is the whole point: it cannot pick an arm it never saw.

    Args:
        early: Rows from before the new arm appeared.
        late: Rows from after, used to fit and evaluate the environment.
        space: The full arm space.
        extra_policies: Adaptive policies to run alongside the two fixed rules.
        n_rounds: Rounds per episode.
        seeds: Seeds to average over.

    Returns:
        The comparison, plus which arm each rule ended up on.
    """
    seed_list = list(range(config.N_SEEDS) if seeds is None else seeds)

    stale_arm = best_historical_arm(early, space)
    late_train, late_test = _stratified_halves(late)
    environment, _ = build_environment(late_train, late_test, space)
    oracle_arm = best_historical_arm(late_train, space)

    lineup: dict[str, PolicyFactory] = {
        f"FixedArm[{stale_arm}]": lambda rng: FixedArm(
            stale_arm, space.n_arms, rng=rng
        ),
        f"FixedArm[{oracle_arm}]": lambda rng: FixedArm(
            oracle_arm, space.n_arms, rng=rng
        ),
        **extra_policies,
    }

    results = [
        run_experiment(factory, environment, n_rounds=n_rounds, seeds=seed_list)
        for factory in lineup.values()
    ]
    table = summarize(results, baseline=f"FixedArm[{stale_arm}]")

    return NewArmScenario(
        table=table,
        stale_arm=stale_arm,
        oracle_arm=oracle_arm,
        available_before=arms_present(early, space),
        environment=environment,
    )


def _stratified_halves(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split of a slice, stratified by target and arm."""
    from sklearn.model_selection import train_test_split

    strata = (
        df[config.TARGET_BINARY].astype(str) + "|" + df[config.ARM_COLUMN].astype(str)
    )
    return train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        shuffle=True,
        stratify=strata,
    )


def logged_mixture(df: pd.DataFrame, space: ArmSpace) -> np.ndarray:
    """Arm mixture of a slice, for building its logging-policy baseline."""
    return arm_distribution(df, space)


@dataclass(frozen=True)
class ChannelConfounding:
    """Size of the temporal confound in the channel effect.

    Attributes:
        by_arm: Observed conversion per arm, pooled and inside the coexistence
            window, side by side.
        pooled_gap: Relative advantage of mobile over landline across the whole
            log.
        coexistence_gap: The same advantage measured only where both channels
            actually ran.
        inflation: How many times larger the pooled gap is. Anything well above
            1 means the pooled number is crediting the channel for the calendar.
        coexistence_rows: Rows in the coexistence window.
    """

    by_arm: pd.DataFrame
    pooled_gap: float
    coexistence_gap: float
    inflation: float
    coexistence_rows: int


def channel_confounding_report(
    df: pd.DataFrame,
    periods: pd.Series,
    space: ArmSpace,
    *,
    channel_column: str = "contact",
    cut: float = 0.25,
) -> ChannelConfounding:
    """Compare the channel effect pooled against the effect where both ran.

    The pooled comparison is the one the main environment sees, and it is
    contaminated: landline is almost entirely confined to the opening months,
    when the campaign converted at a third of its later rate. Restricting to the
    window where both channels were live removes the calendar from the
    comparison, at the cost of dropping the early rows.

    Args:
        df: The prepared frame.
        periods: Period index aligned with ``df``.
        space: The arm space, used to keep the per-arm table in index order.
        channel_column: The coarse action dimension to contrast.
        cut: Share of periods treated as the pre-introduction window.

    Returns:
        The two gaps and the per-arm breakdown behind them.
    """
    _, late = split_by_period(df, periods, cut=cut)

    def _gap(frame: pd.DataFrame) -> float:
        rates = frame.groupby(channel_column, observed=True)[
            config.TARGET_BINARY
        ].mean()
        if len(rates) < 2:
            return float("nan")
        return float(rates.max() / rates.min() - 1.0)

    pooled = df.groupby(config.ARM_COLUMN, observed=True)[config.TARGET_BINARY].mean()
    coexistence = late.groupby(config.ARM_COLUMN, observed=True)[
        config.TARGET_BINARY
    ].mean()

    by_arm = pd.DataFrame(
        {
            "arm": list(space.labels),
            "cvr_pooled": [pooled.get(label, float("nan")) for label in space.labels],
            "cvr_coexistence": [
                coexistence.get(label, float("nan")) for label in space.labels
            ],
        }
    )
    by_arm["delta"] = by_arm["cvr_coexistence"] - by_arm["cvr_pooled"]

    pooled_gap = _gap(df)
    coexistence_gap = _gap(late)

    return ChannelConfounding(
        by_arm=by_arm,
        pooled_gap=pooled_gap,
        coexistence_gap=coexistence_gap,
        inflation=pooled_gap / coexistence_gap if coexistence_gap else float("nan"),
        coexistence_rows=len(late),
    )
