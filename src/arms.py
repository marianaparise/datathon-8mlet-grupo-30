"""The arm space: the set of actions a policy can choose from.

An arm here is a campaign decision — channel crossed with contact window — not a
client attribute. Phase 1 fixed the space at ``config.ARM_COLUMNS`` by observed
support; this module turns it into a first-class object with a stable ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


class UnknownArmError(KeyError):
    """Raised when a label has no index in the arm space."""


@dataclass(frozen=True)
class ArmSpace:
    """The ordered set of arms, plus the label ↔ index mapping.

    The ordering is fixed at construction and never re-derived: policies, the
    environment and the serialized artefacts all index arms by position, so a
    reordering between runs would silently mislabel every result.

    Attributes:
        labels: Arm labels in index order.
    """

    labels: tuple[str, ...]

    @classmethod
    def from_frame(
        cls, df: pd.DataFrame, *, column: str = config.ARM_COLUMN
    ) -> ArmSpace:
        """Build the space from the arms present in a frame, sorted by label."""
        if column not in df.columns:
            raise UnknownArmError(f"Coluna de braço ausente: {column!r}.")
        return cls(tuple(sorted(df[column].dropna().unique())))

    def __len__(self) -> int:
        """Number of arms."""
        return len(self.labels)

    @property
    def n_arms(self) -> int:
        """Number of arms, named for readability at call sites."""
        return len(self.labels)

    def index(self, label: str) -> int:
        """Position of ``label`` in the space.

        Raises:
            UnknownArmError: If the label is not part of the space.
        """
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise UnknownArmError(
                f"Braço {label!r} fora do espaço {self.labels}."
            ) from exc

    def label(self, index: int) -> str:
        """Label at position ``index``.

        Raises:
            UnknownArmError: If the index is out of range.
        """
        if not 0 <= index < len(self.labels):
            raise UnknownArmError(f"Índice {index} fora de [0, {len(self.labels)}).")
        return self.labels[index]

    def encode(self, labels: pd.Series) -> np.ndarray:
        """Map a Series of labels to their integer indices."""
        lookup = {label: i for i, label in enumerate(self.labels)}
        unknown = set(labels.dropna().unique()) - lookup.keys()
        if unknown:
            raise UnknownArmError(f"Braços fora do espaço: {sorted(unknown)}.")
        return labels.map(lookup).to_numpy(dtype=int)


def arm_distribution(
    df: pd.DataFrame, space: ArmSpace, *, column: str = config.ARM_COLUMN
) -> np.ndarray:
    """Empirical share of each arm in the log, in arm-index order.

    This is the logging policy: the mixture the bank actually played. It is the
    project's main baseline, so it comes from the data rather than from a guess.

    Returns:
        Probabilities summing to 1, shape ``(n_arms,)``.
    """
    counts = df[column].value_counts()
    shares = np.array([counts.get(label, 0) for label in space.labels], dtype=float)
    total = shares.sum()
    if total == 0:
        raise UnknownArmError("Nenhum braço do espaço aparece no frame.")
    return shares / total


def best_historical_arm(
    df: pd.DataFrame,
    space: ArmSpace,
    *,
    column: str = config.ARM_COLUMN,
    target: str = config.TARGET_BINARY,
) -> int:
    """Index of the arm with the highest observed conversion rate.

    The hard comparator: a non-contextual bandit converges here, so any claimed
    contextual gain has to be measured against this arm, not against the log.
    """
    rates = df.groupby(column, observed=True)[target].mean()
    ordered = [rates.get(label, float("-inf")) for label in space.labels]
    return int(np.argmax(ordered))
