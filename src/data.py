"""Loading and preparation of the bank-marketing dataset."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


class SchemaError(ValueError):
    """Raised when the raw file does not match the expected schema."""


def load_raw(path: Path | str | None = None, *, validate: bool = True) -> pd.DataFrame:
    """Load the raw CSV with the correct separator.

    The UCI file is semicolon-separated. Reading it with the default comma yields a
    single-column DataFrame *without* raising, so the parse is validated rather than
    trusted.

    Args:
        path: Override for the raw CSV location. Defaults to ``config.RAW_CSV``.
        validate: Whether to check the parsed columns against the expected schema.

    Returns:
        The raw dataset, untouched apart from parsing.

    Raises:
        FileNotFoundError: If the file is missing, with a hint to run ``make data``.
        SchemaError: If the parsed columns do not match ``config.RAW_COLUMNS``.
    """
    csv_path = Path(path) if path is not None else config.RAW_CSV

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} não encontrado. Rode `make data` para baixar a base."
        )

    df = pd.read_csv(csv_path, sep=config.RAW_SEPARATOR)

    if validate:
        _validate_schema(df)

    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Check that the parse produced the expected columns."""
    if len(df.columns) == 1:
        raise SchemaError(
            "O arquivo foi lido como uma única coluna — separador errado. "
            f"O esperado é {config.RAW_SEPARATOR!r}."
        )

    actual = tuple(df.columns)
    if actual != config.RAW_COLUMNS:
        missing = set(config.RAW_COLUMNS) - set(actual)
        extra = set(actual) - set(config.RAW_COLUMNS)
        raise SchemaError(
            f"Schema inesperado. Faltando: {sorted(missing) or 'nenhuma'}. "
            f"Inesperadas: {sorted(extra) or 'nenhuma'}."
        )


def drop_forbidden(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns banned by the challenge statement.

    ``duration`` is only known after the call ends, so using it leaks the outcome.
    """
    present = [c for c in config.FORBIDDEN_COLUMNS if c in df.columns]
    return df.drop(columns=present)


def binarize_target(df: pd.DataFrame) -> pd.Series:
    """Convert the ``y`` column to a 0/1 integer Series."""
    return (df[config.TARGET] == config.TARGET_POSITIVE).astype(int)


def add_week_window(df: pd.DataFrame) -> pd.DataFrame:
    """Add the contact window, collapsing ``day_of_week`` into three levels.

    Raises:
        SchemaError: If a day falls outside ``config.WEEK_WINDOWS``.
    """
    unmapped = set(df["day_of_week"].unique()) - set(config.WEEK_WINDOWS)
    if unmapped:
        raise SchemaError(f"Dias sem janela definida: {sorted(unmapped)}.")

    out = df.copy()
    out[config.WEEK_WINDOW_COLUMN] = df["day_of_week"].map(config.WEEK_WINDOWS)
    return out


def add_first_contact(df: pd.DataFrame) -> pd.DataFrame:
    """Add a flag for ``pdays == 999``, the sentinel for a first-ever contact."""
    out = df.copy()
    is_first = df["pdays"] == config.PDAYS_SENTINEL
    out[config.FIRST_CONTACT_COLUMN] = is_first.astype(int)
    return out


def build_arm(
    df: pd.DataFrame, *, columns: Sequence[str] = config.ARM_COLUMNS
) -> pd.Series:
    """Compose the arm label by joining the action dimensions with ``|``.

    Args:
        df: Frame already carrying every column in ``columns``.
        columns: The action dimensions that define the arm space.

    Returns:
        A Series of arm labels aligned with ``df.index``.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(f"Colunas de braço ausentes: {missing}.")

    parts = [df[c].astype(str) for c in columns]
    return parts[0].str.cat(parts[1:], sep="|") if len(parts) > 1 else parts[0]


def prepare(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Deterministic cleaning pipeline, from raw log to modelling frame.

    Drops the forbidden column, derives the contact window, the first-contact
    flag, the arm label and the binary target. ``unknown`` is left untouched:
    in this dataset it is a recorded answer, not a missing value.

    Args:
        df: Raw frame. Loaded from ``config.RAW_CSV`` when omitted.

    Returns:
        A new frame; the input is never mutated.
    """
    raw = load_raw() if df is None else df

    out = drop_forbidden(raw)
    out = add_week_window(out)
    out = add_first_contact(out)
    out[config.ARM_COLUMN] = build_arm(out)
    out[config.TARGET_BINARY] = binarize_target(out)
    return out


def split_train_test(
    df: pd.DataFrame,
    *,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and test, stratified by target *and* arm.

    Stratifying on the target alone would let a low-volume arm land almost
    entirely on one side; the calibrated environment of the next phase needs
    every arm represented in both folds to estimate ``P(y | context, arm)``.

    Args:
        df: A frame produced by ``prepare``.
        test_size: Share of rows held out.
        seed: Propagated to ``train_test_split`` for reproducibility.

    Returns:
        ``(train, test)``, both preserving the original index.
    """
    strata = (
        df[config.TARGET_BINARY].astype(str) + "|" + df[config.ARM_COLUMN].astype(str)
    )
    return train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True, stratify=strata
    )
