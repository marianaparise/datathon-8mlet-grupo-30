"""Loading and preparation of the bank-marketing dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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
