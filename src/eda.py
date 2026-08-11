"""Descriptive analysis behind the Phase 1 decisions.

Kept apart from ``data.py`` on purpose: that module is on the import path of the
serving API, and it must not pull matplotlib into the container.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src import config


def set_style() -> None:
    """Apply the project plot style. Called explicitly — never on import."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 110


def wilson_interval(
    successes: np.ndarray | int,
    total: np.ndarray | int,
    *,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion, vectorised.

    Preferred over the normal approximation because arm cells with few
    conversions would otherwise get intervals running past 0.
    """
    k = np.asarray(successes, dtype=float)
    n = np.asarray(total, dtype=float)
    z = norm.ppf(0.5 + confidence / 2)

    denominator = n + z**2
    center = (k + z**2 / 2) / denominator
    half = (z / denominator) * np.sqrt(k * (n - k) / n + z**2 / 4)
    return center - half, center + half


def cardinality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Type, distinct-value count and example values, one row per column."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_unique": df.nunique(),
            "n_missing": df.isna().sum(),
            "exemplos": [
                ", ".join(map(str, df[c].drop_duplicates().head(3))) for c in df.columns
            ],
        }
    )


def unknown_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and share of the ``unknown`` token, for the columns that carry it."""
    counts = {
        column: int((df[column] == config.UNKNOWN_TOKEN).sum())
        for column in df.columns
        if df[column].dtype == object
    }
    report = pd.DataFrame(
        {"n_unknown": pd.Series(counts, dtype=int)},
    )
    report["share"] = report["n_unknown"] / len(df)
    return report[report["n_unknown"] > 0].sort_values("share", ascending=False)


def conversion_by(
    df: pd.DataFrame,
    by: str | Sequence[str],
    *,
    target: str = config.TARGET_BINARY,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Events, conversions, CVR and Wilson interval per group."""
    grouped = df.groupby(list([by]) if isinstance(by, str) else list(by), observed=True)
    table = grouped[target].agg(n="size", conversions="sum")
    table["cvr"] = table["conversions"] / table["n"]

    low, high = wilson_interval(
        table["conversions"].to_numpy(), table["n"].to_numpy(), confidence=confidence
    )
    table["cvr_low"] = low
    table["cvr_high"] = high
    return table.sort_values("cvr", ascending=False)


def arm_support(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    target: str = config.TARGET_BINARY,
) -> pd.DataFrame:
    """Per-cell support for a candidate arm space, worst cell first."""
    table = conversion_by(df, columns, target=target)
    return table.sort_values("n")


def screen_arm_spaces(
    df: pd.DataFrame,
    candidates: Sequence[Sequence[str]] = config.ARM_SPACE_CANDIDATES,
    *,
    min_events: int = config.MIN_EVENTS_PER_ARM,
    min_conversions: int = config.MIN_CONVERSIONS_PER_ARM,
    target: str = config.TARGET_BINARY,
) -> pd.DataFrame:
    """Apply the support floor to each candidate arm space.

    This is the function that decides the arm space: the finest candidate whose
    *worst* cell clears both floors wins.

    Returns:
        One row per candidate, with the worst cell observed and whether it passes.
    """
    rows = []
    for columns in candidates:
        table = arm_support(df, list(columns), target=target)
        rows.append(
            {
                "espaco": " × ".join(columns),
                "n_bracos": len(table),
                "min_eventos": int(table["n"].min()),
                "min_conversoes": int(table["conversions"].min()),
                "cvr_min": float(table["cvr"].min()),
                "cvr_max": float(table["cvr"].max()),
                "passa": bool(
                    table["n"].min() >= min_events
                    and table["conversions"].min() >= min_conversions
                ),
            }
        )
    return pd.DataFrame(rows)


def period_index(df: pd.DataFrame, *, column: str = "month") -> pd.Series:
    """Index the campaign periods by walking the file in order.

    The dataset has no date column. Months repeat across the 2008-2010 campaign,
    so calendar order is not chronological order — but a change of month in file
    order marks a new period. See ``month_run_count`` for the evidence that file
    order is chronological in the first place.
    """
    changed = df[column].ne(df[column].shift())
    return changed.cumsum() - 1


def month_run_count(df: pd.DataFrame, *, column: str = "month") -> int:
    """Number of runs of consecutive equal months in file order.

    A chronologically ordered file yields one run per campaign period; a shuffled
    one yields roughly ``n * (1 - 1/k)``.
    """
    return int(df[column].ne(df[column].shift()).sum())


def duration_leakage(
    df: pd.DataFrame,
    *,
    seed: int = config.SEED,
    test_size: float = config.TEST_SIZE,
) -> pd.DataFrame:
    """Quantify the leak: AUC with ``duration``, without it, and from it alone.

    The statement bans the column; this turns that ban into a number.
    """
    target = (df[config.TARGET] == config.TARGET_POSITIVE).astype(int)
    features = df.drop(columns=[config.TARGET])

    encoded = features.copy()
    for column in encoded.columns:
        if encoded[column].dtype == object:
            encoded[column] = encoded[column].astype("category")

    x_train, x_test, y_train, y_test = train_test_split(
        encoded, target, test_size=test_size, random_state=seed, stratify=target
    )

    variants = {
        "com duration": list(encoded.columns),
        "sem duration": [c for c in encoded.columns if c != "duration"],
        "só duration": ["duration"],
    }

    rows = []
    for label, columns in variants.items():
        model = HistGradientBoostingClassifier(
            random_state=seed, categorical_features="from_dtype"
        )
        model.fit(x_train[columns], y_train)
        score = model.predict_proba(x_test[columns])[:, 1]
        rows.append(
            {
                "conjunto": label,
                "n_features": len(columns),
                "auc": roc_auc_score(y_test, score),
            }
        )

    return pd.DataFrame(rows)


def contact_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each channel per campaign period — the confounding check.

    If the channel mix drifts with the calendar, part of what looks like a
    channel effect is really the period it was used in.
    """
    ordered = df.assign(periodo=period_index(df))
    counts = pd.crosstab(
        [ordered["periodo"], ordered["month"]], ordered["contact"], normalize="index"
    )
    counts["n"] = pd.crosstab(
        [ordered["periodo"], ordered["month"]], ordered["contact"]
    ).sum(axis=1)
    return counts


def macro_calendar_report(
    df: pd.DataFrame, *, columns: Sequence[str] = config.MACRO_COLUMNS
) -> pd.DataFrame:
    """Test whether each macro indicator is really a calendar stamp.

    ``periodos_por_valor == 1`` means the period is a function of the indicator:
    knowing the value tells you exactly when the call happened.
    """
    periodo = period_index(df)
    rows = []
    for column in columns:
        pair = pd.DataFrame({"valor": df[column], "periodo": periodo})
        grouped = pair.groupby("periodo", observed=True)["valor"]

        total = float(((pair["valor"] - pair["valor"].mean()) ** 2).sum())
        within = float(((pair["valor"] - grouped.transform("mean")) ** 2).sum())

        rows.append(
            {
                "indicador": column,
                "n_valores": int(df[column].nunique()),
                "periodos_por_valor_max": int(
                    pair.groupby("valor", observed=True)["periodo"].nunique().max()
                ),
                "r2_periodo": 1.0 - within / total,
                "corr_spearman_periodo": float(
                    df[column].corr(periodo, method="spearman")
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_conversion(
    table: pd.DataFrame, *, title: str, xlabel: str = "conversão"
) -> Figure:
    """Horizontal bars with the Wilson interval as error bars."""
    labels = [
        " | ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in table.index
    ]
    positions = np.arange(len(table))
    errors = np.vstack(
        [table["cvr"] - table["cvr_low"], table["cvr_high"] - table["cvr"]]
    )

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(table) + 1.6))
    ax.barh(positions, table["cvr"], xerr=errors, color=sns.color_palette("deep")[0])
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    # Rótulo depois do limite superior do intervalo, senão colide com a barra de erro.
    offset = table["cvr_high"].max() * 0.03
    labelled = zip(table["cvr_high"], table["n"], strict=True)
    for position, (high, n) in enumerate(labelled):
        ax.text(
            high + offset,
            position,
            f"n={n:,}".replace(",", "."),
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, table["cvr_high"].max() * 1.22)

    fig.tight_layout()
    return fig


def plot_contact_over_time(df: pd.DataFrame) -> Figure:
    """Channel mix across campaign periods."""
    shares = contact_by_month(df).drop(columns="n")
    labels = [f"{p}·{m}" for p, m in shares.index]

    fig, ax = plt.subplots(figsize=(11, 3.6))
    bottom = np.zeros(len(shares))
    for column in shares.columns:
        ax.bar(labels, shares[column], bottom=bottom, label=column)
        bottom += shares[column].to_numpy()
    ax.set_ylabel("participação")
    ax.set_xlabel("período · mês")
    ax.set_title("Mix de canal ao longo da campanha")
    ax.legend(loc="lower left")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    return fig


def plot_macro_over_time(
    df: pd.DataFrame, *, columns: Sequence[str] = ("euribor3m", "nr.employed")
) -> Figure:
    """Macro indicators against campaign period — the step shape is the point."""
    periodo = period_index(df)

    fig, axes = plt.subplots(
        len(columns), 1, figsize=(9, 2.4 * len(columns)), sharex=True
    )
    for ax, column in zip(np.atleast_1d(axes), columns, strict=True):
        ax.plot(periodo.to_numpy(), df[column].to_numpy(), linewidth=1.4)
        ax.set_ylabel(column)
    np.atleast_1d(axes)[-1].set_xlabel("período da campanha")
    np.atleast_1d(axes)[0].set_title("Indicadores macro no tempo")
    fig.tight_layout()
    return fig


def plot_duration_leakage(df: pd.DataFrame) -> Figure:
    """Duration distribution split by outcome, on a log scale."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for label, color in (("no", "#8c8c8c"), ("yes", "#2a7fb8")):
        subset = df.loc[df[config.TARGET] == label, "duration"]
        ax.hist(
            subset[subset > 0],
            bins=60,
            alpha=0.65,
            label=f"y = {label}",
            color=color,
            log=True,
        )
    ax.set_xlabel("duração da ligação (s)")
    ax.set_ylabel("frequência (log)")
    ax.set_title("`duration` separa o desfecho porque é medida depois dele")
    ax.legend()
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, name: str, *, directory: Path | None = None) -> Path:
    """Write a figure to ``reports/figures`` and close it."""
    target = (directory or config.FIGURES_DIR) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target
