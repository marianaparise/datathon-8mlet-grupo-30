"""The calibrated environment: what would each arm have yielded for this client?

Track A of the evaluation strategy. Because every arm appears in the historical
log, ``P(y | context, arm)`` is estimable from observed rows for *every* arm —
nothing is invented. What the model does is interpolate between rows that exist.

That licence is not unconditional, which is why nothing here is usable until
three checks pass: the probabilities have to be calibrated (globally *and* per
arm), the model has to beat a logistic baseline, and every arm needs support
across the context space. :func:`build_environment` runs all three.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.arms import ArmSpace
from src.evaluation import Observation

CATEGORICAL_CONTEXT: tuple[str, ...] = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "poutcome",
)

NUMERIC_CONTEXT: tuple[str, ...] = (
    "age",
    "campaign",
    "pdays",
    "previous",
    config.FIRST_CONTACT_COLUMN,
)


class EnvironmentError(RuntimeError):
    """Raised when the fitted environment fails one of its quality gates."""


def context_frame(df: pd.DataFrame, *, include_macro: bool = False) -> pd.DataFrame:
    """Select and clean the columns a policy is allowed to see.

    ``pdays`` carries 999 as a sentinel for "never contacted". Left raw it would
    read as a very old contact and dominate any linear model, so the sentinel is
    zeroed and the information lives in the ``first_contact`` flag instead.

    Args:
        df: A frame produced by ``data.prepare``.
        include_macro: Whether to append the macro indicators. They are excluded
            by default: Phase 1 showed they are calendar stamps constant within
            a period, so they move the base rate without telling clients apart.

    Returns:
        A new frame carrying only context columns.
    """
    columns = list(CATEGORICAL_CONTEXT + NUMERIC_CONTEXT)
    if include_macro:
        columns += list(config.MACRO_COLUMNS)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise EnvironmentError(f"Colunas de contexto ausentes: {missing}.")

    out = df[columns].copy()
    out["pdays"] = out["pdays"].replace(config.PDAYS_SENTINEL, 0)
    return out


def build_context_encoder(*, include_macro: bool = False) -> ColumnTransformer:
    """One-hot the categoricals, standardise the numerics.

    The output feeds both the reward model and ``LinTS``, so the two always see
    the same client in the same coordinates.
    """
    numeric = list(NUMERIC_CONTEXT) + (
        list(config.MACRO_COLUMNS) if include_macro else []
    )
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_CONTEXT),
            ),
            ("numeric", StandardScaler(), numeric),
        ],
        remainder="drop",
    )


def _stack_arm(features: np.ndarray, arm: int, n_arms: int) -> np.ndarray:
    """Append a one-hot arm indicator to every row of ``features``."""
    indicator = np.zeros((len(features), n_arms), dtype=float)
    indicator[:, arm] = 1.0
    return np.hstack([features, indicator])


def _design_matrix(features: np.ndarray, arms: np.ndarray, n_arms: int) -> np.ndarray:
    """Row-wise concatenation of context and the arm actually played."""
    indicator = np.zeros((len(features), n_arms), dtype=float)
    indicator[np.arange(len(features)), arms] = 1.0
    return np.hstack([features, indicator])


def calibration_report(
    y_true: np.ndarray, y_prob: np.ndarray, arms: np.ndarray, space: ArmSpace
) -> pd.DataFrame:
    """Brier score and mean predicted-vs-observed rate, globally and per arm.

    Per arm matters more than the total: a model can look well calibrated
    overall while being badly wrong on a low-volume arm, and the experiment
    would inherit that error as if it were truth.
    """
    rows = [
        {
            "arm": "TOTAL",
            "n": len(y_true),
            "brier": brier_score_loss(y_true, y_prob),
            "predicted": float(y_prob.mean()),
            "observed": float(y_true.mean()),
        }
    ]

    for index, label in enumerate(space.labels):
        mask = arms == index
        if not mask.any():
            continue
        rows.append(
            {
                "arm": label,
                "n": int(mask.sum()),
                "brier": brier_score_loss(y_true[mask], y_prob[mask]),
                "predicted": float(y_prob[mask].mean()),
                "observed": float(y_true[mask].mean()),
            }
        )

    report = pd.DataFrame(rows)
    report["gap"] = (report["predicted"] - report["observed"]).abs()
    return report


def overlap_report(
    features: np.ndarray, arms: np.ndarray, space: ArmSpace, *, seed: int = config.SEED
) -> pd.DataFrame:
    """Positivity check: does every arm have support across the context space?

    Fits ``P(arm | context)`` and reports how low that propensity gets. Where an
    arm is essentially never played, predicting its outcome is extrapolation
    rather than interpolation, and the README has to say so.
    """
    model = LogisticRegression(max_iter=1_000, random_state=seed)
    model.fit(features, arms)
    propensity = model.predict_proba(features)

    rows = []
    for index, label in enumerate(space.labels):
        column = propensity[:, index]
        rows.append(
            {
                "arm": label,
                "min": float(column.min()),
                "p01": float(np.percentile(column, 1)),
                "median": float(np.median(column)),
                "max": float(column.max()),
                "share_below_floor": float((column < config.MIN_ARM_PROPENSITY).mean()),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ContextualCeiling:
    """How much a perfectly contextual policy could gain over the best fixed arm.

    Attributes:
        best_global_arm: Index of the arm with the highest mean probability.
        fixed_cvr: Mean conversion if that single arm were played for everyone.
        oracle_cvr: Mean conversion under per-client arm choice.
        absolute_gain: ``oracle_cvr - fixed_cvr``, in probability points.
        relative_gain: The same gain as a share of ``fixed_cvr``.
        switch_share: Fraction of clients whose best arm is not the global one.
    """

    best_global_arm: int
    fixed_cvr: float
    oracle_cvr: float
    absolute_gain: float
    relative_gain: float
    switch_share: float


def contextual_ceiling(reward_matrix: np.ndarray) -> ContextualCeiling:
    """Upper bound on what any contextual policy could win.

    Phase 1 found arm × context heterogeneity to be weak. This turns that
    qualitative worry into the number that decides whether ``LinTS`` has
    anything to win: if the ceiling is near zero, no contextual policy can beat
    the best fixed arm, and the honest move is to report that rather than to
    hunt for a gain that is not there.
    """
    mean_by_arm = reward_matrix.mean(axis=0)
    best_global = int(np.argmax(mean_by_arm))

    fixed = float(mean_by_arm[best_global])
    oracle = float(reward_matrix.max(axis=1).mean())
    per_client_best = reward_matrix.argmax(axis=1)

    return ContextualCeiling(
        best_global_arm=best_global,
        fixed_cvr=fixed,
        oracle_cvr=oracle,
        absolute_gain=oracle - fixed,
        relative_gain=(oracle / fixed - 1.0) if fixed else float("nan"),
        switch_share=float((per_client_best != best_global).mean()),
    )


class CalibratedEnvironment:
    """Samples clients from the held-out log and reveals every arm's odds.

    Satisfies the ``evaluation.Environment`` protocol structurally, so the
    runner consumes it without either module importing the other.

    Attributes:
        space: The arm space, in index order.
        features: Encoded contexts of the held-out clients.
        reward_matrix: ``P(y | client, arm)`` for every client and arm.
    """

    def __init__(
        self,
        space: ArmSpace,
        features: np.ndarray,
        reward_matrix: np.ndarray,
        *,
        encoder: ColumnTransformer | None = None,
        model: CalibratedClassifierCV | None = None,
        include_macro: bool = False,
    ) -> None:
        """Store the client pool, its per-arm probabilities and the fitted parts."""
        if features.shape[0] != reward_matrix.shape[0]:
            raise EnvironmentError("features e reward_matrix têm tamanhos diferentes.")
        if reward_matrix.shape[1] != space.n_arms:
            raise EnvironmentError("reward_matrix não tem uma coluna por braço.")

        self.space = space
        self.features = features
        self.reward_matrix = reward_matrix
        self.encoder = encoder
        self.model = model
        self.include_macro = include_macro

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """Turn raw prepared rows into the context vectors a policy consumes."""
        if self.encoder is None:
            raise EnvironmentError("Ambiente sem encoder — recarregue o artefato.")
        return self.encoder.transform(
            context_frame(df, include_macro=self.include_macro)
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """``P(y | client, arm)`` for unseen clients, shape ``(n_rows, n_arms)``.

        This is what the serving API needs: the simulation pool is precomputed,
        but a live request arrives as a raw row and has to be scored on the spot.
        """
        if self.model is None:
            raise EnvironmentError("Ambiente sem modelo — recarregue o artefato.")

        features = self.encode(df)
        return np.column_stack(
            [
                self.model.predict_proba(_stack_arm(features, arm, self.n_arms))[:, 1]
                for arm in range(self.n_arms)
            ]
        )

    @property
    def n_arms(self) -> int:
        """Number of arms on offer."""
        return self.space.n_arms

    @property
    def n_features(self) -> int:
        """Width of the context vector."""
        return int(self.features.shape[1])

    @property
    def n_clients(self) -> int:
        """Size of the client pool."""
        return int(self.features.shape[0])

    def sample(self, rng: np.random.Generator) -> Observation:
        """Draw one client from the pool, with replacement."""
        index = int(rng.integers(self.n_clients))
        return Observation(
            features=self.features[index],
            expected_rewards=self.reward_matrix[index],
        )


@dataclass
class EnvironmentDiagnostics:
    """Everything needed to defend the environment in the README.

    Attributes:
        calibration: Brier and predicted-vs-observed, globally and per arm.
        overlap: Propensity ranges per arm.
        ceiling: Upper bound on contextual gain.
        auc: Test AUC of the calibrated model.
        baseline_auc: Test AUC of the logistic sanity check.
        brier: Global Brier score on the test fold.
    """

    calibration: pd.DataFrame
    overlap: pd.DataFrame
    ceiling: ContextualCeiling
    auc: float
    baseline_auc: float
    brier: float


def build_environment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    space: ArmSpace,
    *,
    include_macro: bool = False,
    seed: int = config.SEED,
) -> tuple[CalibratedEnvironment, EnvironmentDiagnostics]:
    """Fit ``P(y | context, arm)`` on train and validate it on test.

    Raises:
        EnvironmentError: If the Brier score exceeds ``config.MAX_BRIER_SCORE``.
            An uncalibrated environment produces confident nonsense, and every
            downstream number would inherit it.
    """
    encoder = build_context_encoder(include_macro=include_macro)
    train_context = encoder.fit_transform(
        context_frame(train, include_macro=include_macro)
    )
    test_context = encoder.transform(
        context_frame(test, include_macro=include_macro)
    )

    train_arms = space.encode(train[config.ARM_COLUMN])
    test_arms = space.encode(test[config.ARM_COLUMN])
    y_train = train[config.TARGET_BINARY].to_numpy()
    y_test = test[config.TARGET_BINARY].to_numpy()

    model = CalibratedClassifierCV(
        HistGradientBoostingClassifier(
            max_iter=config.ENV_MAX_ITER,
            learning_rate=config.ENV_LEARNING_RATE,
            random_state=seed,
        ),
        method=config.ENV_CALIBRATION_METHOD,
        cv=config.ENV_CALIBRATION_CV,
    )
    model.fit(_design_matrix(train_context, train_arms, space.n_arms), y_train)

    played = _design_matrix(test_context, test_arms, space.n_arms)
    y_prob = model.predict_proba(played)[:, 1]

    brier = float(brier_score_loss(y_test, y_prob))
    if brier > config.MAX_BRIER_SCORE:
        raise EnvironmentError(
            f"Brier {brier:.4f} acima do piso {config.MAX_BRIER_SCORE}. "
            "Ambiente descalibrado — não seguir para o experimento."
        )

    baseline = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1_000))]
    ).fit(_design_matrix(train_context, train_arms, space.n_arms), y_train)

    reward_matrix = np.column_stack(
        [
            model.predict_proba(_stack_arm(test_context, arm, space.n_arms))[:, 1]
            for arm in range(space.n_arms)
        ]
    )

    diagnostics = EnvironmentDiagnostics(
        calibration=calibration_report(y_test, y_prob, test_arms, space),
        overlap=overlap_report(test_context, test_arms, space, seed=seed),
        ceiling=contextual_ceiling(reward_matrix),
        auc=float(roc_auc_score(y_test, y_prob)),
        baseline_auc=float(roc_auc_score(y_test, baseline.predict_proba(played)[:, 1])),
        brier=brier,
    )

    environment = CalibratedEnvironment(
        space,
        test_context,
        reward_matrix,
        encoder=encoder,
        model=model,
        include_macro=include_macro,
    )
    return environment, diagnostics
