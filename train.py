"""End-to-end experiment: prepare, calibrate, compete, record.

Run with ``make train``. Produces the comparison table, the figures the README
quotes, the MLflow runs required by Etapa 7 and the serialized environment the
serving API loads.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config, data  # noqa: E402
from src.arms import ArmSpace, arm_distribution, best_historical_arm  # noqa: E402
from src.environment import (  # noqa: E402
    CalibratedEnvironment,
    EnvironmentDiagnostics,
    build_environment,
)
from src.evaluation import (  # noqa: E402
    ExperimentResult,
    log_experiment,
    run_experiment,
    start_tracking,
    summarize,
)
from src.policies import (  # noqa: E402
    UCB1,
    EpsilonGreedy,
    FixedArm,
    LinTS,
    LoggingPolicy,
    Policy,
    ThompsonSampling,
)

PolicyFactory = Callable[[np.random.Generator], Policy]


def build_policies(
    env: CalibratedEnvironment, shares: np.ndarray, best_arm: int
) -> dict[str, tuple[PolicyFactory, dict[str, float]]]:
    """The line-up, with the hyperparameters each one gets logged under.

    ``LoggingPolicy`` is the baseline: it reproduces the mixture the campaign
    actually played, which is the honest counterfactual. ``FixedArm`` on the
    best historical arm is the hard comparator — a non-contextual bandit
    converges there, so that is the bar ``LinTS`` has to clear.
    """
    n_arms, n_features = env.n_arms, env.n_features
    return {
        "LoggingPolicy": (
            lambda rng: LoggingPolicy(shares, rng=rng),
            {},
        ),
        f"FixedArm[{best_arm}]": (
            lambda rng: FixedArm(best_arm, n_arms, rng=rng),
            {"arm": best_arm},
        ),
        "EpsilonGreedy": (
            lambda rng: EpsilonGreedy(n_arms, rng=rng),
            {"epsilon": config.EPSILON},
        ),
        "UCB1": (
            lambda rng: UCB1(n_arms, rng=rng),
            {"c": config.UCB_C},
        ),
        f"ThompsonSampling[{config.TS_ALPHA_PRIOR:g},{config.TS_BETA_PRIOR:g}]": (
            lambda rng: ThompsonSampling(n_arms, rng=rng),
            {"alpha_prior": config.TS_ALPHA_PRIOR, "beta_prior": config.TS_BETA_PRIOR},
        ),
        (
            f"ThompsonSampling[{config.TS_ALPHA_PRIOR_INFORMED:g},"
            f"{config.TS_BETA_PRIOR_INFORMED:g}]"
        ): (
            lambda rng: ThompsonSampling(
                n_arms,
                rng=rng,
                alpha_prior=config.TS_ALPHA_PRIOR_INFORMED,
                beta_prior=config.TS_BETA_PRIOR_INFORMED,
            ),
            {
                "alpha_prior": config.TS_ALPHA_PRIOR_INFORMED,
                "beta_prior": config.TS_BETA_PRIOR_INFORMED,
            },
        ),
        f"LinTS[v={config.LINTS_V:g}]": (
            lambda rng: LinTS(n_arms, n_features, rng=rng),
            {"v": config.LINTS_V, "lambda": config.LINTS_LAMBDA},
        ),
    }


def plot_curves(results: list[ExperimentResult], space: ArmSpace) -> None:
    """Write the three figures the README quotes."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 110
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for result in results:
        ax.plot(result.mean_cumulative_cvr, label=result.policy, linewidth=1.4)
    ax.set_xlabel("rodada")
    ax.set_ylabel("conversão acumulada")
    ax.set_title("Conversão acumulada — média de seeds")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "cvr_acumulada.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for result in results:
        ax.plot(result.mean_cumulative_regret, label=result.policy, linewidth=1.4)
    ax.set_xlabel("rodada")
    ax.set_ylabel("regret acumulado")
    ax.set_title("Regret acumulado contra o oráculo — média de seeds")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "regret_acumulado.png")
    plt.close(fig)

    pulls = pd.DataFrame(
        {r.policy: r.mean_pull_counts / r.episodes[0].n_rounds for r in results},
        index=list(space.labels),
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    pulls.T.plot(kind="barh", stacked=True, ax=ax, width=0.75)
    ax.set_xlabel("share das puxadas")
    ax.set_title("Onde cada política gastou o tráfego")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "puxadas_por_braco.png")
    plt.close(fig)


def report_environment(diagnostics: EnvironmentDiagnostics, space: ArmSpace) -> None:
    """Print the three quality gates before anything downstream is trusted."""
    print("\n=== AMBIENTE ===")
    print(
        f"AUC {diagnostics.auc:.4f} (logística {diagnostics.baseline_auc:.4f}) | "
        f"Brier {diagnostics.brier:.4f} (piso {config.MAX_BRIER_SCORE})"
    )
    print("\ncalibração por braço:")
    print(diagnostics.calibration.round(4).to_string(index=False))
    print("\nsobreposição:")
    print(diagnostics.overlap.round(4).to_string(index=False))

    ceiling = diagnostics.ceiling
    print("\nteto do ganho contextual:")
    print(f"  melhor braço global    {space.label(ceiling.best_global_arm)}")
    print(f"  CVR braço fixo         {ceiling.fixed_cvr:.2%}")
    print(f"  CVR oráculo            {ceiling.oracle_cvr:.2%}")
    print(
        f"  ganho máximo possível  {ceiling.absolute_gain:.2%} "
        f"({ceiling.relative_gain:+.2%} relativo)"
    )
    print(f"  clientes que trocam    {ceiling.switch_share:.2%}")


def main() -> None:
    """Run the whole pipeline."""
    parser = argparse.ArgumentParser(description="Experimento de bandit do TC5.")
    parser.add_argument("--rounds", type=int, default=config.N_ROUNDS)
    parser.add_argument("--seeds", type=int, default=config.N_SEEDS)
    parser.add_argument(
        "--no-mlflow", action="store_true", help="pula o registro no MLflow"
    )
    args = parser.parse_args()

    started = time.time()

    prepared = data.prepare()
    train, test = data.split_train_test(prepared)
    space = ArmSpace.from_frame(prepared)
    shares = arm_distribution(train, space)
    best_arm = best_historical_arm(train, space)

    print(f"braços: {space.labels}")
    print(f"treino {len(train)} | teste {len(test)}")

    env, diagnostics = build_environment(train, test, space)
    report_environment(diagnostics, space)

    if not args.no_mlflow:
        start_tracking()

    print(f"\n=== EXPERIMENTO ({args.rounds} rodadas x {args.seeds} seeds) ===")
    results: list[ExperimentResult] = []
    baseline_cvr = None

    for name, (factory, params) in build_policies(env, shares, best_arm).items():
        tick = time.time()
        result = run_experiment(
            factory, env, n_rounds=args.rounds, seeds=range(args.seeds)
        )
        results.append(result)

        if baseline_cvr is None:
            baseline_cvr = float(result.cvrs.mean())

        if not args.no_mlflow:
            log_experiment(
                result,
                params=params,
                baseline_cvr=baseline_cvr,
                n_rounds=args.rounds,
            )
        print(
            f"  {name:44s} CVR {result.cvrs.mean():.4%}  "
            f"({time.time() - tick:4.1f}s)"
        )

    table = summarize(results, baseline=results[0].policy)
    print("\n=== RESULTADO ===")
    print(table.round(4).to_string(index=False))

    plot_curves(results, space)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(env, config.ENVIRONMENT_ARTIFACT)
    table.to_csv(config.MODELS_DIR / "results.csv", index=False)
    (config.MODELS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "arms": list(space.labels),
                "best_historical_arm": space.label(best_arm),
                "arm_shares": dict(zip(space.labels, shares.round(6), strict=True)),
                "n_rounds": args.rounds,
                "n_seeds": args.seeds,
                "seed": config.SEED,
                "environment": {
                    "auc": round(diagnostics.auc, 4),
                    "baseline_auc": round(diagnostics.baseline_auc, 4),
                    "brier": round(diagnostics.brier, 4),
                    "contextual_ceiling_relative": round(
                        diagnostics.ceiling.relative_gain, 4
                    ),
                    "switch_share": round(diagnostics.ceiling.switch_share, 4),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nartefatos em {config.MODELS_DIR}")
    print(f"figuras em {config.FIGURES_DIR}")
    print(f"tempo total {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
