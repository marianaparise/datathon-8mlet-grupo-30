"""The five worked cases the challenge asks for.

Etapa 4 wants five clients, the recommendation each one got, and whether the
decision made sense. Randomly drawn clients would not answer that — five average
customers all get the same arm and the table proves nothing.

So the five are chosen by criterion, each one exercising a different part of the
decision surface: someone with no history, someone who already said yes, someone
worn down by repeated calls, an unremarkable case, and — the interesting one —
a client whose best arm is *not* the arm that is best on average.

What is shown per client is the score of every arm, not just the winner. The
policies in this project are non-contextual, so the *policy* would hand all five
the same answer; the per-client ranking comes from the calibrated reward model,
which is the Direct Method, not a bandit. The README names that difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.arms import ArmSpace
from src.environment import CalibratedEnvironment

# Abaixo desta diferença os dois melhores braços são tratados como empatados.
# A calibração isotônica produz degraus, então clientes de alta propensão caem
# no mesmo patamar e o argmax entre eles é desempate numérico, não decisão.
# Meio ponto percentual é menor que o desvio de calibração do pior braço.
TIE_THRESHOLD = 0.005

DISPLAY_COLUMNS: tuple[str, ...] = (
    "age",
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "campaign",
    "previous",
    "poutcome",
    config.FIRST_CONTACT_COLUMN,
)


@dataclass(frozen=True)
class GoldenCase:
    """One worked example.

    Attributes:
        criterion: Why this client is in the set.
        row_label: Index of the client in the held-out frame.
        profile: Readable client attributes.
        scores: Estimated conversion of every arm, in arm-index order.
        recommended: Index of the arm with the highest score.
        global_best: Index of the arm that is best on average.
    """

    criterion: str
    row_label: int
    profile: dict[str, object]
    scores: np.ndarray
    recommended: int
    global_best: int

    @property
    def best_score(self) -> float:
        """Score of the recommended arm."""
        return float(self.scores[self.recommended])

    @property
    def worst_score(self) -> float:
        """Score of the weakest arm."""
        return float(self.scores.min())

    @property
    def runner_up_score(self) -> float:
        """Score of the second-best arm."""
        return float(np.sort(self.scores)[-2])

    @property
    def margin(self) -> float:
        """Best minus runner-up — how firmly the top arm is on top."""
        return self.best_score - self.runner_up_score

    @property
    def is_tie(self) -> bool:
        """Whether the top two arms are too close to call.

        Isotonic calibration is a step function, so several arms can land on the
        same plateau for a high-propensity client. Calling that a preference
        would be reading noise.
        """
        return self.margin < TIE_THRESHOLD

    @property
    def gain_over_global(self) -> float:
        """What choosing per client buys over always playing the average-best arm.

        This — not the margin over the runner-up — is what personalisation is
        worth for this client.
        """
        return self.best_score - float(self.scores[self.global_best])

    @property
    def switches(self) -> bool:
        """Whether the best arm differs from the global one *and* the gap is real."""
        return self.recommended != self.global_best and not self.is_tie

    @property
    def spread(self) -> float:
        """Best minus worst — how much the choice is worth for this client."""
        return self.best_score - self.worst_score


def _median_candidate(frame: pd.DataFrame, scores: pd.Series) -> int | None:
    """Pick the median-scoring row so the choice is reproducible, not cherry-picked."""
    if frame.empty:
        return None
    candidates = scores.loc[frame.index].sort_values()
    return int(candidates.index[len(candidates) // 2])


def select_golden_set(
    test: pd.DataFrame, environment: CalibratedEnvironment, space: ArmSpace
) -> list[GoldenCase]:
    """Choose five clients, each for a stated reason.

    Args:
        test: Held-out frame, already prepared.
        environment: Calibrated environment, used to score every arm per client.
        space: The arm space.

    Returns:
        Five cases in a fixed order. Fewer only if a criterion matches nobody.
    """
    matrix = environment.predict(test)
    best_per_client = matrix.argmax(axis=1)
    global_best = int(matrix.mean(axis=0).argmax())
    top_score = pd.Series(matrix.max(axis=1), index=test.index)

    switchers = test[best_per_client != global_best]
    spread = pd.Series(matrix.max(axis=1) - matrix.min(axis=1), index=test.index)

    criteria: dict[str, int | None] = {
        "Sem histórico de contato": _median_candidate(
            test[
                (test[config.FIRST_CONTACT_COLUMN] == 1)
                & (test["poutcome"] == "nonexistent")
            ],
            top_score,
        ),
        "Converteu em campanha anterior": _median_candidate(
            test[test["poutcome"] == "success"], top_score
        ),
        "Muitos contatos nesta campanha": _median_candidate(
            test[test["campaign"] >= 5], top_score
        ),
        "Perfil mediano da base": _median_candidate(test, top_score),
        "Melhor braço difere do global": (
            int(spread.loc[switchers.index].idxmax()) if len(switchers) else None
        ),
    }

    cases = []
    for criterion, row_label in criteria.items():
        if row_label is None:
            continue
        position = test.index.get_loc(row_label)
        scores = matrix[position]
        cases.append(
            GoldenCase(
                criterion=criterion,
                row_label=row_label,
                profile={
                    column: test.loc[row_label, column]
                    for column in DISPLAY_COLUMNS
                    if column in test.columns
                },
                scores=scores,
                recommended=int(scores.argmax()),
                global_best=global_best,
            )
        )
    return cases


def explain(case: GoldenCase, space: ArmSpace) -> str:
    """A business-language sentence about why this recommendation makes sense.

    Built from the numbers rather than written by hand, so it cannot drift away
    from what the model actually said.
    """
    arm = space.label(case.recommended)
    global_arm = space.label(case.global_best)

    if case.is_tie and case.gain_over_global >= TIE_THRESHOLD:
        # Empate no topo, mas os líderes batem o braço médio: a decisão útil é
        # de exclusão — sai o braço global, e entre os empatados decide o custo.
        head = (
            f"Os dois primeiros braços estão **empatados** "
            f"({case.margin * 100:.2f} p.p. entre eles), mas ambos superam "
            f"{global_arm}, o melhor na média, por "
            f"{case.gain_over_global * 100:.2f} p.p. A recomendação útil aqui é "
            f"o que **não** fazer: sair de {global_arm}. Entre os líderes, quem "
            f"decide é o custo do canal."
        )
    elif case.is_tie:
        head = (
            f"Os braços do topo estão **empatados** para este cliente — "
            f"{case.margin * 100:.2f} p.p. separam o primeiro do segundo, e "
            f"nenhum se destaca do melhor braço médio. A decisão é indiferente, "
            f"e quem escolhe deveria ser o custo do canal, não o modelo."
        )
    elif case.switches:
        head = (
            f"Recomenda **{arm}** em vez de {global_arm}, que é o melhor na "
            f"média. Para este cliente a troca vale "
            f"{case.gain_over_global * 100:.2f} p.p."
        )
    else:
        head = (
            f"Recomenda **{arm}**, o mesmo braço que vence na média, com "
            f"{case.margin * 100:.2f} p.p. sobre o segundo colocado."
        )

    # Ordem importa: `first_contact` vale para 96% da base e engoliria os
    # demais casos se viesse antes deles.
    campaign = int(case.profile.get("campaign", 0) or 0)
    if case.profile.get("poutcome") == "success":
        tail = (
            "Já assinou numa campanha anterior — o sinal mais forte da base, e "
            "por isso a probabilidade fica muito acima da taxa-base de 11,27%."
        )
    elif campaign >= 5:
        tail = (
            f"Já recebeu {campaign} ligações nesta campanha, e a estimativa cai "
            "conforme as tentativas se acumulam."
        )
    elif case.profile.get(config.FIRST_CONTACT_COLUMN) == 1:
        tail = (
            "Nunca participou de campanha anterior, então não há histórico para "
            "elevar a estimativa: ela fica perto da taxa-base."
        )
    else:
        tail = (
            "Perfil sem marcador forte em nenhuma direção; a escolha responde "
            "sobretudo ao canal e à janela de contato."
        )

    return f"{head} {tail}"


def golden_set_table(cases: list[GoldenCase], space: ArmSpace) -> pd.DataFrame:
    """One row per case, one column per arm, plus the recommendation."""
    rows = []
    for case in cases:
        row: dict[str, object] = {
            "criterio": case.criterion,
            "cliente": case.row_label,
            "idade": case.profile.get("age"),
            "profissao": case.profile.get("job"),
            "poutcome": case.profile.get("poutcome"),
            "campaign": case.profile.get("campaign"),
        }
        for index, label in enumerate(space.labels):
            row[label] = float(case.scores[index])
        row["recomendado"] = space.label(case.recommended)
        row["troca"] = case.switches
        rows.append(row)
    return pd.DataFrame(rows)
