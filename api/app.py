"""The demonstrable service of Etapa 5.

Loads the environment serialized by ``train.py`` and scores every arm for a
client arriving in a request. What it returns is the calibrated reward model's
ranking — the Direct Method — not a bandit policy: the policies in this project
are non-contextual and would hand every caller the same arm.

The ``explore`` flag layers ε-greedy on top so the exploration/exploitation
trade-off is visible in the demo. It is stateless, though: a real deployment
needs a feedback endpoint and persisted arm statistics for the bandit to
actually learn online. That gap is stated in the README rather than papered over.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from api.schemas import ArmScore, ClientRequest, Health, Recommendation
from src import config
from src.golden_set import TIE_THRESHOLD

STATE: dict[str, Any] = {"environment": None, "version": "unloaded"}


def load_environment() -> None:
    """Read the serialized environment into module state.

    Failure is not fatal on purpose: the container should come up and report
    ``degraded`` on ``/health`` rather than crash-loop, which is far easier to
    diagnose than a restarting pod.
    """
    if not config.ENVIRONMENT_ARTIFACT.exists():
        STATE["environment"] = None
        STATE["version"] = "unloaded"
        return

    STATE["environment"] = joblib.load(config.ENVIRONMENT_ARTIFACT)
    STATE["version"] = config.ENVIRONMENT_ARTIFACT.stat().st_mtime_ns.__str__()[-10:]


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201, ARG001
    """Load the artefact once at startup, never per request."""
    load_environment()
    yield


app = FastAPI(
    title="Plataforma de Experimentação Adaptativa — TC5",
    description=(
        "Recomenda canal e janela de contato para um cliente elegível, "
        "com a probabilidade estimada de conversão de cada braço."
    ),
    version="0.7.0",
    lifespan=lifespan,
)


def _require_environment() -> Any:
    """Fetch the loaded environment or fail with a message that says what to do."""
    environment = STATE["environment"]
    if environment is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ambiente não carregado. Esperado em {config.ENVIRONMENT_ARTIFACT}. "
                "Rode `make train` antes de subir a API."
            ),
        )
    return environment


def _to_frame(client: ClientRequest) -> pd.DataFrame:
    """Turn the request into the single-row frame the environment expects.

    ``first_contact`` is derived here rather than asked of the caller: it is a
    modelling detail about the 999 sentinel, not something a CRM would store.
    """
    row = client.model_dump()
    row[config.FIRST_CONTACT_COLUMN] = int(row["pdays"] == config.PDAYS_SENTINEL)
    return pd.DataFrame([row])


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Send browsers straight to the interactive docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=Health, tags=["operação"])
async def health() -> Health:
    """Liveness, plus which artefact is loaded."""
    environment = STATE["environment"]
    loaded = environment is not None
    return Health(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        n_arms=environment.n_arms if loaded else 0,
        arms=list(environment.space.labels) if loaded else [],
        model_version=STATE["version"],
    )


@app.get("/arms", response_model=list[str], tags=["operação"])
async def arms() -> list[str]:
    """The action space the service can choose from."""
    return list(_require_environment().space.labels)


@app.post("/recommend", response_model=Recommendation, tags=["decisão"])
async def recommend(
    client: ClientRequest,
    explore: bool = Query(  # noqa: B008
        default=False,
        description=(
            "Aplica ε-greedy sobre o ranking: com probabilidade ε devolve um "
            "braço aleatório em vez do topo. É assim que um bandit continua "
            "aprendendo em produção."
        ),
    ),
    seed: int | None = Query(
        default=None, description="Fixa a exploração, para demonstração reprodutível"
    ),
) -> Recommendation:
    """Score every arm for this client and recommend one."""
    environment = _require_environment()
    space = environment.space

    scores = environment.predict(_to_frame(client))[0]
    order = np.argsort(scores)[::-1]

    chosen = int(order[0])
    explored = False
    if explore:
        rng = np.random.default_rng(seed)
        if rng.random() < config.EPSILON:
            chosen = int(rng.integers(len(scores)))
            explored = chosen != int(order[0])

    margin = float(scores[order[0]] - scores[order[1]]) if len(scores) > 1 else 0.0

    return Recommendation(
        recommended_arm=space.label(chosen),
        probability=float(scores[chosen]),
        ranking=[
            ArmScore(
                arm=space.label(int(index)),
                channel=space.label(int(index)).split("|")[0],
                window=space.label(int(index)).split("|")[1],
                probability=float(scores[index]),
            )
            for index in order
        ],
        is_tie=margin < TIE_THRESHOLD,
        margin=margin,
        explored=explored,
        model_version=STATE["version"],
    )
