"""Request and response contracts for the recommendation service.

The categorical fields are typed as literals rather than free strings so an
unknown level is rejected at the door with a 422 that names the allowed values,
instead of reaching the encoder and being silently one-hot-encoded to all zeros.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Job = Literal[
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]
Marital = Literal["divorced", "married", "single", "unknown"]
Education = Literal[
    "basic.4y",
    "basic.6y",
    "basic.9y",
    "high.school",
    "illiterate",
    "professional.course",
    "university.degree",
    "unknown",
]
YesNoUnknown = Literal["no", "unknown", "yes"]
Poutcome = Literal["failure", "nonexistent", "success"]


class ClientRequest(BaseModel):
    """An eligible client the campaign has to decide how to approach.

    Mirrors the raw log's own columns, so a caller holding a customer record can
    fill this in without knowing anything about the model. ``duration`` has no
    field here and never will — it is only knowable after the call ends.
    """

    age: int = Field(ge=17, le=98, examples=[38])
    job: Job = Field(examples=["technician"])
    marital: Marital = Field(examples=["married"])
    education: Education = Field(examples=["university.degree"])
    default: YesNoUnknown = Field(examples=["no"], description="Inadimplente")
    housing: YesNoUnknown = Field(examples=["yes"], description="Crédito imobiliário")
    loan: YesNoUnknown = Field(examples=["no"], description="Crédito pessoal")
    campaign: int = Field(
        ge=1, le=56, examples=[2], description="Contatos nesta campanha"
    )
    pdays: int = Field(
        ge=0,
        le=999,
        examples=[999],
        description="Dias desde o último contato. 999 = nunca contatado",
    )
    previous: int = Field(
        ge=0, le=7, examples=[0], description="Contatos em campanhas anteriores"
    )
    poutcome: Poutcome = Field(
        examples=["nonexistent"], description="Desfecho da campanha anterior"
    )


class ArmScore(BaseModel):
    """One arm and its estimated conversion for this client."""

    arm: str = Field(examples=["cellular|mid"])
    channel: str = Field(examples=["cellular"])
    window: str = Field(
        examples=["mid"], description="early=seg, mid=ter–qui, late=sex"
    )
    probability: float = Field(ge=0.0, le=1.0, examples=[0.1547])


class Recommendation(BaseModel):
    """The answer: what to do, how sure, and the whole ranking behind it.

    The full ranking is returned rather than just the winner because on this
    dataset the top arms are often statistically tied, and a caller that can
    weigh channel cost should be able to see that instead of being handed a
    false precision.
    """

    recommended_arm: str = Field(examples=["cellular|mid"])
    probability: float = Field(ge=0.0, le=1.0, examples=[0.1547])
    ranking: list[ArmScore]
    is_tie: bool = Field(
        description="Os dois primeiros braços estão dentro do erro de calibração"
    )
    margin: float = Field(description="Diferença entre o primeiro e o segundo braço")
    explored: bool = Field(
        description="A escolha veio de exploração aleatória, não do topo do ranking"
    )
    model_version: str


class Health(BaseModel):
    """Liveness plus enough detail to tell which artefact is loaded."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    n_arms: int
    arms: list[str]
    model_version: str
