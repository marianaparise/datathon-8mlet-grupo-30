"""Tests for the recommendation service."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.app import app
from src import config

CLIENT = {
    "age": 38,
    "job": "technician",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
}


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Context manager form, so the lifespan actually loads the artefact."""
    if not config.ENVIRONMENT_ARTIFACT.exists():
        pytest.skip("ambiente ausente — rode `make train`")
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_the_loaded_arms(client) -> None:
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_arms"] == 6
    assert body["model_version"] != "unloaded"


def test_arms_endpoint_lists_the_action_space(client) -> None:
    arms = client.get("/arms").json()

    assert len(arms) == 6
    assert "cellular|mid" in arms


def test_root_redirects_to_the_docs(client) -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_recommend_returns_the_whole_ranking(client) -> None:
    body = client.post("/recommend", json=CLIENT).json()

    assert len(body["ranking"]) == 6
    assert body["recommended_arm"] in [a["arm"] for a in body["ranking"]]
    assert 0.0 <= body["probability"] <= 1.0


def test_ranking_is_sorted_by_probability(client) -> None:
    ranking = client.post("/recommend", json=CLIENT).json()["ranking"]
    probabilities = [a["probability"] for a in ranking]

    assert probabilities == sorted(probabilities, reverse=True)


def test_recommendation_is_the_top_of_the_ranking(client) -> None:
    body = client.post("/recommend", json=CLIENT).json()

    assert body["recommended_arm"] == body["ranking"][0]["arm"]
    assert body["explored"] is False


def test_arms_are_split_into_channel_and_window(client) -> None:
    ranking = client.post("/recommend", json=CLIENT).json()["ranking"]

    for score in ranking:
        assert score["arm"] == f"{score['channel']}|{score['window']}"
        assert score["channel"] in ("cellular", "telephone")
        assert score["window"] in ("early", "mid", "late")


def test_different_clients_get_different_rankings(client) -> None:
    """The whole point of serving the scorer: the payload has to matter."""
    student = {**CLIENT, "age": 18, "job": "student", "education": "high.school"}
    retired = {**CLIENT, "age": 71, "job": "retired", "education": "basic.4y"}

    first = client.post("/recommend", json=student).json()
    second = client.post("/recommend", json=retired).json()

    assert first["probability"] != second["probability"]


def test_prior_success_scores_far_above_the_base_rate(client) -> None:
    converted = {**CLIENT, "poutcome": "success", "pdays": 6, "previous": 1}
    body = client.post("/recommend", json=converted).json()

    assert body["probability"] > 0.30


def test_the_response_flags_a_tie(client) -> None:
    body = client.post("/recommend", json=CLIENT).json()
    ranking = body["ranking"]
    margin = ranking[0]["probability"] - ranking[1]["probability"]

    assert body["margin"] == pytest.approx(margin, abs=1e-9)
    assert body["is_tie"] == (margin < 0.005)


def test_exploration_is_off_by_default(client) -> None:
    for _ in range(20):
        assert client.post("/recommend", json=CLIENT).json()["explored"] is False


def test_exploration_can_pick_off_the_top(client) -> None:
    """ε-greedy on the ranking, so the trade-off is visible in the demo."""
    explored = [
        client.post(f"/recommend?explore=true&seed={seed}", json=CLIENT).json()
        for seed in range(60)
    ]

    assert any(r["explored"] for r in explored), "nenhuma exploração em 60 seeds"
    assert all(0.0 <= r["probability"] <= 1.0 for r in explored)


def test_exploration_is_reproducible_with_a_seed(client) -> None:
    first = client.post("/recommend?explore=true&seed=7", json=CLIENT).json()
    second = client.post("/recommend?explore=true&seed=7", json=CLIENT).json()

    assert first["recommended_arm"] == second["recommended_arm"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job", "astronauta"),
        ("marital", ""),
        ("education", "doutorado"),
        ("poutcome", "talvez"),
        ("default", "maybe"),
        ("age", 5),
        ("age", 120),
        ("campaign", 0),
        ("pdays", -1),
        ("pdays", 1000),
        ("previous", 99),
    ],
)
def test_invalid_payloads_are_rejected(client, field, value) -> None:
    response = client.post("/recommend", json={**CLIENT, field: value})

    assert response.status_code == 422


def test_missing_field_is_rejected(client) -> None:
    incomplete = {k: v for k, v in CLIENT.items() if k != "job"}

    assert client.post("/recommend", json=incomplete).status_code == 422


def test_duration_is_not_part_of_the_contract(client) -> None:
    """The forbidden column must not be reachable, even by a caller who tries."""
    schema = client.get("/openapi.json").json()
    properties = schema["components"]["schemas"]["ClientRequest"]["properties"]

    assert "duration" not in properties


def test_the_serving_path_stays_free_of_heavy_dependencies() -> None:
    """MLflow, matplotlib and friends must never reach the container."""
    import subprocess
    import sys

    probe = (
        "import sys, api.app;"
        "heavy = [m for m in ('mlflow', 'matplotlib', 'seaborn') "
        "if any(k == m or k.startswith(m + '.') for k in sys.modules)];"
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=config.PROJECT_ROOT,
        check=True,
    )

    assert result.stdout.strip() == "", f"import pesado no caminho: {result.stdout}"
