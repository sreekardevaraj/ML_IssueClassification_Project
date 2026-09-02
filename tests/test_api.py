from fastapi.testclient import TestClient
import json
from pathlib import Path

from app.api.main import app

client = TestClient(app)


def test_live_and_ready():
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").json()["status"] in {"ready", "not_ready"}


def test_predict_requires_production_artifacts_or_classifies():
    response = client.post("/v1/predict", json={"case_id": "C-1", "text": "VPN cannot connect"})
    if client.get("/health/ready").json()["status"] == "not_ready":
        assert response.status_code == 503
        assert "artifacts" in response.json()["detail"]
    else:
        assert response.status_code == 200
        assert len(response.json()["candidates"]) == 3


def test_batch_and_validation():
    response = client.post("/v1/predict/batch", json={"cases": [{"text": "Printer broken"}]})
    assert response.status_code in {200, 503}

    invalid = client.post("/v1/predict", json={"text": "x"})
    assert invalid.status_code == 422


def test_model_info_is_non_secret():
    body = client.get("/v1/model-info").json()
    assert body["taxonomy_version"] == "2026.09.0"
    assert "DeBERTa" in body["ambiguity_path"]


def test_production_style_cases_have_expected_top_category():
    cases = {
        "network": "My company VPN disconnects every few minutes and I cannot access internal applications.",
        "account_access": "I am locked out after failed login attempts and need my corporate account unlocked.",
        "security": "A suspicious email asks me to verify my password using an external link.",
        "email": "Outlook receives emails, but I cannot send messages and my calendar is not synchronizing.",
        "printer": "The office printer is not printing documents.",
        "performance": "My laptop takes ten minutes to start and applications freeze frequently.",
    }
    if client.get("/health/ready").json()["status"] != "ready":
        return
    for expected, text in cases.items():
        response = client.post("/v1/predict", json={"case_id": expected, "text": text})
        assert response.status_code == 200
        assert response.json()["candidates"][0]["category"] == expected


def test_prediction_is_persisted_without_raw_text():
    if client.get("/health/ready").json()["status"] != "ready":
        return
    response = client.post("/v1/predict", json={"case_id": "SAVE-001", "text": "The office printer is not printing documents."})
    assert response.status_code == 200
    lines = Path("results/predictions.jsonl").read_text(encoding="utf-8").splitlines()
    saved = json.loads(lines[-1])
    assert saved["case_id"] == "SAVE-001"
    assert saved["request_id"] == response.json()["request_id"]
    assert "case_text" not in saved


def test_drift_report_tracks_dataset_shape(tmp_path):
    import pandas as pd

    from scripts.drift_report import report

    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    output = tmp_path / "drift.json"
    pd.DataFrame({"text": ["vpn issue", "printer issue"], "label": ["network", "printer"]}).to_csv(reference, index=False)
    pd.DataFrame({"text": ["vpn issue"], "label": ["network"]}).to_csv(candidate, index=False)
    result = report(reference, candidate, output)
    assert result["reference_records"] == 2
    assert result["candidate_records"] == 1
    assert output.exists()
