"""Integration tests for the VitalAI FastAPI endpoints."""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api.app import create_app


def test_health_and_symptoms_endpoints() -> None:
    client = TestClient(create_app())

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health = health_resp.json()
    assert health["status"] == "ok"
    assert health["model_loaded"] is True
    assert health["n_features"] == 131
    assert health["n_classes"] == 41
    assert health["kb_diseases"] == 41

    sym_resp = client.get("/symptoms")
    assert sym_resp.status_code == 200
    assert sym_resp.json()["count"] == 131

    search_resp = client.get("/symptoms", params={"q": "chest"})
    assert search_resp.status_code == 200
    keys = [item["key"] for item in search_resp.json()["symptoms"]]
    assert "chest_pain" in keys

    alias_search = client.get("/symptoms", params={"q": "dyspnea"})
    assert alias_search.status_code == 200
    alias_items = alias_search.json()["symptoms"]
    assert any(item["key"] == "breathlessness" and "dyspnea" in item["aliases"] for item in alias_items)

    us_spelling_search = client.get("/symptoms", params={"q": "diarrhea"})
    assert us_spelling_search.status_code == 200
    assert any(item["key"] == "diarrhoea" for item in us_spelling_search.json()["symptoms"])


def test_predict_happy_path_and_normalization() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/predict",
        json={"symptoms": ["Chest Pain", "dyspnea", "diaphoresis", "breathlessness"], "top_k": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["normalized_symptoms"] == ["chest_pain", "breathlessness", "sweating"]
    assert body["predictions"][0]["disease"] == "Heart attack"
    assert len(body["predictions"][0]["supporting_symptoms"]) >= 2
    assert "reasoning_summary" in body["predictions"][0]
    assert "Myocardial Infarction" in body["primary_reasoning"]
    assert len(body["alternative_causes"]) >= 1
    assert len(body["symptom_causes"]) == 3
    assert body["triage"]["urgency"] == "Critical"
    assert len(body["triage"]["red_flags"]) >= 1
    assert len(body["inference_trace"]) >= 3
    assert "not medical advice" in body["disclaimer"].lower()


def test_predict_ambiguous_case_returns_followups() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/predict",
        json={"symptoms": ["yellowish_skin", "dark_urine", "abdominal_pain"], "top_k": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["uncertainty"]["low_confidence"] is True
    assert body["triage"]["differential_alert"] is True
    assert len(body["follow_up_symptoms"]) > 0


def test_predict_rejects_empty_and_unknown_symptoms() -> None:
    client = TestClient(create_app())

    empty_resp = client.post("/predict", json={"symptoms": []})
    assert empty_resp.status_code == 422

    blank_resp = client.post("/predict", json={"symptoms": ["   "]})
    assert blank_resp.status_code == 422

    bad_resp = client.post("/predict", json={"symptoms": ["laser_eyes", "cough"]})
    assert bad_resp.status_code == 400
    assert "laser_eyes" in bad_resp.json()["detail"]["unknown_symptoms"]


def test_metrics_endpoint_and_untrained_503(tmp_path: Path) -> None:
    client = TestClient(create_app())
    metrics_resp = client.get("/metrics", params={"compact": True})
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()
    assert metrics["dataset_diagnostics"]["n_train"] == 4920
    assert len(metrics["model_comparison"]) == 3

    untrained_client = TestClient(create_app(artifact_dir=tmp_path))
    assert untrained_client.get("/health").json()["model_loaded"] is False
    pred_503 = untrained_client.post("/predict", json={"symptoms": ["cough"]})
    assert pred_503.status_code == 503


def test_export_docx_produces_valid_word_consultation_note() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/export-docx",
        json={
            "symptoms": ["Chest Pain", "dyspnea", "sweating"],
            "patient_name": "Jane Doe",
        },
    )
    assert resp.status_code == 200
    assert (
        resp.headers["content-type"]
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert "vitalai_consultation_note.docx" in resp.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = set(zf.namelist())
        assert {
            "[Content_Types].xml",
            "_rels/.rels",
            "word/_rels/document.xml.rels",
            "word/styles.xml",
            "word/document.xml",
        }.issubset(names)
        doc_xml = zf.read("word/document.xml")

    root = ET.fromstring(doc_xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    full_text = "\n".join(
        t.text or "" for t in root.findall(".//w:t", ns)
    )
    assert "VitalAI Clinical Consultation & Prescription Note" in full_text
    assert "Patient: Jane Doe" in full_text
    assert "CRITICAL" in full_text
    assert "Myocardial Infarction (Heart Attack)" in full_text
    assert "Investigations / Lab Tests Ordered:" in full_text
    assert "Rx  —  Medications, Dosage & Duration:" in full_text
    assert "Doctor's Clinical Notes & Follow-Up Plan:" in full_text


def test_parse_symptoms_endpoint_extracts_and_rejects_blank() -> None:
    client = TestClient(create_app())

    resp = client.post(
        "/parse-symptoms",
        json={
            "text": "chest pain, shortness of breath, diarrhea, no cough",
            "provider": "local",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    keys = [item["key"] for item in body["matched_symptoms"]]
    assert keys == ["chest_pain", "breathlessness", "diarrhoea"]
    assert "cough" in body["negated_terms"]
    assert body["mode"] == "local_parser"

    blank = client.post("/parse-symptoms", json={"text": "  "})
    assert blank.status_code == 422


