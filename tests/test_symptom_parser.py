"""Tests for local deterministic symptom extraction and mocked LLM translation."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from backend.ml.symptom_parser import (
    parse_symptoms_locally,
    parse_symptoms_smart,
)

META_PATH = Path("backend/artifacts/nb_model_meta.json")
FEATURES = json.loads(META_PATH.read_text(encoding="utf-8"))["feature_names"]


def test_local_parser_handles_synonyms_typos_and_negations() -> None:
    res = parse_symptoms_locally(
        "chest pain, shortness of breath, feeling sweaty, dizzy, no cough, earache",
        FEATURES,
    )
    assert res.mode == "local_parser"
    assert res.matched_keys == [
        "chest_pain",
        "breathlessness",
        "sweating",
        "dizziness",
    ]
    assert "cough" in res.negated_terms
    assert "earache" in res.unmapped_terms


def test_local_parser_fuzzy_matches_misspellings() -> None:
    res = parse_symptoms_locally("shortnes of breth, diarhea", FEATURES)
    assert "breathlessness" in res.matched_keys
    assert "diarrhoea" in res.matched_keys


def test_mocked_gemini_translator_whitelists_and_blocks_hallucinations() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "generativelanguage.googleapis.com" in str(request.url)
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "matched_symptoms": [
                                            "dyspnea",
                                            "chest_pain",
                                            "hallucinated_key",
                                        ],
                                        "negated_terms": ["cough"],
                                        "unmapped_terms": ["earache"],
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        res = parse_symptoms_smart(
            "short of breath with chest pain, earache, no cough",
            FEATURES,
            provider="gemini",
            api_key="test-key",
            http_client=client,
        )

    assert res.mode == "ai_gemini"
    assert res.matched_keys == ["breathlessness", "chest_pain"]
    assert "hallucinated key" in res.unmapped_terms
    assert "earache" in res.unmapped_terms
    assert "cough" in res.negated_terms


def test_llm_error_falls_back_cleanly_to_local_parser() -> None:
    def failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "upstream down"})

    with httpx.Client(transport=httpx.MockTransport(failing_handler)) as client:
        res = parse_symptoms_smart(
            "chest pain, diarrhea",
            FEATURES,
            provider="openai",
            api_key="sk-test",
            http_client=client,
        )

    assert res.mode == "local_fallback"
    assert res.matched_keys == ["chest_pain", "diarrhoea"]
