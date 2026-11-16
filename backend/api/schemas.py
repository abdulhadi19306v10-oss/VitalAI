"""Pydantic request and response schemas for VitalAI FastAPI service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from backend.ml.symptom_aliases import resolve_symptom_key


class SymptomItem(BaseModel):
    """Canonical symptom entry for client search/selection UIs."""

    key: str
    display_name: str
    aliases: list[str] = Field(default_factory=list)


class SymptomsResponse(BaseModel):
    count: int
    symptoms: list[SymptomItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str | None = None
    alpha: float | None = None
    n_features: int = 0
    n_classes: int = 0
    kb_diseases: int = 0


class PredictRequest(BaseModel):
    """Incoming symptom payload from client applications."""

    symptoms: list[str] = Field(
        ...,
        min_length=1,
        description="List of patient symptom keys (e.g., ['chest_pain', 'sweating']).",
    )
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("symptoms")
    @classmethod
    def normalize_and_require_non_empty(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in value:
            norm = resolve_symptom_key(raw)
            if norm and norm not in seen:
                seen.add(norm)
                cleaned.append(norm)
        if not cleaned:
            raise ValueError("At least one valid non-empty symptom must be provided.")
        return cleaned


class DiseaseCandidate(BaseModel):
    disease: str
    display_name: str
    probability: float
    rank: int
    category: str
    base_urgency: str
    supporting_symptoms: list[str] = Field(default_factory=list)
    missing_hallmarks: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""


class AlternativeCause(BaseModel):
    disease: str
    display_name: str
    probability: float
    base_urgency: str
    matched_symptoms: list[str]
    missing_hallmarks: list[str]
    why_possible: str


class SymptomCauseGroup(BaseModel):
    symptom: str
    display_name: str
    possible_diseases: list[str]


class SymptomContribution(BaseModel):
    symptom: str
    display_name: str
    log_likelihood_lift: float


class FollowUpSymptom(BaseModel):
    symptom: str
    display_name: str
    separation_score: float
    favors: str


class UncertaintySummary(BaseModel):
    normalized_entropy: float
    confidence_margin: float
    low_confidence: bool


class TriageOutcome(BaseModel):
    urgency: str
    specialist: str
    category: str
    red_flags: list[str]
    immediate_actions: list[str]
    precautions: list[str]
    warnings: list[str]
    differential_alert: bool


class InferenceStepOut(BaseModel):
    iteration: int
    rule_name: str
    matched_facts: list[str]
    asserted_facts: list[str]
    explanation: str


class PredictResponse(BaseModel):
    normalized_symptoms: list[str]
    predictions: list[DiseaseCandidate]
    primary_reasoning: str = ""
    alternative_causes: list[AlternativeCause] = Field(default_factory=list)
    symptom_causes: list[SymptomCauseGroup] = Field(default_factory=list)
    uncertainty: UncertaintySummary
    symptom_contributions: list[SymptomContribution]
    follow_up_symptoms: list[FollowUpSymptom]
    triage: TriageOutcome
    inference_trace: list[InferenceStepOut]
    disclaimer: str


class MetricsResponse(BaseModel):
    report: dict[str, Any]
