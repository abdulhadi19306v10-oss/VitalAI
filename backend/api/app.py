"""FastAPI REST API integrating Multinomial Naive Bayes and the FOL Expert System."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from backend.api.docx_export import build_consultation_docx
from backend.api.schemas import (
    AlternativeCause,
    DiseaseCandidate,
    ExportDocxRequest,
    FollowUpSymptom,
    HealthResponse,
    InferenceStepOut,
    ParseSymptomsRequest,
    ParseSymptomsResponse,
    PredictRequest,
    PredictResponse,
    SymptomCauseGroup,
    SymptomContribution,
    SymptomItem,
    SymptomsResponse,
    TriageOutcome,
    UncertaintySummary,
)
from backend.logic.expert_system import TriageExpertSystem
from backend.logic.knowledge_base import DISEASE_KB, URGENCY_RANK, get_disease_profile
from backend.ml.data_loader import format_symptom_display, normalize_symptom_key
from backend.ml.naive_bayes import _BaseNaiveBayes
from backend.ml.symptom_aliases import get_aliases_for_symptom, get_symptom_category
from backend.ml.symptom_parser import parse_symptoms_smart, resolve_llm_config

DEFAULT_ARTIFACT_DIR = Path("backend/artifacts")
MEDICAL_DISCLAIMER = (
    "VitalAI is an academic AI decision-support prototype combining Naive Bayes "
    "classification and rule-based triage logic. It is not medical advice and does "
    "not replace evaluation by a licensed healthcare professional."
)


class _RuntimeContainer:
    """Lazy-initialized runtime container holding loaded ML weights and rule engine."""

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.model: _BaseNaiveBayes | None = None
        self.expert = TriageExpertSystem()
        self.feature_index: dict[str, int] = {}
        self.display_names: dict[str, str] = {}
        self._load_if_available()

    def _load_if_available(self) -> None:
        if (self.artifact_dir / "nb_model.npz").exists() and (
            self.artifact_dir / "nb_model_meta.json"
        ).exists():
            self.model = _BaseNaiveBayes.load(self.artifact_dir, prefix="nb_model")
            self.feature_index = {
                name: idx for idx, name in enumerate(self.model.feature_names_)
            }
            self.display_names = {
                name: format_symptom_display(name)
                for name in self.model.feature_names_
            }

    def require_model(self) -> _BaseNaiveBayes:
        if self.model is None:
            self._load_if_available()
        if self.model is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Trained model artifacts not found in '{self.artifact_dir}'. "
                    "Run `python -m backend.ml.train_model --data-dir data` first."
                ),
            )
        return self.model


def create_app(artifact_dir: Path | None = None) -> FastAPI:
    """Application factory creating the configured VitalAI FastAPI service."""
    load_dotenv(override=False)
    resolved_dir = artifact_dir or DEFAULT_ARTIFACT_DIR
    runtime = _RuntimeContainer(resolved_dir)

    service = FastAPI(
        title="VitalAI: Intelligent Disease Prediction & Triage API",
        version="1.0.0",
        description=(
            "Dual-AI medical decision support combining a self-written Multinomial "
            "Naive Bayes classifier with a Forward-Chaining First-Order Logic Expert System."
        ),
    )

    service.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    service.state.runtime = runtime

    @service.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        model = runtime.model
        provider, _, _, _ = resolve_llm_config()
        return HealthResponse(
            status="ok",
            model_loaded=model is not None,
            model_type=model.model_type if model is not None else None,
            alpha=model.alpha if model is not None else None,
            n_features=len(model.feature_names_) if model is not None else 0,
            n_classes=len(model.classes_) if model is not None else 0,
            kb_diseases=len(DISEASE_KB),
            llm_translator_configured=provider is not None,
            llm_provider=provider,
        )

    def _make_symptom_item(key: str) -> SymptomItem:
        return SymptomItem(
            key=key,
            display_name=runtime.display_names[key],
            aliases=get_aliases_for_symptom(key),
            category=get_symptom_category(key),
        )

    @service.get("/symptoms", response_model=SymptomsResponse)
    def list_symptoms(
        q: str | None = Query(default=None, description="Optional symptom search filter."),
    ) -> SymptomsResponse:
        model = runtime.require_model()
        items = [_make_symptom_item(key) for key in model.feature_names_]
        if q:
            needle = normalize_symptom_key(q)
            q_low = q.strip().lower()
            items = [
                item
                for item in items
                if needle in item.key
                or q_low in item.display_name.lower()
                or any(q_low in alias.lower() for alias in item.aliases)
            ]
        return SymptomsResponse(count=len(items), symptoms=items)

    @service.post("/parse-symptoms", response_model=ParseSymptomsResponse)
    def parse_symptoms_endpoint(payload: ParseSymptomsRequest) -> ParseSymptomsResponse:
        model = runtime.require_model()
        parsed = parse_symptoms_smart(
            payload.text,
            model.feature_names_,
            provider=payload.provider,
            api_key=payload.api_key,
        )
        return ParseSymptomsResponse(
            matched_symptoms=[_make_symptom_item(k) for k in parsed.matched_keys],
            unmapped_terms=parsed.unmapped_terms,
            negated_terms=parsed.negated_terms,
            mode=parsed.mode,
            note=parsed.note,
        )

    def _run_triage_pipeline(payload: PredictRequest) -> PredictResponse:
        model = runtime.require_model()
        unknown = [s for s in payload.symptoms if s not in runtime.feature_index]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "One or more unrecognized symptom keys were supplied.",
                    "unknown_symptoms": unknown,
                },
            )

        x = np.zeros(len(model.feature_names_), dtype=np.int64)
        for sym in payload.symptoms:
            x[runtime.feature_index[sym]] = 1

        analysis = model.predict_detailed(x, top_k=payload.top_k)
        triage = runtime.expert.evaluate(payload.symptoms, analysis)

        def _display_list(keys: list[str]) -> list[str]:
            return [
                runtime.display_names.get(k, format_symptom_display(k))
                for k in keys
            ]

        def _candidate_summary(
            rank: int,
            prob: float,
            matched: list[str],
            missing: list[str],
        ) -> str:
            matched_txt = ", ".join(matched) if matched else "overall symptom pattern"
            if rank == 1:
                base = f"Matched {matched_txt} ({prob * 100:.1f}% probability)."
                if missing:
                    return f"{base} Unreported hallmarks: {', '.join(missing[:3])}."
                return base
            if missing:
                return f"Shares {matched_txt}; ranked lower without {', '.join(missing[:3])}."
            return f"Shares {matched_txt} ({prob * 100:.1f}% probability)."

        candidates = []
        for pred in analysis.top_predictions:
            profile = get_disease_profile(pred["disease"])
            matched_disp = _display_list(pred.get("supporting_symptoms", []))
            missing_disp = _display_list(pred.get("missing_hallmarks", []))
            candidates.append(
                DiseaseCandidate(
                    disease=pred["disease"],
                    display_name=profile.display_name,
                    probability=round(float(pred["probability"]), 6),
                    rank=int(pred["rank"]),
                    category=profile.category,
                    base_urgency=profile.base_urgency,
                    supporting_symptoms=matched_disp,
                    missing_hallmarks=missing_disp,
                    reasoning_summary=_candidate_summary(
                        int(pred["rank"]),
                        float(pred["probability"]),
                        matched_disp,
                        missing_disp,
                    ),
                )
            )

        top_cand = candidates[0]
        if triage.red_flags:
            rule_clause = f" Safety escalation triggered by: {', '.join(triage.red_flags)}."
        elif triage.differential_alert:
            rule_clause = " Differential alert active — consider checking follow-up symptoms."
        else:
            rule_clause = f" Triage routed to {triage.specialist} ({triage.urgency} urgency)."
        primary_reasoning = (
            f"Top assessment is {top_cand.display_name} ({top_cand.probability * 100:.1f}%): "
            f"{top_cand.reasoning_summary}{rule_clause}"
        )

        # Rank other possible causes prioritizing multi-symptom overlap & clinical urgency
        alt_sorted = sorted(
            analysis.alternative_diseases,
            key=lambda item: (
                len(item["matched_symptoms"]),
                URGENCY_RANK[get_disease_profile(item["disease"]).base_urgency],
                float(item["probability"]),
            ),
            reverse=True,
        )
        alternative_causes = []
        for alt in alt_sorted[:5]:
            profile = get_disease_profile(alt["disease"])
            matched_disp = _display_list(alt["matched_symptoms"])
            missing_disp = _display_list(alt["missing_hallmarks"])
            why = f"Can explain {', '.join(matched_disp)}"
            if missing_disp:
                why += f"; typically also presents with {', '.join(missing_disp[:3])}"
            alternative_causes.append(
                AlternativeCause(
                    disease=alt["disease"],
                    display_name=profile.display_name,
                    probability=round(float(alt["probability"]), 6),
                    base_urgency=profile.base_urgency,
                    matched_symptoms=matched_disp,
                    missing_hallmarks=missing_disp,
                    why_possible=why,
                )
            )

        symptom_causes = [
            SymptomCauseGroup(
                symptom=entry["symptom"],
                display_name=runtime.display_names.get(
                    entry["symptom"], format_symptom_display(entry["symptom"])
                ),
                possible_diseases=[
                    get_disease_profile(d).display_name
                    for d in entry["possible_diseases"]
                ],
            )
            for entry in analysis.symptom_causes
        ]

        contributions = [
            SymptomContribution(
                symptom=item["symptom"],
                display_name=runtime.display_names.get(
                    item["symptom"], format_symptom_display(item["symptom"])
                ),
                log_likelihood_lift=float(item["log_likelihood_lift"]),
            )
            for item in analysis.symptom_contributions
        ]

        followups = [
            FollowUpSymptom(
                symptom=item["symptom"],
                display_name=runtime.display_names.get(
                    item["symptom"], format_symptom_display(item["symptom"])
                ),
                separation_score=float(item["separation_score"]),
                favors=item["favors"],
            )
            for item in analysis.discriminating_followups
        ]

        trace_out = [
            InferenceStepOut(
                iteration=step.iteration,
                rule_name=step.rule_name,
                matched_facts=step.matched_facts,
                asserted_facts=step.asserted_facts,
                explanation=step.explanation,
            )
            for step in triage.inference_trace
        ]

        top1_prob = float(analysis.top_predictions[0]["probability"])
        low_confidence = (
            top1_prob < 0.65
            or analysis.confidence_margin < 0.20
            or len(payload.symptoms) <= 1
        )

        return PredictResponse(
            normalized_symptoms=payload.symptoms,
            predictions=candidates,
            primary_reasoning=primary_reasoning,
            alternative_causes=alternative_causes,
            symptom_causes=symptom_causes,
            uncertainty=UncertaintySummary(
                normalized_entropy=round(analysis.normalized_entropy, 6),
                confidence_margin=round(analysis.confidence_margin, 6),
                low_confidence=low_confidence,
            ),
            symptom_contributions=contributions,
            follow_up_symptoms=followups,
            triage=TriageOutcome(
                urgency=triage.urgency,
                specialist=triage.specialist,
                category=triage.category,
                red_flags=triage.red_flags,
                immediate_actions=triage.immediate_actions,
                precautions=triage.precautions,
                warnings=triage.warnings,
                differential_alert=triage.differential_alert,
            ),
            inference_trace=trace_out,
            disclaimer=MEDICAL_DISCLAIMER,
        )

    @service.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        return _run_triage_pipeline(payload)

    @service.post("/export-docx")
    def export_docx(payload: ExportDocxRequest) -> Response:
        result = _run_triage_pipeline(payload)
        symptom_displays = [
            runtime.display_names.get(sym, format_symptom_display(sym))
            for sym in result.normalized_symptoms
        ]
        docx_bytes = build_consultation_docx(
            result=result,
            symptom_display_names=symptom_displays,
            patient_name=payload.patient_name,
        )
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": 'attachment; filename="vitalai_consultation_note.docx"'
            },
        )

    @service.get("/metrics")
    def get_metrics(compact: bool = Query(default=True)) -> dict[str, Any]:
        report_path = runtime.artifact_dir / "evaluation_report.json"
        if not report_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Evaluation report not generated yet. Run evaluate_model first.",
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if compact:
            compact_comparisons = []
            for rec in report.get("model_comparison", []):
                rec_copy = dict(rec)
                if rec_copy.get("test_metrics"):
                    tm = dict(rec_copy["test_metrics"])
                    tm.pop("per_class", None)
                    rec_copy["test_metrics"] = tm
                compact_comparisons.append(rec_copy)
            return {
                "dataset_diagnostics": {
                    k: v
                    for k, v in report.get("dataset_diagnostics", {}).items()
                    if k != "class_counts"
                },
                "recommended_model": report.get("recommended_model"),
                "model_comparison": compact_comparisons,
            }
        return report

    return service


app = create_app()
