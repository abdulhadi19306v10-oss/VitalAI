"""Forward-chaining First-Order Logic (FOL) triage expert system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from backend.logic.knowledge_base import URGENCY_RANK, get_disease_profile
from backend.ml.naive_bayes import PredictionAnalysis


@dataclass(frozen=True)
class Fact:
    """Atomic First-Order Logic predicate fact stored in working memory."""

    predicate: str
    args: tuple[Any, ...] = ()

    def format(self) -> str:
        if not self.args:
            return f"{self.predicate}()"
        rendered_args = ", ".join(repr(a) if isinstance(a, str) else str(a) for a in self.args)
        return f"{self.predicate}({rendered_args})"


@dataclass
class InferenceStep:
    """Audit record of a single rule firing inside the forward-chaining loop."""

    iteration: int
    rule_name: str
    matched_facts: list[str]
    asserted_facts: list[str]
    explanation: str


class WorkingMemory:
    """Set-backed working memory supporting predicate queries and fact assertion."""

    def __init__(self) -> None:
        self._facts: list[Fact] = []
        self._seen: set[Fact] = set()

    def assert_fact(self, predicate: str, *args: Any) -> Fact | None:
        fact = Fact(predicate=predicate, args=args)
        if fact in self._seen:
            return None
        self._seen.add(fact)
        self._facts.append(fact)
        return fact

    def has(self, predicate: str, *args: Any) -> bool:
        return Fact(predicate=predicate, args=args) in self._seen

    def query(self, predicate: str) -> list[Fact]:
        return [f for f in self._facts if f.predicate == predicate]

    @property
    def facts(self) -> list[Fact]:
        return list(self._facts)


@dataclass
class Rule:
    """Production rule evaluated repeatedly until working memory reaches a fixpoint."""

    name: str
    evaluate: Callable[[WorkingMemory], list[tuple[list[Fact], list[tuple[str, tuple[Any, ...]]], str]]]


@dataclass
class TriageResult:
    """Final synthesized clinical triage decision and logical proof trace."""

    urgency: str
    specialist: str
    category: str
    red_flags: list[str]
    immediate_actions: list[str]
    precautions: list[str]
    warnings: list[str]
    differential_alert: bool
    inference_trace: list[InferenceStep] = field(default_factory=list)


def _rule_categorize_top_prediction(
    wm: WorkingMemory,
) -> list[tuple[list[Fact], list[tuple[str, tuple[Any, ...]]], str]]:
    fired = []
    for fact in wm.query("PredictedDisease"):
        disease, prob = fact.args
        profile = get_disease_profile(disease)
        assertions = [
            ("DiseaseCategory", (disease, profile.category)),
            ("BaselineUrgency", (disease, profile.base_urgency)),
            ("BaseSpecialist", (disease, profile.specialist)),
        ]
        if profile.contagious:
            assertions.append(("ContagiousDisease", (disease,)))
        fired.append(
            (
                [fact],
                assertions,
                f"Mapped top ML prediction '{disease}' (p={prob:.3f}) to category "
                f"'{profile.category}', base urgency '{profile.base_urgency}', and specialist '{profile.specialist}'.",
            )
        )
    return fired


def _rule_cardiac_red_flag(
    wm: WorkingMemory,
) -> list[tuple[list[Fact], list[tuple[str, tuple[Any, ...]]], str]]:
    chest = [f for f in wm.query("HasSymptom") if f.args == ("chest_pain",)]
    companions = [
        f
        for f in wm.query("HasSymptom")
        if f.args[0] in ("breathlessness", "sweating", "palpitations")
    ]
    if not chest or not companions:
        return []
    return [
        (
            [chest[0], companions[0]],
            [
                ("RedFlagActive", ("Acute Coronary / Cardiac Red Flag",)),
                ("EscalatedUrgency", ("Critical",)),
                ("OverrideSpecialist", ("Cardiologist / Emergency Medicine",)),
            ],
            f"HasSymptom('chest_pain') combined with HasSymptom('{companions[0].args[0]}') "
            f"triggers acute cardiac emergency escalation regardless of ML top-1 label.",
        )
    ]


def _rule_neuro_stroke_red_flag(
    wm: WorkingMemory,
) -> list[tuple[list[Fact], list[tuple[str, tuple[Any, ...]]], str]]:
    focal = [
        f
        for f in wm.query("HasSymptom")
        if f.args[0] in ("weakness_of_one_body_side", "slurred_speech")
    ]
    if not focal:
        return []
    primary = focal[0]
    neuro_companion = [
        f
        for f in wm.query("HasSymptom")
        if f != primary
        and f.args[0]
        in (
            "altered_sensorium",
            "coma",
            "loss_of_balance",
            "headache",
            "slurred_speech",
            "weakness_of_one_body_side",
        )
    ]
    if not neuro_companion:
        return []
    return [
        (
            [primary, neuro_companion[0]],
            [
                ("RedFlagActive", ("Acute Stroke / Neurological Deficit Red Flag",)),
                ("EscalatedUrgency", ("Critical",)),
                ("OverrideSpecialist", ("Neurologist / Emergency Medicine",)),
            ],
            f"Focal neurological deficit ('{primary.args[0]}') with '{neuro_companion[0].args[0]}' "
            f"requires immediate stroke-protocol emergency triage.",
        )
    ]


def _rule_critical_organ_crisis(
    wm: WorkingMemory,
) -> list[tuple[list[Fact], list[tuple[str, tuple[Any, ...]]], str]]:
    severe = [
        f
        for f in wm.query("HasSymptom")
        if f.args[0] in ("coma", "acute_liver_failure", "stomach_bleeding")
    ]
    if not severe:
        return []
    return [
        (
            [severe[0]],
            [
                ("RedFlagActive", (f"Critical Organ / Hemorrhage Sign ({severe[0].args[0]})",)),
                ("EscalatedUrgency", ("Critical",)),
                ("OverrideSpecialist", ("Emergency Medicine / Critical Care",)),
            ],
            f"Presence of critical clinical sign '{severe[0].args[0]}' mandates immediate ER care.",
        )
    ]


def _rule_meningeal_red_flag(
    wm: WorkingMemory,
) -> list[tuple[list[Fact], list[tuple[str, tuple[Any, ...]]], str]]:
    fever = [f for f in wm.query("HasSymptom") if f.args == ("high_fever",)]
    neck = [f for f in wm.query("HasSymptom") if f.args == ("stiff_neck",)]
    head = [f for f in wm.query("HasSymptom") if f.args == ("headache",)]
    if not (fever and neck and head):
        return []
    return [
        (
            [fever[0], neck[0], head[0]],
            [
                ("RedFlagActive", ("Possible Meningeal Irritation Triad",)),
                ("EscalatedUrgency", ("Critical",)),
                ("OverrideSpecialist", ("Emergency Medicine / Infectious Disease",)),
            ],
            "Triad of high fever, stiff neck, and headache requires urgent rule-out of meningitis.",
        )
    ]


def _rule_pulmonary_hemoptysis(
    wm: WorkingMemory,
) -> list[tuple[list[Fact], list[tuple[str, tuple[Any, ...]]], str]]:
    blood = [f for f in wm.query("HasSymptom") if f.args == ("blood_in_sputum",)]
    febrile = [
        f
        for f in wm.query("HasSymptom")
        if f.args[0] in ("high_fever", "breathlessness", "weight_loss")
    ]
    if not (blood and febrile):
        return []
    return [
        (
            [blood[0], febrile[0]],
            [
                ("RedFlagActive", ("Hemoptysis with Systemic Pulmonary Symptoms",)),
                ("EscalatedUrgency", ("High",)),
                ("OverrideSpecialist", ("Pulmonologist",)),
            ],
            f"Blood in sputum paired with '{febrile[0].args[0]}' escalates triage to High urgency pulmonology.",
        )
    ]


def _rule_severe_dehydration(
    wm: WorkingMemory,
) -> list[tuple[list[Fact], list[tuple[str, tuple[Any, ...]]], str]]:
    dehyd = [f for f in wm.query("HasSymptom") if f.args == ("dehydration",)]
    sunken = [f for f in wm.query("HasSymptom") if f.args == ("sunken_eyes",)]
    gi = [f for f in wm.query("HasSymptom") if f.args[0] in ("vomiting", "diarrhoea")]
    if not (dehyd and sunken and gi):
        return []
    return [
        (
            [dehyd[0], sunken[0], gi[0]],
            [
                ("RedFlagActive", ("Severe Fluid Depletion Risk",)),
                ("EscalatedUrgency", ("High",)),
            ],
            "Dehydration with sunken eyes and GI losses requires urgent rehydration assessment.",
        )
    ]


def _rule_low_confidence_differential(
    wm: WorkingMemory,
) -> list[tuple[list[Fact], list[tuple[str, tuple[Any, ...]]], str]]:
    pred_facts = wm.query("PredictedDisease")
    margin_facts = wm.query("ConfidenceMargin")
    if not pred_facts or not margin_facts:
        return []
    top_prob = float(pred_facts[0].args[1])
    margin = float(margin_facts[0].args[0])
    if top_prob >= 0.65 and margin >= 0.20:
        return []
    return [
        (
            [pred_facts[0], margin_facts[0]],
            [
                ("DifferentialAlert", ("Ambiguous ML Posterior",)),
                (
                    "Warning",
                    (
                        f"Diagnostic confidence is moderate/ambiguous (top_p={top_prob:.2f}, margin={margin:.2f}); "
                        f"check discriminating follow-up symptoms.",
                    ),
                ),
            ],
            f"Low top-1 probability ({top_prob:.3f}) or narrow margin ({margin:.3f}) activates differential review.",
        )
    ]


def _rule_sparse_symptom_guard(
    wm: WorkingMemory,
) -> list[tuple[list[Fact], list[tuple[str, tuple[Any, ...]]], str]]:
    count_facts = wm.query("SymptomCount")
    if not count_facts:
        return []
    count = int(count_facts[0].args[0])
    if count > 1:
        return []
    return [
        (
            [count_facts[0]],
            [
                ("DifferentialAlert", ("Sparse Symptom Input",)),
                (
                    "Warning",
                    (
                        "Only 1 symptom was provided; single-symptom predictions carry high uncertainty.",
                    ),
                ),
            ],
            "Single-symptom input triggers reliability warning and prompts for additional symptoms.",
        )
    ]


def _rule_high_urgency_differential_safety(
    wm: WorkingMemory,
) -> list[tuple[list[Fact], list[tuple[str, tuple[Any, ...]]], str]]:
    if not wm.query("DifferentialAlert"):
        return []
    pred_facts = wm.query("PredictedDisease")
    if not pred_facts:
        return []
    top_disease = str(pred_facts[0].args[0])
    top_rank = URGENCY_RANK[get_disease_profile(top_disease).base_urgency]

    matches = []
    for cand_fact in wm.query("CandidateDisease"):
        cand_disease, cand_prob = cand_fact.args
        if cand_disease == top_disease or float(cand_prob) < 0.15:
            continue
        cand_profile = get_disease_profile(cand_disease)
        if URGENCY_RANK[cand_profile.base_urgency] > top_rank:
            target_urgency = (
                "Critical" if cand_profile.base_urgency == "Critical" else "High"
            )
            matches.append(
                (
                    [wm.query("DifferentialAlert")[0], cand_fact],
                    [
                        ("EscalatedUrgency", (target_urgency,)),
                        (
                            "Precaution",
                            (
                                f"Higher-urgency differential '{cand_disease}' (p={cand_prob:.2f}, "
                                f"urgency={cand_profile.base_urgency}) cannot be excluded without clinical testing.",
                            ),
                        ),
                    ],
                    f"Active differential alert combined with plausible higher-risk candidate '{cand_disease}' "
                    f"(p={cand_prob:.3f}) safely escalates urgency to '{target_urgency}'.",
                )
            )
    return matches


def _rule_infection_control(
    wm: WorkingMemory,
) -> list[tuple[list[Fact], list[tuple[str, tuple[Any, ...]]], str]]:
    contagious_facts = wm.query("ContagiousDisease")
    pred_facts = wm.query("PredictedDisease")
    if not contagious_facts or not pred_facts:
        return []
    disease, prob = pred_facts[0].args
    if float(prob) < 0.45:
        return []
    return [
        (
            [contagious_facts[0], pred_facts[0]],
            [
                (
                    "Precaution",
                    (
                        f"'{disease}' is potentially communicable; practice respiratory/contact precautions "
                        f"and avoid close contact with vulnerable individuals until cleared.",
                    ),
                )
            ],
            f"Predicted communicable condition '{disease}' (p={prob:.3f}) asserts infection-control precaution.",
        )
    ]


def _rule_resolve_final_urgency(
    wm: WorkingMemory,
) -> list[tuple[list[Fact], list[tuple[str, tuple[Any, ...]]], str]]:
    base_facts = wm.query("BaselineUrgency")
    esc_facts = wm.query("EscalatedUrgency")
    all_u = base_facts + esc_facts
    if not all_u:
        return []
    best_fact = max(all_u, key=lambda f: URGENCY_RANK[str(f.args[-1])])
    final_u = str(best_fact.args[-1])
    return [
        (
            all_u,
            [("FinalUrgency", (final_u,))],
            f"Resolved maximum clinical urgency across baseline and rule escalations: '{final_u}'.",
        )
    ]


def _rule_resolve_final_specialist(
    wm: WorkingMemory,
) -> list[tuple[list[Fact], list[tuple[str, tuple[Any, ...]]], str]]:
    overrides = wm.query("OverrideSpecialist")
    bases = wm.query("BaseSpecialist")
    if overrides:
        chosen = str(overrides[0].args[0])
        source = [overrides[0]]
    elif bases:
        chosen = str(bases[0].args[1])
        source = [bases[0]]
    else:
        return []
    return [
        (
            source,
            [("FinalSpecialist", (chosen,))],
            f"Selected specialist routing: '{chosen}'.",
        )
    ]


def _rule_critical_er_action(
    wm: WorkingMemory,
) -> list[tuple[list[Fact], list[tuple[str, tuple[Any, ...]]], str]]:
    finals = [f for f in wm.query("FinalUrgency") if f.args == ("Critical",)]
    if not finals:
        return []
    return [
        (
            [finals[0]],
            [
                (
                    "ImmediateAction",
                    (
                        "CRITICAL TRIAGE: Contact emergency medical services or go to the nearest ER immediately.",
                    ),
                )
            ],
            "FinalUrgency('Critical') asserts mandatory immediate emergency dispatch action.",
        )
    ]


DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("R1_DiseaseTaxonomy", _rule_categorize_top_prediction),
    Rule("R2_CardiacRedFlag", _rule_cardiac_red_flag),
    Rule("R3_NeuroStrokeRedFlag", _rule_neuro_stroke_red_flag),
    Rule("R4_CriticalOrganCrisis", _rule_critical_organ_crisis),
    Rule("R5_MeningealRedFlag", _rule_meningeal_red_flag),
    Rule("R6_PulmonaryHemoptysis", _rule_pulmonary_hemoptysis),
    Rule("R7_SevereDehydration", _rule_severe_dehydration),
    Rule("R8_LowConfidenceDifferential", _rule_low_confidence_differential),
    Rule("R9_SparseSymptomGuard", _rule_sparse_symptom_guard),
    Rule("R10_HighUrgencyDifferentialSafety", _rule_high_urgency_differential_safety),
    Rule("R11_InfectionControl", _rule_infection_control),
    Rule("R12_ResolveFinalUrgency", _rule_resolve_final_urgency),
    Rule("R13_ResolveFinalSpecialist", _rule_resolve_final_specialist),
    Rule("R14_CriticalEmergencyAction", _rule_critical_er_action),
)


class TriageExpertSystem:
    """Forward-chaining FOL production system integrated with Naive Bayes output."""

    def __init__(self, rules: tuple[Rule, ...] = DEFAULT_RULES, max_iterations: int = 10) -> None:
        self.rules = rules
        self.max_iterations = max_iterations

    def evaluate(
        self, symptoms: list[str], prediction: PredictionAnalysis
    ) -> TriageResult:
        """Assert patient and ML facts into working memory and chain until fixpoint."""
        if not symptoms:
            raise ValueError("Expert system requires at least one patient symptom.")
        if not prediction.top_predictions:
            raise ValueError("Expert system requires at least one ML disease prediction.")

        wm = WorkingMemory()
        for sym in symptoms:
            wm.assert_fact("HasSymptom", sym)
        wm.assert_fact("SymptomCount", len(symptoms))

        top1 = prediction.top_predictions[0]
        wm.assert_fact("PredictedDisease", top1["disease"], float(top1["probability"]))
        for cand in prediction.top_predictions:
            wm.assert_fact("CandidateDisease", cand["disease"], float(cand["probability"]))
        wm.assert_fact("ConfidenceMargin", float(prediction.confidence_margin))
        wm.assert_fact("NormalizedEntropy", float(prediction.normalized_entropy))

        trace: list[InferenceStep] = []
        for iteration in range(1, self.max_iterations + 1):
            changed_in_cycle = False
            for rule in self.rules:
                firings = rule.evaluate(wm)
                for matched_facts, to_assert, explanation in firings:
                    newly_asserted: list[Fact] = []
                    for pred_name, pred_args in to_assert:
                        inserted = wm.assert_fact(pred_name, *pred_args)
                        if inserted is not None:
                            newly_asserted.append(inserted)
                    if newly_asserted:
                        changed_in_cycle = True
                        trace.append(
                            InferenceStep(
                                iteration=iteration,
                                rule_name=rule.name,
                                matched_facts=[f.format() for f in matched_facts],
                                asserted_facts=[f.format() for f in newly_asserted],
                                explanation=explanation,
                            )
                        )
            if not changed_in_cycle:
                break

        top_profile = get_disease_profile(str(top1["disease"]))
        urgency = (
            str(
                max(
                    wm.query("FinalUrgency"),
                    key=lambda f: URGENCY_RANK[str(f.args[0])],
                ).args[0]
            )
            if wm.query("FinalUrgency")
            else top_profile.base_urgency
        )
        specialist = (
            str(wm.query("FinalSpecialist")[-1].args[0])
            if wm.query("FinalSpecialist")
            else top_profile.specialist
        )
        red_flags = [str(f.args[0]) for f in wm.query("RedFlagActive")]
        rule_actions = [str(f.args[0]) for f in wm.query("ImmediateAction")]
        actions = list(dict.fromkeys(rule_actions + list(top_profile.actions)))
        precautions = [str(f.args[0]) for f in wm.query("Precaution")]
        warnings = [str(f.args[0]) for f in wm.query("Warning")]
        differential_alert = bool(wm.query("DifferentialAlert"))

        return TriageResult(
            urgency=urgency,
            specialist=specialist,
            category=top_profile.category,
            red_flags=red_flags,
            immediate_actions=actions,
            precautions=precautions,
            warnings=warnings,
            differential_alert=differential_alert,
            inference_trace=trace,
        )
