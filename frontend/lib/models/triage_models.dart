// Strongly-typed clinical data models matching VitalAI FastAPI schemas.

class SymptomItem {
  final String key;
  final String displayName;
  final List<String> aliases;

  const SymptomItem({
    required this.key,
    required this.displayName,
    this.aliases = const [],
  });

  factory SymptomItem.fromJson(Map<String, dynamic> json) {
    return SymptomItem(
      key: json['key'] as String,
      displayName: json['display_name'] as String,
      aliases: List<String>.from(json['aliases'] as List? ?? const []),
    );
  }
}

class HealthStatus {
  final String status;
  final bool modelLoaded;
  final String? modelType;
  final int nFeatures;
  final int nClasses;

  const HealthStatus({
    required this.status,
    required this.modelLoaded,
    this.modelType,
    required this.nFeatures,
    required this.nClasses,
  });

  factory HealthStatus.fromJson(Map<String, dynamic> json) {
    return HealthStatus(
      status: json['status'] as String,
      modelLoaded: json['model_loaded'] as bool? ?? false,
      modelType: json['model_type'] as String?,
      nFeatures: json['n_features'] as int? ?? 0,
      nClasses: json['n_classes'] as int? ?? 0,
    );
  }
}

class DiseaseCandidate {
  final String disease;
  final String displayName;
  final double probability;
  final int rank;
  final String category;
  final String baseUrgency;
  final List<String> supportingSymptoms;
  final List<String> missingHallmarks;
  final String reasoningSummary;

  const DiseaseCandidate({
    required this.disease,
    required this.displayName,
    required this.probability,
    required this.rank,
    required this.category,
    required this.baseUrgency,
    this.supportingSymptoms = const [],
    this.missingHallmarks = const [],
    this.reasoningSummary = '',
  });

  factory DiseaseCandidate.fromJson(Map<String, dynamic> json) {
    return DiseaseCandidate(
      disease: json['disease'] as String,
      displayName: json['display_name'] as String,
      probability: (json['probability'] as num).toDouble(),
      rank: json['rank'] as int,
      category: json['category'] as String,
      baseUrgency: json['base_urgency'] as String,
      supportingSymptoms:
          List<String>.from(json['supporting_symptoms'] as List? ?? const []),
      missingHallmarks:
          List<String>.from(json['missing_hallmarks'] as List? ?? const []),
      reasoningSummary: json['reasoning_summary'] as String? ?? '',
    );
  }
}

class AlternativeCause {
  final String disease;
  final String displayName;
  final double probability;
  final String baseUrgency;
  final List<String> matchedSymptoms;
  final List<String> missingHallmarks;
  final String whyPossible;

  const AlternativeCause({
    required this.disease,
    required this.displayName,
    required this.probability,
    required this.baseUrgency,
    required this.matchedSymptoms,
    required this.missingHallmarks,
    required this.whyPossible,
  });

  factory AlternativeCause.fromJson(Map<String, dynamic> json) {
    return AlternativeCause(
      disease: json['disease'] as String,
      displayName: json['display_name'] as String,
      probability: (json['probability'] as num).toDouble(),
      baseUrgency: json['base_urgency'] as String,
      matchedSymptoms:
          List<String>.from(json['matched_symptoms'] as List? ?? const []),
      missingHallmarks:
          List<String>.from(json['missing_hallmarks'] as List? ?? const []),
      whyPossible: json['why_possible'] as String? ?? '',
    );
  }
}

class SymptomCauseGroup {
  final String symptom;
  final String displayName;
  final List<String> possibleDiseases;

  const SymptomCauseGroup({
    required this.symptom,
    required this.displayName,
    required this.possibleDiseases,
  });

  factory SymptomCauseGroup.fromJson(Map<String, dynamic> json) {
    return SymptomCauseGroup(
      symptom: json['symptom'] as String,
      displayName: json['display_name'] as String,
      possibleDiseases:
          List<String>.from(json['possible_diseases'] as List? ?? const []),
    );
  }
}

class SymptomContribution {
  final String symptom;
  final String displayName;
  final double logLikelihoodLift;

  const SymptomContribution({
    required this.symptom,
    required this.displayName,
    required this.logLikelihoodLift,
  });

  factory SymptomContribution.fromJson(Map<String, dynamic> json) {
    return SymptomContribution(
      symptom: json['symptom'] as String,
      displayName: json['display_name'] as String,
      logLikelihoodLift: (json['log_likelihood_lift'] as num).toDouble(),
    );
  }
}

class FollowUpSymptom {
  final String symptom;
  final String displayName;
  final double separationScore;
  final String favors;

  const FollowUpSymptom({
    required this.symptom,
    required this.displayName,
    required this.separationScore,
    required this.favors,
  });

  factory FollowUpSymptom.fromJson(Map<String, dynamic> json) {
    return FollowUpSymptom(
      symptom: json['symptom'] as String,
      displayName: json['display_name'] as String,
      separationScore: (json['separation_score'] as num).toDouble(),
      favors: json['favors'] as String,
    );
  }
}

class UncertaintySummary {
  final double normalizedEntropy;
  final double confidenceMargin;
  final bool lowConfidence;

  const UncertaintySummary({
    required this.normalizedEntropy,
    required this.confidenceMargin,
    required this.lowConfidence,
  });

  factory UncertaintySummary.fromJson(Map<String, dynamic> json) {
    return UncertaintySummary(
      normalizedEntropy: (json['normalized_entropy'] as num).toDouble(),
      confidenceMargin: (json['confidence_margin'] as num).toDouble(),
      lowConfidence: json['low_confidence'] as bool? ?? false,
    );
  }
}

class TriageOutcome {
  final String urgency;
  final String specialist;
  final String category;
  final List<String> redFlags;
  final List<String> immediateActions;
  final List<String> precautions;
  final List<String> warnings;
  final bool differentialAlert;

  const TriageOutcome({
    required this.urgency,
    required this.specialist,
    required this.category,
    required this.redFlags,
    required this.immediateActions,
    required this.precautions,
    required this.warnings,
    required this.differentialAlert,
  });

  factory TriageOutcome.fromJson(Map<String, dynamic> json) {
    return TriageOutcome(
      urgency: json['urgency'] as String,
      specialist: json['specialist'] as String,
      category: json['category'] as String,
      redFlags: List<String>.from(json['red_flags'] as List? ?? const []),
      immediateActions:
          List<String>.from(json['immediate_actions'] as List? ?? const []),
      precautions: List<String>.from(json['precautions'] as List? ?? const []),
      warnings: List<String>.from(json['warnings'] as List? ?? const []),
      differentialAlert: json['differential_alert'] as bool? ?? false,
    );
  }
}

class InferenceStep {
  final int iteration;
  final String ruleName;
  final List<String> matchedFacts;
  final List<String> assertedFacts;
  final String explanation;

  const InferenceStep({
    required this.iteration,
    required this.ruleName,
    required this.matchedFacts,
    required this.assertedFacts,
    required this.explanation,
  });

  factory InferenceStep.fromJson(Map<String, dynamic> json) {
    return InferenceStep(
      iteration: json['iteration'] as int,
      ruleName: json['rule_name'] as String,
      matchedFacts: List<String>.from(json['matched_facts'] as List? ?? const []),
      assertedFacts:
          List<String>.from(json['asserted_facts'] as List? ?? const []),
      explanation: json['explanation'] as String,
    );
  }
}

class PredictResponse {
  final List<String> normalizedSymptoms;
  final List<DiseaseCandidate> predictions;
  final String primaryReasoning;
  final List<AlternativeCause> alternativeCauses;
  final List<SymptomCauseGroup> symptomCauses;
  final UncertaintySummary uncertainty;
  final List<SymptomContribution> symptomContributions;
  final List<FollowUpSymptom> followUpSymptoms;
  final TriageOutcome triage;
  final List<InferenceStep> inferenceTrace;
  final String disclaimer;

  const PredictResponse({
    required this.normalizedSymptoms,
    required this.predictions,
    this.primaryReasoning = '',
    this.alternativeCauses = const [],
    this.symptomCauses = const [],
    required this.uncertainty,
    required this.symptomContributions,
    required this.followUpSymptoms,
    required this.triage,
    required this.inferenceTrace,
    required this.disclaimer,
  });

  factory PredictResponse.fromJson(Map<String, dynamic> json) {
    return PredictResponse(
      normalizedSymptoms:
          List<String>.from(json['normalized_symptoms'] as List? ?? const []),
      predictions: (json['predictions'] as List)
          .map((e) => DiseaseCandidate.fromJson(e as Map<String, dynamic>))
          .toList(),
      primaryReasoning: json['primary_reasoning'] as String? ?? '',
      alternativeCauses: (json['alternative_causes'] as List? ?? const [])
          .map((e) => AlternativeCause.fromJson(e as Map<String, dynamic>))
          .toList(),
      symptomCauses: (json['symptom_causes'] as List? ?? const [])
          .map((e) => SymptomCauseGroup.fromJson(e as Map<String, dynamic>))
          .toList(),
      uncertainty: UncertaintySummary.fromJson(
          json['uncertainty'] as Map<String, dynamic>),
      symptomContributions: (json['symptom_contributions'] as List? ?? const [])
          .map((e) => SymptomContribution.fromJson(e as Map<String, dynamic>))
          .toList(),
      followUpSymptoms: (json['follow_up_symptoms'] as List? ?? const [])
          .map((e) => FollowUpSymptom.fromJson(e as Map<String, dynamic>))
          .toList(),
      triage: TriageOutcome.fromJson(json['triage'] as Map<String, dynamic>),
      inferenceTrace: (json['inference_trace'] as List? ?? const [])
          .map((e) => InferenceStep.fromJson(e as Map<String, dynamic>))
          .toList(),
      disclaimer: json['disclaimer'] as String? ?? '',
    );
  }
}
