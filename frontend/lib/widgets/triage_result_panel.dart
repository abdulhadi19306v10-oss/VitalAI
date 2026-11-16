import 'package:flutter/material.dart';

import '../models/triage_models.dart';

/// Glanceable clinical triage summary designed to minimize doctor cognitive load.
class TriageResultPanel extends StatelessWidget {
  final PredictResponse? result;
  final ValueChanged<String> onAddFollowUpSymptom;

  const TriageResultPanel({
    super.key,
    required this.result,
    required this.onAddFollowUpSymptom,
  });

  static _UrgencyPalette _paletteFor(String urgency) {
    switch (urgency.toLowerCase()) {
      case 'critical':
        return const _UrgencyPalette(
          color: Color(0xFFB91C1C),
          background: Color(0xFFFEF2F2),
          border: Color(0xFFFECACA),
          icon: Icons.emergency_outlined,
        );
      case 'high':
        return const _UrgencyPalette(
          color: Color(0xFFB45309),
          background: Color(0xFFFFFBEB),
          border: Color(0xFFFDE68A),
          icon: Icons.priority_high_rounded,
        );
      case 'moderate':
        return const _UrgencyPalette(
          color: Color(0xFF1D4ED8),
          background: Color(0xFFEFF6FF),
          border: Color(0xFFBFDBFE),
          icon: Icons.assignment_late_outlined,
        );
      default:
        return const _UrgencyPalette(
          color: Color(0xFF15803D),
          background: Color(0xFFF0FDF4),
          border: Color(0xFFBBF7D0),
          icon: Icons.check_circle_outline,
        );
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final data = result;

    if (data == null) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(28),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(
                Icons.monitor_heart_outlined,
                size: 40,
                color: Color(0xFF0F6E75),
              ),
              const SizedBox(height: 12),
              Text(
                'Ready for Clinical Triage',
                style: theme.textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                'Select one or more patient symptoms on the left and tap Assess Triage for urgency, specialist routing, and top differentials.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: const Color(0xFF64748B),
                ),
              ),
            ],
          ),
        ),
      );
    }

    final triage = data.triage;
    final palette = _paletteFor(triage.urgency);
    final top = data.predictions.isNotEmpty ? data.predictions.first : null;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // 1. Glanceable Urgency & Action Card
        Card(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
            side: BorderSide(color: palette.border, width: 1.5),
          ),
          color: palette.background,
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 5,
                      ),
                      decoration: BoxDecoration(
                        color: palette.color,
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(palette.icon, color: Colors.white, size: 16),
                          const SizedBox(width: 6),
                          Text(
                            '${triage.urgency.toUpperCase()} URGENCY',
                            style: const TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w700,
                              fontSize: 12,
                              letterSpacing: 0.5,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const Spacer(),
                    if (top != null)
                      Text(
                        '${(top.probability * 100).toStringAsFixed(1)}% match',
                        style: theme.textTheme.labelLarge?.copyWith(
                          color: palette.color,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 14),
                if (top != null)
                  Text(
                    top.displayName,
                    style: theme.textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.w800,
                      color: const Color(0xFF0F172A),
                    ),
                  ),
                const SizedBox(height: 4),
                Text(
                  'Refer to: ${triage.specialist}',
                  style: theme.textTheme.titleSmall?.copyWith(
                    color: const Color(0xFF334155),
                    fontWeight: FontWeight.w600,
                  ),
                ),
                if (triage.redFlags.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: palette.border),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(Icons.warning_amber_rounded,
                            color: palette.color, size: 18),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Red Flag: ${triage.redFlags.join(' • ')}',
                            style: theme.textTheme.bodyMedium?.copyWith(
                              color: palette.color,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
                if (triage.immediateActions.isNotEmpty) ...[
                  const SizedBox(height: 14),
                  Text(
                    'Next Steps',
                    style: theme.textTheme.labelLarge?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 6),
                  ...triage.immediateActions.take(3).map(
                        (action) => Padding(
                          padding: const EdgeInsets.only(bottom: 4),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text('• ',
                                  style: TextStyle(fontWeight: FontWeight.w700)),
                              Expanded(child: Text(action)),
                            ],
                          ),
                        ),
                      ),
                ],
                if (triage.precautions.isNotEmpty || triage.warnings.isNotEmpty) ...[
                  const SizedBox(height: 10),
                  ...triage.precautions.map(
                    (p) => _NoticeLine(text: p, color: const Color(0xFF92400E)),
                  ),
                  ...triage.warnings.map(
                    (w) => _NoticeLine(text: w, color: const Color(0xFF475569)),
                  ),
                ],
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),

        // 2. Top Differential Possibilities + Follow-Up Prompts
        Card(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Top Differential Possibilities',
                  style: theme.textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
                if (data.primaryReasoning.isNotEmpty) ...[
                  const SizedBox(height: 6),
                  Text(
                    data.primaryReasoning,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: const Color(0xFF334155),
                    ),
                  ),
                ],
                const SizedBox(height: 12),
                ...data.predictions.map((cand) {
                  final pct = (cand.probability * 100).clamp(0.0, 100.0);
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 10),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Expanded(
                              child: Text(
                                '${cand.rank}. ${cand.displayName}',
                                style: theme.textTheme.bodyMedium?.copyWith(
                                  fontWeight: cand.rank == 1
                                      ? FontWeight.w700
                                      : FontWeight.w500,
                                ),
                              ),
                            ),
                            Text(
                              '${pct.toStringAsFixed(1)}% · ${cand.baseUrgency}',
                              style: theme.textTheme.bodySmall?.copyWith(
                                color: const Color(0xFF475569),
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 4),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(4),
                          child: LinearProgressIndicator(
                            value: cand.probability.clamp(0.0, 1.0),
                            minHeight: 6,
                            backgroundColor: const Color(0xFFE2E8F0),
                            color: cand.rank == 1
                                ? const Color(0xFF0F6E75)
                                : const Color(0xFF94A3B8),
                          ),
                        ),
                        if (cand.reasoningSummary.isNotEmpty) ...[
                          const SizedBox(height: 4),
                          Text(
                            cand.reasoningSummary,
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: const Color(0xFF64748B),
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ],
                    ),
                  );
                }),
                if (data.followUpSymptoms.isNotEmpty) ...[
                  const Divider(height: 20),
                  Text(
                    'Check to Narrow Differential (tap to add)',
                    style: theme.textTheme.labelMedium?.copyWith(
                      color: const Color(0xFF475569),
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: data.followUpSymptoms.take(4).map((f) {
                      return ActionChip(
                        avatar: const Icon(Icons.add, size: 16),
                        label: Text(f.displayName),
                        onPressed: () => onAddFollowUpSymptom(f.symptom),
                      );
                    }).toList(),
                  ),
                ],
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),

        if (data.alternativeCauses.isNotEmpty || data.symptomCauses.isNotEmpty) ...[
          Card(
            child: ExpansionTile(
              title: Text(
                'Other Possible Causes (${data.alternativeCauses.length} conditions)',
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
              childrenPadding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
              expandedCrossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (data.alternativeCauses.isNotEmpty) ...[
                  Text(
                    'Alternative Diseases Sharing These Symptoms',
                    style: theme.textTheme.labelMedium?.copyWith(
                      color: const Color(0xFF475569),
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 8),
                  ...data.alternativeCauses.map(
                    (alt) => Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '${alt.displayName} (${alt.baseUrgency} urgency)',
                            style: theme.textTheme.bodySmall?.copyWith(
                              fontWeight: FontWeight.w700,
                              color: const Color(0xFF0F172A),
                            ),
                          ),
                          Text(
                            alt.whyPossible,
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: const Color(0xFF475569),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
                if (data.symptomCauses.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  Text(
                    'Possible Causes by Symptom',
                    style: theme.textTheme.labelMedium?.copyWith(
                      color: const Color(0xFF475569),
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 6),
                  ...data.symptomCauses.map(
                    (grp) => Padding(
                      padding: const EdgeInsets.only(bottom: 6),
                      child: Text(
                        '${grp.displayName}: ${grp.possibleDiseases.join(', ')}',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: const Color(0xFF334155),
                        ),
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
          const SizedBox(height: 12),
        ],

        // 4. Progressive Disclosure: Rule Logic & Symptom Weights (Collapsed by Default)
        Card(
          child: ExpansionTile(
            title: Text(
              'Rule Logic & Symptom Weights (${data.inferenceTrace.length} rules fired)',
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            childrenPadding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
            expandedCrossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (data.symptomContributions.isNotEmpty) ...[
                Text(
                  'Key Symptom Drivers',
                  style: theme.textTheme.labelMedium?.copyWith(
                    color: const Color(0xFF475569),
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 6),
                Wrap(
                  spacing: 8,
                  runSpacing: 6,
                  children: data.symptomContributions
                      .map(
                        (c) => Chip(
                          label: Text(
                            '${c.displayName} (+${c.logLikelihoodLift.toStringAsFixed(2)})',
                            style: const TextStyle(fontSize: 12),
                          ),
                          visualDensity: VisualDensity.compact,
                        ),
                      )
                      .toList(),
                ),
                const SizedBox(height: 12),
              ],
              Text(
                'Expert System Inference Chain',
                style: theme.textTheme.labelMedium?.copyWith(
                  color: const Color(0xFF475569),
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 6),
              ...data.inferenceTrace.map(
                (step) => Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Text(
                    '[Step ${step.iteration}] ${step.ruleName}: ${step.explanation}',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: const Color(0xFF334155),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 10),
        Text(
          data.disclaimer,
          textAlign: TextAlign.center,
          style: theme.textTheme.bodySmall?.copyWith(
            color: const Color(0xFF64748B),
          ),
        ),
      ],
    );
  }
}

class _NoticeLine extends StatelessWidget {
  final String text;
  final Color color;

  const _NoticeLine({required this.text, required this.color});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Text(
        text,
        style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: color,
              fontWeight: FontWeight.w500,
            ),
      ),
    );
  }
}

class _UrgencyPalette {
  final Color color;
  final Color background;
  final Color border;
  final IconData icon;

  const _UrgencyPalette({
    required this.color,
    required this.background,
    required this.border,
    required this.icon,
  });
}
