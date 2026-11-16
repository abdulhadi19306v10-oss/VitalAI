import 'package:flutter/material.dart';

import '../models/triage_models.dart';

/// Glanceable clinical triage summary supporting both Light and Dark themes cleanly.
class TriageResultPanel extends StatelessWidget {
  final PredictResponse? result;
  final bool isExporting;
  final ValueChanged<String> onAddFollowUpSymptom;
  final ValueChanged<PredictResponse> onCopySummary;
  final ValueChanged<PredictResponse> onExportWord;

  const TriageResultPanel({
    super.key,
    required this.result,
    this.isExporting = false,
    required this.onAddFollowUpSymptom,
    required this.onCopySummary,
    required this.onExportWord,
  });

  static _UrgencyPalette _paletteFor(String urgency, {required bool isDark}) {
    switch (urgency.toLowerCase()) {
      case 'critical':
        return isDark
            ? const _UrgencyPalette(
                badge: Color(0xFFDC2626),
                accentText: Color(0xFFFCA5A5),
                background: Color(0xFF2B1418),
                border: Color(0xFF7F1D1D),
                icon: Icons.emergency_outlined,
              )
            : const _UrgencyPalette(
                badge: Color(0xFFB91C1C),
                accentText: Color(0xFFB91C1C),
                background: Color(0xFFFEF2F2),
                border: Color(0xFFFECACA),
                icon: Icons.emergency_outlined,
              );
      case 'high':
        return isDark
            ? const _UrgencyPalette(
                badge: Color(0xFFD97706),
                accentText: Color(0xFFFCD34D),
                background: Color(0xFF2B1F0E),
                border: Color(0xFF78350F),
                icon: Icons.priority_high_rounded,
              )
            : const _UrgencyPalette(
                badge: Color(0xFFB45309),
                accentText: Color(0xFFB45309),
                background: Color(0xFFFFFBEB),
                border: Color(0xFFFDE68A),
                icon: Icons.priority_high_rounded,
              );
      case 'moderate':
        return isDark
            ? const _UrgencyPalette(
                badge: Color(0xFF2563EB),
                accentText: Color(0xFF93C5FD),
                background: Color(0xFF13223A),
                border: Color(0xFF1E3A8A),
                icon: Icons.assignment_late_outlined,
              )
            : const _UrgencyPalette(
                badge: Color(0xFF1D4ED8),
                accentText: Color(0xFF1D4ED8),
                background: Color(0xFFEFF6FF),
                border: Color(0xFFBFDBFE),
                icon: Icons.assignment_late_outlined,
              );
      default:
        return isDark
            ? const _UrgencyPalette(
                badge: Color(0xFF16A34A),
                accentText: Color(0xFF86EFAC),
                background: Color(0xFF11291D),
                border: Color(0xFF14532D),
                icon: Icons.check_circle_outline,
              )
            : const _UrgencyPalette(
                badge: Color(0xFF15803D),
                accentText: Color(0xFF15803D),
                background: Color(0xFFF0FDF4),
                border: Color(0xFFBBF7D0),
                icon: Icons.check_circle_outline,
              );
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;
    final isDark = theme.brightness == Brightness.dark;
    final data = result;

    if (data == null) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(28),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.monitor_heart_outlined,
                size: 40,
                color: colors.primary,
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
                'Select patient symptoms on the left, paste a clinical note, or load a preset and tap Assess Triage.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: colors.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      );
    }

    final triage = data.triage;
    final palette = _paletteFor(triage.urgency, isDark: isDark);
    final top = data.predictions.isNotEmpty ? data.predictions.first : null;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // 1. Glanceable Urgency, Referral & Export Toolbar
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
                        color: palette.badge,
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
                          color: palette.accentText,
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
                      color: colors.onSurface,
                    ),
                  ),
                const SizedBox(height: 4),
                Text(
                  'Refer to: ${triage.specialist}',
                  style: theme.textTheme.titleSmall?.copyWith(
                    color: colors.onSurfaceVariant,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                if (triage.redFlags.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: colors.surface,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: palette.border),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(Icons.warning_amber_rounded,
                            color: palette.accentText, size: 18),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Red Flag: ${triage.redFlags.join(' • ')}',
                            style: theme.textTheme.bodyMedium?.copyWith(
                              color: palette.accentText,
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
                if (triage.precautions.isNotEmpty ||
                    triage.warnings.isNotEmpty) ...[
                  const SizedBox(height: 10),
                  ...triage.precautions.map(
                    (p) => _NoticeLine(
                      text: p,
                      color: isDark
                          ? const Color(0xFFFCD34D)
                          : const Color(0xFF92400E),
                    ),
                  ),
                  ...triage.warnings.map(
                    (w) => _NoticeLine(text: w, color: colors.onSurfaceVariant),
                  ),
                ],
                const SizedBox(height: 14),
                Wrap(
                  spacing: 10,
                  runSpacing: 8,
                  children: [
                    OutlinedButton.icon(
                      onPressed: () => onCopySummary(data),
                      icon: const Icon(Icons.copy_outlined, size: 16),
                      label: const Text('Copy Note'),
                      style: OutlinedButton.styleFrom(
                        backgroundColor: colors.surface,
                      ),
                    ),
                    FilledButton.tonalIcon(
                      onPressed: isExporting ? null : () => onExportWord(data),
                      icon: isExporting
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.description_outlined, size: 16),
                      label: Text(
                        isExporting ? 'Exporting...' : 'Download Word (.docx)',
                      ),
                    ),
                  ],
                ),
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
                      color: colors.onSurfaceVariant,
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
                                color: colors.onSurfaceVariant,
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
                            backgroundColor: colors.surfaceContainerHighest,
                            color: cand.rank == 1
                                ? colors.primary
                                : colors.outline,
                          ),
                        ),
                        if (cand.reasoningSummary.isNotEmpty) ...[
                          const SizedBox(height: 4),
                          Text(
                            cand.reasoningSummary,
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: colors.onSurfaceVariant,
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
                      color: colors.onSurfaceVariant,
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

        // 3. Progressive Disclosure: Alternative Causes
        if (data.alternativeCauses.isNotEmpty ||
            data.symptomCauses.isNotEmpty) ...[
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
                      color: colors.onSurfaceVariant,
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
                              color: colors.onSurface,
                            ),
                          ),
                          Text(
                            alt.whyPossible,
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: colors.onSurfaceVariant,
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
                      color: colors.onSurfaceVariant,
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
                          color: colors.onSurface,
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
                    color: colors.onSurfaceVariant,
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
                  color: colors.onSurfaceVariant,
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
                      color: colors.onSurface,
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
            color: colors.onSurfaceVariant,
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
  final Color badge;
  final Color accentText;
  final Color background;
  final Color border;
  final IconData icon;

  const _UrgencyPalette({
    required this.badge,
    required this.accentText,
    required this.background,
    required this.border,
    required this.icon,
  });
}
