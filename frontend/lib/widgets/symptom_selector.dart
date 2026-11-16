import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../models/triage_models.dart';

class _ClinicalPreset {
  final String title;
  final List<String> keys;

  const _ClinicalPreset(this.title, this.keys);
}

/// Calm, fast symptom intake widget with system filters, bulk text/AI extraction, and shortcuts.
class SymptomSelector extends StatefulWidget {
  final List<SymptomItem> allSymptoms;
  final Set<String> selectedKeys;
  final bool isLoading;
  final bool isParsing;
  final bool canSubmit;
  final bool llmReady;
  final ParseSymptomsResponse? lastParseFeedback;
  final ValueChanged<String> onToggleSymptom;
  final ValueChanged<List<String>> onApplyPreset;
  final Future<void> Function(String text, bool autoAssess) onBulkExtract;
  final VoidCallback onClearAll;
  final VoidCallback onSubmit;

  const SymptomSelector({
    super.key,
    required this.allSymptoms,
    required this.selectedKeys,
    required this.isLoading,
    this.isParsing = false,
    required this.canSubmit,
    this.llmReady = false,
    this.lastParseFeedback,
    required this.onToggleSymptom,
    required this.onApplyPreset,
    required this.onBulkExtract,
    required this.onClearAll,
    required this.onSubmit,
  });

  @override
  State<SymptomSelector> createState() => _SymptomSelectorState();
}

class _SymptomSelectorState extends State<SymptomSelector> {
  static const List<String> _commonClinicalKeys = [
    'chest_pain',
    'breathlessness',
    'sweating',
    'high_fever',
    'chills',
    'cough',
    'headache',
    'vomiting',
    'nausea',
    'abdominal_pain',
    'yellowish_skin',
    'skin_rash',
    'itching',
    'dizziness',
    'weakness_of_one_body_side',
    'slurred_speech',
  ];

  static const List<String> _filterTabs = [
    'Common',
    'Cardio & Respiratory',
    'GI & Hepatic',
    'Neurological',
    'Skin & Nails',
    'Fever & Systemic',
    'Urinary, Endocrine & Joints',
    'All',
  ];

  static const List<_ClinicalPreset> _presets = [
    _ClinicalPreset(
      'Cardiac Red Flag (Heart Attack)',
      ['chest_pain', 'breathlessness', 'sweating'],
    ),
    _ClinicalPreset(
      'Jaundice Differential (Ambiguous)',
      ['yellowish_skin', 'dark_urine', 'abdominal_pain'],
    ),
    _ClinicalPreset(
      'Meningeal Triad (Critical Rule)',
      ['high_fever', 'stiff_neck', 'headache'],
    ),
    _ClinicalPreset(
      'Dengue Fever Profile',
      ['high_fever', 'joint_pain', 'pain_behind_the_eyes', 'skin_rash'],
    ),
    _ClinicalPreset(
      'Tuberculosis (Isolation Alert)',
      ['cough', 'high_fever', 'blood_in_sputum', 'weight_loss'],
    ),
  ];

  final TextEditingController _searchController = TextEditingController();
  final TextEditingController _bulkController = TextEditingController();
  final FocusNode _searchFocus = FocusNode();
  String _query = '';
  String _activeFilter = 'Common';
  bool _showBulkPaste = false;

  @override
  void dispose() {
    _searchController.dispose();
    _bulkController.dispose();
    _searchFocus.dispose();
    super.dispose();
  }

  List<SymptomItem> _visibleSymptoms() {
    final trimmed = _query.trim().toLowerCase();
    final normQuery = trimmed.replaceAll(RegExp(r'[\s\-]+'), '_');
    bool notSelected(SymptomItem s) => !widget.selectedKeys.contains(s.key);

    if (trimmed.isNotEmpty) {
      return widget.allSymptoms
          .where((s) =>
              notSelected(s) &&
              (s.displayName.toLowerCase().contains(trimmed) ||
                  s.key.contains(normQuery) ||
                  s.aliases.any((a) => a.toLowerCase().contains(trimmed))))
          .take(36)
          .toList();
    }
    if (_activeFilter == 'All') {
      return widget.allSymptoms.where(notSelected).toList();
    }
    if (_activeFilter == 'Common') {
      final byKey = {for (final s in widget.allSymptoms) s.key: s};
      final quick = _commonClinicalKeys
          .where((k) => byKey.containsKey(k) && !widget.selectedKeys.contains(k))
          .map((k) => byKey[k]!)
          .toList();
      return quick.isNotEmpty
          ? quick
          : widget.allSymptoms.where(notSelected).take(16).toList();
    }
    return widget.allSymptoms
        .where((s) => notSelected(s) && s.category == _activeFilter)
        .toList();
  }

  String _chipLabel(SymptomItem item) {
    final trimmed = _query.trim().toLowerCase();
    if (trimmed.isEmpty ||
        item.displayName.toLowerCase().contains(trimmed) ||
        item.key.contains(trimmed.replaceAll(RegExp(r'[\s\-]+'), '_'))) {
      return item.displayName;
    }
    final matchedAlias = item.aliases.firstWhere(
      (a) => a.toLowerCase().contains(trimmed),
      orElse: () => '',
    );
    return matchedAlias.isNotEmpty
        ? '${item.displayName} ($matchedAlias)'
        : item.displayName;
  }

  void _selectTopMatchOnEnter(List<SymptomItem> visible) {
    if (_query.trim().isEmpty || visible.isEmpty) {
      if (widget.canSubmit && !widget.isLoading) {
        widget.onSubmit();
      }
      return;
    }
    widget.onToggleSymptom(visible.first.key);
    _searchController.clear();
    setState(() => _query = '');
    _searchFocus.requestFocus();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;
    final selectedItems = widget.allSymptoms
        .where((s) => widget.selectedKeys.contains(s.key))
        .toList();
    final visible = _visibleSymptoms();
    final knownKeys = {for (final s in widget.allSymptoms) s.key};
    final availablePresets =
        _presets.where((p) => p.keys.every(knownKeys.contains)).toList();

    return CallbackShortcuts(
      bindings: {
        const SingleActivator(LogicalKeyboardKey.enter, control: true): () {
          if (widget.canSubmit && !widget.isLoading) {
            widget.onSubmit();
          }
        },
        const SingleActivator(LogicalKeyboardKey.escape): () {
          if (_query.isNotEmpty) {
            _searchController.clear();
            setState(() => _query = '');
          }
        },
      },
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(
                children: [
                  Text(
                    'Patient Symptoms',
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const Spacer(),
                  if (availablePresets.isNotEmpty)
                    PopupMenuButton<_ClinicalPreset>(
                      tooltip: 'Load clinical demo case',
                      onSelected: (preset) => widget.onApplyPreset(preset.keys),
                      itemBuilder: (ctx) => availablePresets
                          .map(
                            (preset) => PopupMenuItem<_ClinicalPreset>(
                              value: preset,
                              child: Text(preset.title),
                            ),
                          )
                          .toList(),
                      child: Padding(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 8, vertical: 4),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.bolt_outlined,
                                size: 16, color: colors.primary),
                            const SizedBox(width: 4),
                            Text(
                              'Presets',
                              style: theme.textTheme.labelMedium?.copyWith(
                                color: colors.primary,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  const SizedBox(width: 6),
                  Text(
                    '${widget.allSymptoms.length} indexed',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: colors.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _searchController,
                focusNode: _searchFocus,
                textInputAction: TextInputAction.done,
                onChanged: (value) => setState(() => _query = value),
                onSubmitted: (_) => _selectTopMatchOnEnter(visible),
                decoration: InputDecoration(
                  hintText:
                      'Search symptoms or terms (Enter adds top match, Ctrl+Enter runs)',
                  prefixIcon: const Icon(Icons.search, size: 20),
                  suffixIcon: _query.isNotEmpty
                      ? IconButton(
                          tooltip: 'Clear search (Esc)',
                          icon: const Icon(Icons.close, size: 18),
                          onPressed: () {
                            _searchController.clear();
                            setState(() => _query = '');
                          },
                        )
                      : null,
                ),
              ),
              const SizedBox(height: 8),
              Align(
                alignment: Alignment.centerLeft,
                child: TextButton.icon(
                  onPressed: () =>
                      setState(() => _showBulkPaste = !_showBulkPaste),
                  icon: Icon(
                    _showBulkPaste
                        ? Icons.expand_less
                        : Icons.auto_awesome_outlined,
                    size: 16,
                  ),
                  label: Text(
                    _showBulkPaste
                        ? 'Hide bulk symptom extractor'
                        : (widget.llmReady
                            ? 'Paste note / bulk list (AI Translator ready)'
                            : 'Paste note / bulk list'),
                    style: const TextStyle(fontSize: 13),
                  ),
                ),
              ),
              if (_showBulkPaste) ...[
                const SizedBox(height: 6),
                TextField(
                  controller: _bulkController,
                  minLines: 2,
                  maxLines: 4,
                  decoration: const InputDecoration(
                    hintText:
                        'Paste symptom list or clinical note (e.g., chest pain, shortness of breath, sweaty, dizzy, no cough)',
                  ),
                ),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    OutlinedButton.icon(
                      onPressed: widget.isParsing
                          ? null
                          : () {
                              final txt = _bulkController.text.trim();
                              if (txt.length >= 2) {
                                widget.onBulkExtract(txt, false);
                              }
                            },
                      icon: const Icon(Icons.playlist_add_check, size: 17),
                      label: const Text('Extract Chips'),
                    ),
                    FilledButton.tonalIcon(
                      onPressed: widget.isParsing
                          ? null
                          : () {
                              final txt = _bulkController.text.trim();
                              if (txt.length >= 2) {
                                widget.onBulkExtract(txt, true);
                              }
                            },
                      icon: widget.isParsing
                          ? const SizedBox(
                              width: 15,
                              height: 15,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.auto_awesome, size: 16),
                      label: Text(
                        widget.isParsing ? 'Translating...' : 'Extract & Assess',
                      ),
                    ),
                  ],
                ),
                if (widget.lastParseFeedback != null) ...[
                  const SizedBox(height: 8),
                  Text(
                    widget.lastParseFeedback!.note,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: colors.primary,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  if (widget.lastParseFeedback!.unmappedTerms.isNotEmpty)
                    Text(
                      'Not in dataset: ${widget.lastParseFeedback!.unmappedTerms.join(', ')}',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: colors.onSurfaceVariant,
                      ),
                    ),
                  if (widget.lastParseFeedback!.negatedTerms.isNotEmpty)
                    Text(
                      'Ignored negated: ${widget.lastParseFeedback!.negatedTerms.join(', ')}',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: colors.onSurfaceVariant,
                      ),
                    ),
                ],
              ],
              const SizedBox(height: 10),
              if (selectedItems.isNotEmpty) ...[
                Row(
                  children: [
                    Text(
                      'Selected (${selectedItems.length})',
                      style: theme.textTheme.labelLarge?.copyWith(
                        color: colors.primary,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const Spacer(),
                    TextButton(
                      onPressed: widget.onClearAll,
                      child: const Text('Clear all'),
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: selectedItems
                      .map(
                        (item) => InputChip(
                          label: Text(item.displayName),
                          selected: true,
                          showCheckmark: false,
                          onDeleted: () => widget.onToggleSymptom(item.key),
                        ),
                      )
                      .toList(),
                ),
                const SizedBox(height: 14),
              ],
              FilledButton.icon(
                onPressed: widget.canSubmit && !widget.isLoading
                    ? widget.onSubmit
                    : null,
                icon: widget.isLoading
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Icon(Icons.medical_services_outlined, size: 18),
                label: Text(
                  widget.isLoading ? 'Evaluating...' : 'Assess Triage',
                ),
                style: FilledButton.styleFrom(
                  minimumSize: const Size.fromHeight(46),
                ),
              ),
              const Divider(height: 24),
              if (_query.trim().isEmpty) ...[
                Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  children: _filterTabs.map((tab) {
                    final isCurrent = _activeFilter == tab;
                    return ChoiceChip(
                      label: Text(
                        tab,
                        style: const TextStyle(fontSize: 12),
                      ),
                      selected: isCurrent,
                      visualDensity: VisualDensity.compact,
                      onSelected: (_) => setState(() => _activeFilter = tab),
                    );
                  }).toList(),
                ),
                const SizedBox(height: 10),
              ],
              Row(
                children: [
                  Text(
                    _query.trim().isNotEmpty
                        ? 'Matching Symptoms (${visible.length})'
                        : '$_activeFilter (${visible.length})',
                    style: theme.textTheme.labelLarge?.copyWith(
                      color: colors.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              if (visible.isEmpty)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  child: Text(
                    _query.trim().isNotEmpty
                        ? 'No unselected symptom matched "$_query".'
                        : 'All symptoms in this filter are selected.',
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: colors.onSurfaceVariant,
                    ),
                  ),
                )
              else
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: visible
                      .map(
                        (item) => ActionChip(
                          avatar: const Icon(Icons.add, size: 16),
                          label: Text(_chipLabel(item)),
                          onPressed: () => widget.onToggleSymptom(item.key),
                        ),
                      )
                      .toList(),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
