import 'package:flutter/material.dart';

import '../models/triage_models.dart';

/// Calm, low-clutter symptom intake widget for busy clinicians.
class SymptomSelector extends StatefulWidget {
  final List<SymptomItem> allSymptoms;
  final Set<String> selectedKeys;
  final bool isLoading;
  final bool canSubmit;
  final ValueChanged<String> onToggleSymptom;
  final VoidCallback onClearAll;
  final VoidCallback onSubmit;

  const SymptomSelector({
    super.key,
    required this.allSymptoms,
    required this.selectedKeys,
    required this.isLoading,
    required this.canSubmit,
    required this.onToggleSymptom,
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

  final TextEditingController _searchController = TextEditingController();
  String _query = '';
  bool _showAll = false;

  @override
  void dispose() {
    _searchController.dispose();
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
    if (_showAll) {
      return widget.allSymptoms.where(notSelected).toList();
    }
    final byKey = {for (final s in widget.allSymptoms) s.key: s};
    final quick = _commonClinicalKeys
        .where((k) => byKey.containsKey(k) && !widget.selectedKeys.contains(k))
        .map((k) => byKey[k]!)
        .toList();
    return quick.isNotEmpty
        ? quick
        : widget.allSymptoms.where(notSelected).take(16).toList();
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

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final selectedItems = widget.allSymptoms
        .where((s) => widget.selectedKeys.contains(s.key))
        .toList();
    final visible = _visibleSymptoms();

    return Card(
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
                Text(
                  '${widget.allSymptoms.length} indexed',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: const Color(0xFF64748B),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _searchController,
              onChanged: (value) => setState(() => _query = value),
              decoration: InputDecoration(
                hintText: 'Search symptoms or clinical terms (e.g., dyspnea, diarrhea, fever)',
                prefixIcon: const Icon(Icons.search, size: 20),
                suffixIcon: _query.isNotEmpty
                    ? IconButton(
                        tooltip: 'Clear search',
                        icon: const Icon(Icons.close, size: 18),
                        onPressed: () {
                          _searchController.clear();
                          setState(() => _query = '');
                        },
                      )
                    : null,
              ),
            ),
            const SizedBox(height: 14),
            if (selectedItems.isNotEmpty) ...[
              Row(
                children: [
                  Text(
                    'Selected (${selectedItems.length})',
                    style: theme.textTheme.labelLarge?.copyWith(
                      color: const Color(0xFF0F6E75),
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
              onPressed:
                  widget.canSubmit && !widget.isLoading ? widget.onSubmit : null,
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
            Row(
              children: [
                Text(
                  _query.trim().isNotEmpty
                      ? 'Matching Symptoms (${visible.length})'
                      : (_showAll
                          ? 'All Remaining Symptoms (${visible.length})'
                          : 'Common Triage Symptoms'),
                  style: theme.textTheme.labelLarge?.copyWith(
                    color: const Color(0xFF475569),
                  ),
                ),
                const Spacer(),
                if (_query.trim().isEmpty && widget.allSymptoms.length > 16)
                  TextButton(
                    onPressed: () => setState(() => _showAll = !_showAll),
                    child: Text(_showAll ? 'Show common' : 'Browse all'),
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
                      : 'All symptoms in this view are selected.',
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: const Color(0xFF64748B),
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
    );
  }
}
