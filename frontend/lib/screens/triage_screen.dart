import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../models/triage_models.dart';
import '../services/file_saver.dart';
import '../services/vital_api_client.dart';
import '../widgets/symptom_selector.dart';
import '../widgets/triage_result_panel.dart';

typedef DocxSaver = Future<String> Function(Uint8List bytes, String fileName);

/// Main single-workspace clinical triage screen.
class TriageScreen extends StatefulWidget {
  final VitalApiClient apiClient;
  final DocxSaver? onSaveDocx;
  final bool isDarkMode;
  final VoidCallback? onToggleTheme;

  const TriageScreen({
    super.key,
    required this.apiClient,
    this.onSaveDocx,
    this.isDarkMode = false,
    this.onToggleTheme,
  });

  @override
  State<TriageScreen> createState() => _TriageScreenState();
}

class _TriageScreenState extends State<TriageScreen> {
  List<SymptomItem> _symptoms = const [];
  final Set<String> _selectedKeys = {};
  final List<ConsultationEntry> _history = [];
  PredictResponse? _result;
  ParseSymptomsResponse? _lastParseFeedback;
  bool _loadingCatalog = true;
  bool _predicting = false;
  bool _parsing = false;
  bool _exporting = false;
  String? _errorMessage;
  bool _connected = false;
  bool _llmConfiguredOnBackend = false;
  String _sessionLlmKey = '';

  @override
  void initState() {
    super.initState();
    _loadInitialCatalog();
  }

  Future<void> _loadInitialCatalog() async {
    setState(() {
      _loadingCatalog = true;
      _errorMessage = null;
    });
    try {
      final health = await widget.apiClient.fetchHealth();
      final catalog = await widget.apiClient.fetchSymptoms();
      if (!mounted) return;
      setState(() {
        _connected = health.modelLoaded;
        _llmConfiguredOnBackend = health.llmTranslatorConfigured;
        _symptoms = catalog;
        _loadingCatalog = false;
      });
    } catch (err) {
      if (!mounted) return;
      setState(() {
        _connected = false;
        _loadingCatalog = false;
        _errorMessage =
            'Cannot reach backend at ${widget.apiClient.baseUrl}. Start FastAPI (`uvicorn backend.api.app:app`) and tap Retry.';
      });
    }
  }

  void _toggleSymptom(String key) {
    setState(() {
      if (_selectedKeys.contains(key)) {
        _selectedKeys.remove(key);
      } else {
        _selectedKeys.add(key);
      }
    });
  }

  void _applyPreset(List<String> keys) {
    setState(() {
      _selectedKeys
        ..clear()
        ..addAll(keys);
      _lastParseFeedback = null;
    });
    if (_connected) {
      _runPrediction();
    }
  }

  Future<void> _handleBulkExtract(String text, bool autoAssess) async {
    if (!_connected) return;
    setState(() {
      _parsing = true;
      _errorMessage = null;
    });
    try {
      final parsed = await widget.apiClient.parseSymptoms(
        text,
        apiKey: _sessionLlmKey,
      );
      if (!mounted) return;
      setState(() {
        _parsing = false;
        _lastParseFeedback = parsed;
        for (final item in parsed.matchedSymptoms) {
          _selectedKeys.add(item.key);
        }
      });
      if (parsed.matchedSymptoms.isEmpty) {
        ScaffoldMessenger.of(context)
          ..hideCurrentSnackBar()
          ..showSnackBar(
            const SnackBar(
              content: Text('No dataset symptoms matched that description.'),
            ),
          );
        return;
      }
      if (autoAssess && _selectedKeys.isNotEmpty) {
        await _runPrediction();
      }
    } catch (err) {
      if (!mounted) return;
      setState(() {
        _parsing = false;
        _errorMessage = err.toString();
      });
    }
  }

  void _clearAllWithUndo() {
    if (_selectedKeys.isEmpty && _result == null) return;
    final previousKeys = Set<String>.from(_selectedKeys);
    final previousResult = _result;
    setState(() {
      _selectedKeys.clear();
      _result = null;
      _lastParseFeedback = null;
    });
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          content: Text('Cleared ${previousKeys.length} symptom(s)'),
          duration: const Duration(seconds: 4),
          action: SnackBarAction(
            label: 'Undo',
            onPressed: () {
              setState(() {
                _selectedKeys
                  ..clear()
                  ..addAll(previousKeys);
                _result = previousResult;
              });
            },
          ),
        ),
      );
  }

  void _addFollowUpSymptom(String key) {
    if (_selectedKeys.contains(key)) return;
    setState(() => _selectedKeys.add(key));
    _runPrediction();
  }

  Future<void> _runPrediction() async {
    if (_selectedKeys.isEmpty) return;
    setState(() {
      _predicting = true;
      _errorMessage = null;
    });
    try {
      final snapshotKeys = Set<String>.from(_selectedKeys);
      final response = await widget.apiClient.predict(snapshotKeys.toList());
      if (!mounted) return;
      setState(() {
        _result = response;
        _predicting = false;
        _history.removeWhere(
          (e) =>
              e.symptoms.length == snapshotKeys.length &&
              e.symptoms.containsAll(snapshotKeys),
        );
        _history.insert(
          0,
          ConsultationEntry(
            timestamp: DateTime.now(),
            symptoms: snapshotKeys,
            response: response,
          ),
        );
        if (_history.length > 6) {
          _history.removeLast();
        }
      });
    } catch (err) {
      if (!mounted) return;
      setState(() {
        _predicting = false;
        _errorMessage = err.toString();
      });
    }
  }

  Future<void> _copySummary(PredictResponse data) async {
    final displayByKey = {for (final s in _symptoms) s.key: s.displayName};
    final symNames = data.normalizedSymptoms
        .map((k) => displayByKey[k] ?? k)
        .join(', ');
    final top = data.predictions.isNotEmpty ? data.predictions.first : null;
    final lines = <String>[
      'VitalAI Clinical Triage Note',
      'Urgency: ${data.triage.urgency.toUpperCase()} | Specialist: ${data.triage.specialist}',
      'Symptoms: $symNames',
      if (top != null)
        'Primary Assessment: ${top.displayName} (${(top.probability * 100).toStringAsFixed(1)}%)',
      if (data.triage.redFlags.isNotEmpty)
        'Red Flags: ${data.triage.redFlags.join('; ')}',
      'Reasoning: ${data.primaryReasoning}',
      if (data.predictions.length > 1)
        'Differentials: ${data.predictions.skip(1).map((c) => '${c.displayName} (${(c.probability * 100).toStringAsFixed(1)}%)').join(', ')}',
      if (data.triage.immediateActions.isNotEmpty)
        'Next Steps: ${data.triage.immediateActions.join(' | ')}',
    ];
    await Clipboard.setData(ClipboardData(text: lines.join('\n')));
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        const SnackBar(
          content: Text('Clinical triage summary copied to clipboard'),
          duration: Duration(seconds: 3),
        ),
      );
  }

  Future<void> _exportWordNote(PredictResponse data) async {
    final patientName = await showDialog<String>(
      context: context,
      builder: (ctx) => const _TextPromptDialog(
        title: 'Download Consultation Sheet (.docx)',
        labelText: 'Patient Name / ID (optional)',
        hintText: 'Leave blank for handwritten fill-in',
        confirmLabel: 'Download .docx',
        confirmIcon: Icons.download_outlined,
      ),
    );
    if (patientName == null || !mounted) return;

    setState(() => _exporting = true);
    try {
      final bytes = await widget.apiClient.exportDocx(
        data.normalizedSymptoms,
        patientName: patientName,
      );
      final saver = widget.onSaveDocx ?? saveDocumentBytes;
      final savedMsg = await saver(bytes, 'vitalai_consultation_note.docx');
      if (!mounted) return;
      setState(() => _exporting = false);
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          SnackBar(
            content: Text(savedMsg),
            duration: const Duration(seconds: 4),
          ),
        );
    } catch (err) {
      if (!mounted) return;
      setState(() => _exporting = false);
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          SnackBar(content: Text('Export failed: $err')),
        );
    }
  }

  Future<void> _showModelAuditDialog() async {
    try {
      final report = await widget.apiClient.fetchMetrics();
      if (!mounted) return;
      final diag =
          report['dataset_diagnostics'] as Map<String, dynamic>? ?? const {};
      final rec =
          report['recommended_model'] as Map<String, dynamic>? ?? const {};
      final audit =
          rec['overfitting_audit'] as Map<String, dynamic>? ?? const {};
      final comparisons = (report['model_comparison'] as List? ?? const [])
          .cast<Map<String, dynamic>>();

      await showDialog<void>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Model Audit & Evaluation Summary'),
          content: SizedBox(
            width: 480,
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'Dataset: ${diag['n_train'] ?? 0} train rows · '
                    '${(diag['train_dedup'] as Map?)?['unique_pairs'] ?? 0} unique patterns · '
                    '${diag['n_features'] ?? 0} symptoms · ${diag['n_classes'] ?? 0} diseases',
                    style: const TextStyle(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 8),
                  Text('Gate Status: ${audit['status'] ?? 'Verified'}'),
                  const Divider(height: 20),
                  ...comparisons.map((m) {
                    final cv =
                        m['cv_summary'] as Map<String, dynamic>? ?? const {};
                    final cvAcc = ((cv['mean_val_accuracy'] as num? ?? 0) * 100)
                        .toStringAsFixed(2);
                    final noisyAcc =
                        ((cv['mean_noisy_val_accuracy'] as num? ?? 0) * 100)
                            .toStringAsFixed(2);
                    final d20 = ((m['dropout_20_accuracy'] as num? ?? 0) * 100)
                        .toStringAsFixed(2);
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Text(
                        '${m['model_name']}: Deduplicated CV $cvAcc% · '
                        'Noisy CV $noisyAcc% · 20% Dropout Test $d20%',
                      ),
                    );
                  }),
                ],
              ),
            ),
          ),
          actions: [
            FilledButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('Close'),
            ),
          ],
        ),
      );
    } catch (err) {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          SnackBar(content: Text('Unable to load model metrics: $err')),
        );
    }
  }

  Future<void> _editSettings() async {
    final result = await showDialog<_SettingsResult>(
      context: context,
      builder: (ctx) => _SettingsDialog(
        initialUrl: widget.apiClient.baseUrl,
        initialApiKey: _sessionLlmKey,
      ),
    );
    if (result == null) return;
    setState(() => _sessionLlmKey = result.apiKey);
    if (result.url.isNotEmpty && result.url != widget.apiClient.baseUrl) {
      widget.apiClient.baseUrl = result.url;
      await _loadInitialCatalog();
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;
    final isDark = widget.isDarkMode || theme.brightness == Brightness.dark;

    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Icon(Icons.local_hospital_outlined, color: colors.primary),
            const SizedBox(width: 8),
            const Flexible(
              child: Text(
                'VitalAI Clinical Triage',
                overflow: TextOverflow.ellipsis,
                style: TextStyle(fontWeight: FontWeight.w700),
              ),
            ),
          ],
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 4),
            child: Row(
              children: [
                Icon(
                  Icons.circle,
                  size: 10,
                  color: _connected
                      ? const Color(0xFF16A34A)
                      : const Color(0xFFDC2626),
                ),
                const SizedBox(width: 6),
                Text(
                  _connected ? 'API Ready' : 'Offline',
                  style: const TextStyle(fontSize: 13),
                ),
              ],
            ),
          ),
          IconButton(
            tooltip: isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode',
            icon: Icon(
              isDark ? Icons.light_mode_outlined : Icons.dark_mode_outlined,
            ),
            onPressed: widget.onToggleTheme,
          ),
          IconButton(
            tooltip: 'Model Audit & Metrics',
            icon: const Icon(Icons.analytics_outlined),
            onPressed: _connected ? _showModelAuditDialog : null,
          ),
          IconButton(
            tooltip: 'API & AI Translator Settings',
            icon: const Icon(Icons.settings_outlined),
            onPressed: _editSettings,
          ),
        ],
      ),
      body: _loadingCatalog
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 1180),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      if (_errorMessage != null) ...[
                        MaterialBanner(
                          content: Text(_errorMessage!),
                          backgroundColor: isDark
                              ? const Color(0xFF2B1418)
                              : const Color(0xFFFEF2F2),
                          actions: [
                            TextButton(
                              onPressed: _loadInitialCatalog,
                              child: const Text('Retry'),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                      ],
                      if (_history.isNotEmpty) ...[
                        SingleChildScrollView(
                          scrollDirection: Axis.horizontal,
                          child: Row(
                            children: [
                              Text(
                                'Recent:',
                                style: TextStyle(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w600,
                                  color: colors.onSurfaceVariant,
                                ),
                              ),
                              const SizedBox(width: 8),
                              ..._history.map((entry) {
                                final top = entry.response.predictions.first;
                                final pct =
                                    (top.probability * 100).toStringAsFixed(0);
                                return Padding(
                                  padding: const EdgeInsets.only(right: 8),
                                  child: ActionChip(
                                    label: Text(
                                      '${top.disease} ($pct%)',
                                      style: const TextStyle(fontSize: 12),
                                    ),
                                    visualDensity: VisualDensity.compact,
                                    onPressed: () {
                                      setState(() {
                                        _selectedKeys
                                          ..clear()
                                          ..addAll(entry.symptoms);
                                        _result = entry.response;
                                      });
                                    },
                                  ),
                                );
                              }),
                            ],
                          ),
                        ),
                        const SizedBox(height: 12),
                      ],
                      LayoutBuilder(
                        builder: (context, constraints) {
                          final wide = constraints.maxWidth >= 860;
                          final selector = SymptomSelector(
                            allSymptoms: _symptoms,
                            selectedKeys: _selectedKeys,
                            isLoading: _predicting,
                            isParsing: _parsing,
                            canSubmit: _selectedKeys.isNotEmpty && _connected,
                            llmReady: _llmConfiguredOnBackend ||
                                _sessionLlmKey.isNotEmpty,
                            lastParseFeedback: _lastParseFeedback,
                            onToggleSymptom: _toggleSymptom,
                            onApplyPreset: _applyPreset,
                            onBulkExtract: _handleBulkExtract,
                            onClearAll: _clearAllWithUndo,
                            onSubmit: _runPrediction,
                          );
                          final results = TriageResultPanel(
                            result: _result,
                            isExporting: _exporting,
                            onAddFollowUpSymptom: _addFollowUpSymptom,
                            onCopySummary: _copySummary,
                            onExportWord: _exportWordNote,
                          );

                          if (wide) {
                            return Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Expanded(flex: 5, child: selector),
                                const SizedBox(width: 16),
                                Expanded(flex: 6, child: results),
                              ],
                            );
                          }
                          return Column(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              selector,
                              const SizedBox(height: 16),
                              results,
                            ],
                          );
                        },
                      ),
                    ],
                  ),
                ),
              ),
            ),
    );
  }
}

class _TextPromptDialog extends StatefulWidget {
  final String title;
  final String? labelText;
  final String? hintText;
  final String confirmLabel;
  final IconData? confirmIcon;

  const _TextPromptDialog({
    required this.title,
    this.labelText,
    this.hintText,
    required this.confirmLabel,
    this.confirmIcon,
  });

  @override
  State<_TextPromptDialog> createState() => _TextPromptDialogState();
}

class _TextPromptDialogState extends State<_TextPromptDialog> {
  final TextEditingController _controller = TextEditingController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: Text(widget.title),
      content: TextField(
        controller: _controller,
        autofocus: true,
        textInputAction: TextInputAction.done,
        onSubmitted: (value) => Navigator.pop(context, value.trim()),
        decoration: InputDecoration(
          labelText: widget.labelText,
          hintText: widget.hintText,
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        if (widget.confirmIcon != null)
          FilledButton.icon(
            onPressed: () => Navigator.pop(context, _controller.text.trim()),
            icon: Icon(widget.confirmIcon, size: 18),
            label: Text(widget.confirmLabel),
          )
        else
          FilledButton(
            onPressed: () => Navigator.pop(context, _controller.text.trim()),
            child: Text(widget.confirmLabel),
          ),
      ],
    );
  }
}

class _SettingsResult {
  final String url;
  final String apiKey;

  const _SettingsResult({required this.url, required this.apiKey});
}

class _SettingsDialog extends StatefulWidget {
  final String initialUrl;
  final String initialApiKey;

  const _SettingsDialog({
    required this.initialUrl,
    required this.initialApiKey,
  });

  @override
  State<_SettingsDialog> createState() => _SettingsDialogState();
}

class _SettingsDialogState extends State<_SettingsDialog> {
  late final TextEditingController _urlController;
  late final TextEditingController _keyController;

  @override
  void initState() {
    super.initState();
    _urlController = TextEditingController(text: widget.initialUrl);
    _keyController = TextEditingController(text: widget.initialApiKey);
  }

  @override
  void dispose() {
    _urlController.dispose();
    _keyController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('API & AI Translator Settings'),
      content: SizedBox(
        width: 420,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: _urlController,
              decoration: const InputDecoration(
                labelText: 'Backend API URL',
                hintText: 'http://127.0.0.1:8000',
                helperText: 'Use 10.0.2.2:8000 on Android Emulator',
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _keyController,
              obscureText: true,
              decoration: const InputDecoration(
                labelText: 'AI Translator Key (optional, session-only)',
                hintText: 'Gemini (AIza...), Groq (gsk_...), or OpenAI (sk-...)',
                helperText:
                    'Optional: leave blank to use backend .env or free local parser',
              ),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed: () => Navigator.pop(
            context,
            _SettingsResult(
              url: _urlController.text.trim(),
              apiKey: _keyController.text.trim(),
            ),
          ),
          child: const Text('Save'),
        ),
      ],
    );
  }
}
