import 'package:flutter/material.dart';

import '../models/triage_models.dart';
import '../services/vital_api_client.dart';
import '../widgets/symptom_selector.dart';
import '../widgets/triage_result_panel.dart';

/// Main single-workspace clinical triage screen.
class TriageScreen extends StatefulWidget {
  final VitalApiClient apiClient;

  const TriageScreen({super.key, required this.apiClient});

  @override
  State<TriageScreen> createState() => _TriageScreenState();
}

class _TriageScreenState extends State<TriageScreen> {
  List<SymptomItem> _symptoms = const [];
  final Set<String> _selectedKeys = {};
  PredictResponse? _result;
  bool _loadingCatalog = true;
  bool _predicting = false;
  String? _errorMessage;
  bool _connected = false;

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
        _symptoms = catalog;
        _loadingCatalog = false;
      });
    } catch (err) {
      if (!mounted) return;
      setState(() {
        _connected = false;
        _loadingCatalog = false;
        _errorMessage =
            'Cannot reach backend at ${widget.apiClient.baseUrl}. Start FastAPI (`uvicorn backend.api.app:app --reload`) and tap Retry.';
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
      final response = await widget.apiClient.predict(_selectedKeys.toList());
      if (!mounted) return;
      setState(() {
        _result = response;
        _predicting = false;
      });
    } catch (err) {
      if (!mounted) return;
      setState(() {
        _predicting = false;
        _errorMessage = err.toString();
      });
    }
  }

  Future<void> _editServerUrl() async {
    final controller = TextEditingController(text: widget.apiClient.baseUrl);
    final updated = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Backend API URL'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            hintText: 'http://127.0.0.1:8000',
            helperText: 'Use 10.0.2.2:8000 on Android Emulator',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, controller.text.trim()),
            child: const Text('Save & Connect'),
          ),
        ],
      ),
    );
    controller.dispose();
    if (updated != null && updated.isNotEmpty) {
      widget.apiClient.baseUrl = updated;
      await _loadInitialCatalog();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Row(
          children: [
            Icon(Icons.local_hospital_outlined, color: Color(0xFF0F6E75)),
            SizedBox(width: 10),
            Text(
              'VitalAI Clinical Triage',
              style: TextStyle(fontWeight: FontWeight.w700),
            ),
          ],
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 6),
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
            tooltip: 'Configure API URL',
            icon: const Icon(Icons.settings_outlined),
            onPressed: _editServerUrl,
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
                          backgroundColor: const Color(0xFFFEF2F2),
                          actions: [
                            TextButton(
                              onPressed: _loadInitialCatalog,
                              child: const Text('Retry'),
                            ),
                          ],
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
                            canSubmit: _selectedKeys.isNotEmpty && _connected,
                            onToggleSymptom: _toggleSymptom,
                            onClearAll: () => setState(() {
                              _selectedKeys.clear();
                              _result = null;
                            }),
                            onSubmit: _runPrediction,
                          );
                          final results = TriageResultPanel(
                            result: _result,
                            onAddFollowUpSymptom: _addFollowUpSymptom,
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
