import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:vital_ai/main.dart';
import 'package:vital_ai/services/vital_api_client.dart';

MockClient _buildMockHttp({bool failConnection = false}) {
  return MockClient((request) async {
    if (failConnection) {
      return http.Response('Service unavailable', 503);
    }
    const jsonHeaders = {'content-type': 'application/json; charset=utf-8'};
    if (request.url.path == '/health') {
      return http.Response(
        jsonEncode({
          'status': 'ok',
          'model_loaded': true,
          'model_type': 'multinomial',
          'n_features': 131,
          'n_classes': 41,
          'llm_translator_configured': false,
        }),
        200,
        headers: jsonHeaders,
      );
    }
    if (request.url.path == '/parse-symptoms') {
      return http.Response(
        jsonEncode({
          'matched_symptoms': [
            {
              'key': 'chest_pain',
              'display_name': 'Chest Pain',
              'aliases': ['angina'],
              'category': 'Cardio & Respiratory',
            },
            {
              'key': 'sweating',
              'display_name': 'Sweating',
              'aliases': ['diaphoresis'],
              'category': 'Fever & Systemic',
            },
          ],
          'unmapped_terms': ['earache'],
          'negated_terms': ['cough'],
          'mode': 'local_parser',
          'note': 'Parsed locally using clinical synonym & typo matcher.',
        }),
        200,
        headers: jsonHeaders,
      );
    }
    if (request.url.path == '/symptoms') {
      return http.Response(
        jsonEncode({
          'count': 4,
          'symptoms': [
            {
              'key': 'chest_pain',
              'display_name': 'Chest Pain',
              'aliases': ['angina', 'thoracic pain'],
              'category': 'Cardio & Respiratory',
            },
            {
              'key': 'breathlessness',
              'display_name': 'Breathlessness',
              'aliases': ['dyspnea', 'shortness of breath'],
              'category': 'Cardio & Respiratory',
            },
            {
              'key': 'sweating',
              'display_name': 'Sweating',
              'aliases': ['diaphoresis'],
              'category': 'Fever & Systemic',
            },
            {
              'key': 'cough',
              'display_name': 'Cough',
              'aliases': ['coughing'],
              'category': 'Cardio & Respiratory',
            },
          ],
        }),
        200,
        headers: jsonHeaders,
      );
    }
    if (request.url.path == '/export-docx') {
      return http.Response.bytes(
        [80, 75, 3, 4, 1, 2, 3],
        200,
        headers: const {
          'content-type':
              'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        },
      );
    }
    if (request.url.path == '/metrics') {
      return http.Response(
        jsonEncode({
          'dataset_diagnostics': {
            'n_train': 4920,
            'n_features': 131,
            'n_classes': 41,
            'train_dedup': {'unique_pairs': 304},
          },
          'recommended_model': {
            'overfitting_audit': {'status': 'PASS_LOW_MISTAKE_NO_OVERFIT'},
          },
          'model_comparison': [
            {
              'model_name': 'MultinomialNB(alpha=0.1)',
              'cv_summary': {
                'mean_val_accuracy': 1.0,
                'mean_noisy_val_accuracy': 0.9708,
              },
              'dropout_20_accuracy': 0.9805,
            },
          ],
        }),
        200,
        headers: jsonHeaders,
      );
    }
    if (request.url.path == '/predict') {
      return http.Response(
        jsonEncode({
          'normalized_symptoms': ['chest_pain', 'sweating'],
          'predictions': [
            {
              'disease': 'Heart attack',
              'display_name': 'Myocardial Infarction (Heart Attack)',
              'probability': 0.9374,
              'rank': 1,
              'category': 'cardiovascular',
              'base_urgency': 'Critical',
              'supporting_symptoms': ['Chest Pain', 'Sweating'],
              'missing_hallmarks': ['Breathlessness'],
              'reasoning_summary':
                  'Matched Chest Pain, Sweating (93.7% probability). Unreported hallmarks: Breathlessness.',
            },
            {
              'disease': 'Pneumonia',
              'display_name': 'Pneumonia',
              'probability': 0.0476,
              'rank': 2,
              'category': 'respiratory_infectious',
              'base_urgency': 'High',
              'supporting_symptoms': ['Chest Pain', 'Sweating'],
              'missing_hallmarks': ['High Fever', 'Cough'],
              'reasoning_summary':
                  'Shares Chest Pain, Sweating; ranked lower without High Fever, Cough.',
            },
          ],
          'primary_reasoning':
              'Top assessment is Myocardial Infarction (Heart Attack) (93.7%): Matched Chest Pain, Sweating.',
          'alternative_causes': [
            {
              'disease': 'Pneumonia',
              'display_name': 'Pneumonia',
              'probability': 0.0476,
              'base_urgency': 'High',
              'matched_symptoms': ['Chest Pain', 'Sweating'],
              'missing_hallmarks': ['High Fever', 'Cough'],
              'why_possible':
                  'Can explain Chest Pain, Sweating; typically also presents with High Fever, Cough',
            },
          ],
          'symptom_causes': [
            {
              'symptom': 'chest_pain',
              'display_name': 'Chest Pain',
              'possible_diseases': [
                'Myocardial Infarction (Heart Attack)',
                'Pneumonia',
                'Gastroesophageal Reflux Disease (GERD)',
              ],
            },
          ],
          'uncertainty': {
            'normalized_entropy': 0.1,
            'confidence_margin': 0.8898,
            'low_confidence': false,
          },
          'symptom_contributions': [
            {
              'symptom': 'chest_pain',
              'display_name': 'Chest Pain',
              'log_likelihood_lift': 4.12,
            },
          ],
          'follow_up_symptoms': [
            {
              'symptom': 'breathlessness',
              'display_name': 'Breathlessness',
              'separation_score': 1.25,
              'favors': 'Heart attack',
            },
          ],
          'triage': {
            'urgency': 'Critical',
            'specialist': 'Cardiologist / Emergency Medicine',
            'category': 'cardiovascular',
            'red_flags': ['Acute Coronary / Cardiac Red Flag'],
            'immediate_actions': [
              'Call emergency services immediately — do not drive yourself to the hospital.',
            ],
            'precautions': [],
            'warnings': [],
            'differential_alert': false,
          },
          'inference_trace': [
            {
              'iteration': 1,
              'rule_name': 'R2_CardiacRedFlag',
              'matched_facts': ["HasSymptom('chest_pain')", "HasSymptom('sweating')"],
              'asserted_facts': ["EscalatedUrgency('Critical')"],
              'explanation': 'Chest pain with sweating triggers cardiac emergency escalation.',
            },
          ],
          'disclaimer': 'Clinical decision-support prototype only.',
        }),
        200,
        headers: jsonHeaders,
      );
    }
    return http.Response('Not found', 404);
  });
}

void main() {
  testWidgets('Doctor triage flow renders urgency, specialist, and progressive logic',
      (WidgetTester tester) async {
    final client = VitalApiClient(httpClient: _buildMockHttp());
    await tester.pumpWidget(VitalAiApp(apiClient: client));
    await tester.pumpAndSettle();

    expect(find.text('VitalAI Clinical Triage'), findsOneWidget);
    expect(find.text('API Ready'), findsOneWidget);
    expect(find.text('Chest Pain'), findsOneWidget);

    await tester.tap(find.text('Chest Pain'));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.text('Sweating'));
    await tester.tap(find.text('Sweating'));
    await tester.pumpAndSettle();

    final assessBtn = find.text('Assess Triage');
    await tester.ensureVisible(assessBtn);
    await tester.tap(assessBtn);
    await tester.pumpAndSettle();

    expect(find.text('CRITICAL URGENCY'), findsOneWidget);
    expect(find.text('Myocardial Infarction (Heart Attack)'), findsOneWidget);
    expect(find.text('Refer to: Cardiologist / Emergency Medicine'), findsOneWidget);
    expect(find.textContaining('Acute Coronary / Cardiac Red Flag'), findsOneWidget);
    expect(find.textContaining('Shares Chest Pain, Sweating'), findsOneWidget);

    // Alternative causes & rule trace start collapsed to reduce cognitive load.
    final altHeader = find.textContaining('Other Possible Causes');
    await tester.ensureVisible(altHeader);
    await tester.tap(altHeader);
    await tester.pumpAndSettle();
    expect(find.textContaining('typically also presents with High Fever, Cough'),
        findsOneWidget);

    expect(find.textContaining('R2_CardiacRedFlag'), findsNothing);
    final logicHeader = find.textContaining('Rule Logic & Symptom Weights');
    await tester.ensureVisible(logicHeader);
    await tester.tap(logicHeader);
    await tester.pumpAndSettle();
    expect(find.textContaining('R2_CardiacRedFlag'), findsOneWidget);
  });

  testWidgets('Search Enter shortcut, Undo Clear, Word export, and Model Audit work',
      (WidgetTester tester) async {
    String? savedName;
    int? savedLength;
    final client = VitalApiClient(httpClient: _buildMockHttp());
    await tester.pumpWidget(
      VitalAiApp(
        apiClient: client,
        onSaveDocx: (bytes, fileName) async {
          savedName = fileName;
          savedLength = bytes.length;
          return 'Saved $fileName';
        },
      ),
    );
    await tester.pumpAndSettle();

    // Alias search + Enter adds top match
    await tester.enterText(find.byType(TextField), 'dyspnea');
    await tester.pumpAndSettle();
    expect(find.text('Breathlessness (dyspnea)'), findsOneWidget);
    await tester.testTextInput.receiveAction(TextInputAction.done);
    await tester.pumpAndSettle();
    expect(find.text('Selected (1)'), findsOneWidget);

    // Clear all + Undo restores symptom
    await tester.tap(find.text('Clear all'));
    await tester.pumpAndSettle();
    expect(find.text('Selected (1)'), findsNothing);
    await tester.tap(find.text('Undo'));
    await tester.pumpAndSettle();
    expect(find.text('Selected (1)'), findsOneWidget);

    // Filter pill scopes visible chips cleanly
    final filterPill = find.text('Fever & Systemic');
    await tester.ensureVisible(filterPill);
    await tester.tap(filterPill);
    await tester.pumpAndSettle();
    expect(find.text('Sweating'), findsOneWidget);
    expect(find.text('Chest Pain'), findsNothing);

    await tester.ensureVisible(find.text('Sweating'));
    await tester.tap(find.text('Sweating'));
    await tester.pumpAndSettle();

    final assessBtn = find.text('Assess Triage');
    await tester.ensureVisible(assessBtn);
    await tester.tap(assessBtn);
    await tester.pumpAndSettle();

    // In-session history chip appears
    expect(find.text('Recent:'), findsOneWidget);

    // Word .docx download workflow
    final exportBtn = find.text('Download Word (.docx)');
    await tester.ensureVisible(exportBtn);
    await tester.tap(exportBtn);
    await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextField).last, 'John Doe');
    await tester.tap(find.text('Download .docx'));
    await tester.pumpAndSettle();

    expect(savedName, 'vitalai_consultation_note.docx');
    expect(savedLength, 7);
    expect(find.text('Saved vitalai_consultation_note.docx'), findsOneWidget);

    // Model Audit dialog loads metrics
    await tester.tap(find.byIcon(Icons.analytics_outlined));
    await tester.pumpAndSettle();
    expect(find.text('Model Audit & Evaluation Summary'), findsOneWidget);
    expect(find.textContaining('PASS_LOW_MISTAKE_NO_OVERFIT'), findsOneWidget);
  });

  testWidgets('Dark theme toggle and bulk symptom extractor work smoothly',
      (WidgetTester tester) async {
    final client = VitalApiClient(httpClient: _buildMockHttp());
    await tester.pumpWidget(VitalAiApp(apiClient: client));
    await tester.pumpAndSettle();

    // Toggle light -> dark
    expect(find.byIcon(Icons.dark_mode_outlined), findsOneWidget);
    await tester.tap(find.byIcon(Icons.dark_mode_outlined));
    await tester.pumpAndSettle();
    expect(find.byIcon(Icons.light_mode_outlined), findsOneWidget);

    // Open bulk paste extractor, extract & assess
    await tester.tap(find.text('Paste note / bulk list'));
    await tester.pumpAndSettle();
    await tester.enterText(
      find.byType(TextField).last,
      'chest pain, sweaty, earache, no cough',
    );
    await tester.tap(find.text('Extract & Assess'));
    await tester.pumpAndSettle();

    expect(find.text('Selected (2)'), findsOneWidget);
    expect(find.text('Not in dataset: earache'), findsOneWidget);
    expect(find.text('Ignored negated: cough'), findsOneWidget);
    expect(find.text('CRITICAL URGENCY'), findsOneWidget);
  });

  testWidgets('Offline backend displays retry banner cleanly',
      (WidgetTester tester) async {
    final client = VitalApiClient(httpClient: _buildMockHttp(failConnection: true));
    await tester.pumpWidget(VitalAiApp(apiClient: client));
    await tester.pumpAndSettle();

    expect(find.text('Offline'), findsOneWidget);
    expect(find.text('Retry'), findsOneWidget);
  });
}
