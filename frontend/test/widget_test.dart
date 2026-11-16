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
            },
            {
              'key': 'breathlessness',
              'display_name': 'Breathlessness',
              'aliases': ['dyspnea', 'shortness of breath'],
            },
            {
              'key': 'sweating',
              'display_name': 'Sweating',
              'aliases': ['diaphoresis'],
            },
            {
              'key': 'cough',
              'display_name': 'Cough',
              'aliases': ['coughing'],
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

  testWidgets('Search matches clinical synonym aliases cleanly',
      (WidgetTester tester) async {
    final client = VitalApiClient(httpClient: _buildMockHttp());
    await tester.pumpWidget(VitalAiApp(apiClient: client));
    await tester.pumpAndSettle();

    await tester.enterText(find.byType(TextField), 'dyspnea');
    await tester.pumpAndSettle();

    expect(find.text('Breathlessness (dyspnea)'), findsOneWidget);
    await tester.tap(find.text('Breathlessness (dyspnea)'));
    await tester.pumpAndSettle();
    expect(find.text('Selected (1)'), findsOneWidget);
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
