import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../models/triage_models.dart';

class ApiException implements Exception {
  final String message;
  final int? statusCode;

  const ApiException(this.message, {this.statusCode});

  @override
  String toString() => message;
}

/// Lightweight REST client connecting Flutter UI to the VitalAI FastAPI backend.
class VitalApiClient {
  String baseUrl;
  final http.Client _http;

  VitalApiClient({
    this.baseUrl = 'http://127.0.0.1:8000',
    http.Client? httpClient,
  }) : _http = httpClient ?? http.Client();

  Uri _uri(String path) {
    final trimmed = baseUrl.trim().replaceAll(RegExp(r'/+$'), '');
    return Uri.parse('$trimmed$path');
  }

  Future<HealthStatus> fetchHealth() async {
    final response = await _http
        .get(_uri('/health'))
        .timeout(const Duration(seconds: 6));
    if (response.statusCode != 200) {
      throw ApiException(
        _extractError(utf8.decode(response.bodyBytes), 'Health check failed'),
        statusCode: response.statusCode,
      );
    }
    return HealthStatus.fromJson(
      jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>,
    );
  }

  Future<List<SymptomItem>> fetchSymptoms() async {
    final response = await _http
        .get(_uri('/symptoms'))
        .timeout(const Duration(seconds: 6));
    if (response.statusCode != 200) {
      throw ApiException(
        _extractError(
            utf8.decode(response.bodyBytes), 'Unable to load symptom catalog'),
        statusCode: response.statusCode,
      );
    }
    final data =
        jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
    final items = data['symptoms'] as List? ?? const [];
    return items
        .map((e) => SymptomItem.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<PredictResponse> predict(
    List<String> symptoms, {
    int topK = 3,
  }) async {
    final response = await _http
        .post(
          _uri('/predict'),
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode({'symptoms': symptoms, 'top_k': topK}),
        )
        .timeout(const Duration(seconds: 8));
    if (response.statusCode != 200) {
      throw ApiException(
        _extractError(
            utf8.decode(response.bodyBytes), 'Prediction request failed'),
        statusCode: response.statusCode,
      );
    }
    return PredictResponse.fromJson(
      jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>,
    );
  }

  Future<Uint8List> exportDocx(
    List<String> symptoms, {
    String patientName = '',
    int topK = 3,
  }) async {
    final response = await _http
        .post(
          _uri('/export-docx'),
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode({
            'symptoms': symptoms,
            'top_k': topK,
            'patient_name': patientName,
          }),
        )
        .timeout(const Duration(seconds: 10));
    if (response.statusCode != 200) {
      throw ApiException(
        _extractError(
            utf8.decode(response.bodyBytes), 'Word report export failed'),
        statusCode: response.statusCode,
      );
    }
    return response.bodyBytes;
  }

  Future<Map<String, dynamic>> fetchMetrics() async {
    final response = await _http
        .get(_uri('/metrics?compact=true'))
        .timeout(const Duration(seconds: 6));
    if (response.statusCode != 200) {
      throw ApiException(
        _extractError(
            utf8.decode(response.bodyBytes), 'Unable to load model metrics'),
        statusCode: response.statusCode,
      );
    }
    return jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
  }

  Future<ParseSymptomsResponse> parseSymptoms(
    String text, {
    String? apiKey,
    String provider = 'auto',
  }) async {
    final response = await _http
        .post(
          _uri('/parse-symptoms'),
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode({
            'text': text,
            'provider': provider,
            if (apiKey != null && apiKey.trim().isNotEmpty)
              'api_key': apiKey.trim(),
          }),
        )
        .timeout(const Duration(seconds: 15));
    if (response.statusCode != 200) {
      throw ApiException(
        _extractError(
            utf8.decode(response.bodyBytes), 'Unable to parse symptoms'),
        statusCode: response.statusCode,
      );
    }
    return ParseSymptomsResponse.fromJson(
      jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>,
    );
  }

  static String _extractError(String body, String fallback) {
    try {
      final parsed = jsonDecode(body);
      if (parsed is Map<String, dynamic> && parsed['detail'] != null) {
        final detail = parsed['detail'];
        if (detail is String) return detail;
        if (detail is Map && detail['message'] is String) {
          return detail['message'] as String;
        }
      }
    } catch (_) {
      // Fall back to default message when response body is non-JSON.
    }
    return fallback;
  }
}
