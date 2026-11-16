import 'dart:io';
import 'dart:typed_data';

Future<String> saveDocumentBytes(Uint8List bytes, String fileName) async {
  Directory targetDir = Directory.systemTemp;
  final home = Platform.environment['USERPROFILE'] ?? Platform.environment['HOME'];
  if (home != null && home.isNotEmpty) {
    final downloads = Directory('$home${Platform.pathSeparator}Downloads');
    if (downloads.existsSync()) {
      targetDir = downloads;
    }
  }
  final outPath = '${targetDir.path}${Platform.pathSeparator}$fileName';
  await File(outPath).writeAsBytes(bytes, flush: true);
  return 'Saved to $outPath';
}
