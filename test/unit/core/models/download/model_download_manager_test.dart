import 'package:llamadart/llamadart.dart';
import 'package:test/test.dart';

void main() {
  group('model_download_manager export', () {
    test('exports base API and default implementation', () {
      const manager = DefaultModelDownloadManager();

      expect(manager, isA<ModelDownloadManager>());
    });

    test('default IO placeholder throws LlamaUnsupportedException', () async {
      const manager = DefaultModelDownloadManager();

      await expectLater(
        () => manager.ensureModel(
          ModelSource.url(Uri.parse('https://host/model.gguf')),
        ),
        throwsA(isA<LlamaUnsupportedException>()),
      );
    });
  });
}
