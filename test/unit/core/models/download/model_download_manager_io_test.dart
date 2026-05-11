@TestOn('vm')
library;

import 'package:llamadart/src/core/exceptions.dart';
import 'package:llamadart/src/core/models/download/model_download_manager_io.dart';
import 'package:llamadart/src/core/models/model_source.dart';
import 'package:test/test.dart';

void main() {
  group('DefaultModelDownloadManager IO placeholder', () {
    test('throws unsupported exception for ensureModel', () async {
      const manager = DefaultModelDownloadManager();

      await expectLater(
        manager.ensureModel(ModelSource.path('/models/model.gguf')),
        throwsA(isA<LlamaUnsupportedException>()),
      );
    });
  });
}
