import 'package:llamadart/llamadart.dart';
import 'package:test/test.dart';

void main() {
  group('ModelSource', () {
    test('path remains local and does not attempt network resolution', () {
      final source = ModelSource.path('/local/model.gguf');

      expect(source.kind, ModelSourceKind.path);
      expect(source.isLocal, isTrue);
      expect(source.isRemote, isFalse);
      expect(source.path, '/local/model.gguf');
      expect(source.resolvedUri, isNull);
      expect(source.fileName, 'model.gguf');
      expect(source.canonicalKey, 'path:/local/model.gguf');
    });

    test('https URI resolves as HTTP with inferred file name', () {
      final source = ModelSource.url(
        Uri.parse('https://host/path/model.gguf?download=true'),
      );

      expect(source.kind, ModelSourceKind.http);
      expect(source.isRemote, isTrue);
      expect(
        source.resolvedUri,
        Uri.parse('https://host/path/model.gguf?download=true'),
      );
      expect(source.fileName, 'model.gguf');
      expect(source.canonicalKey, 'https://host/path/model.gguf?download=true');
      expect(source.metadataSourceKey, isNot(contains('download=true')));
      expect(source.toString(), isNot(contains('download=true')));
    });

    test('http URLs require an authority and host', () {
      expect(
        () => ModelSource.url(Uri.parse('https:model.gguf')),
        throwsArgumentError,
      );
      expect(() => ModelSource.parse('https:model.gguf'), throwsArgumentError);
    });

    test('remote explicit file names reject traversal and separators', () {
      final invalidNames = <String>[
        '',
        '.',
        '..',
        '../x.gguf',
        'dir/x.gguf',
        r'dir\x.gguf',
        '/x.gguf',
        r'\x.gguf',
        '%2e',
        '%2e%2e',
        '..%2Fx.gguf',
        'dir%2Fx.gguf',
        r'dir%5Cx.gguf',
      ];

      for (final fileName in invalidNames) {
        expect(
          () => ModelSource.url(
            Uri.parse('https://host/model.gguf'),
            fileName: fileName,
          ),
          throwsArgumentError,
          reason: 'url fileName=$fileName',
        );
        expect(
          () => ModelSource.huggingFace(
            repoId: 'owner/repo',
            filePath: 'model.gguf',
            fileName: fileName,
          ),
          throwsArgumentError,
          reason: 'hf fileName=$fileName',
        );
      }
    });

    test('inferred URL file names reject encoded traversal and separators', () {
      final invalidUrls = <String>[
        'https://host/',
        'https://host/.',
        'https://host/..',
        'https://host/%2e',
        'https://host/%2e%2e',
        'https://host/dir%2Fx.gguf',
        'https://host/dir%5Cx.gguf',
        'https://host/%2e%2e%2Fx.gguf',
      ];

      for (final value in invalidUrls) {
        expect(
          () => ModelSource.url(Uri.parse(value)),
          throwsArgumentError,
          reason: value,
        );
      }
    });

    test('hf URI resolves to Hugging Face main download URL', () {
      final source = ModelSource.parse(
        'hf://unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf',
      );

      expect(source.kind, ModelSourceKind.huggingFace);
      expect(source.repoId, 'unsloth/Qwen3.5-0.8B-GGUF');
      expect(source.revision, 'main');
      expect(source.filePath, 'Qwen3.5-0.8B-Q4_K_M.gguf');
      expect(
        source.resolvedUri,
        Uri.parse(
          'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf?download=true',
        ),
      );
      expect(source.fileName, 'Qwen3.5-0.8B-Q4_K_M.gguf');
      expect(
        source.canonicalKey,
        'hf://unsloth/Qwen3.5-0.8B-GGUF@main/Qwen3.5-0.8B-Q4_K_M.gguf',
      );
    });

    test('hf URI resolves explicit revision and nested file path', () {
      final source = ModelSource.parse(
        'hf://owner/repo@rev/sub/dir/model.gguf',
      );

      expect(source.repoId, 'owner/repo');
      expect(source.revision, 'rev');
      expect(source.filePath, 'sub/dir/model.gguf');
      expect(
        source.resolvedUri,
        Uri.parse(
          'https://huggingface.co/owner/repo/resolve/rev/sub/dir/model.gguf?download=true',
        ),
      );
      expect(source.fileName, 'model.gguf');
      expect(source.canonicalKey, 'hf://owner/repo@rev/sub/dir/model.gguf');
    });

    test('parse accepts local paths as explicit convenience', () {
      final source = ModelSource.parse('/local/model.gguf');

      expect(source.kind, ModelSourceKind.path);
      expect(source.path, '/local/model.gguf');
      expect(source.resolvedUri, isNull);
    });

    test('parse rejects invalid schemes', () {
      expect(
        () => ModelSource.parse('ftp://host/model.gguf'),
        throwsArgumentError,
      );
    });

    test('rejects invalid Hugging Face references', () {
      final invalidValues = <String>[
        'hf://owner/repo',
        'hf://owner',
        'hf://owner//model.gguf',
        'hf://owner/repo/../model.gguf',
        'hf://owner/repo//absolute.gguf',
        'hf://owner/repo/%2e%2e/model.gguf',
        'hf://owner/repo/sub/%2E%2E/model.gguf',
        'hf://owner/repo/dir%2Fx.gguf',
        r'hf://owner/repo/dir%5Cx.gguf',
      ];

      for (final value in invalidValues) {
        expect(
          () => ModelSource.parse(value),
          throwsArgumentError,
          reason: value,
        );
      }
    });

    test('canonical sources produce stable deterministic cache keys', () {
      final source = ModelSource.huggingFace(
        repoId: 'owner/repo',
        filePath: 'sub/model.gguf',
      );
      final equivalent = ModelSource.parse('hf://owner/repo/sub/model.gguf');
      final differentRevision = ModelSource.parse(
        'hf://owner/repo@other/sub/model.gguf',
      );
      final differentPath = ModelSource.parse('hf://owner/repo/other.gguf');

      expect(source.canonicalKey, equivalent.canonicalKey);
      expect(source.cacheKey, equivalent.cacheKey);
      expect(source.cacheDirectoryName, equivalent.cacheDirectoryName);
      expect(source.cacheKey, hasLength(64));
      expect(source.cacheDirectoryName, startsWith('model-'));
      expect(
        source.cacheDirectoryName,
        contains(source.cacheKey.substring(0, 12)),
      );
      expect(source.cacheKey, isNot(differentRevision.cacheKey));
      expect(source.cacheKey, isNot(differentPath.cacheKey));
    });
  });
}
