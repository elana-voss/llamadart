@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:llamadart/src/core/exceptions.dart';
import 'package:llamadart/src/platform/io/model_download_manager_io.dart';
import 'package:llamadart/src/core/models/model_load_options.dart';
import 'package:llamadart/src/core/models/model_source.dart';
import 'package:path/path.dart' as path;
import 'package:test/test.dart';

void main() {
  group('DefaultModelDownloadManager IO', () {
    late Directory tempDir;
    late _ModelHttpFixture server;

    setUp(() async {
      tempDir = await Directory.systemTemp.createTemp(
        'llamadart_model_cache_test_',
      );
      server = await _ModelHttpFixture.start();
    });

    tearDown(() async {
      await server.close();
      if (await tempDir.exists()) {
        await tempDir.delete(recursive: true);
      }
    });

    test(
      'downloads remote source, writes metadata, reports progress, and reuses cache',
      () async {
        server.payload = utf8.encode('model-bytes-v1');
        final manager = DefaultModelDownloadManager(
          defaultCacheDirectory: tempDir.path,
        );
        final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');
        final progress = <double>[];

        final entry = await manager.ensureModel(
          source,
          options: ModelLoadOptions(
            bearerToken: 'secret-token',
            headers: const <String, String>{'X-Test-Header': 'present'},
          ),
          onProgress: (event) {
            final fraction = event.fraction;
            if (fraction != null) {
              progress.add(fraction);
            }
          },
        );

        expect(server.requestCount, 1);
        expect(server.lastAuthorization, 'Bearer secret-token');
        expect(server.lastTestHeader, 'present');
        expect(File(entry.filePath).readAsBytesSync(), server.payload);
        expect(entry.bytes, server.payload.length);
        expect(entry.sha256, isNull);
        expect(entry.sourceCanonicalKey, isNot(contains('secret-token')));
        expect(progress, isNotEmpty);
        expect(progress.last, 1);
        expect(
          File(
            path.join(path.dirname(entry.filePath), 'metadata.json'),
          ).existsSync(),
          isTrue,
        );

        server.payload = utf8.encode('model-bytes-v2');
        final cached = await manager.ensureModel(source);

        expect(
          server.requestCount,
          1,
          reason: 'preferCached should not re-download a completed entry',
        );
        expect(cached.filePath, entry.filePath);
        expect(
          File(cached.filePath).readAsBytesSync(),
          utf8.encode('model-bytes-v1'),
        );
      },
    );

    test(
      'cacheOnly fails on cache miss without touching the network',
      () async {
        final manager = DefaultModelDownloadManager(
          defaultCacheDirectory: tempDir.path,
        );
        final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');

        await expectLater(
          manager.ensureModel(
            source,
            options: ModelLoadOptions(cachePolicy: ModelCachePolicy.cacheOnly),
          ),
          throwsA(isA<LlamaStateException>()),
        );

        expect(server.requestCount, 0);
      },
    );

    test('cacheOnly returns an existing completed cache entry', () async {
      final manager = DefaultModelDownloadManager(
        defaultCacheDirectory: tempDir.path,
      );
      final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');

      server.payload = utf8.encode('cached-model');
      final first = await manager.ensureModel(source);
      server.payload = utf8.encode('remote-newer-model');

      final cached = await manager.ensureModel(
        source,
        options: ModelLoadOptions(cachePolicy: ModelCachePolicy.cacheOnly),
      );

      expect(server.requestCount, 1);
      expect(cached.filePath, first.filePath);
      expect(File(cached.filePath).readAsStringSync(), 'cached-model');
    });

    test('management APIs can target the per-call cache directory', () async {
      final manager = DefaultModelDownloadManager(
        defaultCacheDirectory: tempDir.path,
      );
      final customCacheDirectory = path.join(tempDir.path, 'custom-cache');
      server.payload = utf8.encode('custom-cache-model');
      final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');

      final entry = await manager.ensureModel(
        source,
        options: ModelLoadOptions(cacheDirectory: customCacheDirectory),
      );

      expect(await manager.list(), isEmpty);
      expect(
        await manager.list(cacheDirectory: customCacheDirectory),
        hasLength(1),
      );
      expect(
        await manager.get(
          source.cacheKey,
          cacheDirectory: customCacheDirectory,
        ),
        isNotNull,
      );
      expect(File(entry.filePath).existsSync(), isTrue);

      final pruned = await manager.prune(
        maxBytes: 0,
        cacheDirectory: customCacheDirectory,
      );
      expect(pruned.single.cacheKey, source.cacheKey);
      expect(File(entry.filePath).existsSync(), isFalse);

      final refreshed = await manager.ensureModel(
        source,
        options: ModelLoadOptions(cacheDirectory: customCacheDirectory),
      );
      expect(File(refreshed.filePath).existsSync(), isTrue);

      await manager.remove(
        source.cacheKey,
        cacheDirectory: customCacheDirectory,
      );
      expect(await manager.list(cacheDirectory: customCacheDirectory), isEmpty);

      await manager.ensureModel(
        source,
        options: ModelLoadOptions(cacheDirectory: customCacheDirectory),
      );
      await manager.clear(cacheDirectory: customCacheDirectory);
      expect(await manager.list(cacheDirectory: customCacheDirectory), isEmpty);
    });

    test('stores sha256 metadata only when checksum is requested', () async {
      final manager = DefaultModelDownloadManager(
        defaultCacheDirectory: tempDir.path,
      );
      server.payload = utf8.encode('checksummed-model');
      final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');
      final expectedSha256 = sha256.convert(server.payload).toString();

      final entry = await manager.ensureModel(
        source,
        options: ModelLoadOptions(sha256: expectedSha256),
      );

      expect(entry.sha256, expectedSha256);
      expect(
        File(
          path.join(path.dirname(entry.filePath), 'metadata.json'),
        ).readAsStringSync(),
        contains(expectedSha256),
      );
    });

    test('cache hit with sha256 verifies and persists the digest', () async {
      final manager = DefaultModelDownloadManager(
        defaultCacheDirectory: tempDir.path,
      );
      server.payload = utf8.encode('cached-checksum-model');
      final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');
      final first = await manager.ensureModel(source);
      final expectedSha256 = sha256.convert(server.payload).toString();

      expect(first.sha256, isNull);

      final verified = await manager.ensureModel(
        source,
        options: ModelLoadOptions(sha256: expectedSha256),
      );

      expect(server.requestCount, 1);
      expect(verified.filePath, first.filePath);
      expect(verified.sha256, expectedSha256);
      expect(
        File(
          path.join(path.dirname(verified.filePath), 'metadata.json'),
        ).readAsStringSync(),
        contains(expectedSha256),
      );
    });

    test('local path sha256 is verified before metadata is returned', () async {
      final manager = DefaultModelDownloadManager(
        defaultCacheDirectory: tempDir.path,
      );
      final localFile = File(path.join(tempDir.path, 'local-model.gguf'))
        ..writeAsStringSync('local-model');
      final expectedSha256 = sha256
          .convert(utf8.encode('local-model'))
          .toString();

      final entry = await manager.ensureModel(
        ModelSource.path(localFile.path),
        options: ModelLoadOptions(sha256: expectedSha256),
      );

      expect(entry.filePath, localFile.path);
      expect(entry.sha256, expectedSha256);

      await expectLater(
        manager.ensureModel(
          ModelSource.path(localFile.path),
          options: ModelLoadOptions(sha256: List.filled(64, '0').join()),
        ),
        throwsA(isA<LlamaModelException>()),
      );
    });

    test('normalizes local paths before returning metadata', () async {
      final manager = DefaultModelDownloadManager(
        defaultCacheDirectory: tempDir.path,
      );
      await Directory(path.join(tempDir.path, 'nested')).create();
      final localFile = File(path.join(tempDir.path, 'local-model.gguf'))
        ..writeAsStringSync('local-model');
      final traversalPath = path.join(
        tempDir.path,
        'nested',
        '..',
        'local-model.gguf',
      );

      final entry = await manager.ensureModel(ModelSource.path(traversalPath));

      expect(entry.filePath, path.normalize(path.absolute(localFile.path)));
      expect(entry.filePath, isNot(contains('..')));
    });

    test('missing local path errors include the full path', () async {
      final manager = DefaultModelDownloadManager(
        defaultCacheDirectory: tempDir.path,
      );
      final missingPath = path.join(tempDir.path, 'missing', 'model.gguf');

      await expectLater(
        manager.ensureModel(ModelSource.path(missingPath)),
        throwsA(
          isA<LlamaModelException>().having(
            (error) => error.message,
            'message',
            contains(missingPath),
          ),
        ),
      );
    });

    test(
      'noCache creates transient entries that cache cleanup can remove',
      () async {
        final manager = DefaultModelDownloadManager(
          defaultCacheDirectory: tempDir.path,
        );
        final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');

        server.payload = utf8.encode('first-transient');
        final first = await manager.ensureModel(
          source,
          options: ModelLoadOptions(cachePolicy: ModelCachePolicy.noCache),
        );
        server.payload = utf8.encode('second-transient');
        final second = await manager.ensureModel(
          source,
          options: ModelLoadOptions(cachePolicy: ModelCachePolicy.noCache),
        );

        expect(server.requestCount, 2);
        expect(first.filePath, isNot(second.filePath));
        expect(
          path.basename(path.dirname(first.filePath)),
          contains('-nocache-'),
        );
        expect(await manager.list(), hasLength(2));

        await manager.remove(source.cacheKey);

        expect(File(first.filePath).existsSync(), isFalse);
        expect(File(second.filePath).existsSync(), isFalse);
        expect(await manager.list(), isEmpty);
      },
    );

    test(
      'get finds newest matching cache entry without full metadata scan',
      () async {
        final manager = DefaultModelDownloadManager(
          defaultCacheDirectory: tempDir.path,
        );
        final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');

        server.payload = utf8.encode('stable-entry');
        await manager.ensureModel(source);

        final malformedCandidate = Directory(
          path.join(tempDir.path, '${source.cacheDirectoryName}-nocache-bad'),
        )..createSync(recursive: true);
        File(
          path.join(malformedCandidate.path, 'metadata.json'),
        ).writeAsStringSync('{not valid json');

        server.payload = utf8.encode('transient-entry');
        final transient = await manager.ensureModel(
          source,
          options: ModelLoadOptions(cachePolicy: ModelCachePolicy.noCache),
        );

        final found = await manager.get(source.cacheKey);

        expect(found, isNotNull);
        expect(found!.filePath, transient.filePath);
        expect(File(found.filePath).readAsStringSync(), 'transient-entry');
      },
    );

    test('retries retryable HTTP failures and then succeeds', () async {
      final manager = DefaultModelDownloadManager(
        defaultCacheDirectory: tempDir.path,
      );
      final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');
      server.failuresBeforeSuccess = 1;
      server.failureStatusCode = HttpStatus.internalServerError;
      server.payload = utf8.encode('eventual-model');

      final entry = await manager.ensureModel(
        source,
        options: ModelLoadOptions(maxRetries: 1),
      );

      expect(server.requestCount, 2);
      expect(File(entry.filePath).readAsStringSync(), 'eventual-model');
    });

    test('does not retry non-retryable HTTP failures', () async {
      final manager = DefaultModelDownloadManager(
        defaultCacheDirectory: tempDir.path,
      );
      final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');
      server.failuresBeforeSuccess = 1;
      server.failureStatusCode = HttpStatus.notFound;

      await expectLater(
        manager.ensureModel(source, options: ModelLoadOptions(maxRetries: 3)),
        throwsA(isA<LlamaModelException>()),
      );

      expect(server.requestCount, 1);
    });

    test(
      'refresh re-downloads and replaces the cached file atomically',
      () async {
        final manager = DefaultModelDownloadManager(
          defaultCacheDirectory: tempDir.path,
        );
        final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');

        server.payload = utf8.encode('old-model');
        final first = await manager.ensureModel(source);
        expect(File(first.filePath).readAsStringSync(), 'old-model');

        server.payload = utf8.encode('new-model');
        final refreshed = await manager.ensureModel(
          source,
          options: ModelLoadOptions(cachePolicy: ModelCachePolicy.refresh),
        );

        expect(server.requestCount, 2);
        expect(refreshed.filePath, first.filePath);
        expect(File(refreshed.filePath).readAsStringSync(), 'new-model');
        expect(path.basename(refreshed.filePath), 'tiny.gguf');
        expect(
          Directory(
            path.dirname(refreshed.filePath),
          ).listSync().whereType<File>().map((f) => path.basename(f.path)),
          isNot(contains('tiny.gguf.tmp')),
        );
      },
    );

    test(
      'failed refresh preserves the previous completed cache entry',
      () async {
        final manager = DefaultModelDownloadManager(
          defaultCacheDirectory: tempDir.path,
        );
        final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');

        server.payload = utf8.encode('old-model');
        final first = await manager.ensureModel(source);
        final metadataFile = File(
          path.join(path.dirname(first.filePath), 'metadata.json'),
        );
        final metadataBefore = metadataFile.readAsStringSync();

        server.payload = utf8.encode('corrupt-new-model');
        await expectLater(
          manager.ensureModel(
            source,
            options: ModelLoadOptions(
              cachePolicy: ModelCachePolicy.refresh,
              sha256: List.filled(64, '0').join(),
            ),
          ),
          throwsA(isA<LlamaModelException>()),
        );

        expect(File(first.filePath).readAsStringSync(), 'old-model');
        expect(metadataFile.readAsStringSync(), metadataBefore);
        expect((await manager.get(source.cacheKey))?.filePath, first.filePath);
      },
    );

    test('checksum mismatch removes final file and metadata', () async {
      final manager = DefaultModelDownloadManager(
        defaultCacheDirectory: tempDir.path,
      );
      final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');
      server.payload = utf8.encode('bad-checksum-content');

      await expectLater(
        manager.ensureModel(
          source,
          options: ModelLoadOptions(sha256: List.filled(64, '0').join()),
        ),
        throwsA(isA<LlamaModelException>()),
      );

      final cacheDir = Directory(
        path.join(tempDir.path, source.cacheDirectoryName),
      );
      expect(File(path.join(cacheDir.path, 'tiny.gguf')).existsSync(), isFalse);
      expect(
        File(path.join(cacheDir.path, 'metadata.json')).existsSync(),
        isFalse,
      );
    });

    test(
      'resumes a partial download only when the server returns matching 206',
      () async {
        final manager = DefaultModelDownloadManager(
          defaultCacheDirectory: tempDir.path,
        );
        final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');
        server.payload = utf8.encode('0123456789');
        server.supportRanges = true;
        final cacheDir = Directory(
          path.join(tempDir.path, source.cacheDirectoryName),
        )..createSync(recursive: true);
        File(
          path.join(cacheDir.path, 'tiny.gguf.part'),
        ).writeAsStringSync('0123');
        File(path.join(cacheDir.path, 'tiny.gguf.part.json')).writeAsStringSync(
          '${jsonEncode(<String, Object?>{'etag': server.etag})}\n',
        );

        final entry = await manager.ensureModel(source);

        expect(server.lastRange, 'bytes=4-');
        expect(server.lastIfRange, server.etag);
        expect(File(entry.filePath).readAsStringSync(), '0123456789');
      },
    );

    test(
      'restarts validator-less partial downloads instead of appending',
      () async {
        final manager = DefaultModelDownloadManager(
          defaultCacheDirectory: tempDir.path,
        );
        final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');
        server.payload = utf8.encode('fresh-model');
        server.supportRanges = true;
        final cacheDir = Directory(
          path.join(tempDir.path, source.cacheDirectoryName),
        )..createSync(recursive: true);
        File(
          path.join(cacheDir.path, 'tiny.gguf.part'),
        ).writeAsStringSync('stale');

        final entry = await manager.ensureModel(source);

        expect(server.lastRange, isNull);
        expect(File(entry.filePath).readAsStringSync(), 'fresh-model');
      },
    );

    test(
      'restarts from byte zero when a resume request receives HTTP 200',
      () async {
        final manager = DefaultModelDownloadManager(
          defaultCacheDirectory: tempDir.path,
        );
        final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');
        server.payload = utf8.encode('fresh-content');
        server.supportRanges = false;
        final cacheDir = Directory(
          path.join(tempDir.path, source.cacheDirectoryName),
        )..createSync(recursive: true);
        File(
          path.join(cacheDir.path, 'tiny.gguf.part'),
        ).writeAsStringSync('stale-partial');
        File(path.join(cacheDir.path, 'tiny.gguf.part.json')).writeAsStringSync(
          '${jsonEncode(<String, Object?>{'etag': server.etag})}\n',
        );

        final entry = await manager.ensureModel(source);

        expect(server.lastRange, 'bytes=13-');
        expect(server.lastIfRange, server.etag);
        expect(File(entry.filePath).readAsStringSync(), 'fresh-content');
      },
    );

    test('cancellation leaves no completed cache entry or metadata', () async {
      final manager = DefaultModelDownloadManager(
        defaultCacheDirectory: tempDir.path,
      );
      final source = ModelSource.url(server.modelUri, fileName: 'tiny.gguf');
      server.payload = List<int>.generate(64 * 1024, (index) => index % 251);
      final token = ModelDownloadCancelToken();

      await expectLater(
        manager.ensureModel(
          source,
          options: ModelLoadOptions(cancelToken: token),
          onProgress: (event) {
            if (event.receivedBytes > 0) {
              token.cancel();
            }
          },
        ),
        throwsA(isA<LlamaStateException>()),
      );

      final cacheDir = Directory(
        path.join(tempDir.path, source.cacheDirectoryName),
      );
      expect(File(path.join(cacheDir.path, 'tiny.gguf')).existsSync(), isFalse);
      expect(
        File(path.join(cacheDir.path, 'metadata.json')).existsSync(),
        isFalse,
      );
    });

    test(
      'list, get, remove, clear, and prune operate on persisted metadata',
      () async {
        final manager = DefaultModelDownloadManager(
          defaultCacheDirectory: tempDir.path,
        );
        server.payload = utf8.encode('first');
        final firstSource = ModelSource.url(
          server.modelUri,
          fileName: 'first.gguf',
        );
        final first = await manager.ensureModel(firstSource);

        server.payload = utf8.encode('second');
        final secondSource = ModelSource.url(
          server.modelUri.replace(queryParameters: <String, String>{'id': '2'}),
          fileName: 'second.gguf',
        );
        final second = await manager.ensureModel(secondSource);

        expect(
          (await manager.list()).map((entry) => entry.cacheKey),
          containsAll(<String>[first.cacheKey, second.cacheKey]),
        );
        expect((await manager.get(first.cacheKey))?.fileName, 'first.gguf');

        await manager.remove(first.cacheKey);
        expect(await manager.get(first.cacheKey), isNull);
        expect(File(first.filePath).existsSync(), isFalse);
        expect(await manager.get(second.cacheKey), isNotNull);

        final secondMetadataFile = File(
          path.join(path.dirname(second.filePath), 'metadata.json'),
        );
        final secondMetadata = Map<String, Object?>.from(
          jsonDecode(secondMetadataFile.readAsStringSync()) as Map,
        );
        secondMetadata.remove('bytes');
        secondMetadataFile.writeAsStringSync(
          '${const JsonEncoder.withIndent('  ').convert(secondMetadata)}\n',
        );

        final pruned = await manager.prune(maxBytes: 1);
        expect(
          pruned.map((entry) => entry.cacheKey),
          contains(second.cacheKey),
        );
        expect(await manager.list(), isEmpty);

        final third = await manager.ensureModel(
          firstSource,
          options: ModelLoadOptions(cachePolicy: ModelCachePolicy.refresh),
        );
        expect(File(third.filePath).existsSync(), isTrue);
        await manager.clear();
        expect(await manager.list(), isEmpty);
        expect(
          Directory(
            path.join(tempDir.path, firstSource.cacheDirectoryName),
          ).existsSync(),
          isFalse,
        );
      },
    );
  });
}

class _ModelHttpFixture {
  _ModelHttpFixture._(this._server);

  final HttpServer _server;
  List<int> payload = utf8.encode('model-bytes');
  bool supportRanges = false;
  String? etag = '"fixture-v1"';
  int failuresBeforeSuccess = 0;
  int failureStatusCode = HttpStatus.internalServerError;
  int requestCount = 0;
  String? lastRange;
  String? lastIfRange;
  String? lastAuthorization;
  String? lastTestHeader;

  Uri get modelUri => Uri.parse(
    'http://127.0.0.1:${_server.port}/models/tiny.gguf?download=true',
  );

  static Future<_ModelHttpFixture> start() async {
    final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    final fixture = _ModelHttpFixture._(server);
    server.listen(fixture._handleRequest);
    return fixture;
  }

  Future<void> close() => _server.close(force: true);

  void _handleRequest(HttpRequest request) async {
    requestCount += 1;
    lastRange = request.headers.value(HttpHeaders.rangeHeader);
    lastIfRange = request.headers.value(HttpHeaders.ifRangeHeader);
    lastAuthorization = request.headers.value(HttpHeaders.authorizationHeader);
    lastTestHeader = request.headers.value('X-Test-Header');

    final currentEtag = etag;
    if (currentEtag != null) {
      request.response.headers.set(HttpHeaders.etagHeader, currentEtag);
    }

    if (requestCount <= failuresBeforeSuccess) {
      request.response.statusCode = failureStatusCode;
      await request.response.close();
      return;
    }

    final range = lastRange;
    if (supportRanges && range != null && range.startsWith('bytes=')) {
      final startText = range.substring('bytes='.length).split('-').first;
      final start = int.parse(startText);
      request.response.statusCode = HttpStatus.partialContent;
      request.response.headers.set(HttpHeaders.acceptRangesHeader, 'bytes');
      request.response.headers.set(
        HttpHeaders.contentRangeHeader,
        'bytes $start-${payload.length - 1}/${payload.length}',
      );
      request.response.headers.contentLength = payload.length - start;
      request.response.add(payload.sublist(start));
    } else {
      request.response.statusCode = HttpStatus.ok;
      request.response.headers.contentLength = payload.length;
      request.response.add(payload);
    }
    await request.response.close();
  }
}
