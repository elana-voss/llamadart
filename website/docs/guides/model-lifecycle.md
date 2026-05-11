---
title: Model Lifecycle
---

This guide covers model load/unload flows and safe lifecycle patterns.

## Single model lifecycle

```dart
final engine = LlamaEngine(LlamaBackend());
await engine.loadModel('/path/to/model.gguf');

// ...run inference...

await engine.unloadModel();
await engine.dispose();
```

## Switching models

`LlamaEngine.loadModel(...)` requires no currently loaded model. Unload first:

```dart
await engine.unloadModel();
await engine.loadModel('/path/to/another_model.gguf');
```

## Load from URL (web-focused)

```dart
await engine.loadModelFromUrl(
  'https://example.com/model.gguf',
  onProgress: (progress) => print('progress: $progress'),
);
```

`loadModelFromUrl` requires a backend with URL loading support.

## Structured model sources

Use `loadModelSource(...)` when the caller wants to describe the model location
as a value object instead of branching on raw strings:

```dart
await engine.loadModelSource(ModelSource.path('/path/to/model.gguf'));

await engine.loadModelSource(
  ModelSource.url(Uri.parse('https://example.com/model.gguf')),
  onProgress: (progress) {
    final fraction = progress.fraction;
    if (fraction != null) {
      print('download progress: ${(fraction * 100).toStringAsFixed(1)}%');
    }
  },
);

await engine.loadModelSource(
  ModelSource.parse('hf://owner/repo/path/to/model.gguf'),
);
```

`ModelSource.parse(...)` accepts local paths, HTTP(S) URLs, and `hf://`
Hugging Face references. Local path sources require native/file-backed targets;
URL-loading web backends reject explicit local filesystem paths and should use
HTTP(S) or `hf://` sources instead. Source values expose deterministic cache
keys and redacted metadata identities so signed URL query strings are not stored
in logs or cache metadata.

Native/file-backed backends download remote sources into the package-managed
cache, verify the completed file, persist metadata, and then call the existing
local `loadModel(...)` path. URL-capable web backends keep using
`loadModelFromUrl(...)` for simple unauthenticated `preferCached` requests; use
native/file-backed targets when you need authenticated headers, checksum
verification, explicit cache policies, retries, or resume.

```dart
final cancelToken = ModelDownloadCancelToken();

await engine.loadModelSource(
  ModelSource.url(Uri.parse('https://example.com/model.gguf')),
  options: ModelLoadOptions(
    cachePolicy: ModelCachePolicy.preferCached,
    cacheDirectory: '/app/cache/llamadart-models',
    sha256: '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
    bearerToken: 'hf_xxx',
    cancelToken: cancelToken,
    resume: true,
    maxRetries: 3,
  ),
  onProgress: (progress) {
    final fraction = progress.fraction;
    if (fraction != null) {
      print('download progress: ${(fraction * 100).toStringAsFixed(1)}%');
    }
  },
);
```

Cache policies:

- `preferCached`: reuse a completed cache entry when present; otherwise download.
- `refresh`: replace the cached file atomically.
- `cacheOnly`: fail without a network request when the entry is missing.
- `noCache`: download to a temporary manager entry without source-keyed reuse; call
  `remove(entry.cacheKey)`/`clear()` when the returned file is no longer needed, or
  use thresholded `prune(...)` for age/size-based cleanup.

The download manager can also inspect and clean the persisted cache:

```dart
final manager = DefaultModelDownloadManager(
  defaultCacheDirectory: '/app/cache/llamadart-models',
);

final cached = await manager.list();
final entry = await manager.get(ModelSource.parse('hf://owner/repo/model.gguf').cacheKey);
if (entry != null) {
  await manager.remove(entry.cacheKey);
}
await manager.prune(maxAge: const Duration(days: 30), maxBytes: 20 * 1024 * 1024 * 1024);
await manager.clear();
```

Downloaded files are written to `.part` files and promoted to the completed
model path only after the HTTP stream and optional SHA-256 verification succeed.
Retry/resume use HTTP Range only when the partial file has a safe validator
(ETag/Last-Modified) or the caller supplied a SHA-256 checksum; validator-less
partials restart from byte zero. If a server ignores a resume request and
returns `200 OK`, the manager also restarts from byte zero. Signed URL query
strings, fragments, and userinfo are redacted from display strings and metadata.

### Mobile large-download guidance

For Flutter apps, pass an app-controlled `cacheDirectory` from your storage
strategy (for example an application-support or documents directory selected by
`path_provider`). Surface progress/cancel controls in the UI, keep downloads
serialized for large GGUF files, and expect OS backgrounding to interrupt active
requests. The `.part` resume support lets a later foreground session continue
when the server supports Range requests and exposes a safe validator or the
caller supplies SHA-256. On Android, prefer app-private storage unless the app
intentionally exposes model files to the user; on iOS, avoid cache directories
that the OS may purge while a model is still needed.

## Multimodal projector lifecycle

```dart
await engine.loadMultimodalProjector('/path/to/mmproj.gguf');
final canSee = await engine.supportsVision;
final canHear = await engine.supportsAudio;
print('vision=$canSee audio=$canHear');
```

Projector resources are released by `unloadModel()` or `dispose()`.

## LoRA adapters at runtime

```dart
await engine.setLora('/path/to/adapter.gguf', scale: 0.8);
await engine.removeLora('/path/to/adapter.gguf');
await engine.clearLoras();
```

See [LoRA Adapters](./lora-adapters) for scaling strategy, stacking, and
platform-specific behavior.

## Recommended lifecycle checks

- Check `engine.isReady` before inference paths.
- Use `try/finally` to guarantee `dispose()` on shutdown.
- Keep model switch logic serialized to avoid overlapping load/unload calls.
