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
Hugging Face references. Source values expose deterministic cache keys and
redacted metadata identities so signed URL query strings are not stored in logs
or cache metadata.

This release adds the API foundation only. Local sources still use the existing
native `loadModel(...)` path, while remote sources require a backend that already
supports URL loading. Package-managed native download/cache IO, authenticated
requests, checksum verification, refresh/cache-only/no-cache policies, and
custom retry/resume behavior are reserved for a later implementation phase and
currently fail with `LlamaUnsupportedException` when requested through the
default resolver.

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
