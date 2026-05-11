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

## Native state persistence

Native backends can save and restore llama.cpp KV-cache state to avoid
re-evaluating a long raw prompt on resume or when forking a prompt prefix. Gate
the flow with `supportsStatePersistence` because the WebGPU bridge does not
expose llama.cpp state files.

```dart
if (!engine.supportsStatePersistence) {
  throw LlamaUnsupportedException('State persistence is native-only.');
}

final prompt = 'You are a concise assistant. Summarize llamadart.';
final tokens = await engine.tokenize(prompt);

// Populate the KV cache, then persist it with the token sequence that produced
// the state. Use your app's own writable path for the state file.
await engine.generate(
  prompt,
  params: const GenerationParams(reusePromptPrefix: true),
).drain<void>();
await engine.stateSaveFile('/path/to/prompt-prefix.state', tokens: tokens);

// Later, after loading the same model with a compatible native runtime build:
final restored = await engine.stateLoadFile(
  '/path/to/prompt-prefix.state',
  tokenCapacity: await engine.getContextSize(),
);

await for (final token in engine.generate(
  '$prompt Continue from the saved prefix.',
  params: const GenerationParams(reusePromptPrefix: true),
)) {
  print(token);
}

print('Restored ${restored.tokens.length} prompt tokens');
```

Important caveats:

- State files are opaque llama.cpp artifacts. Treat them as tied to the same
  model file and compatible native runtime/build that created them.
- `stateLoadFile(...)` restores native KV-cache state only. It does not rebuild
  `ChatSession` message history; persist and reconstruct chat messages
  separately when using the high-level chat API.
- Pass a `tokenCapacity` large enough for the saved prompt token sequence. The
  current context size is usually a safe default.
- WebGPU sessions should check `supportsStatePersistence` and use a normal
  prompt re-evaluation fallback.

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
