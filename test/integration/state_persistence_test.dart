@TestOn('vm')
@Timeout(Duration(minutes: 5))
library;

import 'dart:io';
import 'package:test/test.dart';
import 'package:llamadart/llamadart.dart';
import '../test_helper.dart';

/// Round-trip integration test for `llama_state_save_file` and
/// `llama_state_load_file` exposed via [LlamaEngine].
///
/// Saves the KV-cache state of a tiny model after seeding it with a
/// known prompt, then verifies that [stateLoadFile] returns the same
/// token sequence and that the destination context can resume
/// generation without re-evaluating the prompt.
void main() async {
  late File modelFile;
  late LlamaEngine engine;
  late LlamaBackend backend;
  late Directory tmpDir;

  setUpAll(() async {
    modelFile = await TestHelper.getTestModel();
    backend = LlamaBackend();
    engine = LlamaEngine(backend);
    await engine.loadModel(
      modelFile.path,
      modelParams: const ModelParams(contextSize: 256),
    );
    tmpDir = Directory.systemTemp.createTempSync('llamadart_state_test_');
  });

  tearDownAll(() async {
    await engine.dispose();
    if (tmpDir.existsSync()) {
      tmpDir.deleteSync(recursive: true);
    }
  });

  test('reports state persistence support on the native backend', () {
    expect(engine.supportsStatePersistence, isTrue);
  });

  test('saves and loads KV state, recovering the original tokens', () async {
    // Seed the context by tokenizing a known prompt and saving state.
    final tokens = await engine.tokenize('Once upon a time in a quiet land');
    expect(tokens, isNotEmpty);

    final savePath = '${tmpDir.path}/state.bin';
    final saved = await engine.stateSaveFile(savePath, tokens: tokens);
    expect(saved, isTrue, reason: 'stateSaveFile should succeed');
    expect(File(savePath).existsSync(), isTrue);
    expect(File(savePath).lengthSync(), greaterThan(0));

    final result = await engine.stateLoadFile(savePath, tokenCapacity: 256);
    expect(result.tokens, equals(tokens));
  });

  test('rejects non-existent state files', () async {
    final missingPath = '${tmpDir.path}/does-not-exist.bin';
    expect(
      () => engine.stateLoadFile(missingPath, tokenCapacity: 256),
      throwsA(isA<Exception>()),
    );
  });

  test('round-trips an empty token sequence', () async {
    final savePath = '${tmpDir.path}/empty.bin';
    final saved = await engine.stateSaveFile(savePath, tokens: const []);
    expect(saved, isTrue);
    final loaded = await engine.stateLoadFile(savePath, tokenCapacity: 64);
    expect(loaded.tokens, isEmpty);
  });

  test('generate after stateLoadFile reuses KV prefix, not full prompt eval', () async {
    const prompt = 'Once upon a time in a land far away';

    // Evaluate the prompt into the KV cache via a short generation.
    await engine
        .generate(
          prompt,
          params: const GenerationParams(
            maxTokens: 3,
            reusePromptPrefix: true,
          ),
        )
        .drain<void>();
    final fullPerf = await engine.getPerformanceContext();
    final fullEvalTokens = fullPerf?.promptEvalTokens ?? 0;
    expect(
      fullEvalTokens,
      greaterThan(0),
      reason: 'first generate must evaluate the prompt',
    );

    // Save the KV state together with the producing token sequence.
    final tokens = await engine.tokenize(prompt);
    final savePath = '${tmpDir.path}/kv_reuse.bin';
    expect(await engine.stateSaveFile(savePath, tokens: tokens), isTrue);

    // Load the state — this must seed cachedPromptTokens.
    final loadResult = await engine.stateLoadFile(savePath, tokenCapacity: 512);
    expect(loadResult.tokens, equals(tokens));

    // Generate with the same prompt again.  If cachedPromptTokens was seeded
    // correctly the KV prefix is already in place and only the empty suffix
    // (zero tokens) should be evaluated, not the full prompt again.
    await engine
        .generate(
          prompt,
          params: const GenerationParams(
            maxTokens: 3,
            reusePromptPrefix: true,
          ),
        )
        .drain<void>();
    final cachedPerf = await engine.getPerformanceContext();
    final cachedEvalTokens = cachedPerf?.promptEvalTokens ?? fullEvalTokens;

    expect(
      cachedEvalTokens,
      lessThan(fullEvalTokens),
      reason:
          'generate after stateLoadFile should reuse the KV prefix and not re-evaluate the full prompt',
    );
  });
}
