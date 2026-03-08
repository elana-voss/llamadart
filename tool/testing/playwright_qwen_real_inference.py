#!/usr/bin/env python3
import argparse

from playwright_qwen_harness import (
    DEFAULT_APP_URL,
    DEFAULT_MODEL_URL,
    DEFAULT_MMPROJ_URL,
    print_json_result,
    run_bridge_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("app_url", nargs="?", default=DEFAULT_APP_URL)
    parser.add_argument("model_url", nargs="?", default=DEFAULT_MODEL_URL)
    parser.add_argument("mmproj_url", nargs="?", default=DEFAULT_MMPROJ_URL)
    parser.add_argument("--channel", type=str, default="chromium")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--load-timeout-ms", type=int, default=8 * 60 * 1000)
    parser.add_argument("--mmproj-timeout-ms", type=int, default=5 * 60 * 1000)
    parser.add_argument("--inference-timeout-ms", type=int, default=4 * 60 * 1000)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--thread-pool-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--micro-batch-size", type=int, default=256)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--echo-console", action="store_true")
    args = parser.parse_args()

    payload = run_bridge_evaluation(
        app_url=args.app_url,
        channel=args.channel,
        headed=args.headed,
        default_timeout_ms=0,
        console_tail_count=40,
        echo_console=args.echo_console,
        evaluate_script=
        """
            async ({ modelUrl, mmprojUrl, loadTimeoutMs, mmprojTimeoutMs, inferenceTimeoutMs, gpuLayers, threadPoolSize, threads, batchSize, microBatchSize, useCache }) => {
              const withTimeout = (promise, ms, label) =>
                Promise.race([
                  promise,
                  new Promise((_, reject) =>
                    setTimeout(() => reject(new Error(`${label} timeout (${ms}ms)`)), ms),
                  ),
                ]);

              const cfg = {
                workerUrl:
                  typeof window.__llamadartBridgeWorkerUrl === 'string'
                    ? window.__llamadartBridgeWorkerUrl
                    : undefined,
                coreModuleUrl:
                  typeof window.__llamadartBridgeCoreModuleUrl === 'string'
                    ? window.__llamadartBridgeCoreModuleUrl
                    : undefined,
                coreModuleUrlMem64:
                  typeof window.__llamadartBridgeCoreModuleUrlMem64 === 'string'
                    ? window.__llamadartBridgeCoreModuleUrlMem64
                    : undefined,
                wasmUrl:
                  typeof window.__llamadartBridgeWasmUrl === 'string'
                    ? window.__llamadartBridgeWasmUrl
                    : undefined,
                wasmUrlMem64:
                  typeof window.__llamadartBridgeWasmUrlMem64 === 'string'
                    ? window.__llamadartBridgeWasmUrlMem64
                    : undefined,
                preferMemory64: window.__llamadartBridgePreferMemory64 === true,
                threadPoolSize,
                logLevel: 3,
              };

              const bridge = new window.LlamaWebGpuBridge(cfg);
              const loadProgress = [];
              const timings = {};
              const startedAt = performance.now();
              const diagnostics = {
                workerUrl: cfg.workerUrl || null,
                coreModuleUrl: cfg.coreModuleUrl || null,
                bridgeAssetSource: window.__llamadartBridgeAssetSource || null,
                bridgeModuleUrl: window.__llamadartBridgeModuleUrl || null,
                configuredWorkerStallTimeoutMs: cfg.workerGenerationStallTimeoutMs || null,
                effectiveWorkerStallTimeoutMs:
                  typeof bridge._workerCompletionStallTimeoutMs === 'function'
                    ? bridge._workerCompletionStallTimeoutMs({
                        parts: [{ type: 'image', bytes: new Uint8Array([1]) }],
                      })
                    : null,
              };

              const probeFetch = async (url, label) => {
                const controller = new AbortController();
                const timer = setTimeout(() => controller.abort(), 45000);
                try {
                  const response = await fetch(url, {
                    method: 'GET',
                    cache: 'no-store',
                    mode: 'cors',
                    signal: controller.signal,
                  });
                  const contentLength = response.headers.get('content-length');
                  await response.body?.cancel?.();
                  diagnostics[`${label}Probe`] = {
                    ok: response.ok,
                    status: response.status,
                    contentLength,
                    redirected: response.redirected,
                    type: response.type,
                  };
                } catch (error) {
                  diagnostics[`${label}Probe`] = {
                    ok: false,
                    error: String(error),
                  };
                } finally {
                  clearTimeout(timer);
                }
              };

              await probeFetch(modelUrl, 'model');
              await probeFetch(mmprojUrl, 'mmproj');

              try {
                const modelLoadStart = performance.now();
                await withTimeout(
                  bridge.loadModelFromUrl(modelUrl, {
                    nCtx: 4096,
                    nGpuLayers: gpuLayers,
                    nThreads: threads,
                    nThreadsBatch: threads,
                    nBatch: batchSize,
                    nUbatch: microBatchSize,
                    useCache,
                    remoteFetchThresholdBytes: 9000000000000,
                    progressCallback: (progress) => {
                      const loaded = Number(progress?.loaded || 0);
                      const total = Number(progress?.total || 0);
                      loadProgress.push({
                        loaded,
                        total,
                        atMs: performance.now() - startedAt,
                      });
                    },
                  }),
                  loadTimeoutMs,
                  'loadModelFromUrl',
                );
                timings.modelLoadMs = Math.round(performance.now() - modelLoadStart);

                const mmprojStart = performance.now();
                await withTimeout(
                  bridge.loadMultimodalProjector(mmprojUrl),
                  mmprojTimeoutMs,
                  'loadMultimodalProjector',
                );
                timings.mmprojLoadMs = Math.round(performance.now() - mmprojStart);

                diagnostics.preInferenceBackend =
                  typeof bridge.getBackendName === 'function'
                    ? bridge.getBackendName()
                    : null;
                diagnostics.preInferenceGpuActive =
                  typeof bridge.isGpuActive === 'function'
                    ? bridge.isGpuActive()
                    : null;
                diagnostics.preInferenceMetadata =
                  typeof bridge.getModelMetadata === 'function'
                    ? bridge.getModelMetadata()
                    : {};

                const buildSyntheticImageBytes = async () => {
                  const width = 3072;
                  const height = 1792;
                  let blob = null;

                  const drawPattern = (ctx) => {
                    ctx.fillStyle = '#0f172a';
                    ctx.fillRect(0, 0, width, height);
                    const grad = ctx.createLinearGradient(0, 0, width, height);
                    grad.addColorStop(0, '#60a5fa');
                    grad.addColorStop(1, '#f97316');
                    ctx.fillStyle = grad;
                    ctx.fillRect(180, 180, width - 360, height - 360);
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 180px sans-serif';
                    ctx.fillText('llamadart multimodal test', 240, 420);
                    ctx.fillStyle = '#111827';
                    ctx.font = '120px sans-serif';
                    ctx.fillText('large synthetic image payload', 240, 620);
                  };

                  if (typeof OffscreenCanvas === 'function') {
                    const canvas = new OffscreenCanvas(width, height);
                    const ctx = canvas.getContext('2d');
                    if (ctx && typeof canvas.convertToBlob === 'function') {
                      drawPattern(ctx);
                      blob = await canvas.convertToBlob({ type: 'image/png' });
                    }
                  }

                  if (!blob && typeof document !== 'undefined' && typeof document.createElement === 'function') {
                    const canvas = document.createElement('canvas');
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                      drawPattern(ctx);
                      blob = await new Promise((resolve, reject) => {
                        canvas.toBlob((value) => {
                          if (value) {
                            resolve(value);
                            return;
                          }
                          reject(new Error('Failed to create PNG blob from canvas'));
                        }, 'image/png');
                      });
                    }
                  }

                  if (!blob) {
                    throw new Error('Failed to synthesize test image payload');
                  }

                  return new Uint8Array(await blob.arrayBuffer());
                };

                const imageBytes = await buildSyntheticImageBytes();
                diagnostics.syntheticImageBytes = imageBytes.length;

                const inferStart = performance.now();
                const output = await withTimeout(
                  bridge.createCompletion('what do you see?', {
                    nPredict: 64,
                    temp: 0.2,
                    topK: 40,
                    topP: 0.95,
                    penalty: 1.1,
                    parts: [
                      {
                        type: 'image',
                        bytes: imageBytes,
                      },
                    ],
                  }),
                  inferenceTimeoutMs,
                  'createCompletion',
                );
                timings.inferenceMs = Math.round(performance.now() - inferStart);

                const metadata = bridge.getModelMetadata?.() || {};
                return {
                  ok: true,
                  output: String(output || ''),
                  timings,
                  metadata,
                  diagnostics,
                  progressSamples: loadProgress.slice(-8),
                };
              } catch (error) {
                const runtimeNotes = Array.isArray(bridge?._runtime?._runtimeNotes)
                  ? bridge._runtime._runtimeNotes.slice(-30)
                  : [];
                return {
                  ok: false,
                  error: String(error),
                  errorStack:
                    error && typeof error === 'object' && typeof error.stack === 'string'
                      ? error.stack
                      : null,
                  timings,
                  diagnostics: {
                    ...diagnostics,
                    workerFallbackReason: window.__llamadartBridgeWorkerFallbackReason || null,
                    bridgeLoadError: window.__llamadartBridgeLoadError || null,
                    workerPendingCalls:
                      Number(bridge?._workerProxy?._pending?.size) || 0,
                    runtimeNotes,
                    runtimeModelBytes: Number(bridge?._runtime?._modelBytes || 0),
                    runtimeModelSource: bridge?._runtime?._modelSource || null,
                    runtimeCoreVariant: bridge?._runtime?._coreVariant || null,
                    runtimeLastCoreError: bridge?._runtime?._lastCoreErrorText || null,
                    runtimeLastCoreHint: bridge?._runtime?._lastCoreErrorHint || null,
                  },
                  progressSamples: loadProgress.slice(-12),
                };
              } finally {
                try {
                  await bridge.dispose();
                } catch (_) {
                  // best-effort disposal only
                }
              }
            }
            """,
        payload={
            "modelUrl": args.model_url,
            "mmprojUrl": args.mmproj_url,
            "loadTimeoutMs": args.load_timeout_ms,
            "mmprojTimeoutMs": args.mmproj_timeout_ms,
            "inferenceTimeoutMs": args.inference_timeout_ms,
            "gpuLayers": args.gpu_layers,
            "threadPoolSize": args.thread_pool_size,
            "threads": args.threads,
            "batchSize": args.batch_size,
            "microBatchSize": args.micro_batch_size,
            "useCache": args.use_cache,
        },
    )

    print_json_result(payload)
    result = payload.get("result", {})
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
