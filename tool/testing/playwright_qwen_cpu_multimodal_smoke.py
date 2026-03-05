#!/usr/bin/env python3
import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright


DEFAULT_APP_URL = "http://127.0.0.1:7357"
DEFAULT_MODEL_URL = (
    "https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/"
    "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf?download=true"
)
DEFAULT_MMPROJ_URL = (
    "https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/"
    "mmproj-Qwen_Qwen3.5-0.8B-f16.gguf?download=true"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("app_url", nargs="?", default=DEFAULT_APP_URL)
    parser.add_argument("model_url", nargs="?", default=DEFAULT_MODEL_URL)
    parser.add_argument("mmproj_url", nargs="?", default=DEFAULT_MMPROJ_URL)
    parser.add_argument("--model-timeout-ms", type=int, default=8 * 60 * 1000)
    parser.add_argument("--mmproj-timeout-ms", type=int, default=5 * 60 * 1000)
    parser.add_argument("--infer-timeout-ms", type=int, default=5 * 60 * 1000)
    parser.add_argument("--n-predict", type=int, default=192)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--media-max-image-pixels", type=int, default=0)
    parser.add_argument("--media-max-image-edge", type=int, default=0)
    parser.add_argument("--channel", type=str, default="chromium")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()

    app_url = args.app_url
    model_url = args.model_url
    mmproj_url = args.mmproj_url
    image_path = args.image_path.strip()

    image_bytes_base64: str | None = None
    image_file_name: str | None = None
    if image_path:
        image_file = Path(image_path).expanduser()
        if not image_file.exists() or not image_file.is_file():
            raise FileNotFoundError(f"Image file not found: {image_file}")

        image_bytes = image_file.read_bytes()
        if not image_bytes:
            raise ValueError(f"Image file is empty: {image_file}")

        image_bytes_base64 = base64.b64encode(image_bytes).decode("ascii")
        image_file_name = image_file.name

    console_logs: list[dict[str, Any]] = []

    with sync_playwright() as playwright:
        launch_kwargs = {
            "headless": not args.headed,
            "args": [
                "--enable-unsafe-webgpu",
                "--disable-vulkan-surface",
                "--enable-features=Vulkan",
            ],
        }
        if args.channel and args.channel != "chromium":
            launch_kwargs["channel"] = args.channel

        browser = playwright.chromium.launch(
            **launch_kwargs,
        )
        page = browser.new_page()
        page.set_default_timeout(120000)

        def on_console(message: Any) -> None:
            try:
                text = message.text
            except Exception:  # pragma: no cover
                text = str(message)
            console_logs.append({"type": message.type, "text": text})
            if "llamadart:" in text or "RuntimeError" in text or "failed" in text.lower():
                print(f"[browser:{message.type}] {text}", flush=True)

        page.on("console", on_console)

        print("[e2e] opening app", flush=True)
        page.goto(app_url, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_function(
            "() => typeof window.LlamaWebGpuBridge === 'function'",
            timeout=120000,
        )
        print("[e2e] app ready", flush=True)

        result = page.evaluate(
            """
            async ({ modelUrl, mmprojUrl, modelTimeoutMs, mmprojTimeoutMs, inferTimeoutMs, nPredict, nGpuLayers, nThreads, imageBytesBase64, imageFileName, mediaMaxImagePixels, mediaMaxImageEdge }) => {
              const withTimeout = (promise, ms, label) =>
                Promise.race([
                  promise,
                  new Promise((_, reject) =>
                    setTimeout(() => reject(new Error(`${label} timeout (${ms}ms)`)), ms),
                  ),
                ]);

              const bridge = new window.LlamaWebGpuBridge({
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
                threadPoolSize:
                  Number.isFinite(Number(window.__llamadartBridgeThreadPoolSize))
                    ? Number(window.__llamadartBridgeThreadPoolSize)
                    : undefined,
                logLevel: 3,
              });

              const timings = {};
              try {
                const loadStart = performance.now();
                await withTimeout(
                  bridge.loadModelFromUrl(modelUrl, {
                    nCtx: 4096,
                    nGpuLayers,
                    nThreads,
                    useCache: true,
                    remoteFetchThresholdBytes: 9000000000000,
                  }),
                  modelTimeoutMs,
                  'loadModelFromUrl',
                );
                timings.modelLoadMs = Math.round(performance.now() - loadStart);

                const mmprojStart = performance.now();
                await withTimeout(
                  bridge.loadMultimodalProjector(mmprojUrl),
                  mmprojTimeoutMs,
                  'loadMultimodalProjector',
                );
                timings.mmprojLoadMs = Math.round(performance.now() - mmprojStart);

                const decodeBase64ToUint8Array = (value) => {
                  const binary = atob(value);
                  const out = new Uint8Array(binary.length);
                  for (let i = 0; i < binary.length; i += 1) {
                    out[i] = binary.charCodeAt(i);
                  }
                  return out;
                };

                let imageBytes = null;
                let imageSource = 'synthetic';
                if (typeof imageBytesBase64 === 'string' && imageBytesBase64.length > 0) {
                  imageBytes = decodeBase64ToUint8Array(imageBytesBase64);
                  imageSource = imageFileName || 'provided';
                }

                if (!imageBytes) {
                  const canvas = document.createElement('canvas');
                  canvas.width = 320;
                  canvas.height = 180;
                  const ctx = canvas.getContext('2d');
                  ctx.fillStyle = '#0f172a';
                  ctx.fillRect(0, 0, canvas.width, canvas.height);
                  ctx.fillStyle = '#22d3ee';
                  ctx.fillRect(16, 16, 288, 148);
                  ctx.fillStyle = '#111827';
                  ctx.font = 'bold 42px sans-serif';
                  ctx.fillText('HELLO', 80, 108);

                  const blob = await new Promise((resolve, reject) => {
                    canvas.toBlob((value) => {
                      if (value) {
                        resolve(value);
                        return;
                      }
                      reject(new Error('toBlob failed'));
                    }, 'image/png');
                  });
                  imageBytes = new Uint8Array(await blob.arrayBuffer());
                }

                const inferStart = performance.now();
                let tokenCount = 0;
                let firstTokenAtMs = null;
                const output = await withTimeout(
                  bridge.createCompletion(
                    'what do you see?',
                    {
                      nPredict,
                      temp: 0.6,
                      topK: 20,
                      topP: 0.95,
                      penalty: 1.0,
                      onToken: () => {
                        if (firstTokenAtMs === null) {
                          firstTokenAtMs = performance.now() - inferStart;
                        }
                        tokenCount += 1;
                      },
                      parts: [{ type: 'image', bytes: imageBytes }],
                      mediaMaxImagePixels:
                        Number.isFinite(Number(mediaMaxImagePixels)) && Number(mediaMaxImagePixels) > 0
                          ? Number(mediaMaxImagePixels)
                          : undefined,
                      mediaMaxImageEdge:
                        Number.isFinite(Number(mediaMaxImageEdge)) && Number(mediaMaxImageEdge) > 0
                          ? Number(mediaMaxImageEdge)
                          : undefined,
                    },
                  ),
                  inferTimeoutMs,
                  'createCompletion',
                );
                timings.inferenceMs = Math.round(performance.now() - inferStart);
                const outputText = String(output || '');
                const trimmedOutput = outputText.trim();

                const metadata =
                  typeof bridge.getModelMetadata === 'function'
                    ? bridge.getModelMetadata() || {}
                    : {};

                if (tokenCount <= 0 || trimmedOutput.length == 0) {
                  return {
                    ok: false,
                    error: 'CPU multimodal returned no visible tokens.',
                    version: window.__llamadartBridgeLocalVersion || null,
                    coi: window.crossOriginIsolated,
                    timings,
                    metadata: {
                      execution: metadata['llamadart.webgpu.execution'] || null,
                      fallbackReason: metadata['llamadart.webgpu.worker_fallback_reason'] || null,
                      mmprojLoaded: metadata['llamadart.webgpu.mmproj_loaded'] || null,
                      supportsVision: metadata['llamadart.webgpu.supports_vision'] || null,
                      nGpuLayers: metadata['llamadart.webgpu.n_gpu_layers'] || null,
                      runtimeNotes: metadata['llamadart.webgpu.runtime_notes'] || null,
                    },
                    debug: {
                      tokenCount,
                      firstTokenLatencyMs:
                        firstTokenAtMs === null ? null : Math.round(firstTokenAtMs),
                      nGpuLayers,
                      nThreads,
                      imageBytes: imageBytes.length,
                      imageSource,
                      mediaMaxImagePixels,
                      mediaMaxImageEdge,
                      rawOutput: outputText.slice(0, 120),
                    },
                  };
                }

                return {
                  ok: true,
                  version: window.__llamadartBridgeLocalVersion || null,
                  coi: window.crossOriginIsolated,
                  output: outputText.slice(0, 280),
                  tokenCount,
                  timings,
                  debug: {
                    firstTokenLatencyMs:
                      firstTokenAtMs === null ? null : Math.round(firstTokenAtMs),
                    nGpuLayers,
                    nThreads,
                    imageBytes: imageBytes.length,
                    imageSource,
                    mediaMaxImagePixels,
                    mediaMaxImageEdge,
                  },
                  metadata: {
                    execution: metadata['llamadart.webgpu.execution'] || null,
                    fallbackReason: metadata['llamadart.webgpu.worker_fallback_reason'] || null,
                    mmprojLoaded: metadata['llamadart.webgpu.mmproj_loaded'] || null,
                    supportsVision: metadata['llamadart.webgpu.supports_vision'] || null,
                    nGpuLayers: metadata['llamadart.webgpu.n_gpu_layers'] || null,
                    runtimeNotes: metadata['llamadart.webgpu.runtime_notes'] || null,
                  },
                };
              } catch (error) {
                const metadata =
                  typeof bridge.getModelMetadata === 'function'
                    ? bridge.getModelMetadata() || {}
                    : {};
                return {
                  ok: false,
                  error: String(error),
                  version: window.__llamadartBridgeLocalVersion || null,
                  coi: window.crossOriginIsolated,
                  timings,
                  metadata: {
                    execution: metadata['llamadart.webgpu.execution'] || null,
                    fallbackReason: metadata['llamadart.webgpu.worker_fallback_reason'] || null,
                    mmprojLoaded: metadata['llamadart.webgpu.mmproj_loaded'] || null,
                    supportsVision: metadata['llamadart.webgpu.supports_vision'] || null,
                    runtimeNotes: metadata['llamadart.webgpu.runtime_notes'] || null,
                  },
                };
              } finally {
                try {
                  await bridge.dispose();
                } catch (_) {
                  // best-effort cleanup only
                }
              }
            }
            """,
            {
                "modelUrl": model_url,
                "mmprojUrl": mmproj_url,
                "modelTimeoutMs": args.model_timeout_ms,
                "mmprojTimeoutMs": args.mmproj_timeout_ms,
                "inferTimeoutMs": args.infer_timeout_ms,
                "nPredict": args.n_predict,
                "nGpuLayers": args.n_gpu_layers,
                "nThreads": args.n_threads,
                "imageBytesBase64": image_bytes_base64,
                "imageFileName": image_file_name,
                "mediaMaxImagePixels": args.media_max_image_pixels,
                "mediaMaxImageEdge": args.media_max_image_edge,
            },
        )

        print("[e2e] evaluation finished", flush=True)

        browser.close()

    print(
        json.dumps(
            {
                "result": result,
                "consoleTail": console_logs[-14:],
            },
            indent=2,
        )
    )
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
