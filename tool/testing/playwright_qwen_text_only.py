#!/usr/bin/env python3
import json
import sys
from typing import Any

from playwright.sync_api import sync_playwright


DEFAULT_APP_URL = "http://127.0.0.1:7357"
DEFAULT_MODEL_URL = (
    "https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/"
    "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf?download=true"
)


def main() -> int:
    app_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_APP_URL
    model_url = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL_URL

    console_logs: list[dict[str, Any]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=False,
            args=[
                "--enable-unsafe-webgpu",
                "--disable-vulkan-surface",
                "--enable-features=Vulkan",
            ],
        )
        page = browser.new_page()
        page.set_default_timeout(0)

        def on_console(message: Any) -> None:
            try:
                text = message.text
            except Exception:  # pragma: no cover
                text = str(message)
            console_logs.append({"type": message.type, "text": text})

        page.on("console", on_console)

        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_function("() => typeof window.LlamaWebGpuBridge === 'function'")

        result = page.evaluate(
            """
            async ({ modelUrl }) => {
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
                threadPoolSize:
                  Number.isFinite(Number(window.__llamadartBridgeThreadPoolSize))
                    ? Number(window.__llamadartBridgeThreadPoolSize)
                    : undefined,
                logLevel: 3,
              };

              const bridge = new window.LlamaWebGpuBridge(cfg);
              const timings = {};

              try {
                const loadStart = performance.now();
                await withTimeout(
                  bridge.loadModelFromUrl(modelUrl, {
                    nCtx: 4096,
                    nGpuLayers: 99,
                    nThreads: 4,
                    useCache: false,
                    remoteFetchThresholdBytes: 9000000000000,
                  }),
                  8 * 60 * 1000,
                  'loadModelFromUrl',
                );
                timings.modelLoadMs = Math.round(performance.now() - loadStart);

                let tokenCount = 0;
                const inferStart = performance.now();
                const output = await withTimeout(
                  bridge.createCompletion(
                    'Write five concise bullet points about WebGPU performance tuning in browsers.',
                    {
                      nPredict: 160,
                      temp: 0.2,
                      topK: 40,
                      topP: 0.95,
                      penalty: 1.1,
                      onToken: () => {
                        tokenCount += 1;
                      },
                    },
                  ),
                  3 * 60 * 1000,
                  'createCompletion',
                );
                timings.inferenceMs = Math.round(performance.now() - inferStart);

                const metadata =
                  typeof bridge.getModelMetadata === 'function'
                    ? bridge.getModelMetadata() || {}
                    : {};
                const tokensPerSecond =
                  timings.inferenceMs > 0 ? (tokenCount * 1000.0) / timings.inferenceMs : null;

                return {
                  ok: true,
                  tokenCount,
                  tokensPerSecond,
                  outputPreview: String(output || '').slice(0, 220),
                  timings,
                  metadata,
                };
              } catch (error) {
                return {
                  ok: false,
                  error: String(error),
                  timings,
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
            {"modelUrl": model_url},
        )

        browser.close()

    print(
        json.dumps(
            {
                "result": result,
                "consoleTail": console_logs[-8:],
            },
            indent=2,
        )
    )
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
