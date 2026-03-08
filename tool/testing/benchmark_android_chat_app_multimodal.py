#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path


PACKAGE = "com.example.llamadart_chat_example"
LOAD_MODEL_TAP = (480, 1380)
TEXTBOX_TAP = (420, 1950)
ADD_ATTACHMENT_TAP = (120, 1955)
SEND_TAP = (835, 1958)


def run(cmd: list[str], *, capture: bool = False) -> str:
    if capture:
      return subprocess.check_output(cmd, text=True)
    subprocess.check_call(cmd)
    return ""


def patch_prefs(
    xml: str,
    *,
    model_path: str,
    mmproj_path: str,
    backend: int,
    gpu_layers: int,
    threads: int,
    context_size: int,
    max_tokens: int,
) -> str:
    replacements = {
        r'<long name="flutter\.preferred_backend" value="\d+" />':
        f'<long name="flutter.preferred_backend" value="{backend}" />',
        r'<long name="flutter\.gpu_layers" value="\d+" />':
        f'<long name="flutter.gpu_layers" value="{gpu_layers}" />',
        r'<long name="flutter\.threads" value="\d+" />':
        f'<long name="flutter.threads" value="{threads}" />',
        r'<long name="flutter\.threads_batch" value="\d+" />':
        f'<long name="flutter.threads_batch" value="{threads}" />',
        r'<string name="flutter\.model_path">.*?</string>':
        f'<string name="flutter.model_path">{model_path}</string>',
        r'<string name="flutter\.mmproj_path">.*?</string>':
        f'<string name="flutter.mmproj_path">{mmproj_path}</string>',
        r'<boolean name="flutter\.tools_enabled" value="(?:true|false)" />':
        '<boolean name="flutter.tools_enabled" value="false" />',
        r'<boolean name="flutter\.thinking_enabled" value="(?:true|false)" />':
        '<boolean name="flutter.thinking_enabled" value="false" />',
        r'<long name="flutter\.context_size" value="\d+" />':
        f'<long name="flutter.context_size" value="{context_size}" />',
        r'<long name="flutter\.max_tokens" value="\d+" />':
        f'<long name="flutter.max_tokens" value="{max_tokens}" />',
    }
    for pattern, replacement in replacements.items():
        xml = re.sub(pattern, replacement, xml)
    return xml


def push_prefs(adb: Path, serial: str, xml: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as handle:
        handle.write(xml)
        temp_path = Path(handle.name)

    try:
        run([str(adb), "-s", serial, "push", str(temp_path), "/data/local/tmp/FlutterSharedPreferences.xml"])
        run(
            [
                str(adb),
                "-s",
                serial,
                "shell",
                "run-as",
                PACKAGE,
                "cp",
                "/data/local/tmp/FlutterSharedPreferences.xml",
                f"/data/user/0/{PACKAGE}/shared_prefs/FlutterSharedPreferences.xml",
            ]
        )
    finally:
        temp_path.unlink(missing_ok=True)


def dump_ui_values(adb: Path, serial: str) -> list[str]:
    run([str(adb), "-s", serial, "shell", "uiautomator", "dump", "/sdcard/bench_mm_ui.xml"])
    xml = run([str(adb), "-s", serial, "shell", "cat", "/sdcard/bench_mm_ui.xml"], capture=True)
    root = ET.fromstring(xml)
    values: list[str] = []
    for node in root.iter("node"):
        value = (node.attrib.get("content-desc", "") or node.attrib.get("text", "")).strip()
        if value:
            values.append(value)
    return values


def tap(adb: Path, serial: str, x: int, y: int) -> None:
    run([str(adb), "-s", serial, "shell", "input", "tap", str(x), str(y)])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mmproj-path", required=True)
    parser.add_argument("--backend", choices=["cpu", "vulkan"], required=True)
    parser.add_argument("--gpu-layers", type=int, required=True)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--prompt", default="what color is the square")
    parser.add_argument("--load-timeout-sec", type=int, default=300)
    parser.add_argument("--infer-timeout-sec", type=int, default=180)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    adb = Path.home() / "Library/Android/sdk/platform-tools/adb"
    backend_index = 1 if args.backend == "cpu" else 2

    original_xml = run(
        [
            str(adb),
            "-s",
            args.serial,
            "shell",
            "run-as",
            PACKAGE,
            "cat",
            "shared_prefs/FlutterSharedPreferences.xml",
        ],
        capture=True,
    )

    try:
        patched_xml = patch_prefs(
            original_xml,
            model_path=args.model_path,
            mmproj_path=args.mmproj_path,
            backend=backend_index,
            gpu_layers=args.gpu_layers,
            threads=args.threads,
            context_size=args.context_size,
            max_tokens=args.max_tokens,
        )
        push_prefs(adb, args.serial, patched_xml)

        run([str(adb), "-s", args.serial, "shell", "am", "force-stop", PACKAGE])
        run(
            [
                str(adb),
                "-s",
                args.serial,
                "shell",
                "monkey",
                "-p",
                PACKAGE,
                "-c",
                "android.intent.category.LAUNCHER",
                "1",
            ]
        )
        time.sleep(4)
        tap(adb, args.serial, *LOAD_MODEL_TAP)

        load_deadline = time.time() + args.load_timeout_sec
        while time.time() < load_deadline:
            values = dump_ui_values(adb, args.serial)
            if "Model loaded successfully! Ready to chat." in values:
                break
            if any("Error loading model" in value or "Something went wrong" in value for value in values):
                print(json.dumps({"ok": False, "stage": "load", "values": values}, indent=2))
                return 2
            time.sleep(5)
        else:
            print(json.dumps({"ok": False, "stage": "load_timeout"}, indent=2))
            return 3

        tap(adb, args.serial, *ADD_ATTACHMENT_TAP)
        time.sleep(2)
        values = dump_ui_values(adb, args.serial)
        if "Remove attachment" not in values:
            print(json.dumps({"ok": False, "stage": "attach_failed", "values": values}, indent=2))
            return 4

        tap(adb, args.serial, *TEXTBOX_TAP)
        time.sleep(1)
        run([str(adb), "-s", args.serial, "shell", "input", "text", args.prompt.replace(" ", "%s")])
        time.sleep(1)
        tap(adb, args.serial, *SEND_TAP)

        infer_deadline = time.time() + args.infer_timeout_sec
        while time.time() < infer_deadline:
            values = dump_ui_values(adb, args.serial)
            if "Stop generation" not in values and any(
                args.prompt.lower() in value.lower() for value in values
            ):
                print(json.dumps({"ok": True, "backend": args.backend, "values": values}, indent=2))
                return 0
            time.sleep(5)

        values = dump_ui_values(adb, args.serial)
        if "Stop generation" in values:
            tap(adb, args.serial, *SEND_TAP)
            time.sleep(2)
            values = dump_ui_values(adb, args.serial)
        print(json.dumps({"ok": False, "stage": "infer_timeout", "values": values}, indent=2))
        return 5
    finally:
        push_prefs(adb, args.serial, original_xml)


if __name__ == "__main__":
    raise SystemExit(main())
