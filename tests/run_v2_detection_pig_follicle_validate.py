from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib import request

import requests
from requests import Response


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATE_ROOT = REPO_ROOT / "AnomalyDetection" / "asset" / "validate"
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
ROUTE_PATH = "/v2/detection_pig_follicle"
PORT = 3015


@dataclass(frozen=True)
class ModelConfig:
    label: str
    backend: str
    weight: str | None


MODEL_CONFIGS = [
    ModelConfig(label="anomaly", backend="anomaly", weight=None),
    ModelConfig(label="yolo_best", backend="yolo", weight="best.pt"),
    ModelConfig(label="yolo_finetune", backend="yolo", weight="best_finetune_YOLO26-cls_Ver2_20260424.pt"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[model.label for model in MODEL_CONFIGS], default=None)
    return parser.parse_args()


def wait_api_ready(port: int, timeout_seconds: int = 90) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            with request.urlopen(f"http://127.0.0.1:{port}/version", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"API not ready on port {port}: {last_error}")


def start_api(model: ModelConfig, port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["MYAPI_PORT"] = str(port)
    env["INSERT_ULTRASOUND_TO_DB"] = "false"
    env["PREGNANCY_DETECT_MODEL_V2"] = model.backend
    if model.weight is not None:
        env["ModelName"] = model.weight

    return subprocess.Popen(
        [
            str(PYTHON_EXE),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def stop_api(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def restart_api(process: subprocess.Popen[str], model: ModelConfig, port: int) -> subprocess.Popen[str]:
    stop_api(process)
    new_process = start_api(model, port)
    wait_api_ready(port)
    return new_process


def iter_validate_files() -> Iterable[tuple[str, Path]]:
    for folder in sorted([path for path in VALIDATE_ROOT.iterdir() if path.is_dir()], key=lambda item: item.name):
        files = sorted([path for path in folder.iterdir() if path.is_file()], key=lambda item: item.name)
        for file_path in files:
            yield folder.name, file_path


def detect_path_kind(path_images: str) -> str:
    if "gemini_" in path_images:
        return "gemini"
    if "anomaly_" in path_images:
        return "anomaly"
    return "other"


def post_detection(file_path: Path, port: int) -> Response:
    mime_type = "image/png" if file_path.suffix.lower() == ".png" else "image/jpeg"
    with file_path.open("rb") as file_obj:
        return requests.post(
            f"http://127.0.0.1:{port}{ROUTE_PATH}",
            files={"files": (file_path.name, file_obj, mime_type)},
            timeout=180,
        )


def run_model_benchmark(model: ModelConfig, port: int) -> list[dict]:
    process = start_api(model, port)
    try:
        wait_api_ready(port)
        rows: list[dict] = []
        for folder_name, file_path in iter_validate_files():
            response = None
            request_error = ""
            for attempt in range(2):
                try:
                    response = post_detection(file_path, port)
                    response.raise_for_status()
                    break
                except Exception as exc:  # noqa: BLE001
                    request_error = str(exc)
                    process = restart_api(process, model, port)
            if response is None:
                rows.append(
                    {
                        "model": model.label,
                        "folder": folder_name,
                        "file": file_path.name,
                        "result": "error",
                        "confidence": None,
                        "fetus": 0,
                        "path_kind": "other",
                        "error": request_error,
                    }
                )
                continue

            payload = response.json()
            item = payload["results"][0]
            rows.append(
                {
                    "model": model.label,
                    "folder": folder_name,
                    "file": file_path.name,
                    "result": item["result"],
                    "confidence": item["confidence"],
                    "fetus": item["number_of_fetus"],
                    "path_kind": detect_path_kind(item["path_images"]),
                    "error": item["error_remark"],
                }
            )
        return rows
    finally:
        stop_api(process)


def summarize(rows: list[dict]) -> list[dict]:
    summary: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["model"], row["folder"])
        if key not in summary:
            summary[key] = {
                "model": row["model"],
                "folder": row["folder"],
                "samples": 0,
                "pregnant": 0,
                "no_pregnant": 0,
                "not_sure": 0,
                "gemini_paths": 0,
                "anomaly_paths": 0,
                "gemini_unusable": 0,
                "confidence_values": [],
            }
        bucket = summary[key]
        bucket["samples"] += 1
        if row["result"] == "pregnant":
            bucket["pregnant"] += 1
        elif row["result"] == "no pregnant":
            bucket["no_pregnant"] += 1
        elif row["result"] == "not sure":
            bucket["not_sure"] += 1
        if row["path_kind"] == "gemini":
            bucket["gemini_paths"] += 1
        if row["path_kind"] == "anomaly":
            bucket["anomaly_paths"] += 1
        if row["error"] == "Gemini did not return usable follicle annotation":
            bucket["gemini_unusable"] += 1
        if row["confidence"] is not None:
            bucket["confidence_values"].append(float(row["confidence"]))

    results = []
    for bucket in summary.values():
        confidence_values = bucket.pop("confidence_values")
        bucket["avg_confidence"] = round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else None
        results.append(bucket)
    return sorted(results, key=lambda item: (item["model"], item["folder"]))


def print_table(title: str, rows: list[dict], columns: list[str]) -> None:
    print(title)
    widths = {column: max(len(column), *(len(str(row.get(column, ""))) for row in rows)) for column in columns}
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(divider)
    for row in rows:
        print(" | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))
    print()


def main() -> int:
    args = parse_args()
    all_rows: list[dict] = []
    selected_models = [model for model in MODEL_CONFIGS if args.model in (None, model.label)]
    for model in selected_models:
        all_rows.extend(run_model_benchmark(model, PORT))

    summary_rows = summarize(all_rows)
    print_table(
        "SUMMARY",
        summary_rows,
        [
            "model",
            "folder",
            "samples",
            "pregnant",
            "no_pregnant",
            "not_sure",
            "gemini_paths",
            "anomaly_paths",
            "gemini_unusable",
            "avg_confidence",
        ],
    )
    print_table(
        "DETAIL",
        all_rows,
        [
            "model",
            "folder",
            "file",
            "result",
            "confidence",
            "fetus",
            "path_kind",
            "error",
        ],
    )
    print("JSON_SUMMARY=" + json.dumps(summary_rows, ensure_ascii=False))
    print("JSON_DETAIL=" + json.dumps(all_rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
