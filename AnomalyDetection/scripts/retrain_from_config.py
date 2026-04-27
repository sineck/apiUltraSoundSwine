from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "retrain_anomaly.json"
TRAIN_SCRIPT = ROOT / "AnomalyDetection" / "scripts" / "train_anomaly_models.py"
INDEX_SCRIPT = ROOT / "AnomalyDetection" / "scripts" / "build_artifact_index.py"
REPORT_SCRIPT = ROOT / "AnomalyDetection" / "scripts" / "generate_report.py"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def build_commands(config: dict) -> list[list[str]]:
    train_command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--batch-size",
        str(config["batch_size"]),
    ]
    feature_sets = config.get("feature_sets")
    if feature_sets:
        train_command.extend(["--feature-sets", str(feature_sets)])
    model_keys = config.get("model_keys")
    if model_keys:
        train_command.extend(["--model-keys", str(model_keys)])

    commands = [train_command]
    if config.get("rebuild_index", True):
        commands.append([sys.executable, str(INDEX_SCRIPT)])
    if config.get("generate_report", True):
        commands.append(
            [
                sys.executable,
                str(REPORT_SCRIPT),
                "--detail-heatmaps",
                str(config.get("detail_heatmaps", "active")),
            ]
        )
    return commands


def main() -> None:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Missing config: {CONFIG_PATH}")

    config = load_config()
    commands = build_commands(config)
    print(f"[CONFIG] {CONFIG_PATH}")
    for command in commands:
        print(f"$ {' '.join(command)}")
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
