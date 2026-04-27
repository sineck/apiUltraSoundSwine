from __future__ import annotations

"""รัน retrain anomaly ตาม config กลางตัวเดียว.

ไฟล์นี้จงใจไม่รับ parameter ยิบย่อยจาก command line เพราะ repo นี้กำหนดให้
`config/retrain_anomaly.json` เป็น single source of truth สำหรับค่า default ของ
การ retrain ทั้งฝั่ง script และฝั่ง API
"""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "retrain_anomaly.json"
TRAIN_MODULE = "AnomalyDetection.scripts.train_anomaly_models"
INDEX_MODULE = "AnomalyDetection.scripts.build_artifact_index"
REPORT_MODULE = "AnomalyDetection.scripts.generate_report"


def load_config() -> dict:
    """อ่าน config retrain กลางจาก `config/retrain_anomaly.json`."""
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def build_commands(config: dict) -> list[list[str]]:
    """แปลง config เป็นรายการคำสั่งที่ต้องรันตามลำดับ.

    ลำดับปัจจุบันคือ:
    1. train anomaly models
    2. rebuild artifact index (ถ้าเปิด)
    3. generate report (ถ้าเปิด)
    """
    train_command = [
        sys.executable,
        "-m",
        TRAIN_MODULE,
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
        commands.append([sys.executable, "-m", INDEX_MODULE])
    if config.get("generate_report", True):
        commands.append(
            [
                sys.executable,
                "-m",
                REPORT_MODULE,
                "--detail-heatmaps",
                str(config.get("detail_heatmaps", "active")),
            ]
        )
    return commands


def main() -> None:
    """entrypoint ของ retrain script."""
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
