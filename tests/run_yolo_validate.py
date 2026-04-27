from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", required=True)
    return parser.parse_args()


def normalize_yolo_label(raw_label: str | None) -> str:
    if raw_label is None:
        return "error"
    label = str(raw_label).strip().lower()
    if "pregnant" in label and not label.startswith("2_"):
        return "pregnant"
    if "noprenant" in label or "no_pregnant" in label or "no pregnant" in label:
        return "no_pregnant"
    if "notsure" in label or "not sure" in label:
        return "not_sure"
    if label == "unknown":
        return "unknown"
    return raw_label


def folder_truth(folder_name: str) -> str:
    mapping = {
        "1_Pregnant": "pregnant",
        "2_NoPregnant": "no_pregnant",
        "3_NotSure": "no_pregnant",
    }
    return mapping.get(folder_name, folder_name)


def main() -> int:
    args = parse_args()
    os.environ["ModelName"] = args.weight

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from app.process_pdf import preprocess_yolo  # noqa: PLC0415

    validate_root = repo_root / "AnomalyDetection" / "asset" / "validate"

    rows: list[dict] = []
    for folder in sorted([path for path in validate_root.iterdir() if path.is_dir()], key=lambda item: item.name):
        for image_path in sorted([path for path in folder.iterdir() if path.is_file()], key=lambda item: item.name):
            raw_label, confidence = preprocess_yolo(str(image_path))
            rows.append(
                {
                    "model": f"yolo:{args.weight}",
                    "folder": folder.name,
                    "truth": folder_truth(folder.name),
                    "file": image_path.name,
                    "raw_prediction": raw_label,
                    "prediction": normalize_yolo_label(raw_label),
                    "confidence": None if confidence is None else float(confidence),
                }
            )

    print(json.dumps(rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
