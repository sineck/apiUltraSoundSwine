from __future__ import annotations

"""สร้างไฟล์ index แบบคนอ่านได้สำหรับ artifact ของ anomaly run หนึ่งรอบ.

ไฟล์ใน `artifacts/models/<run_name>/` มีทั้ง `.joblib`, `*_weights.json`,
`*_predictions_*.json` และ `experiment_results.json` ซึ่งมีประโยชน์ต่อ runtime
แต่คนเปิดโฟลเดอร์ดูด้วยตาเปล่าจะไล่ยาก

script นี้จึงทำหน้าที่:
1. อ่าน `model_registry.json`
2. เลือก run ที่ต้องการหรือ run active
3. สร้างโฟลเดอร์ `_index/`
4. เขียน manifest, scaler summary, csv summary และ README สำหรับ run นั้น
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    """อ่าน JSON แบบ utf-8-sig เพื่อรองรับไฟล์ที่อาจมี BOM."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    """เขียน JSON แบบ utf-8 ลงดิสก์."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """อ่าน argument ของ script.

    - `--registry` = path ไป `model_registry.json`
    - `--run-name` = ถ้าไม่ระบุ จะใช้ run ที่ active อยู่ใน registry
    """
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build human-readable indexes for anomaly model artifacts.")
    parser.add_argument("--registry", type=Path, default=root / "artifacts" / "models" / "model_registry.json")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def active_run_name(registry: dict[str, Any], requested_run: str | None) -> str:
    """หา run ที่จะใช้สร้าง index.

    ถ้าผู้ใช้ส่ง `--run-name` มา จะใช้ค่านั้น
    ถ้าไม่ส่ง จะ fallback ไป run ของ `active_model`
    """
    if requested_run:
        return requested_run
    return str(registry["active_model"]).split("/", 1)[0]


def prediction_path(model_file: str, model_key: str, split: str) -> str:
    """สร้าง path มาตรฐานของ prediction JSON ของ model/split นั้น."""
    return str(Path(model_file).with_name(f"{model_key}_predictions_{split}.json"))


def has_standard_scaler(weights: dict[str, Any]) -> bool:
    """เช็กว่า weights json มี scaler แบบ mean/scale มาตรฐานหรือไม่."""
    return "scaler_mean" in weights and "scaler_scale" in weights


def build_indexes(registry_path: Path, run_name: str | None) -> tuple[Path, dict[str, Any]]:
    """สร้าง `_index` ทั้งชุดสำหรับ run anomaly หนึ่งรอบ.

    output หลักคือ:
    - `artifact_manifest.json`
    - `scaler_index.json`
    - `model_summary.csv`
    - `README.md`
    """
    registry = load_json(registry_path)
    selected_run = active_run_name(registry, run_name)
    run = registry["runs"][selected_run]
    experiment_path = Path(run["experiment_results"])
    experiment = load_json(experiment_path)
    run_dir = experiment_path.parent
    index_dir = run_dir / "_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    results_by_key = {item["model_key"]: item for item in experiment["results"]}
    models = []
    scalers = []
    for model_key, model_ref in sorted(run["models"].items()):
        result = results_by_key[model_key]
        weights_path = Path(model_ref["weights_json"])
        weights = load_json(weights_path)
        model_file = model_ref["model_file"]
        model_entry = {
            "model_key": model_key,
            "feature_set": model_ref["feature_set"],
            "model_name": model_ref["model_name"],
            "threshold": model_ref["threshold"],
            "is_active": model_key == run["active_model"],
            "has_standard_scaler": has_standard_scaler(weights),
            "model_file": model_file,
            "weights_json": model_ref["weights_json"],
            "predictions_validate": prediction_path(model_file, model_key, "validate"),
            "predictions_test": prediction_path(model_file, model_key, "test"),
            "predictions_review": prediction_path(model_file, model_key, "review"),
            "validate_balanced_accuracy": result["validate"]["balanced_accuracy"],
            "test_balanced_accuracy": result["test"]["balanced_accuracy"],
            "test_f1_no_pregnant": result["test"]["f1_no_pregnant"],
            "test_confusion_matrix": result["test"]["confusion_matrix"],
        }
        models.append(model_entry)

        scaler_mean = weights.get("scaler_mean")
        scaler_scale = weights.get("scaler_scale")
        scalers.append(
            {
                "model_key": model_key,
                "feature_set": model_ref["feature_set"],
                "model_name": model_ref["model_name"],
                "has_standard_scaler": has_standard_scaler(weights),
                "scaler_feature_count": len(scaler_mean) if isinstance(scaler_mean, list) else 0,
                "scaler_mean": scaler_mean,
                "scaler_scale": scaler_scale,
                "weights_json": model_ref["weights_json"],
                "model_file": model_file,
            }
        )

    manifest = {
        "schema_version": 1,
        "run_name": selected_run,
        "active_model": run["active_model"],
        "experiment_results": str(experiment_path),
        "dataset_summary": experiment["dataset_summary"],
        "artifact_counts": {
            "models": len(models),
            "weights": len(list(run_dir.glob("*_weights.json"))),
            "prediction_files": len(list(run_dir.glob("*_predictions_*.json"))),
        },
        "models": models,
    }
    write_json(index_dir / "artifact_manifest.json", manifest)
    write_json(index_dir / "scaler_index.json", scalers)

    csv_path = index_dir / "model_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model_key",
                "feature_set",
                "model_name",
                "is_active",
                "has_standard_scaler",
                "threshold",
                "validate_balanced_accuracy",
                "test_balanced_accuracy",
                "test_f1_no_pregnant",
                "weights_json",
                "model_file",
            ],
        )
        writer.writeheader()
        for model in models:
            writer.writerow({field: model[field] for field in writer.fieldnames})

    readme = f"""# Model Artifact Index

Run: `{selected_run}`

Active model: `{run["active_model"]}`

This `_index` folder is a human-readable map for the model files. The real runtime paths are not moved, because `model_registry.json` points directly to the `.joblib` and `*_weights.json` files.

## Files

- `artifact_manifest.json` - every model, weight file, prediction file, key metric, and threshold.
- `scaler_index.json` - scaler values per model. Models with `has_standard_scaler=false` use their own mean/std or covariance-style weights instead.
- `model_summary.csv` - compact table for spreadsheet viewing.

## Naming

- `.joblib` is the loadable runtime artifact.
- `*_weights.json` is the readable audit file.
- `*_predictions_validate.json`, `*_predictions_test.json`, and `*_predictions_review.json` are per-split predictions.

## Counts

- Models: {manifest["artifact_counts"]["models"]}
- Weight JSON files: {manifest["artifact_counts"]["weights"]}
- Prediction JSON files: {manifest["artifact_counts"]["prediction_files"]}
"""
    (index_dir / "README.md").write_text(readme, encoding="utf-8")
    return index_dir, manifest


def main() -> None:
    """entrypoint ของ script."""
    args = parse_args()
    index_dir, manifest = build_indexes(args.registry, args.run_name)
    print(f"[DONE] Index: {index_dir}")
    print(f"[DONE] Models: {manifest['artifact_counts']['models']}")
    print(f"[DONE] Active: {manifest['active_model']}")


if __name__ == "__main__":
    main()
