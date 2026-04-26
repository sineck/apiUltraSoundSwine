from __future__ import annotations

import argparse
from pathlib import Path, PureWindowsPath

import numpy as np

from anomaly_lib import (
    ImageRow,
    extract_patch_handcrafted,
    get_feature_matrix,
    label_prediction,
    load_json,
    load_model_bundle,
    predict_from_scores,
    score_bundle,
    score_patch_bundle,
    write_json,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Predict one ultrasound image with the active anomaly model.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--registry", type=Path, default=root / "artifacts" / "models" / "model_registry.json")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_active_model(registry_path: Path) -> Path:
    registry = load_json(registry_path)
    active = registry["active_model"]
    run_name, model_key = active.split("/", 1)
    model_ref = registry["runs"][run_name]["models"][model_key]
    model_file = str(model_ref["model_file"])
    model_path = Path(model_file)
    if model_path.exists():
        return model_path
    model_filename = PureWindowsPath(model_file).name if "\\" in model_file else model_path.name
    return registry_path.parent / run_name / model_filename


def main() -> None:
    args = parse_args()
    model_path = resolve_active_model(args.registry)
    bundle = load_model_bundle(model_path)
    row = ImageRow(path=args.image, split="inference", label_name="unknown", target=None)
    if bundle["feature_set"] == "patch_handcrafted":
        score = float(score_patch_bundle(bundle, extract_patch_handcrafted([row]))[0])
    else:
        features = get_feature_matrix([row], bundle["feature_set"])
        score = float(score_bundle(bundle, features)[0])
    prediction_target = int(predict_from_scores(np.asarray([score]), float(bundle["threshold"]))[0])
    result = {
        "image": str(args.image),
        "model_file": str(model_path),
        "model_name": bundle["model_name"],
        "feature_set": bundle["feature_set"],
        "prediction": label_prediction(prediction_target),
        "score_no_pregnant": score,
        "threshold": float(bundle["threshold"]),
        "score_meaning": bundle["score_meaning"],
    }

    if args.output:
        write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
