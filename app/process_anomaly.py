from __future__ import annotations

from functools import lru_cache
from pathlib import Path, PureWindowsPath
import sys

import cv2
import numpy as np

from app.image_naming import build_image_filename
from app.process_detection import ImageResult


pathInitial = Path(__file__).resolve().parent.parent
ANOMALY_ROOT = pathInitial / "AnomalyDetection"
ANOMALY_SCRIPT_DIR = ANOMALY_ROOT / "scripts"
ANOMALY_REGISTRY = ANOMALY_ROOT / "artifacts" / "models" / "model_registry.json"

if str(ANOMALY_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(ANOMALY_SCRIPT_DIR))

from anomaly_lib import (  # noqa: E402
    ImageRow,
    extract_patch_handcrafted,
    get_feature_matrix,
    label_prediction,
    load_json,
    load_model_bundle,
    predict_from_scores,
    score_bundle,
    score_patch_bundle,
)


def resolve_active_anomaly_model(registry_path: Path = ANOMALY_REGISTRY) -> Path:
    registry = load_json(registry_path)
    active = registry["active_model"]
    run_name, model_key = active.split("/", 1)
    model_ref = registry["runs"][run_name]["models"][model_key]
    model_file = str(model_ref["model_file"])
    model_path = Path(model_file)

    # Registry files can contain absolute Windows paths from training time.
    # In Docker or another checkout, fall back to the model file beside the registry.
    if model_path.exists():
        return model_path
    model_filename = PureWindowsPath(model_file).name if "\\" in model_file else model_path.name
    return registry_path.parent / run_name / model_filename


@lru_cache(maxsize=1)
def load_active_anomaly_bundle() -> tuple[Path, dict]:
    model_path = resolve_active_anomaly_model()
    return model_path, load_model_bundle(model_path)


def predict_anomaly_image(image_path: Path) -> dict:
    model_path, bundle = load_active_anomaly_bundle()
    row = ImageRow(path=image_path, split="api", label_name="unknown", target=None)

    if bundle["feature_set"] == "patch_handcrafted":
        score = float(score_patch_bundle(bundle, extract_patch_handcrafted([row]))[0])
    else:
        features = get_feature_matrix([row], bundle["feature_set"])
        score = float(score_bundle(bundle, features)[0])

    threshold = float(bundle["threshold"])
    prediction_target = int(predict_from_scores(np.asarray([score]), threshold)[0])
    prediction = label_prediction(prediction_target)

    return {
        "image": str(image_path),
        "model_file": str(model_path),
        "model_name": bundle["model_name"],
        "feature_set": bundle["feature_set"],
        "prediction": prediction,
        "score_no_pregnant": score,
        "threshold": threshold,
        "score_meaning": bundle["score_meaning"],
        "estimator_type": bundle["estimator_type"],
    }


def anomaly_confidence(prediction: dict) -> float:
    score = float(prediction["score_no_pregnant"])
    if prediction.get("estimator_type") == "sklearn_supervised" and 0.0 <= score <= 1.0:
        confidence = score if prediction["prediction"] == "no_pregnant" else 1.0 - score
        return round(max(0.0, min(1.0, confidence)), 2)
    return 0.0


def format_anomaly_result(filename: str, prediction: dict, save_path: str) -> ImageResult:
    if prediction["prediction"] == "pregnant":
        result_text = "pregnant"
    elif prediction["prediction"] == "no_pregnant":
        result_text = "no pregnant"
    else:
        result_text = "not sure"

    return ImageResult(
        path_images=save_path,
        result=result_text,
        confidence=anomaly_confidence(prediction),
        number_of_fetus=0,
        error_remark="",
    )


def save_anomaly_input(img_cv2: np.ndarray, org_filename: str) -> str:
    output_dir = pathInitial / "app" / "detections"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / build_image_filename("anomaly")
    cv2.imwrite(str(out_path), img_cv2)
    return str(out_path)
