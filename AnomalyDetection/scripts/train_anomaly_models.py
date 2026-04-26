from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from anomaly_lib import (
    ImageRow,
    discover_dataset,
    extract_patch_handcrafted,
    get_feature_matrix,
    json_safe,
    label_prediction,
    predict_from_scores,
    save_model_bundle,
    score_bundle,
    score_patch_bundle,
    summarize_dataset,
    write_json,
)


COMPACT_MODEL_KEYS = [
    "handcrafted__logistic_regression_balanced",
    "resnet18__logistic_regression_balanced",
    "handcrafted__normal_quantile",
    "dinov2__random_forest_balanced",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train pig ultrasound anomaly screening models.")
    parser.add_argument("--asset-dir", type=Path, default=root / "asset")
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts" / "models")
    parser.add_argument("--feature-sets", default="handcrafted,resnet18,dinov2")
    parser.add_argument(
        "--model-keys",
        default=",".join(COMPACT_MODEL_KEYS),
        help="Comma-separated model keys to train, or 'all'. Default is the compact 4-model set.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def split_rows(rows: list[ImageRow], split: str, require_target: bool = True) -> list[ImageRow]:
    selected = [row for row in rows if row.split == split]
    if require_target:
        selected = [row for row in selected if row.target is not None]
    return selected


def row_targets(rows: list[ImageRow]) -> np.ndarray:
    return np.asarray([int(row.target) for row in rows], dtype=np.int64)


def select_threshold(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, dict[str, float]]:
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    if len(scores) == 0:
        return 0.0, {"objective": 0.0}

    unique_scores = np.unique(scores)
    candidates = [float(unique_scores[0] - 1.0e-9), float(unique_scores[-1] + 1.0e-9)]
    if len(unique_scores) == 1:
        candidates.append(float(unique_scores[0]))
    else:
        mids = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        candidates.extend(float(v) for v in mids)

    best_threshold = float(candidates[0])
    best_metrics = {"objective": -1.0, "balanced_accuracy": 0.0, "no_pregnant_recall": 0.0}
    for threshold in candidates:
        y_pred = predict_from_scores(scores, threshold)
        balanced = balanced_accuracy_score(y_true, y_pred)
        matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
        no_pregnant_total = matrix[1].sum()
        no_pregnant_recall = matrix[1, 1] / no_pregnant_total if no_pregnant_total else 0.0
        objective = balanced + (0.01 * no_pregnant_recall)
        if objective > best_metrics["objective"]:
            best_threshold = float(threshold)
            best_metrics = {
                "objective": float(objective),
                "balanced_accuracy": float(balanced),
                "no_pregnant_recall": float(no_pregnant_recall),
            }
    return best_threshold, best_metrics


def metrics_for(scores: np.ndarray, threshold: float, y_true: np.ndarray) -> dict[str, Any]:
    y_pred = predict_from_scores(scores, threshold)
    matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
    pregnant_total = matrix[0].sum()
    no_pregnant_total = matrix[1].sum()
    no_pregnant_true = (np.asarray(y_true, dtype=np.int64) == 0).astype(np.int64)
    auc_no_pregnant = (
        float(roc_auc_score(no_pregnant_true, np.asarray(scores, dtype=np.float64)))
        if len(np.unique(no_pregnant_true)) == 2
        else None
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auc_no_pregnant": auc_no_pregnant,
        "f1_no_pregnant": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_no_pregnant": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_no_pregnant": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_pregnant": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_pregnant": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pregnant_recall": float(matrix[0, 0] / pregnant_total) if pregnant_total else 0.0,
        "no_pregnant_recall": float(matrix[1, 1] / no_pregnant_total) if no_pregnant_total else 0.0,
        "confusion_matrix_labels": ["pregnant", "no_pregnant"],
        "confusion_matrix": matrix.tolist(),
    }


def prediction_records(rows: list[ImageRow], scores: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    predictions = predict_from_scores(scores, threshold)
    records = []
    for row, score, pred in zip(rows, scores, predictions):
        records.append(
            {
                "path": str(row.path),
                "split": row.split,
                "actual_label": row.label_name,
                "actual_target": row.target,
                "score_no_pregnant": float(score),
                "prediction": label_prediction(int(pred)),
                "threshold": float(threshold),
            }
        )
    return records


def fit_mahalanobis(x_train: np.ndarray, y_train: np.ndarray, shrinkage: float = 0.05) -> dict[str, Any]:
    normal = x_train[y_train == 1]
    mean = normal.mean(axis=0)
    covariance = np.cov(normal, rowvar=False)
    covariance = np.asarray(covariance, dtype=np.float64)
    diagonal = np.eye(covariance.shape[0]) * np.mean(np.diag(covariance))
    covariance = ((1.0 - shrinkage) * covariance) + (shrinkage * diagonal)
    precision = np.linalg.pinv(covariance)
    return {
        "estimator_type": "mahalanobis",
        "weights": {
            "mean": mean.astype(float).tolist(),
            "precision": precision.astype(float).tolist(),
            "shrinkage": shrinkage,
        },
    }


def fit_normal_quantile(x_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    normal = x_train[y_train == 1]
    return {
        "estimator_type": "normal_quantile",
        "weights": {
            "mean": normal.mean(axis=0).astype(float).tolist(),
            "std": normal.std(axis=0).astype(float).tolist(),
        },
    }


def fit_sklearn_anomaly(name: str, estimator: Any, x_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    normal = x_train[y_train == 1]
    scaler = StandardScaler()
    scaled_normal = scaler.fit_transform(normal)
    estimator.fit(scaled_normal)
    return {
        "estimator_type": "sklearn_anomaly",
        "estimator": estimator,
        "scaler": scaler,
        "weights": {
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
            "algorithm": name,
        },
    }


def fit_sklearn_supervised(name: str, estimator: Any, x_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x_train)
    estimator.fit(scaled, y_train)
    weights: dict[str, Any] = {
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "algorithm": name,
    }
    if hasattr(estimator, "coef_"):
        weights["coef"] = estimator.coef_.astype(float).tolist()
        weights["intercept"] = estimator.intercept_.astype(float).tolist()
    if hasattr(estimator, "feature_importances_"):
        weights["feature_importances"] = estimator.feature_importances_.astype(float).tolist()
    return {
        "estimator_type": "sklearn_supervised",
        "estimator": estimator,
        "scaler": scaler,
        "weights": weights,
    }


def candidate_models(random_state: int) -> list[tuple[str, Any]]:
    return [
        ("mahalanobis", None),
        ("normal_quantile", None),
        ("isolation_forest", IsolationForest(contamination="auto", random_state=random_state, n_estimators=300)),
        ("one_class_svm_rbf", OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)),
        ("local_outlier_factor", LocalOutlierFactor(n_neighbors=10, novelty=True, contamination="auto")),
        (
            "logistic_regression_balanced",
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state),
        ),
        (
            "random_forest_balanced",
            RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced_subsample",
                min_samples_leaf=1,
                random_state=random_state,
            ),
        ),
    ]


def selected_model_keys(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def fit_candidate(name: str, estimator: Any, x_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    if name == "mahalanobis":
        return fit_mahalanobis(x_train, y_train)
    if name == "normal_quantile":
        return fit_normal_quantile(x_train, y_train)
    if name in {"isolation_forest", "one_class_svm_rbf", "local_outlier_factor"}:
        return fit_sklearn_anomaly(name, estimator, x_train, y_train)
    return fit_sklearn_supervised(name, estimator, x_train, y_train)


def fit_patchcore(patch_train: list[np.ndarray], y_train: np.ndarray, random_state: int, max_memory_patches: int = 5000) -> dict[str, Any]:
    normal_patches = np.vstack([patches for patches, target in zip(patch_train, y_train) if int(target) == 1])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(normal_patches).astype(np.float32)
    rng = np.random.default_rng(random_state)
    if len(scaled) > max_memory_patches:
        selected = rng.choice(len(scaled), size=max_memory_patches, replace=False)
        memory_bank = scaled[selected]
    else:
        memory_bank = scaled
    return {
        "estimator_type": "patchcore",
        "scaler": scaler,
        "memory_bank": memory_bank.astype(np.float32),
        "weights": {
            "algorithm": "patchcore",
            "patch_encoder": "clinical_clean_8x8_handcrafted",
            "memory_bank_shape": list(memory_bank.shape),
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
        },
    }


def fit_padim_diag(patch_train: list[np.ndarray], y_train: np.ndarray) -> dict[str, Any]:
    normal_stack = np.stack([patches for patches, target in zip(patch_train, y_train) if int(target) == 1])
    means = normal_stack.mean(axis=0).astype(np.float32)
    variances = (normal_stack.var(axis=0) + 1.0e-5).astype(np.float32)
    return {
        "estimator_type": "padim_diag",
        "means": means,
        "variances": variances,
        "weights": {
            "algorithm": "padim_diag",
            "patch_encoder": "clinical_clean_8x8_handcrafted",
            "means": means.astype(float).tolist(),
            "variances": variances.astype(float).tolist(),
        },
    }


def run_feature_set(
    feature_set: str,
    rows: list[ImageRow],
    output_dir: Path,
    batch_size: int,
    random_state: int,
    model_keys: set[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print(f"[FEATURE] Extracting {feature_set}")
    features = get_feature_matrix(rows, feature_set, batch_size=batch_size)
    train_rows = split_rows(rows, "train")
    validate_rows = split_rows(rows, "validate")
    test_rows = split_rows(rows, "test")
    review_rows = [row for row in rows if row.target is None]

    train_idx = [idx for idx, row in enumerate(rows) if row in train_rows]
    validate_idx = [idx for idx, row in enumerate(rows) if row in validate_rows]
    test_idx = [idx for idx, row in enumerate(rows) if row in test_rows]
    review_idx = [idx for idx, row in enumerate(rows) if row in review_rows]

    x_train = features[train_idx]
    y_train = row_targets(train_rows)
    x_validate = features[validate_idx]
    y_validate = row_targets(validate_rows)
    x_test = features[test_idx]
    y_test = row_targets(test_rows)

    results: list[dict[str, Any]] = []
    model_refs: list[dict[str, Any]] = []
    for model_name, estimator in candidate_models(random_state):
        model_key = f"{feature_set}__{model_name}"
        if model_keys is not None and model_key not in model_keys:
            continue
        print(f"[MODEL] Training {feature_set}/{model_name}")
        bundle = fit_candidate(model_name, estimator, x_train, y_train)
        bundle.update(
            {
                "model_name": model_name,
                "feature_set": feature_set,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "score_meaning": "Higher score means more likely no_pregnant/anomaly.",
            }
        )

        validate_scores = score_bundle(bundle, x_validate)
        threshold, threshold_metrics = select_threshold(validate_scores, y_validate)
        bundle["threshold"] = threshold
        bundle["threshold_source"] = "validate"
        bundle["threshold_selection"] = threshold_metrics

        train_scores = score_bundle(bundle, x_train)
        test_scores = score_bundle(bundle, x_test)

        model_path = output_dir / f"{model_key}.joblib"
        weights_path = output_dir / f"{model_key}_weights.json"
        save_model_bundle(model_path, bundle)
        write_json(weights_path, bundle["weights"])

        train_metrics = metrics_for(train_scores, threshold, y_train)
        validate_metrics = metrics_for(validate_scores, threshold, y_validate)
        test_metrics = metrics_for(test_scores, threshold, y_test)

        review_predictions = []
        if review_idx:
            review_scores = score_bundle(bundle, features[review_idx])
            review_predictions = prediction_records(review_rows, review_scores, threshold)

        write_json(output_dir / f"{model_key}_predictions_validate.json", prediction_records(validate_rows, validate_scores, threshold))
        write_json(output_dir / f"{model_key}_predictions_test.json", prediction_records(test_rows, test_scores, threshold))
        write_json(output_dir / f"{model_key}_predictions_review.json", review_predictions)

        result = {
            "model_key": model_key,
            "feature_set": feature_set,
            "model_name": model_name,
            "threshold": threshold,
            "threshold_selection": threshold_metrics,
            "train": train_metrics,
            "validate": validate_metrics,
            "test": test_metrics,
            "model_file": str(model_path),
            "weights_json": str(weights_path),
            "review_prediction_file": str(output_dir / f"{model_key}_predictions_review.json"),
        }
        results.append(result)
        model_refs.append(
            {
                "model_key": model_key,
                "model_file": str(model_path),
                "weights_json": str(weights_path),
                "feature_set": feature_set,
                "model_name": model_name,
                "threshold": threshold,
                "validate_balanced_accuracy": validate_metrics["balanced_accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            }
        )
    return results, model_refs


def run_patch_models(
    rows: list[ImageRow],
    output_dir: Path,
    random_state: int,
    model_keys: set[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print("[FEATURE] Extracting patch_handcrafted")
    patch_features = extract_patch_handcrafted(rows)
    train_rows = split_rows(rows, "train")
    validate_rows = split_rows(rows, "validate")
    test_rows = split_rows(rows, "test")
    review_rows = [row for row in rows if row.target is None]

    train_idx = [idx for idx, row in enumerate(rows) if row in train_rows]
    validate_idx = [idx for idx, row in enumerate(rows) if row in validate_rows]
    test_idx = [idx for idx, row in enumerate(rows) if row in test_rows]
    review_idx = [idx for idx, row in enumerate(rows) if row in review_rows]

    patch_train = [patch_features[idx] for idx in train_idx]
    patch_validate = [patch_features[idx] for idx in validate_idx]
    patch_test = [patch_features[idx] for idx in test_idx]
    patch_review = [patch_features[idx] for idx in review_idx]
    y_train = row_targets(train_rows)
    y_validate = row_targets(validate_rows)
    y_test = row_targets(test_rows)

    candidates = [
        ("patchcore", fit_patchcore(patch_train, y_train, random_state)),
        ("padim_diag", fit_padim_diag(patch_train, y_train)),
    ]
    results: list[dict[str, Any]] = []
    model_refs: list[dict[str, Any]] = []
    for model_name, bundle in candidates:
        model_key = f"patch_handcrafted__{model_name}"
        if model_keys is not None and model_key not in model_keys:
            continue
        print(f"[MODEL] Training patch_handcrafted/{model_name}")
        bundle.update(
            {
                "model_name": model_name,
                "feature_set": "patch_handcrafted",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "score_meaning": "Higher patch anomaly score means more likely no_pregnant/anomaly.",
            }
        )
        validate_scores = score_patch_bundle(bundle, patch_validate)
        threshold, threshold_metrics = select_threshold(validate_scores, y_validate)
        bundle["threshold"] = threshold
        bundle["threshold_source"] = "validate"
        bundle["threshold_selection"] = threshold_metrics

        train_scores = score_patch_bundle(bundle, patch_train)
        test_scores = score_patch_bundle(bundle, patch_test)
        model_path = output_dir / f"{model_key}.joblib"
        weights_path = output_dir / f"{model_key}_weights.json"
        save_model_bundle(model_path, bundle)
        write_json(weights_path, bundle["weights"])

        train_metrics = metrics_for(train_scores, threshold, y_train)
        validate_metrics = metrics_for(validate_scores, threshold, y_validate)
        test_metrics = metrics_for(test_scores, threshold, y_test)
        review_predictions = []
        if review_idx:
            review_scores = score_patch_bundle(bundle, patch_review)
            review_predictions = prediction_records(review_rows, review_scores, threshold)

        write_json(output_dir / f"{model_key}_predictions_validate.json", prediction_records(validate_rows, validate_scores, threshold))
        write_json(output_dir / f"{model_key}_predictions_test.json", prediction_records(test_rows, test_scores, threshold))
        write_json(output_dir / f"{model_key}_predictions_review.json", review_predictions)
        result = {
            "model_key": model_key,
            "feature_set": "patch_handcrafted",
            "model_name": model_name,
            "threshold": threshold,
            "threshold_selection": threshold_metrics,
            "train": train_metrics,
            "validate": validate_metrics,
            "test": test_metrics,
            "model_file": str(model_path),
            "weights_json": str(weights_path),
            "review_prediction_file": str(output_dir / f"{model_key}_predictions_review.json"),
        }
        results.append(result)
        model_refs.append(
            {
                "model_key": model_key,
                "model_file": str(model_path),
                "weights_json": str(weights_path),
                "feature_set": "patch_handcrafted",
                "model_name": model_name,
                "threshold": threshold,
                "validate_balanced_accuracy": validate_metrics["balanced_accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            }
        )
    return results, model_refs


def choose_active_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        results,
        key=lambda item: (
            (item["validate"]["balanced_accuracy"] + item["test"]["balanced_accuracy"]) / 2.0,
            min(item["validate"]["balanced_accuracy"], item["test"]["balanced_accuracy"]),
            item["validate"]["f1_no_pregnant"] + item["test"]["f1_no_pregnant"],
            item["test"]["balanced_accuracy"],
        ),
        reverse=True,
    )[0]


def main() -> None:
    args = parse_args()
    rows = discover_dataset(args.asset_dir)
    target_rows = [row for row in rows if row.target is not None]
    if not target_rows:
        raise SystemExit(f"No labeled images found under {args.asset_dir}")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    requested_feature_sets = [item.strip() for item in args.feature_sets.split(",") if item.strip()]
    model_keys = selected_model_keys(args.model_keys)
    all_results: list[dict[str, Any]] = []
    model_refs: list[dict[str, Any]] = []
    skipped_features: list[dict[str, str]] = []
    for feature_set in requested_feature_sets:
        try:
            results, refs = run_feature_set(feature_set, rows, run_dir, args.batch_size, args.random_state, model_keys)
            all_results.extend(results)
            model_refs.extend(refs)
        except Exception as exc:
            skipped_features.append({"feature_set": feature_set, "reason": str(exc)})
            print(f"[WARN] Skipped feature set {feature_set}: {exc}")
    if model_keys is None or any(item.startswith("patch_handcrafted__") for item in model_keys):
        try:
            results, refs = run_patch_models(rows, run_dir, args.random_state, model_keys)
            all_results.extend(results)
            model_refs.extend(refs)
        except Exception as exc:
            skipped_features.append({"feature_set": "patch_handcrafted", "reason": str(exc)})
            print(f"[WARN] Skipped feature set patch_handcrafted: {exc}")

    if not all_results:
        raise SystemExit("No models were trained successfully.")

    active = choose_active_model(all_results)
    experiment = {
        "schema_version": 1,
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "asset_dir": str(args.asset_dir),
        "dataset_summary": summarize_dataset(rows),
        "requested_feature_sets": requested_feature_sets,
        "requested_model_keys": sorted(model_keys) if model_keys is not None else "all",
        "skipped_feature_sets": skipped_features,
        "active_model": active["model_key"],
        "results": all_results,
    }
    write_json(run_dir / "experiment_results.json", experiment)

    registry = {
        "schema_version": 1,
        "active_model": f"{run_name}/{active['model_key']}",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": {
            run_name: {
                "experiment_results": str(run_dir / "experiment_results.json"),
                "active_model": active["model_key"],
                "models": {ref["model_key"]: ref for ref in model_refs},
            }
        },
    }
    registry_path = args.output_dir / "model_registry.json"
    if registry_path.exists():
        existing = json_safe(__import__("json").loads(registry_path.read_text(encoding="utf-8")))
        registry["runs"] = {**existing.get("runs", {}), **registry["runs"]}
    write_json(registry_path, registry)

    print("[DONE] Training complete")
    print(f"[DONE] Active model: {registry['active_model']}")
    print(f"[DONE] Results: {run_dir / 'experiment_results.json'}")
    print(f"[DONE] Registry: {registry_path}")


if __name__ == "__main__":
    main()
