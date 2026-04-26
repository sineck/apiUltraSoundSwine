# Anomaly Model Artifacts

This folder keeps trained model artifacts for the ultrasound pregnancy screening workflow.

## Runtime Entry Point

Use `model_registry.json` as the source of truth. It stores the active model and the exact paths for each model artifact.

Current active model:

```text
20260426_114203/handcrafted__logistic_regression_balanced
```

## Run Folder Layout

Each run folder, such as `20260426_114203`, contains:

- `experiment_results.json` - training settings, metrics, thresholds, dataset summary.
- `<feature>__<model>.joblib` - loadable runtime model bundle.
- `<feature>__<model>_weights.json` - readable model/scaler/weight audit file.
- `<feature>__<model>_predictions_validate.json` - validate split predictions.
- `<feature>__<model>_predictions_test.json` - labeled test split predictions.
- `<feature>__<model>_predictions_review.json` - `3_NotSure` review predictions.
- `_index/` - generated human-readable maps for finding artifacts.

## Generated Indexes

Regenerate the index files after a training run:

```powershell
.\.venv\Scripts\python.exe AnomalyDetection\scripts\build_artifact_index.py
```

Useful files:

- `_index/artifact_manifest.json` - paths and metrics for every model.
- `_index/scaler_index.json` - `StandardScaler` mean/scale values where available.
- `_index/model_summary.csv` - compact spreadsheet-friendly summary.

Do not move `.joblib` or `*_weights.json` files manually unless you also update `model_registry.json`.
