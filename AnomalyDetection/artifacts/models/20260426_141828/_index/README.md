# Model Artifact Index

Run: `20260426_141828`

Active model: `handcrafted__logistic_regression_balanced`

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

- Models: 4
- Weight JSON files: 4
- Prediction JSON files: 12
