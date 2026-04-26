# AnomalyDetection Pig Ultrasound Screening

This folder contains a standalone training workflow for pig ultrasound pregnancy screening.

## Dataset Layout

The trainer reads images from:

- `asset/train/1_Pregnant`
- `asset/train/2_NoPregnant`
- `asset/validate/1_Pregnant`
- `asset/validate/2_NoPregnant`
- `asset/validate/3_NotSure`
- `asset/test/1_Pregnant`
- `asset/test/2_NoPregnant`
- `asset/test/3_NotSure`

`3_NotSure` is treated as a review queue. It is predicted and reported, but it is not used as a binary training label.

Before feature extraction, the pipeline applies a `clinical_clean` preprocessing step that keeps the main ultrasound sector and masks common text/measurement overlay zones such as ID, Depth, Gain, date, PT/BF, Freeze, and measurement labels. This reduces the chance that the model learns from screen text instead of ultrasound anatomy.

## Train All Models

From the repository root:

```powershell
.\.venv\Scripts\python.exe AnomalyDetection\scripts\train_anomaly_models.py
```

The default run tries these feature/model combinations:

- Handcrafted ultrasound image features
- ResNet18 ImageNet features, if torchvision can load the pretrained weights
- DINOv2 ViT-S/14 features from Torch Hub
- Mahalanobis anomaly score
- Normal-quantile anomaly score
- Isolation Forest
- One-Class SVM
- Local Outlier Factor
- Balanced logistic regression
- Balanced random forest
- PatchCore-style nearest-neighbor patch anomaly score
- PaDiM-style diagonal Gaussian patch anomaly score

## Outputs

Each run writes to `AnomalyDetection/artifacts/models/<run_name>/`:

- `experiment_results.json`
- `<feature>__<model>.joblib`
- `<feature>__<model>_weights.json`
- `<feature>__<model>_predictions_validate.json`
- `<feature>__<model>_predictions_test.json`
- `<feature>__<model>_predictions_review.json`

The active model pointer is stored in:

```text
AnomalyDetection/artifacts/models/model_registry.json
```

The JSON files are the audit trail. The `.joblib` files are the loadable model artifacts for runtime prediction.
The active model is selected by the strongest combined validation/test balanced accuracy, with no-pregnant F1 as a tie-breaker.

The report generator also writes cleaned input images next to the heatmaps, so the visible report shows the same overlay-masked images used by the models. Heatmaps are occlusion-sensitivity visualizations over the strongest grid cells, not calibrated medical saliency maps.

## Predict One Image

```powershell
.\.venv\Scripts\python.exe AnomalyDetection\scripts\predict_image.py "path\to\image.jpg"
```

Optional JSON output:

```powershell
.\.venv\Scripts\python.exe AnomalyDetection\scripts\predict_image.py "path\to\image.jpg" --output AnomalyDetection\outputs\sample_prediction.json
```
