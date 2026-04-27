# AnomalyDetection Pig Ultrasound Screening

This folder contains a standalone training workflow for pig ultrasound pregnancy screening.

## Scope In This Repository

เอกสารนี้อธิบายเฉพาะฝั่ง **train / model artifact / anomaly runtime backend** ของโปรเจคนี้

ถ้าต้องการภาพรวมทั้งระบบ เช่น V1/V2 route, config หลัก, และวิธีรันระบบ ให้กลับไปอ่าน [README.md](..\README.md) ที่ root ก่อน แล้วค่อยกลับมาอ่านไฟล์นี้เมื่อจะทำงานฝั่ง anomaly โดยตรง

- **V1 runtime** ของระบบหลักอยู่ที่ `POST /upload_pdf/` และไม่ได้เรียก anomaly backend
- **V2 runtime** ที่เรียก anomaly backend ได้จริงคือ:
  - `POST /v2/upload_pdf/`
  - `POST /v2/detection_pig`
  - `POST /v2/detection_pig_follicle`
- การเลือกว่าจะใช้ anomaly หรือ YOLO ใน V2 ถูกคุมจาก `config/.env` ด้วยค่า:

```env
PREGNANCY_DETECT_MODEL_V2=anomaly
```

ถ้าต้องการภาพรวมการรัน Docker, วิธี build image, หรือวิธี deploy ให้ดู [README-Docker.md](..\README-Docker.md)

## คำสั่งใช้บ่อย

ใช้ section นี้เป็น quick reference เวลาต้องการ train anomaly ซ้ำ, rebuild index, หรือเช็ก active model โดยไม่ต้องไล่อ่านทั้งไฟล์

### Train compact set ใหม่

```powershell
.\.venv\Scripts\python.exe -m AnomalyDetection.scripts.train_anomaly_models
```

### Train full research sweep

```powershell
.\.venv\Scripts\python.exe -m AnomalyDetection.scripts.train_anomaly_models --model-keys all
```

### Rebuild artifact index ของ active run

```powershell
.\.venv\Scripts\python.exe -m AnomalyDetection.scripts.build_artifact_index
```

### ทำนายภาพเดียวจาก active model

```powershell
.\.venv\Scripts\python.exe -m AnomalyDetection.scripts.predict_image "tests\mock_data\sample_input.png"
```

### เทียบโมเดลบนชุด validate

ถ้าต้องการเทียบผลของ:

- `anomaly`
- `yolo:best.pt`
- `yolo:best_finetune_YOLO26-cls_Ver2_20260424.pt`
- `ensemble:anomaly+yolo_finetune`

ให้ใช้สคริปต์นี้จาก root repo:

```powershell
.\.venv\Scripts\python.exe tests\run_validate_compare.py
```

สคริปต์นี้จะ:

- รันภาพทุกไฟล์ใน `AnomalyDetection/asset/validate`
- ไม่ผ่าน Gemini
- พิมพ์ summary table และ detail table ลง console
- ใช้กติกา binary เดียวกันทั้งชุด:
  - `1_Pregnant` = `pregnant`
  - `2_NoPregnant` = `no_pregnant`
  - `3_NotSure` = `no_pregnant`

ถ้าต้องการรัน compare แล้วเขียนรายงาน HTML ไปพร้อมกัน ใช้:

```powershell
.\.venv\Scripts\python.exe tests\run_validate_compare.py --write-report
```

คำสั่งนี้จะเขียนไฟล์:

- `AnomalyDetection/outputs/report/index.html`
- `AnomalyDetection/outputs/report/report_data.json`

ถ้ายังมีคนเรียก wrapper เดิมอยู่ ก็ยังใช้ได้:

```powershell
.\.venv\Scripts\python.exe tests\generate_validate_report.py
```

แต่ตอนนี้เส้นหลักที่ครบที่สุดคือ `run_validate_compare.py --write-report`

### เรียก retrain ผ่าน API

เปิด Swagger UI ก่อน:

- `http://127.0.0.1:3014/docs`
- route ที่ใช้:
  - `POST /anomaly/retrain/`
  - `GET /anomaly/retrain/status/`

ถ้าต้องการลองกดจากหน้าเว็บ ให้เริ่มจาก `/docs` ก่อน แล้วค่อยใช้คำสั่งด้านล่างเมื่อจะเรียกจาก terminal หรือ script

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:3014/anomaly/retrain/
```

### เช็กสถานะ retrain job

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:3014/anomaly/retrain/status/
```

### เรียก retrain จาก Python ตรง

ถ้าไม่ต้องการผ่าน API และต้องการให้ train/index/report วิ่งตาม config เดียวกัน:

```powershell
.\.venv\Scripts\python.exe -m AnomalyDetection.scripts.retrain_from_config
```

ทั้ง route และ script นี้จะอ่านค่าจาก:

```text
config/retrain_anomaly.json
```

หลัง train/index/report เสร็จ pipeline จะรัน compare ต่ออัตโนมัติด้วย:

```powershell
.\.venv\Scripts\python.exe tests\run_validate_compare.py --write-report
```

ดังนั้น `AnomalyDetection/outputs/report/index.html` ชุดล่าสุดจะถูกรีเฟรชจากผล compare ทันทีหลัง retrain

## Dataset Layout

โฟลเดอร์นี้พูดถึงเฉพาะ dataset สำหรับ train/validate/test ของ anomaly workflow ไม่ได้อธิบายไฟล์ output ของ V1/V2 runtime

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

ส่วนนี้คือคำสั่ง train ฝั่ง anomaly โดยตรง ไม่ใช่คำสั่ง run API

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m AnomalyDetection.scripts.train_anomaly_models
```

The default run keeps the compact production comparison set:

- `handcrafted__logistic_regression_balanced`
- `resnet18__logistic_regression_balanced`
- `handcrafted__normal_quantile`
- `dinov2__random_forest_balanced`

To run the full research sweep again:

```powershell
.\.venv\Scripts\python.exe -m AnomalyDetection.scripts.train_anomaly_models --model-keys all
```

The full sweep includes:

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

หลัง train เสร็จ ไฟล์สำคัญจะถูกเขียนไว้ใน `AnomalyDetection/artifacts/models/<run_name>/`

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

Runtime ฝั่ง V2 จะอ่าน active model จากไฟล์นี้เสมอ:

```text
AnomalyDetection/artifacts/models/model_registry.json
```

ดังนั้น `model_registry.json` คือ source of truth ของ anomaly backend ใน runtime ปัจจุบัน ไม่ใช่ path ที่ hardcode ไว้ใน code หรือ path absolute จากเครื่อง train เดิม

## สลับ active model ที่มีอยู่แล้ว

ถ้าต้องการสลับไปใช้ anomaly model ตัวอื่นที่มี artifact อยู่แล้ว โดย **ไม่ต้อง train ใหม่**

ให้แก้ field `active_model` ในไฟล์นี้:

```text
AnomalyDetection/artifacts/models/model_registry.json
```

ตัวอย่าง:

```json
{
  "active_model": "20260426_141828/handcrafted__logistic_regression_balanced"
}
```

จากนั้นให้ **restart API** เพื่อให้ runtime โหลด model active ตัวใหม่

สรุป:

- `config/.env`
  - ใช้เลือกว่าจะให้ V2 ใช้ backend `anomaly`, `yolo`, หรือ `ensemble`
- `model_registry.json`
  - ใช้เลือกว่า anomaly backend จะโหลด **model ตัวไหน**

ถ้าจะเปลี่ยนแค่ anomaly model ภายใน backend เดิม ไม่ต้องแก้ `config/.env`; แก้ `active_model` แล้ว restart API ก็พอ

To generate a readable artifact map for the active run:

```powershell
.\.venv\Scripts\python.exe -m AnomalyDetection.scripts.build_artifact_index
```

This writes `_index/artifact_manifest.json`, `_index/scaler_index.json`, and `_index/model_summary.csv` inside the run folder. Use these files to find each model's `.joblib`, readable weights, predictions, metrics, and StandardScaler values without moving runtime artifacts.

The report generator also writes cleaned input images next to the heatmaps, so the visible report shows the same overlay-masked images used by the models. Heatmaps are occlusion-sensitivity visualizations over the strongest grid cells, not calibrated medical saliency maps.

## Retrain From API

ส่วนนี้ใช้เมื่อ API รันอยู่แล้ว และต้องการสั่ง retrain ผ่าน route โดยไม่ต้องเรียก script ตรง

Start a new anomaly training job in the background:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:3014/anomaly/retrain/
```

If you need to run the same retrain flow without API, use:

```powershell
.\.venv\Scripts\python.exe -m AnomalyDetection.scripts.retrain_from_config
```

ทั้งสองทางจะใช้ค่าใน `config/retrain_anomaly.json` เป็น default กลาง ถ้าจะเปลี่ยน compact set, `batch_size`, หรือ heatmap mode ให้แก้ไฟล์นี้ และหลัง retrain เสร็จจะต่อ compare/report ให้อัตโนมัติ

หมายเหตุเรื่อง runtime:

- route retrain นี้เป็นงาน background สำหรับอัปเดต anomaly artifacts
- เมื่อ train สำเร็จและอัปเดต `model_registry.json` แล้ว เส้น V2 ที่ใช้ `PREGNANCY_DETECT_MODEL_V2=anomaly` จะอิง active model ตัวใหม่
- `force=true` ตอนนี้ไม่เปิดงาน train ซ้อนกับ job ที่กำลังรันอยู่ เพื่อกัน artifact/registry ชนกัน
- ถ้ารัน train/retrain ใหม่ ระบบจะ **เลือก active model ให้เอง** ตาม logic ใน `AnomalyDetection/scripts/train_anomaly_models.py`
- เกณฑ์ปัจจุบันคือเลือก model ที่ดีที่สุดจากผล validation/test แล้วเขียนกลับเข้า `model_registry.json` อัตโนมัติ

Check status:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:3014/anomaly/retrain/status/
```

`detail_heatmaps` can be `none`, `active`, or `all`. Use `all` only when you really want heatmaps for every test image of every model, because it is slow.

## Predict One Image

ใช้คำสั่งนี้เมื่อต้องการเช็ก active model แบบเร็ว ๆ จากไฟล์ภาพเดียว โดยไม่ต้องยิงผ่าน API

```powershell
.\.venv\Scripts\python.exe -m AnomalyDetection.scripts.predict_image "path\to\image.jpg"
```

Optional JSON output:

```powershell
.\.venv\Scripts\python.exe -m AnomalyDetection.scripts.predict_image "path\to\image.jpg" --output AnomalyDetection\outputs\sample_prediction.json
```

CLI นี้ใช้ logic resolve active model แบบเดียวกับ runtime API:

- อ่าน `AnomalyDetection/artifacts/models/model_registry.json`
- รองรับ registry ที่ยังเก็บ absolute Windows path จากเครื่อง train เดิม
- ถ้า path เดิมไม่มีอยู่แล้ว จะ fallback ไปหาไฟล์ `.joblib` ข้าง registry run folder ปัจจุบัน

## Runtime Contract Notes

ส่วนนี้เอาไว้ผูกให้เห็นว่า artifact/training ฝั่ง anomaly ไปเชื่อมกับ route V2 ของระบบหลักอย่างไร

เมื่อ V2 route ใช้ anomaly backend:

- `POST /v2/upload_pdf/`
  - รับ PDF
  - render/crop เป็นภาพใน `app/asset/`
  - เรียก anomaly backend ทีละหน้า
  - insert DB ด้วย legacy labels เดิม (`1_Pregnant` / `2_NoPrenant_or_NotSure`)
  - ตอบกลับแบบ legacy route เดิม `{"status":"complete"}` หรือ `{"status":"error"}`

- `POST /v2/detection_pig`
  - รับหลายรูป
  - เรียก anomaly backend ทีละรูป
  - คืน response shape แบบเดียวกับ `/detect_follicle/`
  - ถ้าเปิด `INSERT_ULTRASOUND_TO_DB=true` จะเขียน DB ด้วย legacy labels เดียวกัน

- `POST /v2/detection_pig_follicle`
  - รับหลายรูป
  - ใช้ anomaly backend เป็น gate ก่อน
  - ถ้าผล gate เป็น `pregnant` ค่อยเรียก Gemini เพื่อพยายามสร้าง annotation
  - ถ้า Gemini ไม่คืน annotation ที่ใช้ได้ route จะยังคืนผล gate เดิม แต่ใส่ `error_remark`

ดังนั้นฝั่ง anomaly training/artifact ในโฟลเดอร์นี้ไม่ได้ถูกเรียกโดย V1 โดยตรง แต่เป็น backend ที่ V2 route เลือกใช้ได้ผ่าน config เท่านั้น
