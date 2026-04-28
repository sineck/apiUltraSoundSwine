# System Specification V2

## Purpose

`apiUltraSoundSwine` เป็น FastAPI สำหรับงานอัลตราซาวด์สุกร โดยมี 2 กลุ่มงานหลัก:

1. รับ PDF ultrasound แล้วแปลงเป็นภาพ, OCR, ประเมินผล, และเขียน DB ตาม config
2. รับภาพ ultrasound โดยตรงแล้วประเมินผลผ่าน API

เอกสารนี้เป็น owner doc ของพฤติกรรมระบบระดับ route/config/runtime ในภาพรวม  
ถ้าต้องการรายละเอียดฝั่ง anomaly training โดยตรง ให้ดู `AnomalyDetection/README.md`

## Runtime Entry Points

### V1

- `GET /upload_form`
  - หน้า HTML เดิมสำหรับอัปโหลด PDF
- `POST /upload_pdf/`
  - รับ PDF 1 ไฟล์
  - แปลงเป็นภาพใน `app/asset/`
  - OCR ข้อมูลบนภาพ
  - ใช้ YOLO legacy
  - เขียน DB ถ้า `INSERT_ULTRASOUND_TO_DB=true`
- `GET /detect_form`
  - หน้า HTML เดิมสำหรับอัปโหลดภาพ
- `POST /detect_follicle/`
  - รับหลายภาพ
  - ใช้ Gemini วง follicle / gestational sac
  - ตอบ JSON แบบ detection response

### V2

- `POST /v2/upload_pdf/`
  - รับ PDF เหมือน V1
  - ใช้ backend จาก `PREGNANCY_DETECT_MODEL_V2`
  - ตอบ `{"status":"complete"}` หรือ `{"status":"error"}`
  - ถ้าเปิด DB จะเขียนลงตารางเดิมด้วย legacy labels

- `POST /v2/detection_pig`
  - รับหลายภาพ
  - ทำ precheck กันภาพนอกโดเมนก่อนเข้า model
  - ถ้า precheck ไม่ผ่าน จะตอบ `unknown`
  - ถ้าผ่าน precheck จะใช้ backend จาก `PREGNANCY_DETECT_MODEL_V2`
  - ถ้าเปิด DB จะเขียนลงตารางเดิมด้วย legacy labels

- `POST /v2/detection_pig_follicle`
  - รับหลายภาพ
  - ทำ precheck แบบเดียวกับ `/v2/detection_pig`
  - ใช้ model gate แบบเดียวกับ `/v2/detection_pig`
  - ถ้า gate = `pregnant` ค่อยเรียก Gemini เพื่อทำ annotation
  - ถ้า Gemini ไม่ได้ผล usable จะคืนผล gate เดิมพร้อม `error_remark`

### System

- `GET /`
  - redirect ไป `/docs`
- `GET /openapi.json`
  - schema ของ API
- `GET /docs`
  - Swagger UI
- `GET /version`
  - ชื่อและ version ของแอป
- `GET /health`
  - สถานะระบบ
  - DB status
  - safe runtime config summary

## V2 Backend Modes

ค่าที่รองรับใน `config/.env`:

```env
PREGNANCY_DETECT_MODEL_V2=anomaly
```

หรือ

```env
PREGNANCY_DETECT_MODEL_V2=yolo
```

หรือ

```env
PREGNANCY_DETECT_MODEL_V2=ensemble
```

### anomaly

- ใช้ anomaly backend จาก active model ใน:
  - `AnomalyDetection/artifacts/models/model_registry.json`

### yolo

- ใช้ YOLO model จาก:
  - `model/${ModelName}`

### ensemble

- ใช้ anomaly + YOLO ร่วมกัน
- final result จะเป็น `pregnant` ก็ต่อเมื่อทั้ง anomaly และ YOLO ตอบ `pregnant` พร้อมกัน
- ถ้าไม่ตรงกัน หรือมีฝั่งใดไม่ใช่ `pregnant` จะถือเป็น `no pregnant`
- ถ้า backend พัง จะคืน `error`

## V2 Precheck

Precheck ใช้กับ:

- `POST /v2/detection_pig`
- `POST /v2/detection_pig_follicle`

กติกาปัจจุบัน:

- ภาพควรมี low color complexity
- ภาพควรมี dark/gray dominant ratio สูงพอ
- edge density ต้องไม่ฟุ้งแบบภาพธรรมชาติ

ถ้าไม่ผ่าน:

- ระบบจะไม่ส่งภาพเข้า model
- ระบบจะไม่เรียก Gemini
- ระบบจะไม่เขียน DB
- ระบบจะบันทึกรูปลง `app/detections/` ด้วย prefix `unknown_`
- ระบบจะตอบ:

```json
{
  "result": "unknown",
  "confidence": 0.0
}
```

พร้อม `error_remark` อธิบายเหตุผลที่ reject

## Database Write Policy

คุมด้วย:

```env
INSERT_ULTRASOUND_TO_DB=true
```

หรือ

```env
INSERT_ULTRASOUND_TO_DB=false
```

- `true`
  - V1 และ V2 route ที่รองรับ DB write จะเขียน DB
- `false`
  - ใช้เป็น dry-run mode
  - route ยังตอบผลตามปกติ แต่ไม่เขียน DB

## Health Contract

`GET /health` คืน:

- `status`
- `db`
- `app.name`
- `app.version`
- `config.config_path`
- `config.configured_myapi_port`
- `config.insert_ultrasound_to_db`
- `config.pregnancy_detect_model_v2`
- `config.yolo_model_name`
- `config.gemini_model`
- `config.max_images`

หมายเหตุ:

- `configured_myapi_port` เป็นค่า config ไม่ใช่ actual listen port ที่ bind อยู่จริง

## Anomaly Retrain Flow

entrypoint มี 2 แบบ:

1. API
   - `POST /anomaly/retrain/`
   - `GET /anomaly/retrain/status/`
2. Python script
   - `python -m AnomalyDetection.scripts.retrain_from_config`

ทั้งสองเส้นใช้ค่า default เดียวกันจาก:

```text
config/retrain_anomaly.json
```

pipeline หลัง retrain:

1. train model set ตาม config
2. rebuild artifact index
3. generate anomaly report
4. run validate compare report

compare/report หลัง retrain จะถูกเรียกอัตโนมัติด้วย:

```powershell
.\.venv\Scripts\python.exe tests\run_validate_compare.py --write-report
```

ผล compare จะเขียนไปที่:

- `AnomalyDetection/outputs/report/index.html`
- `AnomalyDetection/outputs/report/report_data.json`

## Retrain Status Model

retrain job status รองรับ:

- `idle`
- `queued`
- `running`
- `succeeded`
- `succeeded_with_warnings`
- `failed`

field สำคัญ:

- `phase`
- `last_completed_step`
- `failed_step`
- `warnings`

ความหมาย:

- ถ้า train สำเร็จ แต่ compare/report พัง
  - job จะเป็น `succeeded_with_warnings`
  - active model ใหม่ยังถือว่าใช้งานได้

## Source Of Truth

- runtime config จริง:
  - `config/.env`
- retrain default config:
  - `config/retrain_anomaly.json`
- active anomaly model:
  - `AnomalyDetection/artifacts/models/model_registry.json`
- dependency source:
  - `requirements.txt`
