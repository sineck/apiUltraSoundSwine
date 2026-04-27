# apiUltraSoundSwine

## ระบบนี้คืออะไร

`apiUltraSoundSwine` คือ FastAPI สำหรับงานอัลตราซาวด์สุกร มี 2 งานหลัก:

1. รับ **PDF ultrasound** แล้วแปลงเป็นภาพ, OCR ข้อความบนภาพ, ให้ AI ตีความผล, และบันทึกลง MySQL ตาม config
2. รับ **ภาพ ultrasound** แล้วให้ AI ตีความผลแบบ API

ระบบนี้มีทั้ง **V1** ของเดิมที่ยังต้องคงไว้ และ **V2** ที่เลือก backend model ได้จาก config

## เส้นทางที่ใช้งานจริง

### V1

- `GET /upload_form`
  - หน้า HTML เดิมสำหรับอัปโหลด PDF แบบ manual
- `POST /upload_pdf/`
  - รับ PDF
  - แปลงเป็นภาพใน `app/asset/`
  - OCR ข้อมูลบนภาพ
  - ใช้ YOLO รุ่นเดิม
  - เขียน DB ตามค่า `INSERT_ULTRASOUND_TO_DB`
- `GET /detect_form`
  - หน้า HTML เดิมสำหรับอัปโหลดภาพแบบ manual
- `POST /detect_follicle/`
  - รับหลายภาพ
  - ใช้ Gemini ตรวจถุงน้ำคร่ำ
  - ตอบ JSON แบบ detection response

### V2

- `POST /v2/upload_pdf/`
  - รับ PDF เหมือน V1
  - เลือก backend จาก `PREGNANCY_DETECT_MODEL_V2`
    - `anomaly`
    - `yolo`
    - `ensemble`
  - ตอบกลับแบบ legacy เดิม `{"status":"complete"}` หรือ `{"status":"error"}`
  - ถ้าเปิด DB จะเขียนลงตารางเดิมด้วย legacy labels

- `POST /v2/detection_pig`
  - รับหลายภาพ
  - เลือก backend จาก `PREGNANCY_DETECT_MODEL_V2`
  - มี precheck กันภาพนอกโดเมนก่อนเข้า model ถ้าภาพไม่เข้าลักษณะ ultrasound จะตอบ `unknown`
  - ตอบ JSON shape เดียวกับ `/detect_follicle/`
  - ถ้าเปิด DB จะเขียนลงตารางเดิมด้วย legacy labels

- `POST /v2/detection_pig_follicle`
  - รับหลายภาพ
  - ใช้ model gate แบบเดียวกับ `/v2/detection_pig`
  - มี precheck กันภาพนอกโดเมนก่อนเข้า model ถ้าภาพไม่เข้าลักษณะ ultrasound จะตอบ `unknown`
  - ถ้าผล gate เป็น `pregnant` ค่อยเรียก Gemini เพื่อวงรูป
  - ถ้า Gemini ให้ผล usable จะบันทึกรูป annotated ลง `app/detections/`
  - ถ้า Gemini ไม่ให้ annotation ที่ใช้ได้ จะคืนผล gate เดิมพร้อม `error_remark`

### System

- `GET /`
  - redirect ไป `/docs`
- `GET /openapi.json`
  - schema ของ API
- `GET /docs`
  - Swagger UI
- `GET /version`
  - คืนชื่อและเวอร์ชันของแอป
- `GET /health`
  - เช็ก MySQL และคืน safe runtime config summary โดยไม่เปิด secret

## AI backend ที่ระบบใช้

### YOLO legacy

- ใช้กับ V1 `/upload_pdf/` เสมอ
- ใช้กับ V2 ได้ถ้าตั้ง:

```env
PREGNANCY_DETECT_MODEL_V2=yolo
```

- โหลดโมเดลจาก `model/${ModelName}`

### Anomaly backend

- ใช้กับ V2 เท่านั้น
- ตั้งค่า:

```env
PREGNANCY_DETECT_MODEL_V2=anomaly
```

### Ensemble backend

- ใช้กับ V2 เท่านั้น
- ตั้งค่า:

```env
PREGNANCY_DETECT_MODEL_V2=ensemble
ModelName=best_finetune_YOLO26-cls_Ver2_20260424.pt
```

- หลักการ:
  - anomaly และ YOLO ต้องตอบ `pregnant` พร้อมกัน จึงถือว่า final result เป็น `pregnant`
  - ถ้าไม่ตรงกัน หรือมีฝั่งใดไม่ใช่ `pregnant` จะถือเป็น `no pregnant`
  - ถ้ามี backend พัง จะคืน `error` ตรง ๆ ไม่กลบผล

## V2 precheck

- ใช้กับ `POST /v2/detection_pig` และ `POST /v2/detection_pig_follicle`
- กรองภาพก่อนเข้า model ด้วยกติกาเบื้องต้น:
  - ภาพควรมี low color complexity
  - histogram ควรมีโทนดำ/เทาเด่น
  - edge density ต้องไม่ฟุ้งแบบภาพธรรมชาติ
- ถ้าไม่ผ่าน precheck ระบบจะตอบ:
- ระบบจะยังบันทึกรูปไว้ใน `app/detections/` ด้วย prefix `unknown_` เพื่อใช้ trace/debug
- ถ้าไม่ผ่าน precheck ระบบจะตอบ:

```json
{
  "result": "unknown",
  "confidence": 0.0
}
```

พร้อม `error_remark` ว่าไม่ถูกมองเป็นภาพ ultrasound

- active model อ่านจาก:

```text
AnomalyDetection/artifacts/models/model_registry.json
```

## Config ที่สำคัญที่สุด

ใช้ `config/.env` เป็น config จริงเสมอ  
`config/.env.example` เป็นแค่ template

ค่าที่ต้องรู้:

```env
MYAPI_PORT=3014
INSERT_ULTRASOUND_TO_DB=true
PREGNANCY_DETECT_MODEL_V2=anomaly
ModelName=best.pt
Min_Score_th=0.50
MYSQL_HOST=...
MYSQL_PORT=3306
MYSQL_DATABASE=...
MYSQL_USER=...
MYSQL_PASSWORD=...
GEMINI_API_KEY=...
```

ความหมายสั้น ๆ:

- `INSERT_ULTRASOUND_TO_DB`
  - `true` = เขียน MySQL
  - `false` = dry-run ไม่เขียน DB
- `PREGNANCY_DETECT_MODEL_V2`
  - `anomaly` = ใช้ backend ใหม่ของ V2
  - `yolo` = ใช้ backend YOLO เดิมของ V2
  - `ensemble` = ใช้ anomaly + YOLO ร่วมกัน โดยต้องเห็นตรงกันว่า `pregnant`
- `config/retrain_anomaly.json`
  - source of truth ของ default retrain สำหรับทั้ง API และ Python wrapper

## เริ่มใช้แบบเร็วที่สุด

ถ้าจะรันแอปตรงจาก source โดยไม่ใช้ Docker:

สร้าง virtual environment ก่อน:

```powershell
python -m venv .venv
```

activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

แล้วติดตั้ง dependency:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

จากนั้นค่อยรันแอป:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 3014 --reload
```

หรือถ้าไม่ใช้ `--reload`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 3014
```

ถ้าจะติดตั้ง dependency ใหม่ ให้ใช้ไฟล์เดียว:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

หมายเหตุ:
- ใช้ `config/.env` เป็น runtime config จริง
- ถ้าจะเปลี่ยน port ให้แก้ `MYAPI_PORT` ใน `config/.env` และเปลี่ยน `--port` ให้ตรงกัน

ถ้าจะรันระบบด้วย Docker ให้ใช้:

```powershell
docker compose --env-file config/.env up -d --build
```

ถ้าจะรันจาก image ที่ build แล้ว ให้ใช้:

```powershell
docker compose -f docker-compose.image.yml --env-file config/.env up -d --pull never
```

หลังรันแล้ว เปิดได้ที่:

- API docs: `http://localhost:3014/docs`
- OpenAPI JSON: `http://localhost:3014/openapi.json`
- V1 upload form: `http://localhost:3014/upload_form`
- V1 detect form: `http://localhost:3014/detect_form`
- version: `http://localhost:3014/version`
- health: `http://localhost:3014/health`

ก่อนยิงจริงทุกครั้ง ให้เช็ก `config/.env` ก่อนว่า

```env
INSERT_ULTRASOUND_TO_DB=true
```

หรือ

```env
INSERT_ULTRASOUND_TO_DB=false
```

ตรงกับสิ่งที่ต้องการจริง

## Retrain แบบจำสั้น

ถ้า API รันอยู่แล้วและแค่ต้องการสั่ง retrain:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:3014/anomaly/retrain/
```

เช็กสถานะ:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:3014/anomaly/retrain/status/
```

ถ้าต้องการรันจาก Python ตรงโดยไม่ผ่าน API ให้ใช้คำสั่งเดียวนี้:

```powershell
.\.venv\Scripts\python.exe AnomalyDetection\scripts\retrain_from_config.py
```

ทั้ง API และ Python wrapper จะอ่านค่า default จาก:

```text
config/retrain_anomaly.json
```

ดังนั้นถ้าจะเปลี่ยนชุด model, `batch_size`, การ rebuild index, หรือการ generate report ให้แก้ไฟล์นี้ไฟล์เดียว

ถ้าต้องการสลับ **anomaly model ที่มีอยู่แล้ว** โดยไม่ train ใหม่:

- แก้ `active_model` ใน  
  [D:\apiUltraSoundSwine\AnomalyDetection\artifacts\models\model_registry.json](D:\apiUltraSoundSwine\AnomalyDetection\artifacts\models\model_registry.json)
- แล้ว restart API

ถ้ารัน train/retrain ใหม่ ระบบจะเลือก active anomaly model ให้เอง แล้วอัปเดต `model_registry.json` อัตโนมัติ

## Health Response

`GET /health` ตอนนี้ไม่ได้คืนแค่สถานะ DB อย่างเดียว แต่แนบ `config` summary ที่ปลอดภัยกลับมาด้วย เช่น path ของ config จริง, backend ที่ active, ชื่อ YOLO model, Gemini model, และ `max_images`

ตัวอย่าง shape:

```json
{
  "status": "ok",
  "db": "connected",
  "app": {
    "name": "apiUltraSoundSwine",
    "version": "0.1.0"
  },
  "config": {
    "config_path": "D:/apiUltraSoundSwine/config/.env",
    "myapi_port": 3014,
    "insert_ultrasound_to_db": true,
    "pregnancy_detect_model_v2": "anomaly",
    "yolo_model_name": "best.pt",
    "gemini_model": "gemini-3-flash-preview",
    "max_images": 5
  }
}
```

secret เช่น `MYSQL_PASSWORD` จะไม่ถูกคืนใน payload นี้

## โครงสร้างโปรเจคที่ควรรู้

```text
app/
  main.py                FastAPI routes หลัก
  process_pdf.py         V1 PDF flow + YOLO legacy + OCR + DB insert
  process_detection.py   Gemini detection flow
  process_anomaly.py     anomaly runtime backend
  asset/                 ภาพที่แยกจาก PDF
  detections/            ภาพ output ของ detection route

AnomalyDetection/
  scripts/               train / report / predict scripts
  artifacts/models/      model registry และ anomaly artifacts

config/
  .env                   config จริง
  .env.example           template
  retrain_anomaly.json   default retrain config สำหรับ API และ Python wrapper

tests/
  mock_data/             ไฟล์ทดสอบจริง
  unit/                  unit tests
```

## เอกสารที่ควรอ่านต่อ

- [README-Docker.md](./README-Docker.md)
  - วิธี build/run/deploy ด้วย Docker
  - smoke test
  - image transfer

- [AnomalyDetection/README.md](./AnomalyDetection/README.md)
  - การ train anomaly model
  - model registry
  - retrain API

- [AGENTS.MD](./AGENTS.MD)
  - กติกาการทำงานของ coding agent ใน repo นี้

## หลักการอ่าน repo นี้

- source of truth ของ route อยู่ที่ `app/main.py`
- source of truth ของ config runtime อยู่ที่ `config/.env`
- source of truth ของ anomaly model active อยู่ที่ `AnomalyDetection/artifacts/models/model_registry.json`
- เอกสารเป็นตัวช่วย แต่ถ้า doc กับ code ขัดกัน ให้เช็ก code/runtime/tests ก่อน
