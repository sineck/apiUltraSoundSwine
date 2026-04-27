# UltrasoundPig Docker Deployment Guide

## ภาพรวม
- FastAPI API สำหรับรับไฟล์ PDF, แปลงเป็น PNG, และเขียนข้อมูลลง MySQL Cloud (AWS RDS)
- ใช้ Docker Compose สำหรับ deployment
- ไม่ใช้ phpMyAdmin, nginx, หรือ service อื่น

## ข้อกำหนดเบื้องต้น
- Docker Desktop (Windows/Mac) หรือ Docker Engine (Linux)
- Docker Compose
- Git (สำหรับ clone โปรเจค)
- MySQL Cloud (AWS RDS) พร้อมใช้งาน

## โครงสร้างโปรเจค
```
project-root/
│
├── app/
│   ├── main.py
│   ├── process_pdf.py
│   ├── process_detection.py
│   ├── uploads/
│   ├── asset/
│   └── detections/
├── config/
│   ├── .env
│   └── .env.example
├── logs/
│   └── .gitkeep
├── model/
├── tests/
│   └── mock_data/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README-Docker.md
└── AGENTS.MD
```

## การตั้งค่า Environment
- ใช้ `config/.env` เป็น runtime configuration จริงเสมอ ทั้ง local, Docker Compose, และ smoke test
- `config/.env.example` เป็น template/mockup เท่านั้น ใช้เพื่อดูว่าต้องมี key อะไรบ้าง แล้วคัดลอก/ปรับค่าไปที่ `config/.env`
- ห้ามใส่ inline comment ต่อท้ายค่าใน `config/.env` เพราะ Docker `--env-file` จะอ่าน comment นั้นเป็นส่วนหนึ่งของค่า เช่นให้เขียน comment แยกบรรทัดแทน
- ตัวอย่าง key หลัก:
  ```env
  MYAPI_PORT=3014
  MYSQL_HOST=your-rds-endpoint
  MYSQL_PORT=3306
  MYSQL_DATABASE=your_db
  MYSQL_USER=your_user
  MYSQL_PASSWORD=your_password
  INSERT_ULTRASOUND_TO_DB=true
  ModelName=best.pt
  Min_Score_th=0.70
  PREGNANCY_DETECT_MODEL_V2=anomaly
  ```
- .env ต้องถูก ignore จาก git

## เริ่มต้นใช้งาน (Quick Start)

โปรเจคนี้แยกการใช้งานเป็น 3 วิธี:

- **วิธี Local: รันตรงด้วย Uvicorn** ใช้เมื่อจะ debug หรือพัฒนา API จาก source โดยไม่ผ่าน Docker
- **วิธี A: Clone โปรเจคมาทำงานต่อ** ใช้เมื่อจะเข้ามาอ่านโค้ด, แก้โค้ด, รัน test, หรือเตรียม build image ใหม่จาก source
- **วิธี B1: สร้าง Image** ใช้เมื่อมี source code ครบและต้องการ build image จาก `Dockerfile`
- **วิธี B2: Image Transfer** ใช้เมื่อมี image ที่ build แล้วและต้องการเอาไปรันอีกเครื่อง โดยไม่ build ใหม่

ทุกวิธีที่รัน container ต้องมี `config/.env` เพราะเป็น runtime configuration จริง. `config/.env.example` เป็น template เท่านั้น.

### วิธี Local: รันตรงด้วย Uvicorn

ใช้วิธีนี้เมื่อจะ debug หรือพัฒนา API จาก source โดยไม่ผ่าน Docker

สิ่งที่ต้องมี:

```text
project-root/
├── app/
├── AnomalyDetection/
├── config/
│   ├── .env
│   └── .env.example
├── model/
├── requirements.txt
└── Dockerfile
```

สร้าง virtual environment:

```powershell
python -m venv .venv
```

activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

ติดตั้ง dependency:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

จากนั้นถ้าใช้ virtual environment ใน repo:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 3014 --reload
```

ถ้าไม่ต้องการ `--reload`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 3014
```

หมายเหตุ:
- route จะอ่าน config จาก `config/.env` โดยตรง
- ถ้าจะเปลี่ยน port ให้แก้ `MYAPI_PORT` ใน `config/.env` และเปลี่ยน `--port` ให้ตรงกัน
- ถ้าเปิด `INSERT_ULTRASOUND_TO_DB=true` การยิง route ที่เกี่ยวกับ PDF/Detection จะเขียน DB จริง
- ถ้าจะ smoke test แบบไม่ลง DB ให้ตั้ง:

```env
INSERT_ULTRASOUND_TO_DB=false
```

### วิธี A: Clone โปรเจคมาทำงานต่อ

วิธีนี้คือ workflow ฝั่ง source code ไม่ใช่ deployment โดยตรง:

```powershell
git clone <repo-url>
cd apiUltraSoundSwine
```

หลัง clone แล้วค่อยเลือกต่อว่าจะ build image ใหม่ด้วย **วิธี B1** หรือจะใช้ image ที่มีอยู่แล้วด้วย **วิธี B2**

### วิธี B1: สร้าง Image

เครื่องที่จะ build ต้องมี source code และ runtime artifacts ครบก่อนรัน:

```text
project-root/
├── app/
├── AnomalyDetection/
│   ├── scripts/
│   └── artifacts/
│       └── models/
│           ├── model_registry.json
│           └── 20260426_114203/
├── config/
│   ├── .env
│   └── .env.example
├── model/
│   ├── best.pt
│   └── best_finetune_YOLO26-cls_Ver2_20260424.pt
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

คำสั่งเดียวสำหรับ build และ run:

```powershell
docker compose --env-file config/.env up -d --build
```

คำสั่งนี้จะ build image ใหม่จาก `Dockerfile`, copy `model/` และ `AnomalyDetection/artifacts/models/` เข้า image, แล้ว start container. หลัง build แล้ว container ไม่ต้อง mount model/artifacts/logs จาก host.

### วิธี B2: Image Transfer

ใช้วิธีนี้เมื่อ build image เสร็จจากเครื่องต้นทางแล้วต้องเอาไปรันเครื่องอื่น. เครื่องปลายทางไม่ต้องมี source code, `model/`, หรือ `AnomalyDetection/artifacts/models/` เพราะอยู่ใน image แล้ว.

เครื่องต้นทาง export image:

```powershell
docker save apiultrasoundswine-api:latest -o apiultrasoundswine-api.tar
```

เครื่องปลายทางต้องมีอย่างน้อย:

```text
deploy-folder/
├── apiultrasoundswine-api.tar
├── docker-compose.image.yml
└── config/
    └── .env
```

เครื่องปลายทาง load image:

```powershell
docker load -i apiultrasoundswine-api.tar
```

คำสั่งเดียวสำหรับ run จาก image ที่ load แล้ว:

```powershell
docker compose -f docker-compose.image.yml --env-file config/.env up -d
```

วิธีนี้ไม่ใช้ `--build` และไม่ต้องมี `model/` บนเครื่องปลายทาง.

### ตรวจสอบการทำงานหลังรัน

- **เปิดหน้า API**: `http://localhost:${MYAPI_PORT}/docs` (default: 3014)
- **เปิดหน้าอัปโหลด**: `http://localhost:${MYAPI_PORT}/upload_form`
- **เปิดหน้าตรวจรูปด้วย Gemini**: `http://localhost:${MYAPI_PORT}/detect_form`
- **API อัปโหลด PDF V1**: `POST http://localhost:${MYAPI_PORT}/upload_pdf/`
- **API อัปโหลด PDF V2 ด้วย pregnancy model**: `POST http://localhost:${MYAPI_PORT}/v2/upload_pdf/`
- **API ตรวจรูปหมูด้วย pregnancy model**: `POST http://localhost:${MYAPI_PORT}/v2/detection_pig`
- **ดูชื่อและเวอร์ชันแอป**: `http://localhost:${MYAPI_PORT}/version`
- **ตรวจสุขภาพ API + MySQL**: `http://localhost:${MYAPI_PORT}/health`
- **ดู OpenAPI schema**: `http://localhost:${MYAPI_PORT}/openapi.json`
- **ดู Log การทำงาน**:
  ```bash
  docker compose logs -f api
  ```

---

## คำสั่ง Docker Compose อื่นๆ ที่ใช้บ่อย
- **หยุดการทำงาน:** `docker compose down`
- **รีสตาร์ท:** `docker compose restart`

## คำสั่ง Docker ที่ใช้บ่อย

```powershell
# build image ใหม่จาก source
docker build -t apiultrasoundswine-api:latest .

# run จาก source compose
docker compose --env-file config/.env up -d --build

# run จาก image compose
docker compose -f docker-compose.image.yml --env-file config/.env up -d --pull never

# หยุด service ที่รันจาก image compose
docker compose -f docker-compose.image.yml --env-file config/.env down

# ดู logs ของ service api
docker compose -f docker-compose.image.yml --env-file config/.env logs -f api

# export image เป็นไฟล์ .tar
docker save apiultrasoundswine-api:latest -o apiultrasoundswine-api.tar

# import image จากไฟล์ .tar
docker load -i apiultrasoundswine-api.tar
```

## Health และ Version

- `GET /version`: เช็กชื่อและเวอร์ชันแอปอย่างเดียว ไม่แตะ MySQL เหมาะสำหรับดูว่า container ที่รันอยู่เป็น release ไหน
- `GET /health`: เช็ก MySQL ด้วย และแนบข้อมูล `app.name` / `app.version` พร้อม `config` summary ที่ปลอดภัยกลับมาใน response
- เวอร์ชันถูกกำหนดใน `app/version.py` ไม่ใช่ `.env` เพราะเป็นข้อมูลของ code/release

ตัวอย่าง:

```json
{
  "status": "ok",
  "db": "connected",
  "app": {
    "name": "apiUltraSoundSwine",
    "version": "0.1.0"
  }
}
```

## Smoke Test แบบไม่ลง DB

ก่อนทดสอบ PDF โดยไม่เขียน MySQL ให้ตั้งใน `config/.env`:

```env
INSERT_ULTRASOUND_TO_DB=false
```

จากนั้น build และรัน smoke test ด้วย config จริง:

```powershell
docker build -t apiultrasoundswine-test .
docker run --rm --env-file config/.env `
  -v "${PWD}/tests/mock_data:/app/tests/mock_data:ro" `
  -v "${PWD}/app/asset:/app/app/asset" `
  apiultrasoundswine-test sh -c "python - <<'PY'
from pathlib import Path
from app.process_pdf import convert_pdf_to_png, should_insert_ultrasound_to_db

pdf_path = next(Path('/app/tests/mock_data').glob('*.pdf'))
print(f'db_insert_enabled={should_insert_ultrasound_to_db()}')
print(f'pdf={pdf_path}')
print(f'result={convert_pdf_to_png(str(pdf_path))}')
PY"
```

ผลที่คาดหวัง: `db_insert_enabled=False`, `result=True`, และมี PNG ใหม่ใน `app/asset/`

### Smoke Test V1 / V2 แบบไม่ลง DB

ไม่ว่าจะใช้ **วิธี B1: สร้าง Image** หรือ **วิธี B2: Image Transfer** ให้ตั้งค่าเดียวกันก่อน:

```env
INSERT_ULTRASOUND_TO_DB=false
```

จากนั้นรัน container ตามวิธีที่เลือก:

- **B1: สร้าง Image**
  ```powershell
  docker compose --env-file config/.env up -d --build
  ```
- **B2: Image Transfer**
  ```powershell
  docker load -i apiultrasoundswine-api.tar
  docker compose -f docker-compose.image.yml --env-file config/.env up -d
  ```

แล้วทดสอบ route หลักด้วย mock data:

- **V1 PDF**
  ```powershell
  curl -X POST "http://127.0.0.1:3014/upload_pdf/" -F "file=@tests/mock_data/sample_input.pdf;type=application/pdf"
  ```
  ค่าที่คาดหวัง:
  ```json
  {"status":"complete"}
  ```

- **V2 PDF**
  ```powershell
  curl -X POST "http://127.0.0.1:3014/v2/upload_pdf/" -F "file=@tests/mock_data/sample_input.pdf;type=application/pdf"
  ```
  ค่าที่คาดหวัง:
  ```json
  {"status":"complete"}
  ```

- **V2 รูป**
  ```powershell
  curl -X POST "http://127.0.0.1:3014/v2/detection_pig" -F "files=@tests/mock_data/sample_input.png;type=image/png"
  ```
  ค่าที่คาดหวังคือ response shape แบบ `/detect_follicle/` และ `main_results` ต้องเป็น `success`

- **V2 รูป + follicle annotate**
  - route นี้ต้องใช้ Gemini และจะ annotate เฉพาะภาพที่ pregnancy gate ให้ผลเป็น `pregnant`
  - ถ้าจะทดสอบ route นี้จริง ต้องเตรียม `GEMINI_API_KEY` ใน `config/.env`
  ```powershell
  curl -X POST "http://127.0.0.1:3014/v2/detection_pig_follicle" -F "files=@tests/mock_data/sample_input.png;type=image/png"
  ```

เมื่อ `INSERT_ULTRASOUND_TO_DB=false`:

- V1 `/upload_pdf/` ต้องแปลง PDF และเขียนไฟล์ output ได้ตามปกติ
- V2 `/v2/upload_pdf/` ต้อง infer และตอบ legacy shape ได้ตามปกติ
- V2 `/v2/detection_pig` ต้อง infer และตอบ JSON shape เดิมได้ตามปกติ
- ทั้ง 3 เส้นต้อง **ไม่** เขียนข้อมูลลง MySQL

---

## Pregnancy Model API

Runtime ปัจจุบันเหลือ 2 รุ่นหลัก:

- `POST /upload_pdf/` คือ V1 เดิมของระบบ รับ field `file` สำหรับ PDF หนึ่งไฟล์ แล้ววิ่งเข้า pipeline `convert_pdf_to_png()` ตามเดิม โดยใช้ env `INSERT_ULTRASOUND_TO_DB` คุมว่าจะเขียน DB หรือไม่
- `POST /v2/upload_pdf/` คือ V2 ฝั่ง PDF รับ field `file` สำหรับ PDF หนึ่งไฟล์ เหมือน `/upload_pdf/` เดิม แต่เลือก backend จาก `PREGNANCY_DETECT_MODEL_V2`, insert DB แบบ PDF flow เดิม, และตอบ legacy shape `{"status":"complete"}` หรือ `{"status":"error"}`
- `POST /v2/detection_pig` คือ V2 ฝั่งรูป รับ field `files` สำหรับไฟล์รูปหลายไฟล์ แล้วเลือก backend จาก `PREGNANCY_DETECT_MODEL_V2` โดยคืน JSON แบบเดียวกับ `/detect_follicle/`
- `POST /v2/detection_pig_follicle` คือ V2 ฝั่งรูปที่ใช้ pregnancy model เป็น gate ก่อน แล้วค่อยเรียก Gemini annotation เฉพาะรายการที่ gate เป็น `pregnant`

`/v2/detection_pig` ทำงานได้ทั้ง backend `anomaly` และ `yolo` และคืน JSON shape เดียวกันเสมอ:

```json
{
  "main_results": "success",
  "error_massage": "",
  "results": [
    {
      "path_images": "app/detections/preg_pdf_20260426_130909_693921_p001_4e243620.png",
      "result": "pregnant",
      "confidence": 1.0,
      "number_of_fetus": 0,
      "error_remark": ""
    }
  ]
}
```

เลือก backend ด้วย `config/.env`:

```env
# anomaly = ใช้โมเดลใหม่จาก AnomalyDetection/artifacts/models/model_registry.json
# yolo    = ใช้โมเดลเก่าจาก model/${ModelName}
PREGNANCY_DETECT_MODEL_V2=anomaly
```

ถ้าตั้ง `PREGNANCY_DETECT_MODEL_V2=anomaly` ค่า active model จะอ่านจาก `AnomalyDetection/artifacts/models/model_registry.json` ภายใต้ project root ปัจจุบันเสมอ. อย่าใช้ absolute path จากเครื่อง train เป็น runtime source of truth. โค้ดรองรับ registry ที่ยังมี Windows path เก่าอยู่ใน `model_file` โดย resolve กลับมาที่ project root ปัจจุบันแทน.

ถ้าตั้ง `PREGNANCY_DETECT_MODEL_V2=yolo` จะใช้โมเดลเก่าใน `model/` ตามค่า `ModelName` และ threshold จาก `Min_Score_th`.

ถ้าตั้ง `PREGNANCY_DETECT_MODEL_V2=ensemble` จะใช้ anomaly + YOLO ร่วมกัน:
- ทั้ง anomaly และ YOLO ต้องตอบ `pregnant` พร้อมกัน จึงถือว่า final result เป็น `pregnant`
- ถ้าไม่ตรงกัน หรือมีฝั่งใดไม่ใช่ `pregnant` จะถือเป็น `no pregnant`
- YOLO ฝั่ง ensemble จะใช้ weight จาก `ModelName` โดยตรง ดังนั้นถ้าจะจับคู่กับรุ่น finetune ให้ตั้ง `ModelName=best_finetune_YOLO26-cls_Ver2_20260424.pt`

ถ้า `INSERT_ULTRASOUND_TO_DB=true` เส้น V1 `/upload_pdf/`, V2 `/v2/upload_pdf/`, และ V2 `/v2/detection_pig` จะ insert ผลลงตารางเดียวกับ PDF flow เดิม (`UltraSoudPigAI`) ต่อรายการ. สำหรับ V2 ทั้งสองเส้น ค่า `results_ai` จะถูก normalize ให้เป็น legacy label เดียวกันเสมอ (`1_Pregnant` หรือ `2_NoPrenant_or_NotSure`) ไม่ว่าจะใช้ backend `yolo` หรือ `anomaly`; ส่วนรูปที่แยกจาก PDF จะถูกเก็บไว้ใต้ `app/asset/`.

`/v2/detection_pig_follicle` ไม่ใช่ route insert DB. route นี้เน้นคืนผล gate + Gemini annotation path กลับมาให้ client ใช้งานต่อ

หมายเหตุ: transition routes เดิม เช่น `/detect_pregnancy_pdf/`, `/detect_anomaly_pdf/`, `/detect_pregnancy_pic/`, และ `/detect_anomaly_pic/` ถูกถอดออกแล้ว เพื่อให้ runtime เหลือเฉพาะ V1 กับ V2 ที่ใช้งานจริง.

หมายเหตุ: โฟลเดอร์จริงของระบบนี้คือ `AnomalyDetection` ใต้ project root ไม่ใช่ `Anomaly` หรือชื่ออื่น. active model ปัจจุบันเป็น `handcrafted__logistic_regression_balanced` จึงไม่เรียก DINOv2 ใน runtime ปกติ.

### Deep feature weights

โค้ดสร้าง anomaly artifacts จริงมี deep feature 2 กลุ่มที่ต้องใช้ pretrained weights ถ้าเลือกเป็น active model:

- `resnet18__...` เรียก `ResNet18_Weights.DEFAULT` จาก torchvision
- `dinov2__...` เรียก `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`

Dockerfile ตั้งค่า default ให้ pre-download weights ทั้งสองกลุ่มเข้า `TORCH_HOME=/app/.cache/torch` ระหว่าง build:

- `BAKE_DINOV2_WEIGHTS=true`
- `BAKE_RESNET18_WEIGHTS=true`

เหตุผลคือถ้าวันหลังเปลี่ยน `model_registry.json` ให้ active model เป็นกลุ่ม `resnet18__...` หรือ `dinov2__...` แล้ว production ถูก block network, container จะยังมี cache สำหรับ feature extractor อยู่แล้ว.

ถ้าต้องการ image เล็กกว่าและแน่ใจว่าไม่ใช้ deep feature models ให้ build ด้วย:

```powershell
docker build --build-arg BAKE_DINOV2_WEIGHTS=false --build-arg BAKE_RESNET18_WEIGHTS=false -t apiultrasoundswine-api:latest .
```

ถ้าเปลี่ยน active model เป็น `resnet18__...` หรือ `dinov2__...` ต้องมีอย่างใดอย่างหนึ่ง:

- ใช้ image ที่ build ด้วย deep weight bake args เป็น `true`
- server มี network ให้ torchvision หรือ `torch.hub` ดาวน์โหลด weights ตอนรันครั้งแรก
- เตรียม torch hub cache ไว้ล่วงหน้าแล้ว mount เข้า container

สำหรับ production ที่ network ถูก block ไม่ควรพึ่ง first-run download ของ pretrained weights.

---

## Docker Image Size

ตอนนี้ repo ใช้ `requirements.txt` ไฟล์เดียวทั้ง local และ Docker:

- ติดตั้ง PyTorch แบบ CPU-only จาก `https://download.pytorch.org/whl/cpu`
- Docker build จะอ่านจาก `requirements.txt` ไฟล์เดียว แต่กรอง `ultralytics` ออกไปติดตั้งแบบ `--no-deps` แยก เพื่อไม่ดึง `opencv-python` ตัวเต็มและ dependency GUI เกินจำเป็น
- ใช้ `opencv-python-headless` ชุดเดียวสำหรับ API runtime
- ใช้ Python stdlib สำหรับ Docker healthcheck แทน `curl`
- ไม่ copy `config/.env` เข้า image
- local Docker build copy `model/` และ `AnomalyDetection/artifacts/models/` เข้า image เพื่อให้ย้าย image ไปเครื่องอื่นได้โดยไม่ต้องย้าย model แยก
- ค่า default จะ bake DINOv2 และ ResNet18 weights/cache เข้า image ด้วย `BAKE_DINOV2_WEIGHTS=true` และ `BAKE_RESNET18_WEIGHTS=true`; ถ้าปิดจะลดขนาดลง แต่จะรัน active model กลุ่ม `dinov2__...` หรือ `resnet18__...` แบบ offline ไม่ได้

ขนาด image ที่ตรวจล่าสุดหลัง bake DINOv2 + ResNet18 และ copy runtime artifacts: `apiultrasoundswine-api:latest` ประมาณ `2.06GB`. Cache ใน `TORCH_HOME` ตรวจได้ประมาณ `132 MB` และมีไฟล์ `dinov2_vits14_pretrain.pth` กับ `resnet18-f37072fd.pth` อยู่จริง.

### Runtime model artifacts

ตอนนี้ repo เก็บ `model/` และ `AnomalyDetection/artifacts/models/` แล้ว ดังนั้นถ้าใช้ **วิธี B1: สร้าง Image** หลัง clone repo มาก็ build ต่อได้เลย โดยไม่ต้องหา model artifacts เพิ่มแยกต่างหาก

- `model/` ใช้สำหรับ YOLO PDF classifier เช่น `best_finetune_YOLO26-cls_Ver2_20260424.pt`
- `AnomalyDetection/artifacts/models/` ใช้สำหรับ `model_registry.json` และ `.joblib` anomaly artifacts

ส่วน `config/.env` ยังเป็น runtime file จริงที่ต้องเตรียมเองเสมอ และไม่ควรเก็บค่า secret จริงไว้ใน Git

ถ้าใช้ **วิธี B2: Image Transfer** เครื่องปลายทางก็ยังไม่ต้องมีสองโฟลเดอร์นี้บนเครื่องแยกต่างหาก เพราะถูกฝังอยู่ใน image ตั้งแต่เครื่องต้นทางแล้ว.

---

## คำอธิบายการตั้งค่า Docker (Docker Configuration Details)

เพื่อให้เข้าใจการทำงานของ Docker กับโปรเจคนี้มากขึ้น นี่คือคำอธิบายการตั้งค่าสำคัญในไฟล์ `docker-compose.yml`

### การเชื่อมต่อ Port (`ports`)

```yaml
ports:
  - "${MYAPI_PORT:-3014}:${MYAPI_PORT:-3014}"
```
- **ความหมาย:** เป็นการสร้าง "สะพาน" เชื่อม Port ระหว่างเครื่องคอมพิวเตอร์ของคุณ (Host) กับ Port ภายใน Docker Container
- **ฝั่งซ้าย (`${MYAPI_PORT:-3014}`):** คือ Port บนเครื่องของคุณ เมื่อคุณเข้า `localhost:3014` การเชื่อมต่อจะถูกส่งต่อไปยัง Container
- **ฝั่งขวา (`:${MYAPI_PORT:-3014}`):** คือ Port ที่แอปพลิเคชัน FastAPI กำลังทำงานอยู่ *ภายใน* Container

### การเชื่อมต่อโฟลเดอร์ (`volumes`)

```yaml
volumes:
  - ./app/uploads:/app/app/uploads
  - ./app/asset:/app/app/asset
  - ./app/detections:/app/app/detections
```
- **ความหมาย:** เป็นการ "เชื่อมโฟลเดอร์" (Mount) ระหว่างโปรเจคของคุณกับโฟลเดอร์ภายใน Container ทำให้ข้อมูลไม่หายไปเมื่อปิด Container
- **`./app/uploads:/app/app/uploads`**: ไฟล์ PDF ที่อัปโหลดเข้าไปใน Container จะถูกบันทึกไว้ที่โฟลเดอร์ `app/uploads` ในโปรเจคของคุณด้วย
- **`./app/asset:/app/app/asset`**: ไฟล์รูปภาพ PNG ที่ถูกแปลง จะถูกบันทึกไว้ที่ `app/asset` ในโปรเจคของคุณ
- **`./app/detections:/app/app/detections`**: รูป annotated จาก `/detect_follicle/` จะถูกบันทึกกลับออกมาที่ host
- route `/v2/detection_pig_follicle` ก็ใช้ output path ใต้ `app/detections/` ชุดเดียวกัน เมื่อ Gemini คืน annotation ที่ใช้งานได้
- ไม่ mount `model/`, `AnomalyDetection/artifacts/models/`, หรือ `logs/` ใน compose ปัจจุบัน. Model อยู่ใน image; log อยู่ใน container และดูผ่าน `docker compose logs -f api`.

### หมายเหตุ
- สคริปต์ utility เช่น run_api.sh, setup_env.sh, .bat ต่างๆ จะถูกรวมไว้ในโฟลเดอร์ `scripts/` เพื่อความเป็นระเบียบ 

## การจัดการ Log

แอปเขียน runtime log ลง `/app/logs` ภายใน container และใช้ `RotatingFileHandler` เพื่อกันไฟล์ใหญ่เกินไป

- `logs/app.log`: FastAPI upload, convert, health check, และ error หลัก
- `logs/precess.log`: PDF render, crop, YOLO classify, OCR, และ DB skip/insert ของ PDF flow
- `logs/detect.log`: Gemini image detection และ annotated image save

ไฟล์ log จริงถูก ignore จาก git/docker แล้ว. Compose ปัจจุบันไม่ mount `logs/`; ให้ดู log ด้วย `docker compose logs -f api`. ถ้าต้องการเก็บไฟล์ log บน host ค่อยเพิ่ม volume `./logs:/app/logs` เอง.

## Deployment Package

ถ้า deploy ด้วย Image Transfer ให้เตรียมอย่างน้อย:

1. `apiultrasoundswine-api.tar` จาก `docker save`
2. `docker-compose.image.yml`
3. `README-Docker.md`
4. `config/.env.example`
5. `config/.env` ที่เป็นค่าจริงของ server
6. โฟลเดอร์เปล่าสำหรับ output volume: `app/uploads/`, `app/asset/`, `app/detections/`

หลังแตกไฟล์บน server ให้สร้าง `config/.env` จาก template แล้วรัน:

```bash
docker load -i apiultrasoundswine-api.tar
docker compose -f docker-compose.image.yml --env-file config/.env up -d
```

## หมายเหตุสำคัญ (Important Notes)
- container_name ใน docker-compose.yml คือ `ai-ultrasound-swine`
- `docker-compose.yml` ใช้ `env_file: config/.env`; ห้ามทดสอบ Docker โดยไม่โหลดไฟล์นี้ เพราะจะ fallback ไปใช้ค่า default ในโค้ดและอาจโหลดคนละ model/threshold
- .dockerignore ตอนนี้ ignore `app/asset/` แล้ว (asset จะไม่ถูก copy เข้า image)
- ถ้าต้องการ clean build จริง ๆ ให้ลบ container, image, volume เดิมก่อน build ใหม่
- ถ้าไม่มีไฟล์ `config/.env` หรือไม่ได้ส่ง `--env-file config/.env` ตอน `docker run` จะใช้ default ในโค้ด ซึ่งไม่ใช่ source of truth ของโปรเจคนี้
