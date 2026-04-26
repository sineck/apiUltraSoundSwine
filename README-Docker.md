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
  ```
- .env ต้องถูก ignore จาก git

## เริ่มต้นใช้งาน (Quick Start)

ทำตามขั้นตอนต่อไปนี้เพื่อรันแอปพลิเคชัน

### ขั้นตอนที่ 1: เตรียมไฟล์ Environment (`.env`)

ก่อนรันโปรเจค คุณต้องสร้างไฟล์สำหรับเก็บค่า Environment Variables ก่อน

1.  **เตรียมไฟล์ `config/.env` จาก `config/.env.example`**
2.  แก้ไขค่าในไฟล์ `config/.env` ให้ตรงกับระบบของคุณ

    ```env
    MYAPI_PORT=3014
    MYSQL_HOST=your-rds-endpoint.amazonaws.com
    MYSQL_PORT=3306
    MYSQL_DATABASE=your_database_name
    MYSQL_USER=your_username
    MYSQL_PASSWORD=your_password
    INSERT_ULTRASOUND_TO_DB=true
    ```

### ขั้นตอนที่ 2: รันแอปพลิเคชัน (เลือกวิธีใดวิธีหนึ่ง)

#### **วิธี A: Build จากซอร์สโค้ด (สำหรับนักพัฒนา)**
วิธีนี้จะสร้าง Docker Image ขึ้นมาใหม่จาก `Dockerfile` เหมาะสำหรับการเริ่มต้นโปรเจคครั้งแรกหรือเมื่อมีการแก้ไขโค้ด

```bash
# คำสั่งนี้จะ build image และรัน container ใน background
docker compose --env-file config/.env up --build -d
```

#### **วิธี B: Load จากไฟล์ Image (สำหรับย้ายไปรันเครื่องอื่น)**
ในกรณีที่คุณมีไฟล์ Image อยู่แล้ว (เช่น `ultrasoundpig-api.tar`) สามารถโหลดและรันได้เลยโดยไม่ต้อง build ใหม่

```bash
# 1. โหลด Image จากไฟล์ .tar
docker load -i ultrasoundpig-api.tar

# 2. รัน container จาก image ที่โหลดมา (ไม่ต้องใช้ --build)
docker compose --env-file config/.env up -d
```

### ขั้นตอนที่ 3: ตรวจสอบการทำงาน

- **เปิดหน้า API**: `http://localhost:${MYAPI_PORT}/docs` (default: 3014)
- **เปิดหน้าอัปโหลด**: `http://localhost:${MYAPI_PORT}/upload_form`
- **เปิดหน้าตรวจรูปด้วย Gemini**: `http://localhost:${MYAPI_PORT}/detect_form`
- **API ตรวจรูปด้วย Anomaly model**: `POST http://localhost:${MYAPI_PORT}/detect_anomaly/`
- **ดูชื่อและเวอร์ชันแอป**: `http://localhost:${MYAPI_PORT}/version`
- **ตรวจสุขภาพ API + MySQL**: `http://localhost:${MYAPI_PORT}/health`
- **ดู Log การทำงาน**:
  ```bash
  docker compose logs -f api
  ```
- **ดูไฟล์ Log ที่ mount ออกมาจาก container**:
  ```powershell
  Get-ChildItem logs
  Get-Content -Encoding UTF8 logs\app.log -Tail 50
  ```

---

## คำสั่ง Docker Compose อื่นๆ ที่ใช้บ่อย
- **หยุดการทำงาน:** `docker compose down`
- **รีสตาร์ท:** `docker compose restart`

## Health และ Version

- `GET /version`: เช็กชื่อและเวอร์ชันแอปอย่างเดียว ไม่แตะ MySQL เหมาะสำหรับดูว่า container ที่รันอยู่เป็น release ไหน
- `GET /health`: เช็ก MySQL ด้วย และแนบข้อมูล `app.name` / `app.version` กลับมาใน response
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

---

## Anomaly Detection API

`POST /detect_anomaly/` รับ field `files` แบบเดียวกับ `/detect_follicle/` และคืน JSON shape เดียวกัน:

```json
{
  "main_results": "success",
  "error_massage": "",
  "results": [
    {
      "path_images": "app/detections/20260426_scan_anomaly_xxxxxx.png",
      "result": "pregnant",
      "confidence": 1.0,
      "number_of_fetus": 0,
      "error_remark": ""
    }
  ]
}
```

ค่า model active อ่านจาก `AnomalyDetection/artifacts/models/model_registry.json` ภายใต้ project root ปัจจุบันเสมอ. อย่าใช้ absolute path จากเครื่อง train เป็น runtime source of truth. โค้ดรองรับ registry ที่ยังมี Windows path เก่าอยู่ใน `model_file` โดย resolve กลับมาที่ project root ปัจจุบันแทน.

หมายเหตุ: โฟลเดอร์จริงของระบบนี้คือ `AnomalyDetection` ใต้ project root ไม่ใช่ `Anomaly` หรือชื่ออื่น. active model ปัจจุบันเป็น `handcrafted__logistic_regression_balanced` จึงไม่เรียก DINOv2 ใน runtime ปกติ.

### DINOv2 weights

Dockerfile ตั้ง `BAKE_DINOV2_WEIGHTS=true` เป็นค่าเริ่มต้น เพื่อ pre-download DINOv2 ผ่าน `torch.hub` เข้า `TORCH_HOME=/app/.cache/torch` ระหว่าง build. เหตุผลคือถ้าวันหลังเปลี่ยน `model_registry.json` ให้ active model เป็นกลุ่ม `dinov2__...` แล้ว production ถูก block network, container จะยังมี cache สำหรับ DINOv2 อยู่แล้ว.

ถ้าต้องการ image เล็กกว่าและแน่ใจว่าไม่ใช้ `dinov2__...` ให้ build ด้วย:

```powershell
docker build --build-arg BAKE_DINOV2_WEIGHTS=false -t apiultrasoundswine-api:latest .
```

ถ้าเปลี่ยน active model เป็น `dinov2__...` ต้องมีอย่างใดอย่างหนึ่ง:

- ใช้ image ที่ build ด้วย `BAKE_DINOV2_WEIGHTS=true`
- server มี network ให้ `torch.hub` ดาวน์โหลด DINOv2 code/weights ตอนรันครั้งแรก
- เตรียม torch hub cache ไว้ล่วงหน้าแล้ว mount เข้า container

สำหรับ production ที่ network ถูก block ไม่ควรพึ่ง first-run download ของ DINOv2.

---

## Docker Image Size

Dockerfile ใช้ `requirements-docker.txt` แยกจาก `requirements.txt` เพื่อลด dependency ใน image:

- ติดตั้ง PyTorch แบบ CPU-only จาก `https://download.pytorch.org/whl/cpu`
- ติดตั้ง `ultralytics` แบบ `--no-deps` เพื่อไม่ดึง `opencv-python` ตัวเต็มและ `polars`
- ใช้ `opencv-python-headless` ชุดเดียวสำหรับ API runtime
- ใช้ Python stdlib สำหรับ Docker healthcheck แทน `curl`
- ไม่ copy `config/.env` เข้า image
- ค่า default จะ bake DINOv2 weights/cache เข้า image ด้วย `BAKE_DINOV2_WEIGHTS=true`; ถ้าปิดจะลดขนาดลงประมาณ 100-150 MB แต่จะรัน active model กลุ่ม `dinov2__...` แบบ offline ไม่ได้

ขนาด image ที่ตรวจล่าสุดก่อน bake DINOv2: `apiultrasoundswine-api:latest` ประมาณ `1.78GB`. เมื่อ bake DINOv2 แล้วคาดว่าจะเพิ่มประมาณ `100-150 MB` ขึ้นกับ torch hub cache ที่ดาวน์โหลดได้จริง.

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
  - ./logs:/app/logs
```
- **ความหมาย:** เป็นการ "เชื่อมโฟลเดอร์" (Mount) ระหว่างโปรเจคของคุณกับโฟลเดอร์ภายใน Container ทำให้ข้อมูลไม่หายไปเมื่อปิด Container
- **`./app/uploads:/app/app/uploads`**: ไฟล์ PDF ที่อัปโหลดเข้าไปใน Container จะถูกบันทึกไว้ที่โฟลเดอร์ `app/uploads` ในโปรเจคของคุณด้วย
- **`./app/asset:/app/app/asset`**: ไฟล์รูปภาพ PNG ที่ถูกแปลง จะถูกบันทึกไว้ที่ `app/asset` ในโปรเจคของคุณ
- **`./app/detections:/app/app/detections`**: รูป annotated จาก `/detect_follicle/` จะถูกบันทึกกลับออกมาที่ host
- **`./logs:/app/logs`**: ไฟล์ log ของแอปจะถูกเขียนออกมาที่ `logs/` ในโปรเจค

### หมายเหตุ
- สคริปต์ utility เช่น run_api.sh, setup_env.sh, .bat ต่างๆ จะถูกรวมไว้ในโฟลเดอร์ `scripts/` เพื่อความเป็นระเบียบ 

## การจัดการ Log

แอปเขียน runtime log ลงโฟลเดอร์ `logs/` ที่ root ของโปรเจค และใช้ `RotatingFileHandler` เพื่อกันไฟล์ใหญ่เกินไป

- `logs/app.log`: FastAPI upload, convert, health check, และ error หลัก
- `logs/precess.log`: PDF render, crop, YOLO classify, OCR, และ DB skip/insert ของ PDF flow
- `logs/detect.log`: Gemini image detection และ annotated image save

ไฟล์ log จริงถูก ignore จาก git/docker แล้ว เหลือ `logs/.gitkeep` เพื่อให้มีโฟลเดอร์ใน repo. ใน Docker Compose ให้ mount `./logs:/app/logs` เพื่อเก็บ log ไว้บน host.

## Deployment Package

สำหรับส่งไปรันบน server ให้เตรียมอย่างน้อย:

1. `ultrasoundpig-api.tar` จาก `docker save`
2. `docker-compose.yml`
3. `README-Docker.md`
4. `config/.env.example`
5. โฟลเดอร์เปล่าสำหรับ volume: `app/uploads/`, `app/asset/`, `app/detections/`, และ `logs/`

หลังแตกไฟล์บน server ให้สร้าง `config/.env` จาก template แล้วรัน:

```bash
docker load -i ultrasoundpig-api.tar
docker compose --env-file config/.env up -d
```

## หมายเหตุสำคัญ (Important Notes)
- container_name ใน docker-compose.yml คือ `ai-ultrasound-swine`
- `docker-compose.yml` ใช้ `env_file: config/.env`; ห้ามทดสอบ Docker โดยไม่โหลดไฟล์นี้ เพราะจะ fallback ไปใช้ค่า default ในโค้ดและอาจโหลดคนละ model/threshold
- .dockerignore ตอนนี้ ignore `app/asset/` แล้ว (asset จะไม่ถูก copy เข้า image)
- ถ้าต้องการ clean build จริง ๆ ให้ลบ container, image, volume เดิมก่อน build ใหม่
- ถ้าไม่มีไฟล์ `config/.env` หรือไม่ได้ส่ง `--env-file config/.env` ตอน `docker run` จะใช้ default ในโค้ด ซึ่งไม่ใช่ source of truth ของโปรเจคนี้
