
 
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse 
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials 
from fastapi.middleware.cors import CORSMiddleware   

from typing import Annotated
import os
import logging
from logging.handlers import RotatingFileHandler
from PIL import Image  # เผื่อใช้ในอนาคตและให้ requirements.txt ตรงกับโค้ด
from app.process_pdf import convert_pdf_to_png  

from app.process_detection import (
    analyze_ultrasound_core,
    save_annotation_result,
    insert_detection_to_db,
    Format_Result,
    DetectionRespone,  # ✅ Import Schema จาก process_detection
    ImageResult)
from app.process_anomaly import (
    format_anomaly_result,
    predict_anomaly_image,
    save_anomaly_input,
)
from app.version import APP_NAME, APP_VERSION

import pymysql 
import uvicorn 
import cv2
import base64 
import numpy as np  
from typing import List 
from pathlib import Path 
from dotenv import load_dotenv 

# Central runtime path used by all modules to find config, uploads, and assets.
pathInitial = Path(__file__).resolve().parent.parent   

# Load config once at import time so Docker, local run, and tests share the same defaults.
if pathInitial.exists() :
    path = pathInitial /"config" / ".env" 
    load_dotenv(dotenv_path = str(path))

max_images = int(os.getenv("Max_Images","5"))

# --- FastAPI app and shared middleware ---
app = FastAPI()

# --- Add CORS Middleware ---
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]
cors_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}

app.add_middleware(               
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uploaded PDFs are stored temporarily here, then deleted after conversion.
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Converted PDF pages are saved in asset/ and exposed for direct image access.
asset_dir = os.path.join(os.path.dirname(__file__), "asset")
if not os.path.exists(asset_dir):
    os.makedirs(asset_dir)

app.mount("/asset", StaticFiles(directory=asset_dir), name="asset")

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
LOG_DIR = pathInitial / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / "app.log"
# Rotate log file when it reaches 10 MB, keep 5 old files.
my_handler = RotatingFileHandler(str(log_file), mode='a', maxBytes=10*1024*1024, backupCount=5, encoding="utf-8", delay=0)
my_handler.setFormatter(log_formatter)
my_handler.setLevel(logging.INFO)

app_log = logging.getLogger('root')
app_log.setLevel(logging.INFO)
app_log.addHandler(my_handler)


def image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', image)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}" 


def app_metadata() -> dict:
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
    }

# --- HTML Form Endpoint ---
@app.get("/upload_form", response_class=HTMLResponse)
async def upload_form():
    """แสดงหน้าเว็บสำหรับอัปโหลดไฟล์ PDF"""
    return """
    <html>
        <head>
            <title>Upload PDF</title>
        </head>
        <body>
            <h2>Upload PDF File</h2>
            <form action='/upload_pdf/' enctype='multipart/form-data' method='post'>
                <input name='file' type='file' accept='application/pdf'>
                <input type='submit' value='Upload'>
            </form>
        </body>
    </html>
    """

#=========================================
# --- Upload API Endpoint 1 (Classify) ---
#=========================================
@app.post(
    "/upload_pdf/",
    summary="Upload PDF File",
    description="อัปโหลดไฟล์ PDF เพื่อแปลงเป็น PNG",
    tags=["PDF"],
)
async def upload_pdf(
    file: Annotated[
        UploadFile,
        File(description="เลือกไฟล์ PDF ที่ต้องการอัปโหลด"),
    ]
):
    # Keep the PDF endpoint strict because the downstream converter expects a real PDF file.
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="อนุญาตเฉพาะไฟล์ .pdf เท่านั้น")
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Content type ต้องเป็น application/pdf")

    file_location = os.path.join(UPLOAD_DIR, file.filename)
    app_log.info(f"[UPLOAD] get file: {file.filename} -> {file_location}")

    # Save the upload first; convert_pdf_to_png works with a filesystem path.
    try:
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)
        app_log.info(f"[SAVE] Save file PDF Successfully: {file_location}")
    except Exception as e:
        app_log.error(f"[ERROR] Save file PDF Failed: {e}")
        return {"status": "error", "detail": f"save failed: {e}"}

    # The conversion function handles page extraction, YOLO classification, OCR, and DB insert.
    try:
        app_log.info(f"[CONVERT] Start the process PDF to PNG wait..: {file_location}")
        result = convert_pdf_to_png(file_location)
        app_log.info(f"[CONVERT] convert_pdf_to_png result: {result}")
    except Exception as e:
        app_log.error(f"[ERROR] Failed the Process PDF to PNG: {e}")
        return {"status": "error", "detail": f"convert failed: {e}"}

    # Remove the source PDF after processing; converted PNGs remain in app/asset.
    try:
        os.remove(file_location)
        app_log.info(f"[CLEANUP] The original PDF file has been deleted.: {file_location}")
    except Exception as e:
        app_log.warning(f"[WARN] Unable to delete file PDF: {e}")

    return {"status": "complete" if result else "error"}


#=========================================
# --- Upload API Endpoint 2 (Detection) ---
#=========================================
@app.post("/detect_follicle/", 
          summary="Detect ถุงน้ำคร่ำ",
          description="เลือกภาพที่สกัดจาก PDF file มาตรวจหาตำแหน่งถุงน้ำคร่ำ",
          tags=["Detection"])
async def detect_follicle_api(
    files: List[UploadFile] = File(description="ภาพ Ultrasound สำหรับ Detect ถุงน้ำคร่ำ"),
):
    # ================================================================
    # 🌟 1. ตั้งค่าขีดจำกัด ( จำกัดสูงสุด 10 รูปต่อครั้ง/หรือตาม config ใน .env สามารถเพิ่ม/ลด ใน config )
    # ================================================================
    if len(files) > max_images :
        raise HTTPException(
            status_code=400,
            detail=f"อัปโหลดได้สูงสุด {max_images} รูปต่อครั้ง (ส่งมา {len(files)} รูป)"
        )
    # ==========================================
    results=[]
    for file in files:
        filename = file.filename
        try :
            contents = await file.read()
            # Decode each uploaded image in memory; invalid image bytes become an item-level error.
            img_cv2 = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            if img_cv2 is None : 
                results.append(ImageResult(
                    path_images=filename,
                    result="error",
                    confidence=0.0,
                    number_of_fetus=0,
                    error_remark="อ่านไฟล์ภาพไม่ได้"
                ))
                continue

            # analyze_ultrasound_core is the single Gemini detection boundary.
            app_log.info(f"[DETECT] Processing: {file}") # เก็บ log

            ai = analyze_ultrasound_core(img_cv2)  

            # Persist the annotated image and return its path in the response.
            save_path = save_annotation_result(ai["annotated_img"] , filename) 
            app_log.info(f"[SAVE] path: {save_path}") 

            """  
            NOTE : หากต้องการ save img annotation ลง DB  
            try:
                insert_detection_to_db(
                    filename=filename,
                    saved_path=saved_path,
                    status=ai["status"],
                    sac_count=ai["sac_count"],
                    detections=ai["detections"]
                )
                app_log.info(f"[DB] Inserted: {filename}")
            except Exception as db_err:
                # ✅ DB Error ไม่หยุดการทำงาน แค่ Log ไว้
                app_log.error(f"[DB ERROR] {filename}: {db_err}")
            """

            results.append(Format_Result(filename ,ai ,save_path))

            # base64_image = image_to_base64(ai["annotated_img"])  
            # if ai["status"] == "1_Pregnant":
            #         color, text, detail = "green", "ตั้งครรภ์", f"พบถุงน้ำ: <b>{ai['sac_count']}</b> จุด"
            # elif ai["status"] == "2_NoPregnant":
            #     color, text, detail = "red", "ไม่ตั้งครรภ์", "* ไม่พบสัญลักษณ์ถุงน้ำ"
            # else:
            #     color, text, detail = "orange", "ไม่แน่ใจ", "* ควรตรวจซ้ำ"  
            # app_log.info(f"[DETECT] Result: {filename} → {ai['status']} (sac={ai['sac_count']})") 

        except Exception as e :
            app_log.error(f"[ERROR] {filename}: {e}") 
            results.append(ImageResult(
                path_images=filename,
                result="error",
                confidence=0.0,
                number_of_fetus=0,
                error_remark=str(e)
            ))

    # Return Response
    return  DetectionRespone(
        main_results="success",
        error_massage="",
        results=results)


@app.post(
    "/detect_anomaly/",
    summary="Detect pregnancy with Anomaly model",
    description="เลือกภาพ Ultrasound ส่งเข้า Anomaly model แล้วคืน JSON รูปแบบเดียวกับ detect_follicle",
    tags=["Detection"],
)
async def detect_anomaly_api(
    files: List[UploadFile] = File(description="ภาพ Ultrasound สำหรับ Anomaly model"),
):
    if len(files) > max_images:
        raise HTTPException(
            status_code=400,
            detail=f"อัปโหลดได้สูงสุด {max_images} รูปต่อครั้ง (ส่งมา {len(files)} รูป)"
        )

    results = []
    for file in files:
        filename = file.filename
        try:
            contents = await file.read()
            img_cv2 = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            if img_cv2 is None:
                results.append(ImageResult(
                    path_images=filename,
                    result="error",
                    confidence=0.0,
                    number_of_fetus=0,
                    error_remark="อ่านไฟล์ภาพไม่ได้"
                ))
                continue

            save_path = save_anomaly_input(img_cv2, filename)
            prediction = predict_anomaly_image(Path(save_path))
            app_log.info(
                "[ANOMALY] %s -> %s score=%s threshold=%s model=%s/%s",
                filename,
                prediction["prediction"],
                prediction["score_no_pregnant"],
                prediction["threshold"],
                prediction.get("feature_set", "unknown"),
                prediction.get("model_name", "unknown"),
            )
            results.append(format_anomaly_result(filename, prediction, save_path))

        except Exception as e:
            app_log.error(f"[ANOMALY ERROR] {filename}: {e}")
            results.append(ImageResult(
                path_images=filename,
                result="error",
                confidence=0.0,
                number_of_fetus=0,
                error_remark=str(e)
            ))

    return DetectionRespone(
        main_results="success",
        error_massage="",
        results=results,
    )


@app.get("/detect_form", response_class=HTMLResponse, tags=["Detection"])
async def detect_form():
    """หน้า Web สำหรับ Upload ภาพ Ultrasound หลายไฟล์พร้อมกัน"""
    return f"""
    <html>
        <head>
            <link href="https://fonts.googleapis.com/css2?family=Kanit&display=swap" rel="stylesheet">
            <title>Detect ถุงน้ำคร่ำ</title>
        </head>
        <body style="font-family:'Kanit',sans-serif; text-align:center; 
                     padding:40px; background:#f8fafc;">

            <h2>🔬 Detect ถุงน้ำคร่ำ Ultrasound</h2>
            <p style="color:#64748b;">เลือกภาพที่สกัดจาก PDF ได้สูงสุด {max_images} ภาพต่อครั้ง</p>

            <form action="/detect_follicle/" 
                  enctype="multipart/form-data" 
                  method="post"
                  target="_blank">

                <!-- ✅ multiple คือ key สำคัญที่ทำให้เลือกหลายไฟล์ได้ -->
                <input name="files" 
                       type="file" 
                       accept="image/*"
                       multiple
                       style="padding:10px; margin:10px; border:1px solid #ddd; 
                              border-radius:8px; font-family:'Kanit',sans-serif;">
                <br>
                <input type="submit" value="🔍 วิเคราะห์ภาพ"
                       style="padding:12px 30px; background:#2563eb; color:white;
                              border:none; border-radius:8px; cursor:pointer; 
                              font-size:16px; font-family:'Kanit',sans-serif; 
                              margin-top:10px;">
            </form>
        </body>
    </html>
    """


#=========================================
# ------------ Health Check --------------
#=========================================
@app.get("/version", tags=["System"])
def version_check():
    return app_metadata()


@app.get("/health", tags=["System"])
def health_check():
    # Health is intentionally DB-aware: API is healthy only when MySQL accepts SELECT 1.
    host = os.environ.get("MYSQL_HOST", "invalidhost")
    port = int(os.environ.get("MYSQL_PORT", 3306))
    user = os.environ.get("MYSQL_USER", "root")
    password = os.environ.get("MYSQL_PASSWORD", "")
    database = os.environ.get("MYSQL_DATABASE", "")
    app_log.info(f"[HEALTHCHECK] Try connect MySQL host={host} port={port} user={user} db={database}")
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            connect_timeout=3
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        finally:
            conn.close()
        return {"status": "ok", "db": "connected", "app": app_metadata()}
    except Exception as e:
        app_log.error(f"[HEALTHCHECK] DB connect error: {e}")
        return {"status": "error", "db": "unreachable", "detail": str(e), "app": app_metadata()}

if __name__ == "__main__":
    myapi_port = int(os.environ.get("MYAPI_PORT", 3014))
    uvicorn.run(app, host="0.0.0.0", port=myapi_port) 
