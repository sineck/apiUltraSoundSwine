
 
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware   

from typing import Annotated
import json
import os
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from datetime import datetime
from PIL import Image  # เผื่อใช้ในอนาคตและให้ requirements.txt ตรงกับโค้ด
from pydantic import BaseModel, Field
from app.anomaly_training import (
    AnomalyTrainingAlreadyRunning,
    current_anomaly_training_job,
    start_anomaly_training,
)
from app.process_pdf import convert_pdf_to_png  
from app.process_pdf import ModelUnavailableError
from app.process_pdf import preprocess_yolo
from app.process_pdf import (
    CVCODE,
    USER_ID,
    default_ocr_info,
    crop_real_image,
    extract_info_from_image,
    insert_ultrasound_to_db,
    render_pdf_pages,
    should_insert_ultrasound_to_db,
)

from app.process_detection import (
    analyze_ultrasound_core,
    save_annotation_result,
    Format_Result,
    DetectionRespone,  # ✅ Import Schema จาก process_detection
    ImageResult)
from app.process_anomaly import (
    format_anomaly_result,
    predict_anomaly_image,
    save_detection_input,
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
from app.image_naming import build_image_filename

# Central runtime path used by all modules to find config, uploads, and assets.
pathInitial = Path(__file__).resolve().parent.parent   
RETRAIN_CONFIG_PATH = pathInitial / "config" / "retrain_anomaly.json"

# Load config once at import time so Docker, local run, and tests share the same defaults.
if pathInitial.exists() :
    path = pathInitial /"config" / ".env" 
    load_dotenv(dotenv_path = str(path))

max_images = int(os.getenv("Max_Images","5"))

# --- FastAPI app and shared middleware ---
app = FastAPI(
    openapi_tags=[
        {"name": "V1", "description": "เส้นใช้งานเดิมที่ยังต้องคงไว้"},
        {"name": "V2", "description": "เส้นใหม่ที่เลือก backend ได้จาก PREGNANCY_DETECT_MODEL_V2"},
        {"name": "Anomaly Train", "description": "งาน train/retrain anomaly model และเช็กสถานะ"},
        {"name": "System", "description": "เส้นระบบสำหรับตรวจ version และ health"},
    ]
)

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


def patch_openapi_file_schema(schema: dict) -> None:
    """แปลง schema file upload ให้ Swagger UI แสดงเป็น file picker ได้ถูกต้อง."""
    if not isinstance(schema, dict):
        return

    if schema.get("type") == "string" and "contentMediaType" in schema:
        schema["format"] = "binary"
        schema.pop("contentMediaType", None)

    for value in schema.values():
        if isinstance(value, dict):
            patch_openapi_file_schema(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    patch_openapi_file_schema(item)


def custom_openapi():
    """ปรับ OpenAPI หลัง generate เพื่อให้ Swagger docs ของ file upload อ่านง่ายและใช้งานได้."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
        description=app.description,
        tags=app.openapi_tags,
    )
    patch_openapi_file_schema(openapi_schema)
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


def image_to_base64(image: np.ndarray) -> str:
    """แปลงภาพ OpenCV เป็น data URL เพื่อใช้แสดงผลหรือ debug ในหน้าเว็บ."""
    _, buffer = cv2.imencode('.jpg', image)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}" 


def app_metadata() -> dict:
    """คืน metadata กลางของแอปไว้ใช้ใน route ระบบ เช่น /version และ /health."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
    }


def runtime_config_summary() -> dict:
    """คืน config runtime ที่ปลอดภัยสำหรับแสดงใน /health โดยไม่เปิดเผย secret."""
    return {
        "config_path": str(pathInitial / "config" / ".env"),
        "myapi_port": int(os.environ.get("MYAPI_PORT", 3014)),
        "insert_ultrasound_to_db": should_insert_ultrasound_to_db(),
        "pregnancy_detect_model_v2": selected_pregnancy_model(),
        "yolo_model_name": os.environ.get("ModelName", "best.pt"),
        "gemini_model": os.environ.get("MODEL_GEMINI", "gemini-3-flash-preview"),
        "max_images": max_images,
    }


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/docs")


@dataclass(frozen=True)
class UltrasoundPrecheckResult:
    """ผล precheck สำหรับกันภาพนอกโดเมนก่อนส่งเข้า V2 backend."""
    is_ultrasound: bool
    reason: str


def reload_runtime_config() -> None:
    """คงไว้เพื่อ compatibility แต่ไม่ทับ process env ระหว่าง request อีกแล้ว."""
    return None


def anomaly_error_result(path_images: str, error_remark: str) -> ImageResult:
    """สร้างผล error มาตรฐานของสาย detect รูป."""
    return ImageResult(
        path_images=path_images,
        result="error",
        confidence=0.0,
        number_of_fetus=0,
        error_remark=error_remark,
    )


def ultrasound_unknown_result(path_images: str, error_remark: str) -> ImageResult:
    """สร้างผล unknown สำหรับภาพที่ไม่ผ่าน precheck ว่าไม่ใช่ ultrasound."""
    return ImageResult(
        path_images=path_images,
        result="unknown",
        confidence=0.0,
        number_of_fetus=0,
        error_remark=error_remark,
    )


def precheck_ultrasound_image(img_cv2: np.ndarray) -> UltrasoundPrecheckResult:
    """กรองภาพนอกโดเมนด้วยกติกาง่าย ๆ: low-color, โทนมืด/เทา, edge ไม่ฟุ้งแบบภาพธรรมชาติ."""
    if img_cv2 is None or img_cv2.size == 0:
        return UltrasoundPrecheckResult(False, "Input image is empty")

    if len(img_cv2.shape) == 2:
        gray = img_cv2
        color_complexity = 0.0
    else:
        blue = img_cv2[:, :, 0].astype(np.float32)
        green = img_cv2[:, :, 1].astype(np.float32)
        red = img_cv2[:, :, 2].astype(np.float32)
        color_complexity = float(np.mean(np.abs(blue - green) + np.abs(green - red) + np.abs(blue - red)) / 3.0)
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    dark_gray_ratio = float(np.count_nonzero(gray <= 185) / gray.size)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 140)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    failed_checks = []
    if color_complexity > 5.0:
        failed_checks.append(f"color_complexity={color_complexity:.2f}>5.00")
    if dark_gray_ratio < 0.86:
        failed_checks.append(f"dark_gray_ratio={dark_gray_ratio:.2f}<0.86")
    if edge_density > 0.05:
        failed_checks.append(f"edge_density={edge_density:.2f}>0.05")

    if failed_checks:
        return UltrasoundPrecheckResult(
            False,
            "Input is not recognized as an ultrasound image: " + ", ".join(failed_checks),
        )
    return UltrasoundPrecheckResult(True, "")


def legacy_pdf_ai_label_from_result(result_text: str) -> str:
    """map ข้อความผลลัพธ์ให้กลับเป็น label legacy ที่ตาราง UltraSoudPigAI ใช้อยู่."""
    legacy_label = "2_NoPrenant_or_NotSure"
    if result_text == "pregnant":
        legacy_label = "1_Pregnant"
    return legacy_label


def detect_saved_anomaly_image(display_name: str, save_path: str) -> ImageResult:
    """รัน anomaly backend กับภาพที่บันทึกแล้ว และ normalize ให้อยู่ใน ImageResult เดียวกัน."""
    prediction = predict_anomaly_image(Path(save_path))
    app_log.info(
        "[ANOMALY] %s -> %s score=%s threshold=%s model=%s/%s",
        display_name,
        prediction["prediction"],
        prediction["score_no_pregnant"],
        prediction["threshold"],
        prediction.get("feature_set", "unknown"),
        prediction.get("model_name", "unknown"),
    )
    return format_anomaly_result(display_name, prediction, save_path)


@dataclass(frozen=True)
class PregnancyDetectionOutcome:
    """ผลกลางของสาย V2 ที่เก็บทั้งผลตอบกลับและ label legacy สำหรับ DB."""
    result: ImageResult
    legacy_ai_label: str


def detect_saved_yolo_image(display_name: str, save_path: str) -> PregnancyDetectionOutcome:
    """รัน YOLO backend แล้ว normalize ผลให้อยู่ใน outcome กลางเดียวกับ anomaly."""
    result_name, confidence = preprocess_yolo(save_path)
    app_log.info("[PREGNANCY YOLO] %s -> %s confidence=%s", display_name, result_name, confidence)
    result = format_yolo_pregnancy_result(display_name, save_path, result_name, confidence)
    legacy_ai_label = result_name or legacy_pdf_ai_label_from_result(result.result)
    return PregnancyDetectionOutcome(result=result, legacy_ai_label=legacy_ai_label)


def detect_saved_ensemble_image(display_name: str, save_path: str) -> PregnancyDetectionOutcome:
    """ใช้ anomaly + YOLO ร่วมกัน โดยถือ pregnant เฉพาะเมื่อทั้งสองฝั่งตรงกันว่า pregnant."""
    anomaly_result = detect_saved_anomaly_image(display_name, save_path)
    yolo_outcome = detect_saved_yolo_image(display_name, save_path)

    if anomaly_result.result == "error":
        return PregnancyDetectionOutcome(
            result=anomaly_result,
            legacy_ai_label=legacy_pdf_ai_label_from_result(anomaly_result.result),
        )
    if yolo_outcome.result.result == "error":
        return yolo_outcome

    final_result = "no pregnant"
    legacy_ai_label = "2_NoPrenant_or_NotSure"
    confidence = max(float(anomaly_result.confidence or 0.0), float(yolo_outcome.result.confidence or 0.0))

    if anomaly_result.result == "pregnant" and yolo_outcome.result.result == "pregnant":
        final_result = "pregnant"
        legacy_ai_label = "1_Pregnant"
        confidence = min(float(anomaly_result.confidence or 0.0), float(yolo_outcome.result.confidence or 0.0))

    return PregnancyDetectionOutcome(
        result=ImageResult(
            path_images=save_path,
            result=final_result,
            confidence=round(confidence, 2),
            number_of_fetus=0,
            error_remark="",
        ),
        legacy_ai_label=legacy_ai_label,
    )


def detect_saved_pregnancy_image_with_backend(display_name: str, save_path: str) -> PregnancyDetectionOutcome:
    """เลือก backend จาก config แล้วรวมผลให้อยู่รูปกลางเดียวก่อนส่งต่อไป response/DB."""
    backend = selected_pregnancy_model()

    if backend in {"anomaly", "new"}:
        result = detect_saved_anomaly_image(display_name, save_path)
        return PregnancyDetectionOutcome(
            result=result,
            legacy_ai_label=legacy_pdf_ai_label_from_result(result.result),
        )
    if backend in {"yolo", "old", "legacy"}:
        return detect_saved_yolo_image(display_name, save_path)
    if backend == "ensemble":
        return detect_saved_ensemble_image(display_name, save_path)
    raise ValueError("PREGNANCY_DETECT_MODEL_V2 must be anomaly, yolo, or ensemble")


def selected_pregnancy_model() -> str:
    """อ่าน backend ที่ active จริงจาก config runtime."""
    return os.getenv("PREGNANCY_DETECT_MODEL_V2", "anomaly").strip().lower()


def format_yolo_pregnancy_result(display_name: str, save_path: str, result_name: str | None, confidence: float | None) -> ImageResult:
    """แปลง label ของ YOLO ให้เป็นข้อความมาตรฐานแบบเดียวกับ route ตรวจรูป."""
    result_value = result_name or "Unknown"
    result_text = "not sure"
    if result_value == "1_Pregnant":
        result_text = "pregnant"
    elif "NoPregnant" in result_value or "NoPrenant" in result_value:
        result_text = "no pregnant"

    return ImageResult(
        path_images=save_path,
        result=result_text,
        confidence=round(float(confidence or 0.0), 2),
        number_of_fetus=0,
        error_remark="",
    )


def detect_saved_pregnancy_image(display_name: str, save_path: str) -> ImageResult:
    """compat helper สำหรับจุดที่ต้องการแค่ ImageResult โดยไม่ต้องยุ่งกับ label ฝั่ง DB."""
    return detect_saved_pregnancy_image_with_backend(display_name, save_path).result


def annotate_pregnant_detection_with_gemini(
    filename: str,
    img_cv2: np.ndarray,
    outcome: PregnancyDetectionOutcome,
) -> ImageResult:
    """ใช้ Gemini วงรูปเฉพาะภาพที่ gate ว่า pregnant และบันทึกไฟล์ไว้ path เดียวกับ /detect_follicle/."""
    if outcome.result.result != "pregnant":
        return outcome.result

    annotated_result = ImageResult(**outcome.result.model_dump())
    try:
        ai = analyze_ultrasound_core(img_cv2)
        gemini_status = ai.get("status", "")
        gemini_sac_count = int(ai.get("sac_count", 0) or 0)
        if gemini_status == "1_Pregnant" and gemini_sac_count > 0:
            save_path = save_annotation_result(ai["annotated_img"], filename)
            annotated_result.path_images = save_path
            annotated_result.number_of_fetus = gemini_sac_count
        else:
            annotated_result.error_remark = "Gemini did not return usable follicle annotation"
    except Exception as gemini_error:
        app_log.error("[V2 FOLLICLE ERROR] %s: %s", filename, gemini_error)
        annotated_result.error_remark = f"Gemini annotate failed: {gemini_error}"
    return annotated_result


def build_ultrasound_db_payload(
    *,
    source_name: str,
    save_path: str,
    result: ImageResult,
    legacy_ai_label: str,
    path_for_db: str,
    fallback_id_value: str,
) -> dict:
    """ประกอบ payload กลางสำหรับ insert UltraSoudPigAI ให้ image/PDF ใช้กติกา OCR ชุดเดียวกัน."""
    ocr_result = extract_info_from_image(save_path)
    if ocr_result is None:
        ocr_result = default_ocr_info()
    dt_val, _dt_in_img_match_val, pregnant_val, id_val, depth_val, gain_val = ocr_result

    cleaned_id = (id_val or "").strip() or fallback_id_value
    return {
        "create_date": datetime.now(),
        "workdate": datetime.now().strftime("%Y-%m-%d"),
        "time": dt_val,
        "pregnant_p": pregnant_val,
        "id_val": cleaned_id,
        "pdfFileName": source_name,
        "depth_val": depth_val,
        "gain_val": gain_val,
        "path_val": path_for_db,
        "file_name": Path(save_path).name,
        "results_ai": legacy_ai_label,
        "conf_score": result.confidence,
        "cvcode": CVCODE,
        "user_id": USER_ID,
    }


def persist_image_detection_to_db(
    source_name: str,
    save_path: str,
    outcome: PregnancyDetectionOutcome,
) -> ImageResult:
    """เขียนผลของ V2 สายรูปลง DB โดยใช้ label legacy เดียวกับสาย PDF."""
    reload_runtime_config()
    if not should_insert_ultrasound_to_db() or outcome.result.result == "error":
        return outcome.result

    payload = build_ultrasound_db_payload(
        source_name=source_name,
        save_path=save_path,
        result=outcome.result,
        legacy_ai_label=outcome.legacy_ai_label,
        path_for_db=str(Path(save_path).parent),
        fallback_id_value=Path(save_path).stem,
    )

    try:
        insert_ultrasound_to_db(**payload)
        app_log.info("[PREGNANCY DB] inserted %s -> %s", source_name, save_path)
    except Exception as db_error:
        app_log.error("[PREGNANCY DB ERROR] %s: %s", source_name, db_error)
        outcome.result.error_remark = f"DB insert failed: {db_error}"
    return outcome.result


class AnomalyTrainRequest(BaseModel):
    feature_sets: str | None = Field(
        default=None,
        description="เช่น handcrafted,resnet18,dinov2 ถ้าไม่ส่งจะใช้ default ของ trainer",
    )
    model_keys: str | None = Field(
        default=None,
        description="ระบุ model_key คั่นด้วย comma หรือ all ถ้าไม่ส่งจะใช้ compact 4-model default",
    )
    batch_size: int = Field(default=16, ge=1, le=128)
    generate_report: bool = True
    detail_heatmaps: str = Field(default="active", pattern="^(none|active|all)$")
    rebuild_index: bool = True
    force: bool = False


def load_retrain_anomaly_config() -> AnomalyTrainRequest:
    """โหลดค่า default ของ retrain จากไฟล์ JSON ฝั่ง runtime."""
    if not RETRAIN_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing retrain config: {RETRAIN_CONFIG_PATH}")
    with RETRAIN_CONFIG_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return AnomalyTrainRequest.model_validate(payload)


def save_pdf_page_asset(crop_img: Image.Image, page_number: int) -> str:
    # บันทึกรูปที่ตัดจาก PDF ลง asset เพื่อให้ V2 PDF flow เก็บผลลัพธ์แบบเดียวกับสายเดิม
    out_name = build_image_filename("pdf", page_number=page_number)
    out_path = Path(asset_dir) / out_name
    crop_rgb = np.array(crop_img.convert("RGB"))
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), crop_bgr)
    return str(out_path)


def render_pdf_pages_to_asset(pdf_path: Path) -> list[tuple[str, str]]:
    # ใช้ render/crop แบบเดียวกับงาน PDF เดิม แต่แยก helper ไว้ให้ V2 เรียกซ้ำได้
    images = []
    for idx, page in enumerate(render_pdf_pages(str(pdf_path)), start=1):
        crop = crop_real_image(page)
        save_path = save_pdf_page_asset(crop, idx)
        images.append((f"{pdf_path.name}#page={idx}", save_path))
    return images


def normalize_db_id(id_val: str, save_path: str) -> str:
    # ถ้า OCR หา ID ไม่เจอ ให้ใช้ค่าเดียวกันทุก backend เพื่อให้ filter/report ทำงานต่อได้
    cleaned = (id_val or "").strip()
    if cleaned:
        return cleaned
    return "IDUnknown"


def persist_v2_pdf_result_to_db(
    source_name: str,
    save_path: str,
    result: ImageResult,
    legacy_ai_label: str,
) -> None:
    # V2 PDF ต้องยังเก็บผลลง DB ในรูปแบบ legacy label เพื่อให้รายงาน/ระบบเดิมอ่านต่อได้
    reload_runtime_config()
    if not should_insert_ultrasound_to_db() or result.result == "error":
        return

    payload = build_ultrasound_db_payload(
        source_name=source_name,
        save_path=save_path,
        result=result,
        legacy_ai_label=legacy_ai_label,
        path_for_db=asset_dir,
        fallback_id_value="IDUnknown",
    )
    payload["id_val"] = normalize_db_id(payload["id_val"], save_path)
    insert_ultrasound_to_db(**payload)


def process_v2_pdf_upload(pdf_path: Path, source_name: str) -> tuple[bool, str | None]:
    # แกนของ /v2/upload_pdf/: render PDF -> เลือก model -> insert DB ตาม env -> คืนสถานะ legacy
    try:
        for _page_name, save_path in render_pdf_pages_to_asset(pdf_path):
            outcome = detect_saved_pregnancy_image_with_backend(source_name, save_path)
            persist_v2_pdf_result_to_db(source_name, save_path, outcome.result, outcome.legacy_ai_label)
        return True, None
    except ModelUnavailableError:
        raise
    except Exception as exc:
        app_log.error("[V2 PDF ERROR] %s: %s", source_name, exc)
        return False, str(exc)

# --- HTML Form Endpoint ---
@app.get("/upload_form", response_class=HTMLResponse, tags=["V1"])
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
    tags=["V1"],
)
async def upload_pdf(
    file: Annotated[
        UploadFile,
        File(description="เลือกไฟล์ PDF ที่ต้องการอัปโหลด"),
    ]
):
    # V1 เดิม: รับ PDF แล้ววิ่งเข้า pipeline convert_pdf_to_png โดยให้ env เป็นตัวคุม DB insert
    # Keep the PDF endpoint strict because the downstream converter expects a real PDF file.
    source_name = Path(file.filename or "").name
    if Path(source_name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="อนุญาตเฉพาะไฟล์ .pdf เท่านั้น")
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Content type ต้องเป็น application/pdf")

    stored_filename = build_image_filename("upload_pdf", extension=".pdf")
    file_location = os.path.join(UPLOAD_DIR, stored_filename)
    app_log.info(f"[UPLOAD] get file: {source_name} -> {file_location}")

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
    except ModelUnavailableError as e:
        app_log.error(f"[ERROR] PDF model unavailable: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e
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
          tags=["V1"])
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
    "/v2/upload_pdf/",
    summary="Upload PDF and detect pregnancy",
    description="V2 ของ /upload_pdf/: รับ PDF เหมือนเดิม เลือก backend จาก PREGNANCY_DETECT_MODEL_V2 แล้วตอบแบบ legacy route เดิม",
    tags=["V2"],
)
async def upload_pdf_v2_api(
    file: UploadFile = File(description="ไฟล์ PDF Ultrasound สำหรับ pregnancy model"),
):
    # V2: รับ PDF เหมือน V1 แต่เลือก backend จาก env และตอบกลับแบบ legacy route เดิม
    source_name = Path(file.filename or "").name
    if Path(source_name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="อนุญาตเฉพาะไฟล์ .pdf เท่านั้น")
    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Content type ต้องเป็น application/pdf")

    stored_filename = build_image_filename("upload_pdf_v2", extension=".pdf")
    file_location = os.path.join(UPLOAD_DIR, stored_filename)
    try:
        with open(file_location, "wb") as f:
            f.write(await file.read())
        result, detail = process_v2_pdf_upload(Path(file_location), source_name)
    except ModelUnavailableError as e:
        app_log.error(f"[ERROR] V2 PDF model unavailable: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        app_log.error(f"[ERROR] Save or process V2 PDF failed: {e}")
        return {"status": "error", "detail": str(e)}
    finally:
        try:
            if os.path.exists(file_location):
                os.remove(file_location)
        except Exception as cleanup_error:
            app_log.warning(f"[WARN] Unable to delete V2 PDF file: {cleanup_error}")

    if result:
        return {"status": "complete"}
    return {"status": "error", "detail": detail or "process failed"}


@app.post(
    "/v2/detection_pig",
    summary="Detect pregnancy from images with configurable model",
    description="รับรูป ultrasound หลายรูป แล้วเลือก backend จาก PREGNANCY_DETECT_MODEL_V2 โดยคืน response shape แบบเดียวกับ /detect_follicle/",
    tags=["V2"],
)
async def detect_pig_v2_api(
    files: List[UploadFile] = File(description="ภาพ Ultrasound สำหรับ pregnancy model"),
):
    # V2 รูป: รับหลายภาพแล้ว map ผลทุก model ให้เป็น response shape เดียวกับ detect_follicle
    if len(files) > max_images:
        raise HTTPException(
            status_code=400,
            detail=f"อัปโหลดได้สูงสุด {max_images} รูปต่อครั้ง (ส่งมา {len(files)} รูป)"
        )

    results = []
    for file in files:
        filename = file.filename or "upload_image"
        try:
            contents = await file.read()
            img_cv2 = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            if img_cv2 is None:
                results.append(anomaly_error_result(filename, "อ่านไฟล์ภาพไม่ได้"))
                continue
            precheck = precheck_ultrasound_image(img_cv2)
            if not precheck.is_ultrasound:
                unknown_path = save_detection_input(img_cv2, filename, kind="unknown")
                results.append(ultrasound_unknown_result(unknown_path, precheck.reason))
                continue

            # บันทึกรูปที่อัปโหลดไว้ก่อน เพื่อให้ทั้ง yolo/anomaly ใช้ path เดียวกัน
            save_path = save_detection_input(img_cv2, filename)
            outcome = detect_saved_pregnancy_image_with_backend(filename, save_path)
            results.append(persist_image_detection_to_db(filename, save_path, outcome))
        except Exception as e:
            app_log.error(f"[V2 DETECTION PIG ERROR] {filename}: {e}")
            results.append(anomaly_error_result(filename, str(e)))

    return DetectionRespone(
        main_results="success",
        error_massage="",
        results=results,
    )


@app.post(
    "/v2/detection_pig_follicle",
    summary="Detect pregnancy from images and annotate follicle when pregnant",
    description="ใช้ model gate แบบเดียวกับ /v2/detection_pig ก่อน ถ้าผลเป็น pregnant ค่อยเรียก Gemini ให้วงรูป และบันทึก path เดียวกับ /detect_follicle/",
    tags=["V2"],
)
async def detect_pig_follicle_v2_api(
    files: List[UploadFile] = File(description="ภาพ Ultrasound สำหรับ pregnancy model และ Gemini annotation"),
):
    # V2 วงรูป: ใช้ model gate ก่อนเพื่อคัดภาพที่ควรถาม Gemini ต่อและเก็บ path ลง detections แบบ route เดิม
    if len(files) > max_images:
        raise HTTPException(
            status_code=400,
            detail=f"อัปโหลดได้สูงสุด {max_images} รูปต่อครั้ง (ส่งมา {len(files)} รูป)"
        )

    results = []
    for file in files:
        filename = file.filename or "upload_image"
        try:
            contents = await file.read()
            img_cv2 = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            if img_cv2 is None:
                results.append(anomaly_error_result(filename, "อ่านไฟล์ภาพไม่ได้"))
                continue
            precheck = precheck_ultrasound_image(img_cv2)
            if not precheck.is_ultrasound:
                unknown_path = save_detection_input(img_cv2, filename, kind="unknown")
                results.append(ultrasound_unknown_result(unknown_path, precheck.reason))
                continue

            save_path = save_detection_input(img_cv2, filename)
            outcome = detect_saved_pregnancy_image_with_backend(filename, save_path)
            annotated_result = annotate_pregnant_detection_with_gemini(filename, img_cv2, outcome)
            results.append(annotated_result)
        except Exception as e:
            app_log.error(f"[V2 DETECTION PIG FOLLICLE ERROR] {filename}: {e}")
            results.append(anomaly_error_result(filename, str(e)))

    return DetectionRespone(
        main_results="success",
        error_massage="",
        results=results,
    )

@app.post(
    "/anomaly/retrain/",
    summary="Train new anomaly pregnancy model",
    description="เริ่ม train anomaly model ใหม่แบบ background โดยอ่านค่าจาก config/retrain_anomaly.json",
    tags=["Anomaly Train"],
    status_code=202,
)
async def retrain_anomaly_api():
    payload = load_retrain_anomaly_config()
    try:
        job = start_anomaly_training(
            feature_sets=payload.feature_sets,
            model_keys=payload.model_keys,
            batch_size=payload.batch_size,
            generate_report=payload.generate_report,
            detail_heatmaps=payload.detail_heatmaps,
            rebuild_index=payload.rebuild_index,
            force=payload.force,
        )
        app_log.info("[ANOMALY TRAIN] started job=%s", job["job_id"])
        return job
    except AnomalyTrainingAlreadyRunning as exc:
        raise HTTPException(
            status_code=409,
            detail=f"มีงาน train anomaly กำลังทำอยู่แล้ว: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/anomaly/retrain/status/",
    summary="Check anomaly training job status",
    tags=["Anomaly Train"],
)
async def retrain_anomaly_status_api():
    return current_anomaly_training_job()


@app.get("/detect_form", response_class=HTMLResponse, tags=["V1"])
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
    config_summary = runtime_config_summary()
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
        return {"status": "ok", "db": "connected", "app": app_metadata(), "config": config_summary}
    except Exception as e:
        app_log.error(f"[HEALTHCHECK] DB connect error: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "db": "unreachable", "detail": str(e), "app": app_metadata(), "config": config_summary},
        )

if __name__ == "__main__":
    myapi_port = int(os.environ.get("MYAPI_PORT", 3014))
    uvicorn.run(app, host="0.0.0.0", port=myapi_port) 
