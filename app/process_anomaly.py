"""Runtime adapter สำหรับ anomaly backend ของ V2.

ไฟล์นี้ไม่ใช่ที่รวม logic train/model core ทั้งหมดเอง แต่ทำหน้าที่เป็น
"สะพาน" ระหว่าง API runtime ใน `app/` กับ utility ฝั่ง anomaly ที่อยู่ใน
`AnomalyDetection/scripts/anomaly_lib.py`

โครงงานตั้งใจแยกชั้นแบบนี้:
1. `anomaly_lib.py`
   - จัดการ feature extraction, model bundle, scoring, thresholding
   - ใช้ได้ทั้ง script ฝั่ง train/validate และ runtime
2. `process_anomaly.py`
   - จัดการเรื่องที่ API runtime ต้องรู้ เช่น
     - model active ตัวปัจจุบันอยู่ไฟล์ไหน
     - path ใน repo นี้อยู่ตรงไหน
     - save ภาพ input/output ลง `app/detections`
     - format ผลให้เป็น `ImageResult` ที่ route ใช้ร่วมกัน

ดังนั้นเวลามาไล่โค้ด:
- ถ้าสงสัย "โมเดล anomaly คิดคะแนนยังไง" ให้ดู `anomaly_lib.py`
- ถ้าสงสัย "route V2 เรียก anomaly backend ยังไง" ให้ดูไฟล์นี้
"""

from functools import lru_cache
from pathlib import Path, PureWindowsPath
import sys

import cv2
import numpy as np

from app.image_naming import build_image_filename
from app.process_detection import ImageResult
from app.process_pdf import crop_real_image, render_pdf_pages


pathInitial = Path(__file__).resolve().parent.parent
ANOMALY_ROOT = pathInitial / "AnomalyDetection"
ANOMALY_SCRIPT_DIR = ANOMALY_ROOT / "scripts"
ANOMALY_REGISTRY = ANOMALY_ROOT / "artifacts" / "models" / "model_registry.json"

# สำคัญ:
# - `process_anomaly.py` อยู่ใต้ `app/`
# - แต่ anomaly core library จริงอยู่ที่
#   `D:\apiUltraSoundSwine\AnomalyDetection\scripts\anomaly_lib.py`
#
# ดังนั้นถ้าเปิดไฟล์นี้แล้วเห็น `from anomaly_lib import ...` จะหาไฟล์ชื่อ
# `app/anomaly_lib.py` ไม่เจอ ซึ่งเป็นเรื่องปกติ เพราะ import นี้ไม่ได้อิง path
# เดิมของ package `app`
#
# ที่มัน import ได้ เพราะเราเติม folder
# `D:\apiUltraSoundSwine\AnomalyDetection\scripts`
# เข้า `sys.path` ก่อน ทำให้ Python มองว่าไฟล์ `anomaly_lib.py` ใน folder นั้น
# เป็น top-level module ชื่อ `anomaly_lib`
#
# สรุปสั้น:
# - ชื่อ import ที่เห็น: `anomaly_lib`
# - ไฟล์จริงบนดิสก์: `AnomalyDetection/scripts/anomaly_lib.py`
# - เหตุผลที่ทำแบบนี้: reuse logic train/validate/runtime ชุดเดียวกัน
if str(ANOMALY_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(ANOMALY_SCRIPT_DIR))

from anomaly_lib import (  # noqa: E402
    # import จาก `AnomalyDetection/scripts/anomaly_lib.py` โดยตรง หลังเติม sys.path แล้ว
    # ImageRow = row มาตรฐานที่ lib anomaly ใช้แทนภาพหนึ่งใบพร้อม metadata
    ImageRow,
    # extract_patch_handcrafted = แตก feature แบบ patch-based handcrafted
    extract_patch_handcrafted,
    # get_feature_matrix = router ของ feature extraction ปกติ เช่น handcrafted/resnet18/dinov2
    get_feature_matrix,
    # label_prediction = แปลง target เลข 0/1 เป็น label ข้อความ no_pregnant/pregnant
    label_prediction,
    # load_json = loader JSON แบบ utf-8-sig ใช้อ่าน registry/model metadata
    load_json,
    # load_model_bundle = โหลด `.joblib` bundle ที่ train ไว้แล้ว
    load_model_bundle,
    # predict_from_scores = เอา score ไปเทียบ threshold แล้วคืน class target
    predict_from_scores,
    # score_bundle = คิด score สำหรับ model bundle ทั่วไปที่ใช้ feature matrix ตรง ๆ
    score_bundle,
    # score_patch_bundle = คิด score สำหรับ bundle แบบ patchcore/padim_diag
    score_patch_bundle,
)


def resolve_active_anomaly_model(registry_path: Path = ANOMALY_REGISTRY) -> Path:
    """หาไฟล์ model active จาก registry.

    Registry เก็บ `active_model` ในรูป `run_name/model_key` แล้วชี้ไปไฟล์
    `.joblib` จริงอีกชั้นหนึ่ง

    จุดที่ต้องระวังคือ model registry อาจถูกสร้างจากเครื่อง train เดิมและเก็บ
    absolute Windows path ไว้ ดังนั้น runtime ต้องรองรับสองกรณี:
    1. path เดิมยังใช้ได้ -> ใช้ตรง ๆ
    2. path เดิมใช้ไม่ได้ เช่นอยู่ใน Docker / checkout คนละเครื่อง
       -> fallback ไปหาไฟล์ชื่อเดียวกันใต้ folder ข้าง registry
    """
    registry = load_json(registry_path)
    active = registry["active_model"]
    run_name, model_key = active.split("/", 1)
    model_ref = registry["runs"][run_name]["models"][model_key]
    model_file = str(model_ref["model_file"])
    model_path = Path(model_file)

    # Registry files can contain absolute Windows paths from training time.
    # In Docker or another checkout, fall back to the model file beside the registry.
    if model_path.exists():
        return model_path
    model_filename = PureWindowsPath(model_file).name if "\\" in model_file else model_path.name
    return registry_path.parent / run_name / model_filename


@lru_cache(maxsize=1)
def load_active_anomaly_bundle() -> tuple[Path, dict]:
    """โหลด active anomaly bundle ครั้งเดียวต่อ process.

    Runtime V2 อาจเรียก route ตรวจภาพซ้ำหลายครั้งใน process เดียวกัน ถ้าโหลด
    `.joblib` ใหม่ทุก request จะช้าโดยไม่จำเป็น จึง cache ไว้ที่ชั้นนี้
    """
    model_path = resolve_active_anomaly_model()
    return model_path, load_model_bundle(model_path)


def predict_anomaly_image(image_path: Path) -> dict:
    """รัน anomaly inference กับภาพเดียว.

    ลำดับจริงของ runtime anomaly คือ:
    1. โหลด active bundle
    2. สร้าง `ImageRow` ชั่วคราวแทนภาพ input จาก API
    3. เลือกทาง scoring ตาม `feature_set`
       - `patch_handcrafted` -> `extract_patch_handcrafted` + `score_patch_bundle`
       - อื่น ๆ -> `get_feature_matrix` + `score_bundle`
    4. เอา score ไปเทียบ threshold
    5. แปลง class target เป็นข้อความ `pregnant` / `no_pregnant`

    ค่าที่คืนจากฟังก์ชันนี้ยังเป็น "ผลดิบ" ของ anomaly backend
    route หรือ formatter ชั้นบนจะเป็นคนแปลงต่อเป็น `ImageResult`
    """
    model_path, bundle = load_active_anomaly_bundle()
    row = ImageRow(path=image_path, split="api", label_name="unknown", target=None)

    if bundle["feature_set"] == "patch_handcrafted":
        score = float(score_patch_bundle(bundle, extract_patch_handcrafted([row]))[0])
    else:
        features = get_feature_matrix([row], bundle["feature_set"])
        score = float(score_bundle(bundle, features)[0])

    threshold = float(bundle["threshold"])
    prediction_target = int(predict_from_scores(np.asarray([score]), threshold)[0])
    prediction = label_prediction(prediction_target)

    return {
        "image": str(image_path),
        "model_file": str(model_path),
        "model_name": bundle["model_name"],
        "feature_set": bundle["feature_set"],
        "prediction": prediction,
        "score_no_pregnant": score,
        "threshold": threshold,
        "score_meaning": bundle["score_meaning"],
        "estimator_type": bundle["estimator_type"],
    }


def anomaly_confidence(prediction: dict) -> float:
    """คำนวณ confidence สเกล 0..1 สำหรับ response กลางของ route.

    สำหรับ supervised estimator ใน repo นี้ score ถูกตีความเป็น
    `probability of no_pregnant` จึงต้องกลับด้านเมื่อ prediction เป็น
    `pregnant`
    """
    score = float(prediction["score_no_pregnant"])
    confidence = 0.0
    if prediction.get("estimator_type") == "sklearn_supervised" and 0.0 <= score <= 1.0:
        confidence = score if prediction["prediction"] == "no_pregnant" else 1.0 - score
        confidence = round(max(0.0, min(1.0, confidence)), 2)
    return confidence


def format_anomaly_result(filename: str, prediction: dict, save_path: str) -> ImageResult:
    """แปลงผลดิบของ anomaly เป็น `ImageResult`.

    ชั้น route ใน `app/main.py` ใช้ `ImageResult` เป็น contract กลางของสาย
    detection รูป ดังนั้นไม่ว่าผลจะมาจาก yolo หรือ anomaly ก็ต้องถูกจัดรูป
    ให้เหมือนกันที่ชั้นนี้
    """
    result_text = "not sure"
    if prediction["prediction"] == "pregnant":
        result_text = "pregnant"
    elif prediction["prediction"] == "no_pregnant":
        result_text = "no pregnant"

    return ImageResult(
        path_images=save_path,
        result=result_text,
        confidence=anomaly_confidence(prediction),
        number_of_fetus=0,
        error_remark="",
    )


def save_anomaly_cv2(img_cv2: np.ndarray, kind: str = "anomaly", page_number: int | None = None) -> str:
    """บันทึกภาพผลลัพธ์/ภาพกลางลงโฟลเดอร์ detections พร้อมตั้งชื่อไฟล์มาตรฐาน."""
    output_dir = pathInitial / "app" / "detections"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / build_image_filename(kind, page_number=page_number)
    cv2.imwrite(str(out_path), img_cv2)
    return str(out_path)


def save_detection_input(img_cv2: np.ndarray, org_filename: str, kind: str = "anomaly") -> str:
    """บันทึกภาพ input ของสาย detect รูป.

    ตอนนี้ `org_filename` ไม่ถูกใช้ในการตั้งชื่อแล้ว เพราะชื่อไฟล์มาตรฐานถูกสร้าง
    จาก `build_image_filename()` ทั้งหมด แต่ยังคง parameter นี้ไว้เพื่อให้ call
    site ฝั่ง route อ่านตามเจตนาได้ว่า save จาก "input file เดิม"
    """
    return save_anomaly_cv2(img_cv2, kind=kind)


def render_anomaly_pdf_images(pdf_path: Path) -> list[tuple[str, str]]:
    """render และ crop PDF ทีละหน้าแล้วบันทึกเป็นภาพสำหรับสาย anomaly/V2."""
    images = []
    for idx, page in enumerate(render_pdf_pages(str(pdf_path)), start=1):
        crop = crop_real_image(page)
        crop_rgb = np.array(crop.convert("RGB"))
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        save_path = save_anomaly_cv2(crop_bgr, kind="preg_pdf", page_number=idx)
        images.append((f"{pdf_path.name}#page={idx}", save_path))
    return images
