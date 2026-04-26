from typing import List
from PIL import Image
from pathlib import Path 
from google import genai  
from google.genai import types
from dotenv import load_dotenv
import os 
import numpy as np 
import cv2
from pydantic import BaseModel 
import pymysql 

# Ultralytics for annotation
from ultralytics.utils.plotting import Annotator, colors 

import json
import logging
from logging.handlers import RotatingFileHandler  
from app.image_naming import build_image_filename

# Resolve project root once so config and output folders stay stable in Docker/local runs.
pathInitial = Path(__file__).resolve().parent.parent    
if pathInitial.exists() :
    path = pathInitial /"config" / ".env" 
    load_dotenv(dotenv_path = str(path))

#=====================
# ---- Config env ----
#=====================
API_PORT = int(os.getenv("MYAPI_PORT", 3014))
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY") 
MODEL_GEMINI=os.getenv("MODEL_GEMINI", "gemini-3-flash-preview")  
min_score_threshold = float(os.getenv("min_score",'0.50'))


class GeminiAnalysisError(RuntimeError):
    """Raised when Gemini cannot return a usable detection payload."""


IMG_RESULT_DIR = pathInitial / "app" / "detections"
IMG_RESULT_DIR.mkdir(parents=True ,exist_ok=True)

def save_annotation_result(img_cv2: np.ndarray ,org_filename : str) -> str :
    """
    Save ภาพหลัง Gemini Predict แล้ว
    Return: path ของไฟล์ที่ save
    """
    try:
        out_name = build_image_filename("gemini")
        out_path = IMG_RESULT_DIR / out_name

        # ✅ Save ภาพ
        cv2.imwrite(str(out_path), img_cv2)
        detect_log.info(f"[SAVE] Saved annotated image: {out_path}")

        return str(out_path)

    except Exception as e:
        detect_log.error(f"[ERROR] Save image failed: {e}")
        return build_image_filename("gemini_error")


#======================
# --- Logging Setup --- 
#======================
log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
LOG_DIR = pathInitial / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / "detect.log"
my_handler = RotatingFileHandler(
    str(log_file), mode="a", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8", delay=0
)
my_handler.setFormatter(log_formatter)
my_handler.setLevel(logging.INFO)

detect_log = logging.getLogger("detection")
detect_log.setLevel(logging.INFO)
detect_log.addHandler(my_handler) 

#================================
# -------- Results Json --------
#================================
class ImageResult(BaseModel) :
    path_images : str 
    result : str 
    confidence : float 
    number_of_fetus : int 
    error_remark : str

class DetectionRespone(BaseModel) :
    main_results : str 
    error_massage : str 
    results : List[ImageResult]

def Format_Result(filename:str ,ai:dict ,save_path:str) -> ImageResult :
    """ NOTE แปลงผลลัพธ์จาก analyze_ultrasound_core() เป็น ImageResult """
    try:
        # แปลง status → result text
        if ai["status"] == "1_Pregnant":
            result_text = "pregnant"
        elif ai["status"] == "2_NoPregnant":
            result_text = "no pregnant"
        else:
            result_text = "not sure"

        # Response confidence is the mean confidence from accepted Gemini detections.
        detections = ai.get("detections", [])
        if detections:
            confidences = [float(d.get("confidence", 1.0)) for d in detections]
            avg_confidence = round(sum(confidences) / len(confidences), 2)
        else:
            avg_confidence = 0.0

        return ImageResult(
            path_images=save_path,    # ส่ง path ที่ save ภาพ img annotation
            result=result_text,
            confidence=avg_confidence,
            number_of_fetus=ai["sac_count"],
            error_remark=""
        )

    except Exception as e:
        return ImageResult(
            path_images=save_path,
            result="error",
            confidence=0.0,
            number_of_fetus=0,
            error_remark=str(e)
        )

def insert_detection_to_db (
        filename : str,
        saved_path : str,
        status : str,
        sac_count : str,
        detections : str
) :
    path_env = os.path.join(os.getcwd(),"config",".env")  
    load_dotenv(dotenv_path=path_env)
    MYSQL_HOST = os.getenv("MYSQL_HOST")
    MYSQL_PORT = os.getenv("MYSQL_PORT") 
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
    MYSQL_USER = os.getenv("MYSQL_USER")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")  

    connection=pymysql.connect(
        host=MYSQL_HOST,
        port=int(MYSQL_PORT) if MYSQL_PORT else 3306,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
    try:
        with connection.cursor() as cursor : 
            #! --- dummy --- #
            """ NOTE ตาราง UltraSoudPigAI_Detection (ยังไม่ได้สร้าง) """

            sql = """
                INSERT INTO  UltraSoudPigAI_Detection 
                (Filename,Saved_path,Status,Sac_count,Detections) 
                VALUES (%s, %s, %s, %s, %s)
            """  
            values = (filename, 
                        saved_path,
                        status,
                        sac_count,
                        detections, 
                        ) 
            cursor.execute(sql,values)
            connection.commit() 
            print("Insert Suscessfully")
    finally:
        connection.close()
    
#================================
#      Gemini client and prompt contract
#================================
client = None
GEMINI_CLIENT_ERROR: str | None = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        GEMINI_CLIENT_ERROR = str(e)
else:
    GEMINI_CLIENT_ERROR = "GEMINI_API_KEY is not configured"


def get_gemini_client():
    if client is None:
        raise GeminiAnalysisError(f"Gemini client is not available: {GEMINI_CLIENT_ERROR or 'unknown'}")
    return client

# ==========================================
# 🧠 The Inference by GEMINI   (สร้าง Prompt Gemini)
# ==========================================
gemini_prompt = """   
Analyze the image for pregnancy
    To analyze the image for pregnancy and developing piglets, a computer vision system (like a CNN with Object Detection and Segmentation heads) would use the following structured hierarchy of tasks:

    I. Pre-processing & Global Context:
    1. Optical Character Recognition (OCR): Identify text overlays for image context (e.g., ID: M205903, Depth: 160mm). These set the physical scale of the pixels.
    2. Image Normalization: Standardize grayscale values and reduce noise using a speckle-reducing filter, while preserving key edges.

    II. Region Proposal (Object Detection):
    1. Gestational Sac Detection (Early Stage): Scan for large, low-variance, low-intensity (anechoic) blobs (like those in image_0.png) and assign a 'Uterine Fluid Sac' probability.
    2. Fetal Candidate Detection (Later Stage): Identify nested regions where high-intensity (hyperechoic) points (bone candidates) are clustered within medium-intensity (gray) soft tissue, which is itself within an anechoic (fluid) background (the pattern in image_1.png).
    3. Fetal Counting: Employ non-maximum suppression (NMS) to differentiate and count distinct clusters (fetuses). In complex images like image_1.png, the model looks for repeating patterns (e.g., a spine or rib cage).

    III. Feature Extraction & Classification (Per Bounding Box): For each detected region (bounding box), extract and classify features using a deep neural network (e.g., ResNet-50) trained on veterinary ultrasound databases:
    1. Shape and Structure: Match observed high-contrast curves to known anatomical shapes (e.g., spine arc, skull ellipse, limb bones).
    2. Texture: Segment different tissues: fluid (smooth, black), bone (rough, bright white), and soft tissue (grainy, gray).

    IV. Advanced Segmentation & Measurement:
    1. Semantic Segmentation: Create a mask isolating the entire body of each piglet from the surrounding fluid.
    2. Measurement (Biometry): Use the known physical scale (from OCR Depth) to measure the longest segmented diameter, providing an estimated Crown-Rump Length (CRL) for gestational age prediction.
"""
output_prompt = """
IMPORTANT RULES FOR OUTPUT:
Return ONLY a JSON List of dictionaries. Do not use Markdown.
If objects are found, format like this:
[
  {"box_2d": [ymin, xmin, ymax, xmax], "label": "gestational sac"}
]
If NO signs of pregnancy are found, return exactly an empty list:
[]
""" 


def scale_bbox(box_2d: list, img_width: int, img_height: int) -> tuple:
    # Gemini returns normalized coordinates in a 0-1000 space; convert them to pixels.
    ymin, xmin, ymax, xmax = box_2d
    x1, x2 = (xmin / 1000.0 * img_width), (xmax / 1000.0 * img_width)
    y1, y2 = (ymin / 1000.0 * img_height), (ymax / 1000.0 * img_height)
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    return int(x1), int(y1), int(x2), int(y2)

def annotate_detections(image: np.ndarray, detections: list) -> np.ndarray:
    # Draw accepted detections on a copy of the original OpenCV image.
    height, width = image.shape[:2] 
    annotator = Annotator(image, line_width=2, font_size=16)
    for idx, item in enumerate(detections):
        box = item.get("box_2d", [])
        if isinstance(box, list) and len(box) == 4:
            x1, y1, x2, y2 = scale_bbox(box, width, height)
            annotator.box_label([int(x1), int(y1), int(x2), int(y2)], label=item.get("label", "gestational sac"), color=colors(idx, True))
    return annotator.result()

def analyze_ultrasound_core(img_cv2: np.ndarray):
    """ฟังก์ชันสมองกลาง ให้ทั้งเว็บและ PDF เรียกใช้งานร่วมกัน"""
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img_rgb)
    
    # 1. Ask Gemini to return only JSON so the API response remains deterministic.
    try:
        response = get_gemini_client().models.generate_content(
            model=MODEL_GEMINI, contents=[gemini_prompt + output_prompt, image_pil],
            config=types.GenerateContentConfig(temperature=0.1)
        )
        text = response.text.replace("```json", "").replace("```", "").strip()
        raw_detections = json.loads(text) 
    except Exception as e:
        detect_log.error(f"Gemini API Error: {e}")
        raise GeminiAnalysisError(f"Gemini analysis failed: {e}") from e

    # 2. Accept both the expected list shape and a single dict fallback.
    raw_list = []
    if isinstance(raw_detections, list):
        for item in raw_detections:
            if isinstance(item, dict) and isinstance(item.get("box_2d"), list) and len(item["box_2d"]) == 4:
                raw_list.append(item)
    elif isinstance(raw_detections, dict) and isinstance(raw_detections.get("box_2d"), list):
        if len(raw_detections["box_2d"]) == 4 and isinstance(raw_detections["box_2d"][0], (int, float)):
            raw_list.append(raw_detections)

    # 3. Keep only labels that indicate pregnancy-related structures.
    detections = []
    is_pregnant = False

    # รวม class name ที่ gemini เจนออกมา
    valid_keywords = ["gestational sac", "fetus", "fluid", "amniotic", "pregnan", "piglet", "black"] 
                  
    for d in raw_list:
        conf = float(d.get("confidence", 1.0))
        if conf >= min_score_threshold :
            lbl = str(d.get("label", "")).lower()
            if any(k in lbl for k in valid_keywords) or lbl in ["", "unknown"]:
                is_pregnant = True
                d["label"] = "gestational sac"
                detections.append(d)

    # 4. Summarize accepted detections into the internal status contract.
    annotated_img = img_cv2.copy()
    if len(detections) == 0:
        status = "2_NoPregnant" 

    # ถ้า gemini เจอ class ใดหนึ่งใน valid_keywords ให้ระบุเป็นหมูท้อง (1_Pregnant)
    elif is_pregnant:
        annotated_img = annotate_detections(img_cv2, detections)
        status = "1_Pregnant"
    else:
        annotated_img = annotate_detections(img_cv2, detections)
        status = "NotSure"

    return {"status": status, "sac_count": len(detections), "annotated_img": annotated_img, "detections": detections}
