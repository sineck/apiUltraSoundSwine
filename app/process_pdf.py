import fitz  # PyMuPDF สำหรับสกัดภาพจาก ไฟล์ PDF 
from pdf2image import convert_from_path
import io
from PIL import Image
import os
from datetime import datetime
import numpy as np
import cv2
import pytesseract
import re
# import pyodbc
from dotenv import load_dotenv
import pymysql 
import uuid # ใช้สำหรับสุ่มชื่อ 
from pathlib import Path   
from ultralytics import YOLO 
import torch 
from app.image_naming import build_image_filename

import logging
from logging.handlers import RotatingFileHandler

# Resolve project root once so model, config, logs, and output folders stay relative to the repo.
pathInitial = Path(__file__).resolve().parent.parent

# --- Logging Setup ---
log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
LOG_DIR = pathInitial / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / "precess.log"
my_handler = RotatingFileHandler(
    str(log_file), mode="a", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8", delay=0
)
my_handler.setFormatter(log_formatter)
my_handler.setLevel(logging.INFO)

process_log = logging.getLogger("root")
process_log.setLevel(logging.INFO)
process_log.addHandler(my_handler)

if pathInitial.exists() :
    path = pathInitial /"config" / ".env" 
    load_dotenv(dotenv_path = str(path))

# Runtime knobs are read at import time because the YOLO model is loaded once per process.
CVCODE = int(os.getenv("cv_code","1"))  
USER_ID=int(os.getenv("UserID","9"))
min_score_threshold = float(os.getenv("Min_Score_th","0.50")) 
model_name = os.getenv("ModelName","best_finetune_YOLO26-cls_Ver2_20260424.pt") 
show_yolo_results = os.getenv("SHOW_YOLO_RESULTS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}


def should_insert_ultrasound_to_db() -> bool:
    """Read the DB-write switch at call time so .env changes apply without code edits."""
    value = os.getenv("INSERT_ULTRASOUND_TO_DB", "true").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def default_ocr_info():
    """Fallback OCR tuple used when Tesseract cannot read the converted page."""
    return ("The Text was not found.", "", "Unknown", "", "The Text was not found.", "The Text was not found.")


# Load the classifier once and reuse it for every converted PDF page.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use Device : {device}")

try:
    # 2. โหลดโมเดล
    path_model = pathInitial / "model" / model_name
    model = YOLO(path_model)
    
    # 3. ส่งเข้า Device
    model.to(device)
    print("Load Model Successfully!")

except Exception as e:
    print(f"load model failed: {e}")


create_static = pathInitial/"app"/"static" 
create_static.mkdir(parents=True,exist_ok=True)


def preprocess_yolo(path_img ,target_size=640):
    """ NOTE 
    ** ทำ Pre-Process ลด Noise , clear ภาพให้เนียน  
    เทคนิคที่ใช้	    สถานะ	                    ผลกระทบต่อความเป็นจริง
    -Grayscale	 ✅ ดีมาก	         ลดข้อมูลสีที่เป็น Noise ออกไป (Ultrasound จริงก็ไม่มีสี)
    -CLAHE	     ✅ ดีมาก             ช่วยให้โมเดลแยก "น้ำ" ออกจาก "เนื้อเยื่อ" ได้เหมือนตาผู้เชี่ยวชาญ
    -Letterbox	 ✅ ดีมาก	         ป้องกันการบิดเบี้ยว ของสัดส่วนถุงน้ำ 100%
    -Resize 640   ✅ พอดี              ละเอียดพอที่จะเห็นถุงน้ำ แต่ไม่ใหญ่จนเทรนไม่ไหว
    """  
    try :
        #! ทำ Pre-Process เพื่อลด Noise
        img_path = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE) 
        if img_path is None : 
            return None  
        
        # 2. Enhance Contrast (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img_path)

        # 3. Letterbox Padding (ทำให้เป็นจัตุรัส)
        h, w = img.shape[:2]
        max_dim = max(h, w)
        pad_img = np.zeros((max_dim, max_dim), dtype=np.uint8)

        # วางภาพไว้ตรงกลาง 
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2 

        # เพื่อรักษา "สัดส่วนเดิม" (Maintain Aspect Ratio) 
        # เพื่อป้องกัน "ข้อมูลสูญหาย" (Information Loss)
        pad_img[y_offset:y_offset+h, x_offset:x_offset+w] = img

        # 4. Resize to target (640) ก่อนเข้าโมเดล(เหมือนตอนเทรน)
        final_img = cv2.resize(pad_img, (target_size, target_size), interpolation=cv2.INTER_AREA)  

        final_img_bgr = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)

        # Inference expects a BGR 640x640 image matching the training preprocessing.
        results=model(final_img_bgr)  

        if show_yolo_results:
            results[0].show()

        pred_class = results[0].probs.top1  
        pred_names = model.names[int(pred_class)]

        # ค่าความมั่นใจ
        conf_score = results[0].probs.top1conf.item() 

        # Keep the original behavior: low-confidence predictions become "Unknown".
        if conf_score >=min_score_threshold :
            results_ai = pred_names
        else :
            results_ai = "Unknown" 

        return results_ai , conf_score

    except Exception as e :
        process_log.error(f"inference fail as {e}.")
        return None,None


def render_pdf_pages(pdf_path):
    """Render PDF pages with pdf2image first, then fall back to PyMuPDF for local runs."""
    try:
        return convert_from_path(pdf_path)
    except Exception as pdf2image_error:
        process_log.warning(f"[PDF] pdf2image failed, fallback to PyMuPDF: {pdf2image_error}")

    doc = fitz.open(pdf_path)
    try:
        pages = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        return pages
    finally:
        doc.close()


""" ฟังชั่นนี้สกัดทั้งหน้าต้นฉบับจาก PDF ไฟล์ ที่มี Text Results(ผลการตรวจจาก UltraSound โดยตรง)  """
def crop_real_image(pil_img, debug=False, debug_path=None):
    # แปลงภาพ PIL เป็น grayscale numpy arrar
    img = np.array(pil_img.convert("L"))  # แปลงเป็น grayscale
    # ใช้ threshold ต่ำ(ต้องหาค่าเฉพาะของแต่ละภาพ ปรับค่า threshold เพื่อหาเกณฑ์ที่เหมาะสมใช้ได้ครอบคลุมเกือบทุกภาพ) fan shape (ultrasound) ออกจากพื้นหลัง
    _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    # หา contour ทั้งหมดในภาพ (ลักษณะรูปร่าง(ตัวอ่อน,รังไข่) shape เป็น contour ที่ใหญ่สุด)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # เลือก contour ที่มีพื้นที่มากที่สุด (คาดว่าเป็น fan shape [ลักษณะรูปร่าง(ตัวอ่อน,รังไข่] )
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        # ตรวจสอบขนาด bounding box ถ้าเล็กผิดปกติให้ fallback ไปใช้ threshold ขาว
        if w < 100 or h < 100:
            print(f"[WARN] bounding box เล็กผิดปกติ: x={x}, y={y}, w={w}, h={h}")
            # fallback: ใช้ threshold สูงเพื่อหาเฉพาะส่วนที่มืด (fan shape)
            _, thresh2 = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
            coords = cv2.findNonZero(thresh2)
            x2, y2, w2, h2 = cv2.boundingRect(coords)
            # crop เฉพาะส่วนที่เจอ
            cropped = pil_img.crop((x2, y2, x2+w2, y2+h2))
            if debug and debug_path:
                img_color = np.array(pil_img.convert("RGB"))
                # วาดกรอบสี่เหลี่ยมแสดง bounding box ที่ crop
                cv2.rectangle(img_color, (x2, y2), (x2+w2, y2+h2), (255,0,0), 2)
                cv2.imwrite(str(debug_path), cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR))
            return cropped
        # crop เฉพาะ bounding box ของ contour ที่ใหญ่ที่สุด
        cropped = pil_img.crop((x, y, x+w, y+h))
        if debug and debug_path:
            img_color = np.array(pil_img.convert("RGB"))
            # วาด contour และกรอบสี่เหลี่ยมแสดง bounding box
            cv2.drawContours(img_color, [largest], -1, (0,255,0), 2)
            cv2.rectangle(img_color, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.imwrite(str(debug_path), cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR))
        return cropped
    else:
        # fallback: ถ้าไม่เจอ contour เลย ใช้ threshold ขาวเพื่อหาโซนที่ไม่ใช่ขอบขาว
        _, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)
        # crop เฉพาะส่วนที่เจอ
        cropped = pil_img.crop((x, y, x+w, y+h))
        if debug and debug_path:
            img_color = np.array(pil_img.convert("RGB"))
            # วาดกรอบสี่เหลี่ยมแสดง bounding box ที่ crop
            cv2.rectangle(img_color, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.imwrite(str(debug_path), cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR))
        return cropped

""" ฟังชั่นนี้สกัดเฉพาะ pdf to img เท่านั้น  """
# def extract_pdf_to_images(pdf_path, output_folder): 
#     # 1. สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#         print(f"Created directory: {output_folder}")

#     try:
#         # 2. เปิดไฟล์ PDF
#         doc = fitz.open(pdf_path)
#         print(f"Processing PDF: {os.path.basename(pdf_path)} ({len(doc)} pages)") 

#         # สร้าง ID สุ่ม 6 ตัวอักษร สำหรับ PDF ไฟล์นี้
#         unique_id = uuid.uuid4().hex[:6] 

#         for page_index in range(len(doc)):
#             page = doc[page_index]
#             image_list = page.get_images(full=True)

#             for img_index, img in enumerate(image_list):
#                 xref = img[0]
#                 base_image = doc.extract_image(xref)
#                 image_bytes = base_image["image"]
#                 image_ext = base_image["ext"]
                
#                 # แปลงเป็น PIL Image เพื่อจัดการต่อหรือบันทึก
#                 image = Image.open(io.BytesIO(image_bytes))
              
#                 # 5. ตั้งชื่อไฟล์และบันทึก
#                 # ใช้ f-string จัดรูปแบบชื่อไฟล์ให้เรียงง่าย เช่น page_001.png
#                 file_name = f"page_{page_index+1:03d}_{unique_id}.png"
#                 save_path = os.path.join(output_folder, file_name)
                
#                 image.save(save_path)
             
#         doc.close()
#         print(f"Successfully extracted {len(doc)} images to {output_folder}")
#         return True

#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return False

# Normalize OCR confusions inside ID-like text before saving it to the database.
def correct_ocr_text_and_numbers(val):
    if not val:
        return ""
    # mapping เฉพาะกับกลุ่มตัวเลข
    def fix_digits(match):
        s = match.group()
        table = str.maketrans({'e':'8','E':'8','a':'8','s':'8','o':'0','O':'0','i':'1','I':'1','l':'1','b': '6'})
        return s.translate(table)
    # แทนที่เฉพาะเลข (กลุ่ม digit หรือกลุ่มปน)
    return re.sub(r'\d+|[a-zA-Z0-9]+', fix_digits, val) 


def extract_info_from_image(image_path): 
    try:
        if image_path is None:
            process_log.error("No image path provided.")
            return None 
        
        # ถ้าเป็น path
        if isinstance(image_path, (str,Path)):
            image_cv  = cv2.imread(str(image_path)) 

        # ถ้าเป็น PIL.Image object
        elif isinstance(image_path, Image.Image):
            image_cv  = np.array(image_path.convert("RGB")) 

        # ถ้าเป็น np.ndarray อยู่แล้ว
        elif isinstance(image_path, np.ndarray):
            image_cv  = image_path

        else:
            process_log.error("Unsupported image input type.")
            return None

        if image_cv is None:
            process_log.error(f"Can't read image")
            return None
        
        # OCR reads the ultrasound overlay metadata before DB persistence.
        text = pytesseract.image_to_string(image_cv)  
        # print("OCR Output:", text)

        # ใช้ pytesseract หาข้อความในภาพ ก่อน crop images
        dt_match = re.search(r"Time[:\s]+([^\n\r]+)", text)                                 # <--- อ่าน ว/ด/ป (Time) ที่ระบุอยู่ในภาพ pdf ก่อน crop 
        dt_in_img_match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2} \d{2}:\d{2})", text)         # <--- อ่าน ว/ด/ป  ที่ระบุอยู่ในภาพ
        pregnant_match = re.search(r"Pregnant[:\s]+(\w+)", text)                            # <--- อ่าน pregnant  ที่ระบุอยู่ในภาพ
        # id_match = re.search(r"ID[:\s]+(\w+)", text)                                      # <--- อ่าน ID  ที่ระบุอยู่ในภาพ
        id_match = re.search(r"[^\w]*[Il1][DdpP][:.\s]+(\w+)", text, flags=re.IGNORECASE)   #! <--- อ่าน ID  ที่ระบุอยู่ในภาพ หากมีสัญลักษณ์อื่นอยู่หน้า     
        depth_match = re.search(r"Depth[:\s]+([\d]+mm)", text)                              # <--- อ่าน depth  ที่ระบุอยู่ในภาพ
        gain_match = re.search(r"Gain[:\s]+(\d+)\s*(?:dB|4B)", text)                             # <--- อ่าน Gain  ที่ระบุอยู่ในภาพ 
        

        # จัดกรุ๊ป
        dt_val = dt_match.group(1) if dt_match else "The Text was not found."
        dt_in_img_match_val = dt_in_img_match.group(1) if dt_in_img_match else ""
        pregnant_val = pregnant_match.group(1) if pregnant_match else "Unknown"
        id_val = id_match.group(1) if id_match else "" 
        # Post-Process 
        id_val =  correct_ocr_text_and_numbers(id_val)

        depth_val = depth_match.group(1) if depth_match else "The Text was not found."
        gain_val = gain_match.group(1) + "dB" if gain_match else "The Text was not found." 

        return dt_val,dt_in_img_match_val, pregnant_val, id_val, depth_val, gain_val
    
    except Exception as e:
        process_log.error(f"[Error] in funtion extract_info_from_image as {e}")
        return None

def insert_ultrasound_to_db(create_date, 
                            workdate,
                            time,
                            pregnant_p,
                            id_val, 
                            pdfFileName, 
                            depth_val, 
                            gain_val, 
                            path_val, 
                            file_name,
                            results_ai,
                            conf_score,
                            cvcode,
                            user_id
                            ):

    # Reload env here so direct function calls and long-running API workers use current DB settings.
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
            sql = """
                INSERT INTO  UltraSoudPigAI 
                (CreateDate,WorkDate,Time,Pregnant,ID,PdfFileName, Depth, Gain, Path, FileName ,Results_Ai,ConfScore,CVCODE,UserID) 
                VALUES (%s, %s, %s, %s, %s, %s, %s ,%s, %s, %s, %s, %s,%s,%s)
            """  
            values = (create_date, 
                        workdate,
                        time,
                        pregnant_p,
                        id_val, 
                        pdfFileName, 
                        depth_val, 
                        gain_val, 
                        path_val, 
                        file_name,
                        results_ai,
                        conf_score,
                        cvcode,
                        user_id
                        ) 
            cursor.execute(sql,values)
            connection.commit() 
            print("Insert Suscessfully")
    finally:
        connection.close()
        
def convert_pdf_to_png(pdf_path):
    """
    แปลง PDF เป็น PNG (crop เฉพาะส่วนที่เป็นภาพจริง) และบันทึกไฟล์ลง app/asset/
    Args:
        pdf_path (str): path ของไฟล์ PDF
    Returns:
        bool: True ถ้าสำเร็จ, False ถ้าเกิดข้อผิดพลาด
    """
    try:
        # กำหนดวันที่ปัจจุบันในรูปแบบ yyyy-mm-dd
        # date_folder = datetime.now().strftime("%Y-%m-%d")   # กรณีต้องการสร้างโฟลเดอร์pย่อย date ใน asset
        output_dir = os.path.join(os.path.dirname(__file__), "asset")
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: render every PDF page to PIL images.
        images = render_pdf_pages(pdf_path)  

        work_date = datetime.now().strftime("%Y-%m-%d")
        for idx, img in enumerate(images): 

            # Step 2: crop out page margins and save the ultrasound image as PNG.
            crop = crop_real_image(img)
            out_name = build_image_filename("pdf", page_number=idx + 1)
            out_path = os.path.join(output_dir, out_name) 
            crop_np = np.array(crop.convert("RGB"))         # PIL → numpy RGB
            crop_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)  # RGB → BGR (สำหรับ OpenCV)
            cv2.imwrite(out_path, crop_bgr) 
            process_log.info(f"[SAVE] Saved image: {out_path}")

            # Step 3: classify the saved ultrasound image with the YOLO classifier.
            results_ai , conf_score = preprocess_yolo(out_path)

            # Step 4: OCR the same saved image for metadata written to MySQL.
            ocr_result = extract_info_from_image(out_path)
            if ocr_result is None:
                process_log.warning(f"[OCR SKIP] fallback metadata used for {out_name}")
                ocr_result = default_ocr_info()
            dt_val,dt_in_img_match_val, pregnant_val, id_val, depth_val, gain_val = ocr_result  

            # Step 5: persist page-level AI, OCR, and file metadata when enabled.
            if should_insert_ultrasound_to_db():
                insert_ultrasound_to_db(
                    create_date=datetime.now(),
                    workdate=work_date,
                    time=dt_val,
                    pregnant_p=pregnant_val,
                    id_val=id_val,
                    pdfFileName=os.path.basename(pdf_path),
                    depth_val=depth_val,
                    gain_val=gain_val,
                    path_val=output_dir,
                    file_name=out_name,
                    results_ai=results_ai ,
                    conf_score=conf_score,
                    cvcode=CVCODE,
                    user_id=USER_ID
                )
            else:
                process_log.info(f"[DB SKIP] INSERT_ULTRASOUND_TO_DB disabled for {out_name}")
        return True
    except Exception as e:
        print(f"[ERROR] convert_pdf_to_png: {e}")
        return False 
