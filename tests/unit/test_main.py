from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app import main


def test_upload_pdf_rejects_non_pdf_extension():
    client = TestClient(main.app)

    response = client.post(
        "/upload_pdf/",
        files={"file": ("scan.txt", b"not a pdf", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "อนุญาตเฉพาะไฟล์ .pdf เท่านั้น"


def test_upload_pdf_uses_generated_storage_name(monkeypatch, tmp_path):
    seen = {}

    def fake_convert(path):
        seen["path"] = path
        return True

    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(main, "convert_pdf_to_png", fake_convert)
    monkeypatch.setattr(main, "build_image_filename", lambda *args, **kwargs: "generated.pdf")
    client = TestClient(main.app)

    response = client.post(
        "/upload_pdf/",
        files={"file": ("..\\evil.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "complete"}
    assert seen["path"] == str(tmp_path / "generated.pdf")
    assert not (tmp_path.parent / "evil.pdf").exists()


def test_cors_default_does_not_allow_credentials():
    client = TestClient(main.app)

    response = client.options(
        "/health",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


def test_version_returns_app_metadata(monkeypatch):
    client = TestClient(main.app)

    response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {"name": main.APP_NAME, "version": main.APP_VERSION}


def test_root_redirects_to_docs():
    client = TestClient(main.app, follow_redirects=False)

    response = client.get("/")

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_selected_pregnancy_model_uses_process_env_directly(monkeypatch):
    monkeypatch.setenv("PREGNANCY_DETECT_MODEL_V2", "yolo")
    monkeypatch.setattr(main, "reload_runtime_config", lambda: (_ for _ in ()).throw(AssertionError("should not reload .env during request")))

    assert main.selected_pregnancy_model() == "yolo"


def test_health_includes_version_when_db_unreachable(monkeypatch):
    def fail_connect(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(main, "APP_NAME", "test-api")
    monkeypatch.setattr(main, "APP_VERSION", "9.9.9")
    monkeypatch.setattr(
        main,
        "runtime_config_summary",
        lambda: {
            "config_path": "D:/apiUltraSoundSwine/config/.env",
            "configured_myapi_port": 3014,
            "insert_ultrasound_to_db": False,
            "pregnancy_detect_model_v2": "anomaly",
            "yolo_model_name": "best.pt",
            "gemini_model": "gemini-3-flash-preview",
            "max_images": 5,
        },
    )
    monkeypatch.setattr(main.pymysql, "connect", fail_connect)
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["db"] == "unreachable"
    assert payload["app"] == {"name": "test-api", "version": "9.9.9"}
    assert payload["config"]["pregnancy_detect_model_v2"] == "anomaly"
    assert payload["config"]["insert_ultrasound_to_db"] is False
    assert "MYSQL_PASSWORD" not in response.text
    assert "mysql_host" not in payload["config"]


def test_detect_follicle_rejects_more_than_configured_max_images(monkeypatch):
    monkeypatch.setattr(main, "max_images", 1)
    client = TestClient(main.app)

    response = client.post(
        "/detect_follicle/",
        files=[
            ("files", ("a.png", b"not image", "image/png")),
            ("files", ("b.png", b"not image", "image/png")),
        ],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "อัปโหลดได้สูงสุด 1 รูปต่อครั้ง (ส่งมา 2 รูป)"


def test_detect_follicle_returns_item_error_when_ai_raises(monkeypatch):
    def fail_analysis(img):
        raise RuntimeError("ai failed")

    monkeypatch.setattr(main, "max_images", 1)
    monkeypatch.setattr(main, "analyze_ultrasound_core", fail_analysis)
    client = TestClient(main.app)
    image_bytes = BytesIO()
    Image.new("RGB", (4, 4), color="black").save(image_bytes, format="PNG")

    response = client.post(
        "/detect_follicle/",
        files=[("files", ("scan.png", image_bytes.getvalue(), "image/png"))],
    )

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["path_images"] == "scan.png"
    assert item["result"] == "error"
    assert item["error_remark"] == "ai failed"


def test_upload_pdf_returns_503_when_model_unavailable(monkeypatch, tmp_path):
    def fail_convert(path):
        raise main.ModelUnavailableError("missing model")

    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(main, "convert_pdf_to_png", fail_convert)
    monkeypatch.setattr(main, "build_image_filename", lambda *args, **kwargs: "generated.pdf")
    client = TestClient(main.app)

    response = client.post(
        "/upload_pdf/",
        files={"file": ("scan.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "missing model"


def test_v2_detection_pig_returns_existing_detection_response_shape(monkeypatch, tmp_path):
    def fake_save(img_cv2, filename):
        return str(tmp_path / f"saved_{filename}")

    def fake_predict(path):
        return {
            "prediction": "no_pregnant",
            "score_no_pregnant": 0.91,
            "threshold": 0.5,
            "estimator_type": "sklearn_supervised",
        }

    monkeypatch.setenv("PREGNANCY_DETECT_MODEL_V2", "anomaly")
    monkeypatch.setenv("INSERT_ULTRASOUND_TO_DB", "false")
    monkeypatch.setattr(main, "reload_runtime_config", lambda: None)
    monkeypatch.setattr(main, "max_images", 1)
    monkeypatch.setattr(main, "save_detection_input", fake_save)
    monkeypatch.setattr(main, "predict_anomaly_image", fake_predict)
    client = TestClient(main.app)
    image_bytes = BytesIO()
    Image.new("RGB", (4, 4), color="black").save(image_bytes, format="PNG")

    response = client.post(
        "/v2/detection_pig",
        files=[("files", ("scan.png", image_bytes.getvalue(), "image/png"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["main_results"] == "success"
    assert payload["error_massage"] == ""
    assert len(payload["results"]) == 1
    item = payload["results"][0]
    assert item["path_images"].endswith("saved_scan.png")
    assert item["result"] == "no pregnant"
    assert item["confidence"] == 0.91
    assert item["number_of_fetus"] == 0
    assert item["error_remark"] == ""


def test_v2_detection_pig_skips_db_insert_when_disabled(monkeypatch, tmp_path):
    calls = {"insert": 0}

    def fake_save(img_cv2, filename):
        return str(tmp_path / f"saved_{filename}")

    def fake_predict(path):
        return {
            "prediction": "pregnant",
            "score_no_pregnant": 0.12,
            "threshold": 0.5,
            "estimator_type": "sklearn_supervised",
        }

    def fake_insert(**kwargs):
        calls["insert"] += 1

    monkeypatch.setenv("PREGNANCY_DETECT_MODEL_V2", "anomaly")
    monkeypatch.setenv("INSERT_ULTRASOUND_TO_DB", "false")
    monkeypatch.setattr(main, "reload_runtime_config", lambda: None)
    monkeypatch.setattr(main, "max_images", 1)
    monkeypatch.setattr(main, "save_detection_input", fake_save)
    monkeypatch.setattr(main, "predict_anomaly_image", fake_predict)
    monkeypatch.setattr(main, "insert_ultrasound_to_db", fake_insert)
    client = TestClient(main.app)
    image_bytes = BytesIO()
    Image.new("RGB", (4, 4), color="black").save(image_bytes, format="PNG")

    response = client.post(
        "/v2/detection_pig",
        files=[("files", ("scan.png", image_bytes.getvalue(), "image/png"))],
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["result"] == "pregnant"
    assert calls["insert"] == 0


def test_v2_detection_pig_can_use_legacy_yolo_backend(monkeypatch, tmp_path):
    def fake_save(img_cv2, filename):
        return str(tmp_path / f"saved_{filename}")

    def fake_yolo(path):
        assert path.endswith("saved_scan.png")
        return "1_Pregnant", 0.876

    monkeypatch.setenv("PREGNANCY_DETECT_MODEL_V2", "yolo")
    monkeypatch.setenv("INSERT_ULTRASOUND_TO_DB", "false")
    monkeypatch.setattr(main, "reload_runtime_config", lambda: None)
    monkeypatch.setattr(main, "max_images", 1)
    monkeypatch.setattr(main, "save_detection_input", fake_save)
    monkeypatch.setattr(main, "preprocess_yolo", fake_yolo)
    client = TestClient(main.app)
    image_bytes = BytesIO()
    Image.new("RGB", (4, 4), color="black").save(image_bytes, format="PNG")

    response = client.post(
        "/v2/detection_pig",
        files=[("files", ("scan.png", image_bytes.getvalue(), "image/png"))],
    )

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["path_images"].endswith("saved_scan.png")
    assert item["result"] == "pregnant"
    assert item["confidence"] == 0.88
    assert item["number_of_fetus"] == 0
    assert item["error_remark"] == ""


def test_detect_saved_pregnancy_image_with_backend_supports_ensemble_pregnant(monkeypatch):
    monkeypatch.setenv("PREGNANCY_DETECT_MODEL_V2", "ensemble")
    monkeypatch.setattr(
        main,
        "detect_saved_anomaly_image",
        lambda display_name, save_path: main.ImageResult(
            path_images=save_path,
            result="pregnant",
            confidence=0.92,
            number_of_fetus=0,
            error_remark="",
        ),
    )
    monkeypatch.setattr(
        main,
        "detect_saved_yolo_image",
        lambda display_name, save_path: main.PregnancyDetectionOutcome(
            result=main.ImageResult(
                path_images=save_path,
                result="pregnant",
                confidence=0.81,
                number_of_fetus=0,
                error_remark="",
            ),
            legacy_ai_label="1_Pregnant",
        ),
    )

    outcome = main.detect_saved_pregnancy_image_with_backend("scan.png", "saved_scan.png")

    assert outcome.result.result == "pregnant"
    assert outcome.result.confidence == 0.81
    assert outcome.legacy_ai_label == "1_Pregnant"


def test_detect_saved_pregnancy_image_with_backend_supports_ensemble_no_pregnant_on_disagreement(monkeypatch):
    monkeypatch.setenv("PREGNANCY_DETECT_MODEL_V2", "ensemble")
    monkeypatch.setattr(
        main,
        "detect_saved_anomaly_image",
        lambda display_name, save_path: main.ImageResult(
            path_images=save_path,
            result="pregnant",
            confidence=0.92,
            number_of_fetus=0,
            error_remark="",
        ),
    )
    monkeypatch.setattr(
        main,
        "detect_saved_yolo_image",
        lambda display_name, save_path: main.PregnancyDetectionOutcome(
            result=main.ImageResult(
                path_images=save_path,
                result="not sure",
                confidence=0.6,
                number_of_fetus=0,
                error_remark="",
            ),
            legacy_ai_label="Unknown",
        ),
    )

    outcome = main.detect_saved_pregnancy_image_with_backend("scan.png", "saved_scan.png")

    assert outcome.result.result == "no pregnant"
    assert outcome.result.confidence == 0.92
    assert outcome.legacy_ai_label == "2_NoPrenant_or_NotSure"


def test_detect_saved_pregnancy_image_with_backend_supports_ensemble_error_passthrough(monkeypatch):
    monkeypatch.setenv("PREGNANCY_DETECT_MODEL_V2", "ensemble")
    monkeypatch.setattr(
        main,
        "detect_saved_anomaly_image",
        lambda display_name, save_path: main.ImageResult(
            path_images=save_path,
            result="error",
            confidence=0.0,
            number_of_fetus=0,
            error_remark="anomaly failed",
        ),
    )
    monkeypatch.setattr(
        main,
        "detect_saved_yolo_image",
        lambda display_name, save_path: main.PregnancyDetectionOutcome(
            result=main.ImageResult(
                path_images=save_path,
                result="pregnant",
                confidence=0.9,
                number_of_fetus=0,
                error_remark="",
            ),
            legacy_ai_label="1_Pregnant",
        ),
    )

    outcome = main.detect_saved_pregnancy_image_with_backend("scan.png", "saved_scan.png")

    assert outcome.result.result == "error"
    assert outcome.result.error_remark == "anomaly failed"
    assert outcome.legacy_ai_label == "2_NoPrenant_or_NotSure"


def test_v2_detection_pig_inserts_db_when_enabled(monkeypatch, tmp_path):
    calls = {}

    def fake_save(img_cv2, filename):
        return str(tmp_path / f"saved_{filename}")

    def fake_predict(path):
        return {
            "prediction": "pregnant",
            "score_no_pregnant": 0.12,
            "threshold": 0.5,
            "estimator_type": "sklearn_supervised",
        }

    def fake_insert(**kwargs):
        calls["insert"] = kwargs

    monkeypatch.setenv("PREGNANCY_DETECT_MODEL_V2", "anomaly")
    monkeypatch.setenv("INSERT_ULTRASOUND_TO_DB", "false")
    monkeypatch.setenv("INSERT_ULTRASOUND_TO_DB", "true")
    monkeypatch.setattr(main, "reload_runtime_config", lambda: None)
    monkeypatch.setattr(main, "max_images", 1)
    monkeypatch.setattr(main, "save_detection_input", fake_save)
    monkeypatch.setattr(main, "predict_anomaly_image", fake_predict)
    monkeypatch.setattr(main, "extract_info_from_image", lambda path: main.default_ocr_info())
    monkeypatch.setattr(main, "insert_ultrasound_to_db", fake_insert)
    client = TestClient(main.app)
    image_bytes = BytesIO()
    Image.new("RGB", (4, 4), color="black").save(image_bytes, format="PNG")

    response = client.post(
        "/v2/detection_pig",
        files=[("files", ("scan.png", image_bytes.getvalue(), "image/png"))],
    )

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["result"] == "pregnant"
    assert item["error_remark"] == ""
    assert calls["insert"]["pdfFileName"] == "scan.png"
    assert calls["insert"]["pregnant_p"] == "Unknown"
    assert calls["insert"]["id_val"] == "saved_scan"
    assert calls["insert"]["results_ai"] == "1_Pregnant"
    assert calls["insert"]["conf_score"] == 0.88


def test_v2_detection_pig_returns_db_error_in_same_json_shape(monkeypatch, tmp_path):
    def fake_save(img_cv2, filename):
        return str(tmp_path / f"saved_{filename}")

    def fake_predict(path):
        return {
            "prediction": "no_pregnant",
            "score_no_pregnant": 0.91,
            "threshold": 0.5,
            "estimator_type": "sklearn_supervised",
        }

    def fail_insert(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setenv("PREGNANCY_DETECT_MODEL_V2", "anomaly")
    monkeypatch.setenv("INSERT_ULTRASOUND_TO_DB", "true")
    monkeypatch.setattr(main, "reload_runtime_config", lambda: None)
    monkeypatch.setattr(main, "max_images", 1)
    monkeypatch.setattr(main, "save_detection_input", fake_save)
    monkeypatch.setattr(main, "predict_anomaly_image", fake_predict)
    monkeypatch.setattr(main, "extract_info_from_image", lambda path: main.default_ocr_info())
    monkeypatch.setattr(main, "insert_ultrasound_to_db", fail_insert)
    client = TestClient(main.app)
    image_bytes = BytesIO()
    Image.new("RGB", (4, 4), color="black").save(image_bytes, format="PNG")

    response = client.post(
        "/v2/detection_pig",
        files=[("files", ("scan.png", image_bytes.getvalue(), "image/png"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["main_results"] == "success"
    item = payload["results"][0]
    assert item["result"] == "no pregnant"
    assert item["confidence"] == 0.91
    assert item["error_remark"] == "DB insert failed: db down"


def test_v2_detection_pig_returns_item_error_for_invalid_image(monkeypatch):
    monkeypatch.setattr(main, "max_images", 1)
    client = TestClient(main.app)

    response = client.post(
        "/v2/detection_pig",
        files=[("files", ("scan.png", b"not image", "image/png"))],
    )

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["path_images"] == "scan.png"
    assert item["result"] == "error"
    assert item["confidence"] == 0.0
    assert item["error_remark"] == "อ่านไฟล์ภาพไม่ได้"


def test_v2_detection_pig_rejects_more_than_configured_max_images(monkeypatch):
    monkeypatch.setattr(main, "max_images", 1)
    client = TestClient(main.app)

    response = client.post(
        "/v2/detection_pig",
        files=[
            ("files", ("a.png", b"not image", "image/png")),
            ("files", ("b.png", b"not image", "image/png")),
        ],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "อัปโหลดได้สูงสุด 1 รูปต่อครั้ง (ส่งมา 2 รูป)"


def test_precheck_ultrasound_image_rejects_colorful_natural_like_input():
    blue = Image.new("RGB", (96, 96), color=(255, 0, 0))
    green = Image.new("RGB", (96, 96), color=(0, 255, 0))
    image = Image.new("RGB", (192, 96))
    image.paste(blue, (0, 0))
    image.paste(green, (96, 0))
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    img_cv2 = main.cv2.imdecode(main.np.frombuffer(image_bytes.getvalue(), main.np.uint8), main.cv2.IMREAD_COLOR)

    result = main.precheck_ultrasound_image(img_cv2)

    assert result.is_ultrasound is False
    assert "Input is not recognized as an ultrasound image" in result.reason


def test_v2_detection_pig_returns_unknown_for_non_ultrasound(monkeypatch):
    def fake_save(img_cv2, filename, kind="anomaly"):
        return f"D:/apiUltraSoundSwine/app/detections/{kind}_saved.png"

    monkeypatch.setattr(main, "max_images", 1)
    monkeypatch.setattr(main, "save_detection_input", fake_save)
    client = TestClient(main.app)
    image_bytes = BytesIO()
    image = Image.new("RGB", (192, 96))
    image.paste(Image.new("RGB", (96, 96), color=(255, 0, 0)), (0, 0))
    image.paste(Image.new("RGB", (96, 96), color=(0, 255, 0)), (96, 0))
    image.save(image_bytes, format="PNG")

    response = client.post(
        "/v2/detection_pig",
        files=[("files", ("car_like.png", image_bytes.getvalue(), "image/png"))],
    )

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["result"] == "unknown"
    assert item["confidence"] == 0.0
    assert item["path_images"].endswith("unknown_saved.png")
    assert "Input is not recognized as an ultrasound image" in item["error_remark"]


def test_v2_detection_pig_follicle_returns_unknown_for_non_ultrasound(monkeypatch):
    def fake_save(img_cv2, filename, kind="anomaly"):
        return f"D:/apiUltraSoundSwine/app/detections/{kind}_saved.png"

    monkeypatch.setattr(main, "max_images", 1)
    monkeypatch.setattr(main, "save_detection_input", fake_save)
    client = TestClient(main.app)
    image_bytes = BytesIO()
    image = Image.new("RGB", (192, 96))
    image.paste(Image.new("RGB", (96, 96), color=(255, 0, 0)), (0, 0))
    image.paste(Image.new("RGB", (96, 96), color=(0, 255, 0)), (96, 0))
    image.save(image_bytes, format="PNG")

    response = client.post(
        "/v2/detection_pig_follicle",
        files=[("files", ("car_like.png", image_bytes.getvalue(), "image/png"))],
    )

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["result"] == "unknown"
    assert item["confidence"] == 0.0
    assert item["path_images"].endswith("unknown_saved.png")
    assert "Input is not recognized as an ultrasound image" in item["error_remark"]


def test_v2_detection_pig_follicle_saves_annotation_to_detect_follicle_path(monkeypatch, tmp_path):
    def fake_save_input(img_cv2, filename):
        return str(tmp_path / f"saved_{filename}")

    def fake_detect(filename, save_path):
        return main.PregnancyDetectionOutcome(
            result=main.ImageResult(
                path_images=save_path,
                result="pregnant",
                confidence=0.95,
                number_of_fetus=0,
                error_remark="",
            ),
            legacy_ai_label="1_Pregnant",
        )

    def fake_analyze(img_cv2):
        return {
            "annotated_img": img_cv2,
            "sac_count": 2,
            "status": "1_Pregnant",
            "detections": [{"label": "gestational sac"}],
        }

    def fake_save_annotation(img_cv2, filename):
        return str(tmp_path / f"gemini_{filename}")

    monkeypatch.setattr(main, "max_images", 1)
    monkeypatch.setattr(main, "save_detection_input", fake_save_input)
    monkeypatch.setattr(main, "detect_saved_pregnancy_image_with_backend", fake_detect)
    monkeypatch.setattr(main, "analyze_ultrasound_core", fake_analyze)
    monkeypatch.setattr(main, "save_annotation_result", fake_save_annotation)
    client = TestClient(main.app)
    image_bytes = BytesIO()
    Image.new("RGB", (4, 4), color="black").save(image_bytes, format="PNG")

    response = client.post(
        "/v2/detection_pig_follicle",
        files=[("files", ("scan.png", image_bytes.getvalue(), "image/png"))],
    )

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["result"] == "pregnant"
    assert item["path_images"].endswith("gemini_scan.png")
    assert item["number_of_fetus"] == 2
    assert item["error_remark"] == ""


def test_v2_detection_pig_follicle_skips_gemini_when_not_pregnant(monkeypatch, tmp_path):
    calls = {"analyze": 0}

    def fake_save_input(img_cv2, filename):
        return str(tmp_path / f"saved_{filename}")

    def fake_detect(filename, save_path):
        return main.PregnancyDetectionOutcome(
            result=main.ImageResult(
                path_images=save_path,
                result="no pregnant",
                confidence=0.81,
                number_of_fetus=0,
                error_remark="",
            ),
            legacy_ai_label="2_NoPrenant_or_NotSure",
        )

    def fail_analyze(img_cv2):
        calls["analyze"] += 1
        raise AssertionError("Gemini should not be called")

    monkeypatch.setattr(main, "max_images", 1)
    monkeypatch.setattr(main, "save_detection_input", fake_save_input)
    monkeypatch.setattr(main, "detect_saved_pregnancy_image_with_backend", fake_detect)
    monkeypatch.setattr(main, "analyze_ultrasound_core", fail_analyze)
    client = TestClient(main.app)
    image_bytes = BytesIO()
    Image.new("RGB", (4, 4), color="black").save(image_bytes, format="PNG")

    response = client.post(
        "/v2/detection_pig_follicle",
        files=[("files", ("scan.png", image_bytes.getvalue(), "image/png"))],
    )

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["result"] == "no pregnant"
    assert item["path_images"].endswith("saved_scan.png")
    assert item["number_of_fetus"] == 0
    assert item["error_remark"] == ""
    assert calls["analyze"] == 0


def test_v2_detection_pig_follicle_skips_gemini_when_ensemble_disagrees(monkeypatch, tmp_path):
    calls = {"analyze": 0}

    def fake_save_input(img_cv2, filename):
        return str(tmp_path / f"saved_{filename}")

    def fake_detect(filename, save_path):
        return main.PregnancyDetectionOutcome(
            result=main.ImageResult(
                path_images=save_path,
                result="no pregnant",
                confidence=0.92,
                number_of_fetus=0,
                error_remark="",
            ),
            legacy_ai_label="2_NoPrenant_or_NotSure",
        )

    def fail_analyze(img_cv2):
        calls["analyze"] += 1
        raise AssertionError("Gemini should not be called")

    monkeypatch.setattr(main, "max_images", 1)
    monkeypatch.setattr(main, "save_detection_input", fake_save_input)
    monkeypatch.setattr(main, "detect_saved_pregnancy_image_with_backend", fake_detect)
    monkeypatch.setattr(main, "analyze_ultrasound_core", fail_analyze)
    client = TestClient(main.app)
    image_bytes = BytesIO()
    Image.new("RGB", (4, 4), color="black").save(image_bytes, format="PNG")

    response = client.post(
        "/v2/detection_pig_follicle",
        files=[("files", ("scan.png", image_bytes.getvalue(), "image/png"))],
    )

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["result"] == "no pregnant"
    assert item["path_images"].endswith("saved_scan.png")
    assert calls["analyze"] == 0


def test_v2_detection_pig_follicle_keeps_gate_output_when_gemini_annotation_not_usable(monkeypatch, tmp_path):
    def fake_save_input(img_cv2, filename):
        return str(tmp_path / f"saved_{filename}")

    def fake_detect(filename, save_path):
        return main.PregnancyDetectionOutcome(
            result=main.ImageResult(
                path_images=save_path,
                result="pregnant",
                confidence=0.93,
                number_of_fetus=0,
                error_remark="",
            ),
            legacy_ai_label="1_Pregnant",
        )

    def fake_analyze(img_cv2):
        return {
            "annotated_img": img_cv2,
            "sac_count": 0,
            "status": "2_NoPregnant",
            "detections": [],
        }

    monkeypatch.setattr(main, "max_images", 1)
    monkeypatch.setattr(main, "save_detection_input", fake_save_input)
    monkeypatch.setattr(main, "detect_saved_pregnancy_image_with_backend", fake_detect)
    monkeypatch.setattr(main, "analyze_ultrasound_core", fake_analyze)
    client = TestClient(main.app)
    image_bytes = BytesIO()
    Image.new("RGB", (4, 4), color="black").save(image_bytes, format="PNG")

    response = client.post(
        "/v2/detection_pig_follicle",
        files=[("files", ("scan.png", image_bytes.getvalue(), "image/png"))],
    )

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["result"] == "pregnant"
    assert item["path_images"].endswith("saved_scan.png")
    assert item["number_of_fetus"] == 0
    assert item["error_remark"] == "Gemini did not return usable follicle annotation"


def test_openapi_uses_binary_file_schema_for_v2_follicle_upload():
    client = TestClient(main.app)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    payload = response.json()
    request_schema = payload["paths"]["/v2/detection_pig_follicle"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    ref_name = request_schema["$ref"].split("/")[-1]
    files_schema = payload["components"]["schemas"][ref_name]["properties"]["files"]
    assert files_schema["type"] == "array"
    assert files_schema["items"]["type"] == "string"
    assert files_schema["items"]["format"] == "binary"
    assert "contentMediaType" not in files_schema["items"]


def test_legacy_detect_anomaly_route_is_removed():
    client = TestClient(main.app)

    response = client.post("/detect_anomaly/")

    assert response.status_code == 404


def test_detect_pregnancy_pic_route_is_removed():
    client = TestClient(main.app)

    response = client.post("/detect_pregnancy_pic/")

    assert response.status_code == 404


def test_detect_anomaly_pic_route_is_removed():
    client = TestClient(main.app)

    response = client.post("/detect_anomaly_pic/")

    assert response.status_code == 404


def test_detect_pregnancy_pdf_route_is_removed():
    client = TestClient(main.app)

    response = client.post("/detect_pregnancy_pdf/")

    assert response.status_code == 404


def test_detect_anomaly_pdf_route_is_removed():
    client = TestClient(main.app)

    response = client.post("/detect_anomaly_pdf/")

    assert response.status_code == 404


def test_upload_pdf_v2_returns_legacy_success_shape(monkeypatch, tmp_path):
    seen = {}

    def fake_process(pdf_path, source_name):
        seen["pdf_path"] = str(pdf_path)
        seen["source_name"] = source_name
        return True, None

    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(main, "build_image_filename", lambda *args, **kwargs: "generated_v2.pdf")
    monkeypatch.setattr(main, "process_v2_pdf_upload", fake_process)
    client = TestClient(main.app)

    response = client.post(
        "/v2/upload_pdf/",
        files={"file": ("scan.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "complete"}
    assert seen["pdf_path"] == str(tmp_path / "generated_v2.pdf")
    assert seen["source_name"] == "scan.pdf"


def test_process_v2_pdf_upload_maps_anomaly_to_legacy_db_fields(monkeypatch, tmp_path):
    saved_page = tmp_path / "page_001.png"
    calls = []

    def fake_render(pdf_path):
        assert pdf_path == tmp_path / "generated_v2.pdf"
        return [("scan.pdf#page=1", str(saved_page))]

    def fake_predict(path):
        assert path == saved_page
        return {
            "prediction": "pregnant",
            "score_no_pregnant": 0.12,
            "threshold": 0.5,
            "estimator_type": "sklearn_supervised",
        }

    def fake_insert(**kwargs):
        calls.append(kwargs)

    monkeypatch.setenv("PREGNANCY_DETECT_MODEL_V2", "anomaly")
    monkeypatch.setenv("INSERT_ULTRASOUND_TO_DB", "true")
    monkeypatch.setattr(main, "reload_runtime_config", lambda: None)
    monkeypatch.setattr(main, "render_pdf_pages_to_asset", fake_render)
    monkeypatch.setattr(main, "predict_anomaly_image", fake_predict)
    monkeypatch.setattr(main, "extract_info_from_image", lambda path: main.default_ocr_info())
    monkeypatch.setattr(main, "insert_ultrasound_to_db", fake_insert)

    ok, detail = main.process_v2_pdf_upload(tmp_path / "generated_v2.pdf", "scan.pdf")

    assert ok is True
    assert detail is None
    assert len(calls) == 1
    assert calls[0]["pdfFileName"] == "scan.pdf"
    assert calls[0]["path_val"] == main.asset_dir
    assert calls[0]["results_ai"] == "1_Pregnant"
    assert calls[0]["id_val"] == "IDUnknown"
    assert calls[0]["conf_score"] == 0.88


def test_process_v2_pdf_upload_keeps_legacy_yolo_label(monkeypatch, tmp_path):
    saved_page = tmp_path / "page_001.png"
    calls = []

    def fake_render(pdf_path):
        return [("scan.pdf#page=1", str(saved_page))]

    def fake_yolo(path):
        assert path == str(saved_page)
        return "2_NoPrenant_or_NotSure", 0.73

    def fake_insert(**kwargs):
        calls.append(kwargs)

    monkeypatch.setenv("PREGNANCY_DETECT_MODEL_V2", "yolo")
    monkeypatch.setenv("INSERT_ULTRASOUND_TO_DB", "true")
    monkeypatch.setattr(main, "reload_runtime_config", lambda: None)
    monkeypatch.setattr(main, "render_pdf_pages_to_asset", fake_render)
    monkeypatch.setattr(main, "preprocess_yolo", fake_yolo)
    monkeypatch.setattr(main, "extract_info_from_image", lambda path: main.default_ocr_info())
    monkeypatch.setattr(main, "insert_ultrasound_to_db", fake_insert)

    ok, detail = main.process_v2_pdf_upload(tmp_path / "generated_v2.pdf", "scan.pdf")

    assert ok is True
    assert detail is None
    assert len(calls) == 1
    assert calls[0]["results_ai"] == "2_NoPrenant_or_NotSure"
    assert calls[0]["path_val"] == main.asset_dir


def test_process_v2_pdf_upload_supports_ensemble_legacy_mapping(monkeypatch, tmp_path):
    saved_page = tmp_path / "page_001.png"
    calls = []

    def fake_render(pdf_path):
        return [("scan.pdf#page=1", str(saved_page))]

    def fake_detect(source_name, save_path):
        return main.PregnancyDetectionOutcome(
            result=main.ImageResult(
                path_images=save_path,
                result="no pregnant",
                confidence=0.92,
                number_of_fetus=0,
                error_remark="",
            ),
            legacy_ai_label="2_NoPrenant_or_NotSure",
        )

    def fake_insert(**kwargs):
        calls.append(kwargs)

    monkeypatch.setenv("PREGNANCY_DETECT_MODEL_V2", "ensemble")
    monkeypatch.setenv("INSERT_ULTRASOUND_TO_DB", "true")
    monkeypatch.setattr(main, "render_pdf_pages_to_asset", fake_render)
    monkeypatch.setattr(main, "detect_saved_pregnancy_image_with_backend", fake_detect)
    monkeypatch.setattr(main, "extract_info_from_image", lambda path: main.default_ocr_info())
    monkeypatch.setattr(main, "insert_ultrasound_to_db", fake_insert)

    ok, detail = main.process_v2_pdf_upload(tmp_path / "generated_v2.pdf", "scan.pdf")

    assert ok is True
    assert detail is None
    assert len(calls) == 1
    assert calls[0]["results_ai"] == "2_NoPrenant_or_NotSure"
    assert calls[0]["conf_score"] == 0.92
    assert calls[0]["path_val"] == main.asset_dir


def test_process_v2_pdf_upload_skips_db_insert_when_disabled(monkeypatch, tmp_path):
    saved_page = tmp_path / "page_001.png"
    calls = {"insert": 0}

    def fake_render(pdf_path):
        return [("scan.pdf#page=1", str(saved_page))]

    def fake_predict(path):
        return {
            "prediction": "pregnant",
            "score_no_pregnant": 0.12,
            "threshold": 0.5,
            "estimator_type": "sklearn_supervised",
        }

    def fake_insert(**kwargs):
        calls["insert"] += 1

    monkeypatch.setenv("PREGNANCY_DETECT_MODEL_V2", "anomaly")
    monkeypatch.setenv("INSERT_ULTRASOUND_TO_DB", "false")
    monkeypatch.setattr(main, "reload_runtime_config", lambda: None)
    monkeypatch.setattr(main, "render_pdf_pages_to_asset", fake_render)
    monkeypatch.setattr(main, "predict_anomaly_image", fake_predict)
    monkeypatch.setattr(main, "insert_ultrasound_to_db", fake_insert)

    ok, detail = main.process_v2_pdf_upload(tmp_path / "generated_v2.pdf", "scan.pdf")

    assert ok is True
    assert detail is None
    assert calls["insert"] == 0


def test_retrain_anomaly_starts_background_job(monkeypatch):
    seen = {}

    def fake_start(**kwargs):
        seen.update(kwargs)
        return {
            "job_id": "job-1",
            "status": "queued",
            "commands": [["python", "train_anomaly_models.py"]],
        }

    monkeypatch.setattr(
        main,
        "load_retrain_anomaly_config",
        lambda: main.AnomalyTrainRequest(
            feature_sets="handcrafted",
            model_keys="handcrafted__logistic_regression_balanced",
            batch_size=8,
            generate_report=False,
            detail_heatmaps="none",
            rebuild_index=True,
            force=False,
        ),
    )
    monkeypatch.setattr(main, "start_anomaly_training", fake_start)
    client = TestClient(main.app)

    response = client.post("/anomaly/retrain/")

    assert response.status_code == 202
    assert response.json()["job_id"] == "job-1"
    assert seen == {
        "feature_sets": "handcrafted",
        "model_keys": "handcrafted__logistic_regression_balanced",
        "batch_size": 8,
        "generate_report": False,
        "detail_heatmaps": "none",
        "rebuild_index": True,
        "force": False,
    }


def test_retrain_anomaly_rejects_running_job(monkeypatch):
    def fake_start(**kwargs):
        raise main.AnomalyTrainingAlreadyRunning("job-running")

    monkeypatch.setattr(main, "start_anomaly_training", fake_start)
    client = TestClient(main.app)

    response = client.post("/anomaly/retrain/")

    assert response.status_code == 409
    assert "job-running" in response.json()["detail"]


def test_retrain_anomaly_status_returns_current_job(monkeypatch):
    monkeypatch.setattr(
        main,
        "current_anomaly_training_job",
        lambda: {"job_id": "job-1", "status": "running"},
    )
    client = TestClient(main.app)

    response = client.get("/anomaly/retrain/status/")

    assert response.status_code == 200
    assert response.json() == {"job_id": "job-1", "status": "running"}


def test_openapi_retrain_has_no_request_body():
    client = TestClient(main.app)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    post_schema = response.json()["paths"]["/anomaly/retrain/"]["post"]
    assert "requestBody" not in post_schema

