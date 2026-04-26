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


def test_health_includes_version_when_db_unreachable(monkeypatch):
    def fail_connect(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(main, "APP_NAME", "test-api")
    monkeypatch.setattr(main, "APP_VERSION", "9.9.9")
    monkeypatch.setattr(main.pymysql, "connect", fail_connect)
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["db"] == "unreachable"
    assert payload["app"] == {"name": "test-api", "version": "9.9.9"}


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


def test_detect_anomaly_returns_existing_detection_response_shape(monkeypatch, tmp_path):
    def fake_save(img_cv2, filename):
        return str(tmp_path / f"saved_{filename}")

    def fake_predict(path):
        return {
            "prediction": "no_pregnant",
            "score_no_pregnant": 0.91,
            "threshold": 0.5,
            "estimator_type": "sklearn_supervised",
        }

    monkeypatch.setattr(main, "max_images", 1)
    monkeypatch.setattr(main, "save_anomaly_input", fake_save)
    monkeypatch.setattr(main, "predict_anomaly_image", fake_predict)
    client = TestClient(main.app)
    image_bytes = BytesIO()
    Image.new("RGB", (4, 4), color="black").save(image_bytes, format="PNG")

    response = client.post(
        "/detect_anomaly/",
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


def test_detect_anomaly_returns_item_error_for_invalid_image(monkeypatch):
    monkeypatch.setattr(main, "max_images", 1)
    client = TestClient(main.app)

    response = client.post(
        "/detect_anomaly/",
        files=[("files", ("scan.png", b"not image", "image/png"))],
    )

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["path_images"] == "scan.png"
    assert item["result"] == "error"
    assert item["confidence"] == 0.0
    assert item["error_remark"] == "อ่านไฟล์ภาพไม่ได้"


def test_detect_anomaly_rejects_more_than_configured_max_images(monkeypatch):
    monkeypatch.setattr(main, "max_images", 1)
    client = TestClient(main.app)

    response = client.post(
        "/detect_anomaly/",
        files=[
            ("files", ("a.png", b"not image", "image/png")),
            ("files", ("b.png", b"not image", "image/png")),
        ],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "อัปโหลดได้สูงสุด 1 รูปต่อครั้ง (ส่งมา 2 รูป)"
