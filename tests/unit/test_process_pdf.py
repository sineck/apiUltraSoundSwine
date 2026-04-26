import numpy as np
from PIL import Image

from app import process_pdf


def test_correct_ocr_text_and_numbers_preserves_expected_digit_shape():
    assert process_pdf.correct_ocr_text_and_numbers("OIlbse") == "011688"


def test_ensure_yolo_model_ready_raises_clear_error(monkeypatch):
    monkeypatch.setattr(process_pdf, "model", None)
    monkeypatch.setattr(process_pdf, "MODEL_LOAD_ERROR", "missing file")

    try:
        process_pdf.ensure_yolo_model_ready()
    except process_pdf.ModelUnavailableError as exc:
        assert "YOLO model is not available" in str(exc)
        assert "missing file" in str(exc)
    else:
        raise AssertionError("ModelUnavailableError was not raised")


def test_extract_info_from_image_parses_ocr_text(monkeypatch):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    ocr_text = "\n".join(
        [
            "Time: 2026-04-26 10:30",
            "Pregnant: YES",
            "ID: M2O5l",
            "Depth: 160mm",
            "Gain: 70 dB",
        ]
    )

    monkeypatch.setattr(process_pdf.cv2, "imread", lambda path: image)
    monkeypatch.setattr(process_pdf.pytesseract, "image_to_string", lambda img: ocr_text)

    parsed = process_pdf.extract_info_from_image("dummy.png")

    assert parsed == (
        "2026-04-26 10:30",
        "2026-04-26 10:30",
        "YES",
        "M2051",
        "160mm",
        "70dB",
    )


def test_insert_ultrasound_to_db_uses_expected_table_and_values(monkeypatch):
    calls = {}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, values):
            calls["sql"] = sql
            calls["values"] = values

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            calls["committed"] = True

        def close(self):
            calls["closed"] = True

    def fake_connect(**kwargs):
        calls["connect_kwargs"] = kwargs
        return FakeConnection()

    monkeypatch.setenv("MYSQL_HOST", "db.local")
    monkeypatch.setenv("MYSQL_PORT", "3306")
    monkeypatch.setenv("MYSQL_DATABASE", "ultrasound")
    monkeypatch.setenv("MYSQL_USER", "tester")
    monkeypatch.setenv("MYSQL_PASSWORD", "secret")
    monkeypatch.setattr(process_pdf.pymysql, "connect", fake_connect)

    process_pdf.insert_ultrasound_to_db(
        create_date="created",
        workdate="workdate",
        time="time",
        pregnant_p="YES",
        id_val="M2051",
        pdfFileName="scan.pdf",
        depth_val="160mm",
        gain_val="70dB",
        path_val="app/asset",
        file_name="scan.png",
        results_ai="1_Pregnant",
        conf_score=0.91,
        cvcode=1,
        user_id=9,
    )

    assert "UltraSoudPigAI" in calls["sql"]
    assert calls["values"] == (
        "created",
        "workdate",
        "time",
        "YES",
        "M2051",
        "scan.pdf",
        "160mm",
        "70dB",
        "app/asset",
        "scan.png",
        "1_Pregnant",
        0.91,
        1,
        9,
    )
    assert calls["connect_kwargs"]["host"] == "db.local"
    assert calls["committed"] is True
    assert calls["closed"] is True


def test_should_insert_ultrasound_to_db_defaults_to_enabled(monkeypatch):
    monkeypatch.delenv("INSERT_ULTRASOUND_TO_DB", raising=False)

    assert process_pdf.should_insert_ultrasound_to_db() is True


def test_should_insert_ultrasound_to_db_can_be_disabled(monkeypatch):
    monkeypatch.setenv("INSERT_ULTRASOUND_TO_DB", "false")

    assert process_pdf.should_insert_ultrasound_to_db() is False


def test_default_ocr_info_matches_extract_info_fallback_shape():
    assert process_pdf.default_ocr_info() == (
        "The Text was not found.",
        "",
        "Unknown",
        "",
        "The Text was not found.",
        "The Text was not found.",
    )


def test_render_pdf_pages_falls_back_to_pymupdf(monkeypatch):
    class FakePixmap:
        width = 2
        height = 1
        samples = bytes([255, 0, 0, 0, 255, 0])

    class FakePage:
        def get_pixmap(self, matrix, alpha):
            return FakePixmap()

    class FakeDoc:
        def __iter__(self):
            return iter([FakePage()])

        def close(self):
            pass

    monkeypatch.setattr(
        process_pdf,
        "convert_from_path",
        lambda pdf_path: (_ for _ in ()).throw(RuntimeError("no poppler")),
    )
    monkeypatch.setattr(process_pdf.fitz, "open", lambda pdf_path: FakeDoc())

    pages = process_pdf.render_pdf_pages("scan.pdf")

    assert len(pages) == 1
    assert isinstance(pages[0], Image.Image)
    assert pages[0].size == (2, 1)


def test_convert_pdf_to_png_uses_idunknown_when_ocr_id_is_blank(monkeypatch):
    calls = []
    page = Image.new("RGB", (4, 4), color="black")

    def fake_build_image_filename(kind, page_number=None, extension=".png"):
        return f"pdf_page_{page_number:03d}.png"

    def fake_insert(**kwargs):
        calls.append(kwargs)

    monkeypatch.setenv("INSERT_ULTRASOUND_TO_DB", "true")
    monkeypatch.setattr(process_pdf, "render_pdf_pages", lambda path: [page, page])
    monkeypatch.setattr(process_pdf, "crop_real_image", lambda image: image)
    monkeypatch.setattr(process_pdf, "build_image_filename", fake_build_image_filename)
    monkeypatch.setattr(process_pdf.cv2, "imwrite", lambda path, image: True)
    monkeypatch.setattr(process_pdf, "preprocess_yolo", lambda path: ("pregnant", 0.91))
    monkeypatch.setattr(process_pdf, "extract_info_from_image", lambda path: process_pdf.default_ocr_info())
    monkeypatch.setattr(process_pdf, "insert_ultrasound_to_db", fake_insert)

    result = process_pdf.convert_pdf_to_png("scan.pdf")

    assert result is True
    assert [call["id_val"] for call in calls] == ["IDUnknown", "IDUnknown"]


def test_convert_pdf_to_png_skips_db_insert_when_disabled(monkeypatch):
    page = Image.new("RGB", (4, 4), color="black")
    called = {"insert": 0}

    def fake_insert(**kwargs):
        called["insert"] += 1

    monkeypatch.setenv("INSERT_ULTRASOUND_TO_DB", "false")
    monkeypatch.setattr(process_pdf, "render_pdf_pages", lambda path: [page])
    monkeypatch.setattr(process_pdf, "crop_real_image", lambda image: image)
    monkeypatch.setattr(process_pdf, "build_image_filename", lambda kind, page_number=None, extension=".png": "pdf_page_001.png")
    monkeypatch.setattr(process_pdf.cv2, "imwrite", lambda path, image: True)
    monkeypatch.setattr(process_pdf, "preprocess_yolo", lambda path: ("1_Pregnant", 0.91))
    monkeypatch.setattr(process_pdf, "extract_info_from_image", lambda path: process_pdf.default_ocr_info())
    monkeypatch.setattr(process_pdf, "insert_ultrasound_to_db", fake_insert)

    result = process_pdf.convert_pdf_to_png("scan.pdf")

    assert result is True
    assert called["insert"] == 0
