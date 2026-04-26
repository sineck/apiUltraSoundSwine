import numpy as np
from PIL import Image

from app import process_pdf


def test_correct_ocr_text_and_numbers_preserves_expected_digit_shape():
    assert process_pdf.correct_ocr_text_and_numbers("OIlbse") == "011688"


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
