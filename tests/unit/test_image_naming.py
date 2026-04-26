import re

from app.image_naming import build_image_filename


def test_build_image_filename_does_not_use_source_name():
    filename = build_image_filename("Gemini Upload")

    assert filename.endswith(".png")
    assert filename.startswith("gemini_upload_")
    assert "scan" not in filename
    assert re.match(r"^gemini_upload_\d{8}_\d{6}_\d{6}_[0-9a-f]{8}\.png$", filename)


def test_build_image_filename_includes_padded_page_number():
    filename = build_image_filename("pdf", page_number=3)

    assert re.match(r"^pdf_\d{8}_\d{6}_\d{6}_p003_[0-9a-f]{8}\.png$", filename)
