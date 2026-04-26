import json

import numpy as np

from app import process_detection as detection


class _GeminiResponse:
    def __init__(self, text):
        self.text = text


class _GeminiModels:
    def __init__(self, text):
        self._text = text

    def generate_content(self, **kwargs):
        return _GeminiResponse(self._text)


class _GeminiClient:
    def __init__(self, text):
        self.models = _GeminiModels(text)


class _FailingGeminiModels:
    def generate_content(self, **kwargs):
        raise RuntimeError("service unavailable")


class _FailingGeminiClient:
    models = _FailingGeminiModels()


def test_scale_bbox_converts_normalized_coordinates_to_pixels():
    assert detection.scale_bbox([100, 200, 900, 800], img_width=1000, img_height=500) == (
        200,
        50,
        800,
        450,
    )


def test_analyze_ultrasound_core_marks_valid_detection_as_pregnant(monkeypatch):
    payload = [{"box_2d": [100, 200, 600, 700], "label": "gestational sac", "confidence": 0.95}]
    monkeypatch.setattr(detection, "client", _GeminiClient(json.dumps(payload)))

    image = np.zeros((120, 160, 3), dtype=np.uint8)
    result = detection.analyze_ultrasound_core(image)

    assert result["status"] == "1_Pregnant"
    assert result["sac_count"] == 1
    assert result["detections"][0]["label"] == "gestational sac"
    assert result["annotated_img"].shape == image.shape


def test_analyze_ultrasound_core_marks_empty_detection_as_not_pregnant(monkeypatch):
    monkeypatch.setattr(detection, "client", _GeminiClient("[]"))

    image = np.zeros((120, 160, 3), dtype=np.uint8)
    result = detection.analyze_ultrasound_core(image)

    assert result["status"] == "2_NoPregnant"
    assert result["sac_count"] == 0
    assert result["detections"] == []


def test_analyze_ultrasound_core_raises_when_gemini_fails(monkeypatch):
    monkeypatch.setattr(detection, "client", _FailingGeminiClient())

    image = np.zeros((120, 160, 3), dtype=np.uint8)

    try:
        detection.analyze_ultrasound_core(image)
    except detection.GeminiAnalysisError as exc:
        assert "service unavailable" in str(exc)
    else:
        raise AssertionError("GeminiAnalysisError was not raised")


def test_analyze_ultrasound_core_raises_when_client_unavailable(monkeypatch):
    monkeypatch.setattr(detection, "client", None)
    monkeypatch.setattr(detection, "GEMINI_CLIENT_ERROR", "missing key")

    image = np.zeros((120, 160, 3), dtype=np.uint8)

    try:
        detection.analyze_ultrasound_core(image)
    except detection.GeminiAnalysisError as exc:
        assert "missing key" in str(exc)
    else:
        raise AssertionError("GeminiAnalysisError was not raised")


def test_format_result_maps_internal_status_to_response_model():
    ai_result = {
        "status": "1_Pregnant",
        "sac_count": 2,
        "detections": [{"confidence": 0.8}, {"confidence": 1.0}],
    }

    response = detection.Format_Result("source.png", ai_result, "app/detections/result.png")

    assert response.path_images == "app/detections/result.png"
    assert response.result == "pregnant"
    assert response.confidence == 0.9
    assert response.number_of_fetus == 2
    assert response.error_remark == ""
