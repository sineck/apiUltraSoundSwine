from app import process_anomaly


def test_format_anomaly_result_maps_no_pregnant_to_api_schema():
    prediction = {
        "prediction": "no_pregnant",
        "score_no_pregnant": 0.83,
        "estimator_type": "sklearn_supervised",
    }

    result = process_anomaly.format_anomaly_result("scan.png", prediction, "app/detections/scan.png")

    assert result.path_images == "app/detections/scan.png"
    assert result.result == "no pregnant"
    assert result.confidence == 0.83
    assert result.number_of_fetus == 0
    assert result.error_remark == ""


def test_format_anomaly_result_maps_pregnant_confidence_from_no_pregnant_score():
    prediction = {
        "prediction": "pregnant",
        "score_no_pregnant": 0.12,
        "estimator_type": "sklearn_supervised",
    }

    result = process_anomaly.format_anomaly_result("scan.png", prediction, "app/detections/scan.png")

    assert result.result == "pregnant"
    assert result.confidence == 0.88


def test_resolve_active_anomaly_model_falls_back_from_windows_registry_path(tmp_path):
    run_dir = tmp_path / "20260426_113305"
    run_dir.mkdir()
    model_file = run_dir / "active.joblib"
    model_file.write_bytes(b"model")
    registry = tmp_path / "model_registry.json"
    registry.write_text(
        """
        {
          "active_model": "20260426_113305/dinov2__logistic_regression_balanced",
          "runs": {
            "20260426_113305": {
              "models": {
                "dinov2__logistic_regression_balanced": {
                  "model_file": "D:\\\\apiUltraSoundSwine\\\\AnomalyDetection\\\\artifacts\\\\models\\\\20260426_113305\\\\active.joblib"
                }
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )

    assert process_anomaly.resolve_active_anomaly_model(registry) == model_file
