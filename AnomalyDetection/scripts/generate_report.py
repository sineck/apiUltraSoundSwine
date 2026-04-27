from __future__ import annotations

import argparse
import html
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from AnomalyDetection.scripts.anomaly_lib import (
    ImageRow,
    clean_ultrasound_image,
    discover_dataset,
    extract_patch_handcrafted,
    get_feature_matrix,
    label_prediction,
    load_json,
    load_model_bundle,
    predict_from_scores,
    score_bundle,
    score_patch_bundle,
    ultrasound_sector_mask,
    write_json,
)


COMPACT_MODEL_KEYS = [
    "handcrafted__logistic_regression_balanced",
    "resnet18__logistic_regression_balanced",
    "handcrafted__normal_quantile",
    "dinov2__random_forest_balanced",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate an HTML model comparison report with heatmaps.")
    parser.add_argument("--registry", type=Path, default=root / "artifacts" / "models" / "model_registry.json")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "report")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--top-heatmap-cells", type=int, default=8)
    parser.add_argument("--detail-heatmaps", choices=["none", "active", "all"], default="active")
    parser.add_argument(
        "--model-keys",
        default=",".join(COMPACT_MODEL_KEYS),
        help="Comma-separated model keys to include, or 'all'. Default is the compact 4-model set.",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    return parser.parse_args()


def load_active_run(registry_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    registry = load_json(registry_path)
    run_name, _ = registry["active_model"].split("/", 1)
    run = registry["runs"][run_name]
    experiment_path = Path(run["experiment_results"])
    experiment = load_json(experiment_path)
    return registry, experiment, experiment_path.parent


def choose_sample_images(rows: list[ImageRow]) -> list[ImageRow]:
    samples: list[ImageRow] = []
    for label_name in ("1_Pregnant", "2_NoPregnant"):
        match = next((row for row in rows if row.split == "test" and row.label_name == label_name), None)
        if match:
            samples.append(match)
    return samples


def read_color_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def make_occlusion_rows(sample: ImageRow, temp_dir: Path, grid_size: int) -> tuple[list[ImageRow], list[tuple[int, int]], np.ndarray]:
    original = clean_ultrasound_image(sample.path)
    height, width = original.shape[:2]
    baseline = int(np.median(original))
    rows = [ImageRow(path=sample.path, split="heatmap", label_name=sample.label_name, target=sample.target)]
    cells: list[tuple[int, int]] = []

    for row_idx in range(grid_size):
        y1 = int(row_idx * height / grid_size)
        y2 = int((row_idx + 1) * height / grid_size)
        for col_idx in range(grid_size):
            x1 = int(col_idx * width / grid_size)
            x2 = int((col_idx + 1) * width / grid_size)
            occluded = original.copy()
            occluded[y1:y2, x1:x2] = baseline
            occluded_path = temp_dir / f"{sample.path.stem}_{row_idx}_{col_idx}.png"
            write_image(occluded_path, occluded)
            rows.append(ImageRow(path=occluded_path, split="heatmap", label_name=sample.label_name, target=sample.target))
            cells.append((row_idx, col_idx))
    return rows, cells, original


def build_overlay(original: np.ndarray, heatmap_grid: np.ndarray, out_path: Path, top_cells: int) -> None:
    height, width = original.shape[:2]
    if top_cells > 0:
        positive = heatmap_grid[heatmap_grid > 0]
        if len(positive) > top_cells:
            cutoff = np.sort(positive)[-top_cells]
            heatmap_grid = np.where(heatmap_grid >= cutoff, heatmap_grid, 0)
    heatmap = cv2.resize(heatmap_grid.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    if float(heatmap.max()) > float(heatmap.min()):
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)
    sector_mask = ultrasound_sector_mask(original)
    visible_mask = (cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) > 8).astype(np.uint8)
    heatmap = heatmap * sector_mask * visible_mask
    heatmap_u8 = np.uint8(np.clip(heatmap * 255.0, 0, 255))
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    overlay = original.copy()
    blend_mask = heatmap > 0.02
    blended = cv2.addWeighted(original, 0.62, colored, 0.38, 0)
    overlay[blend_mask] = blended[blend_mask]
    write_image(out_path, overlay)


def score_sample_with_heatmap(
    bundle: dict[str, Any],
    features: Any,
    cells: list[tuple[int, int]],
    grid_size: int,
) -> tuple[float, str, np.ndarray]:
    if bundle["feature_set"] == "patch_handcrafted":
        scores = score_patch_bundle(bundle, features)
    else:
        scores = score_bundle(bundle, features)
    base_score = float(scores[0])
    prediction_target = int(predict_from_scores(np.asarray([base_score]), float(bundle["threshold"]))[0])
    base_prediction = label_prediction(prediction_target)
    deltas = np.abs(scores[1:] - base_score)
    heatmap_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for delta, (row_idx, col_idx) in zip(deltas, cells):
        heatmap_grid[row_idx, col_idx] = float(delta)
    return base_score, base_prediction, heatmap_grid


def css() -> str:
    return """
:root {
  --bg: #f5f3ef;
  --panel: #ffffff;
  --text: #202124;
  --muted: #65605a;
  --line: #d8d2c8;
  --accent: #0f766e;
  --accent-soft: #dff3ee;
  --warn: #b45309;
  --good: #15803d;
  --bad: #b91c1c;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Arial, Helvetica, sans-serif;
  background: var(--bg);
  color: var(--text);
}
header {
  padding: 30px 32px 22px;
  border-bottom: 1px solid var(--line);
  background: #fffaf2;
}
h1 { margin: 0 0 8px; font-size: 30px; line-height: 1.15; letter-spacing: 0; }
h2 { margin: 26px 0 12px; font-size: 21px; letter-spacing: 0; }
h3 { margin: 0 0 8px; font-size: 17px; letter-spacing: 0; }
p { margin: 0; color: var(--muted); line-height: 1.55; }
main { padding: 22px 32px 42px; max-width: 1480px; margin: 0 auto; }
.summary {
  display: grid;
  grid-template-columns: repeat(4, minmax(160px, 1fr));
  gap: 12px;
  margin: 18px 0 8px;
}
.metric, .method, .panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.metric strong { display: block; font-size: 22px; margin-top: 6px; }
.muted { color: var(--muted); }
table {
  width: 100%;
  border-collapse: collapse;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}
th, td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
  font-size: 14px;
}
th { background: #ece7dd; color: #26221f; }
tr:last-child td { border-bottom: 0; }
tbody tr:hover { background: #fbfaf7; }
.badge {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 700;
  font-size: 12px;
}
.badge.warn { background: #fff1d6; color: var(--warn); }
.badge.good { background: #e8f5e9; color: var(--good); }
.methods {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(390px, 1fr));
  gap: 16px;
}
.method.active { border-color: var(--accent); box-shadow: inset 0 3px 0 var(--accent); }
.method-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}
.figures {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
figure { margin: 0; }
img {
  display: block;
  width: 100%;
  aspect-ratio: 4 / 3;
  object-fit: cover;
  border-radius: 6px;
  border: 1px solid var(--line);
  background: #111;
}
figcaption {
  margin-top: 6px;
  font-size: 12px;
  line-height: 1.35;
  color: var(--muted);
}
.small { font-size: 12px; }
.pass { color: var(--good); font-weight: 700; }
.fail { color: var(--bad); font-weight: 700; }
.actions { white-space: nowrap; }
.button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 30px;
  padding: 6px 10px;
  border-radius: 6px;
  background: var(--accent);
  color: #fff;
  text-decoration: none;
  font-size: 12px;
  font-weight: 700;
}
.button.secondary {
  background: #eee8dc;
  color: #2d2a26;
}
.note {
  margin-top: 10px;
  padding: 10px 12px;
  border-left: 4px solid var(--accent);
  background: #f8fbf8;
  color: var(--muted);
}
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
}
.thumb {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
}
.thumb img { aspect-ratio: 4 / 3; }
.thumb-pair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.thumb-pair figcaption { margin-top: 4px; }
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
@media (max-width: 760px) {
  header, main { padding-left: 16px; padding-right: 16px; }
  .summary { grid-template-columns: 1fr 1fr; }
  .methods { grid-template-columns: 1fr; }
  .figures { grid-template-columns: 1fr; }
  .thumb-pair { grid-template-columns: 1fr; }
  .topbar { display: block; }
}
"""


def metric_cell(value: float) -> str:
    return f"{value:.4f}"


def optional_metric_cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def method_rank(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        results,
        key=lambda item: (
            (item["validate"]["balanced_accuracy"] + item["test"]["balanced_accuracy"]) / 2.0,
            item["test"]["balanced_accuracy"],
        ),
        reverse=True,
    )


def selected_model_keys(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def filter_results(results: list[dict[str, Any]], model_keys: set[str] | None) -> list[dict[str, Any]]:
    if model_keys is None:
        return results
    filtered = [item for item in results if item["model_key"] in model_keys]
    missing = sorted(model_keys.difference({item["model_key"] for item in filtered}))
    if missing:
        raise SystemExit(f"Missing selected model keys in experiment results: {', '.join(missing)}")
    return filtered


def expected_label(actual_target: int) -> str:
    return "pregnant" if int(actual_target) == 1 else "no_pregnant"


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    y_true = np.asarray([0 if row["actual"] == "no_pregnant" else 1 for row in rows], dtype=np.int64)
    y_pred = np.asarray([0 if row["prediction"] == "no_pregnant" else 1 for row in rows], dtype=np.int64)
    no_pregnant_scores = np.asarray([row["score_no_pregnant"] for row in rows], dtype=np.float64)
    no_pregnant_true = (y_true == 0).astype(np.int64)
    auc_no_pregnant: float | None
    if len(np.unique(no_pregnant_true)) == 2:
        auc_no_pregnant = float(roc_auc_score(no_pregnant_true, no_pregnant_scores))
    else:
        auc_no_pregnant = None

    return {
        "auc_no_pregnant": auc_no_pregnant,
        "precision_no_pregnant": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_no_pregnant": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_pregnant": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_pregnant": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def model_page_name(model_key: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in model_key)
    return f"{safe}.html"


def build_model_test_results(model_result: dict[str, Any]) -> dict[str, Any]:
    model_key = model_result["model_key"]
    predictions_path = Path(model_result["model_file"]).with_name(f"{model_key}_predictions_test.json")
    records = load_json(predictions_path)
    rows = []
    correct = 0
    for record in records:
        actual = expected_label(record["actual_target"])
        prediction = record["prediction"]
        passed = prediction == actual
        if passed:
            correct += 1
        rows.append(
            {
                "filename": Path(record["path"]).name,
                "source_path": record["path"],
                "actual": actual,
                "prediction": prediction,
                "score_no_pregnant": float(record["score_no_pregnant"]),
                "threshold": float(record["threshold"]),
                "passed": passed,
            }
        )
    return {
        "model_key": model_key,
        "predictions_file": str(predictions_path),
        "total": len(rows),
        "correct": correct,
        "incorrect": len(rows) - correct,
        "metrics": classification_metrics(rows),
        "rows": rows,
    }


def build_test_thumbnails(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnails: dict[str, str] = {}
    for idx, row in enumerate(rows, start=1):
        source_path = Path(row["source_path"])
        image = clean_ultrasound_image(source_path)
        height, width = image.shape[:2]
        target_width = 360
        target_height = max(1, int(height * target_width / max(width, 1)))
        thumbnail = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        safe_stem = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in source_path.stem)[:80]
        out_path = output_dir / f"{idx:03d}_{safe_stem}.jpg"
        write_image(out_path, thumbnail)
        thumbnails[str(source_path)] = str(out_path)
    return thumbnails


def rows_to_image_rows(rows: list[dict[str, Any]]) -> list[ImageRow]:
    return [
        ImageRow(
            path=Path(row["source_path"]),
            split="test",
            label_name=str(row["actual"]),
            target=0 if row["actual"] == "no_pregnant" else 1,
        )
        for row in rows
    ]


def build_detail_heatmaps(
    model_result: dict[str, Any],
    bundle: dict[str, Any],
    rows: list[dict[str, Any]],
    output_dir: Path,
    grid_size: int,
    top_cells: int,
    batch_size: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmaps: dict[str, str] = {}
    feature_set = model_result["feature_set"]
    with tempfile.TemporaryDirectory() as temp_name:
        temp_dir = Path(temp_name)
        for idx, row in enumerate(rows_to_image_rows(rows), start=1):
            rows_for_heatmap, cells, original = make_occlusion_rows(row, temp_dir / f"{idx:03d}", grid_size)
            if feature_set == "patch_handcrafted":
                features = extract_patch_handcrafted(rows_for_heatmap)
            else:
                features = get_feature_matrix(rows_for_heatmap, feature_set, batch_size=batch_size)
            _, _, heatmap_grid = score_sample_with_heatmap(bundle, features, cells, grid_size)
            safe_stem = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in row.path.stem)[:80]
            out_path = output_dir / f"{idx:03d}_{safe_stem}_heatmap.jpg"
            build_overlay(original, heatmap_grid, out_path, top_cells)
            heatmaps[str(row.path)] = str(out_path)
    return heatmaps


def render_model_detail_html(
    model_result: dict[str, Any],
    test_results: dict[str, Any],
    thumbnails: dict[str, str],
    detail_heatmaps: dict[str, dict[str, str]],
    page_dir: Path,
) -> str:
    cards = []
    model_heatmaps = detail_heatmaps.get(model_result["model_key"], {})
    for row in test_results["rows"]:
        thumb = thumbnails.get(row["source_path"], "")
        image_rel = Path(os.path.relpath(thumb, page_dir)).as_posix() if thumb else ""
        heatmap = model_heatmaps.get(row["source_path"], "")
        heatmap_rel = Path(os.path.relpath(heatmap, page_dir)).as_posix() if heatmap else ""
        status_class = "pass" if row["passed"] else "fail"
        status_label = "ถูก" if row["passed"] else "ผิด"
        if heatmap_rel:
            image_html = (
                "<div class='thumb-pair'>"
                f"<figure><img src='{html.escape(image_rel)}' alt='Input {html.escape(row['filename'])}'><figcaption>ภาพที่โมเดลเห็น</figcaption></figure>"
                f"<figure><img src='{html.escape(heatmap_rel)}' alt='Heatmap {html.escape(row['filename'])}'><figcaption>Heatmap</figcaption></figure>"
                "</div>"
            )
        else:
            image_html = f"<img src='{html.escape(image_rel)}' alt='{html.escape(row['filename'])}'>"
        cards.append(
            "<article class='thumb'>"
            f"{image_html}"
            f"<h3>{html.escape(row['filename'])}</h3>"
            f"<p class='small'>จริง: <b>{html.escape(row['actual'])}</b><br>"
            f"ทำนาย: <b>{html.escape(row['prediction'])}</b><br>"
            f"score no-pregnant: {row['score_no_pregnant']:.6g}<br>"
            f"threshold: {row['threshold']:.6g}<br>"
            f"ผล: <span class='{status_class}'>{status_label}</span></p>"
            "</article>"
        )

    return f"""<!doctype html>
<html lang="th">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(model_result['model_key'])} - Test Images</title>
  <style>{css()}</style>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <h1>{html.escape(model_result['model_key'])}</h1>
        <p>รูปทั้งหมดในชุด test ของโมเดลนี้ ใช้ภาพที่ลบ overlay แล้วเหมือนตอน train/predict</p>
      </div>
      <a class="button" href="../index.html">กลับหน้าสรุป</a>
    </div>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><span class="muted">Feature</span><strong>{html.escape(model_result['feature_set'])}</strong></div>
      <div class="metric"><span class="muted">Test balanced</span><strong>{metric_cell(model_result['test']['balanced_accuracy'])}</strong></div>
      <div class="metric"><span class="muted">AUC ไม่ท้อง</span><strong>{optional_metric_cell(test_results['metrics']['auc_no_pregnant'])}</strong></div>
      <div class="metric"><span class="muted">Recall ไม่ท้อง</span><strong>{metric_cell(test_results['metrics']['recall_no_pregnant'])}</strong></div>
      <div class="metric"><span class="muted">ถูก</span><strong>{test_results['correct']} / {test_results['total']}</strong></div>
      <div class="metric"><span class="muted">ผิด</span><strong>{test_results['incorrect']}</strong></div>
    </section>
    <p class="note">ค่าคะแนน `score no-pregnant` ยิ่งสูง หมายถึงโมเดลเอนเอียงไปทางไม่ท้องมากขึ้น เมื่อคะแนนสูงกว่า threshold จะทำนายเป็น no_pregnant</p>
    <p class="note">ถ้าการ์ดมี Heatmap แสดงว่ารูปนั้นถูกคำนวณ occlusion heatmap แล้ว สีสว่างคือบริเวณที่ปิดทับแล้วคะแนนเปลี่ยนมาก</p>
    <h2>รูปทั้งหมดของโมเดลนี้</h2>
    <section class="gallery">{''.join(cards)}</section>
  </main>
</body>
</html>
"""


def write_model_detail_pages(
    ranked_results: list[dict[str, Any]],
    model_tests: dict[str, dict[str, Any]],
    thumbnails: dict[str, str],
    detail_heatmaps: dict[str, dict[str, str]],
    output_dir: Path,
) -> None:
    page_dir = output_dir / "models"
    page_dir.mkdir(parents=True, exist_ok=True)
    for result in ranked_results:
        page_path = page_dir / model_page_name(result["model_key"])
        page_path.write_text(
            render_model_detail_html(result, model_tests[result["model_key"]], thumbnails, detail_heatmaps, page_dir),
            encoding="utf-8",
        )


def render_html(report: dict[str, Any], output_dir: Path) -> str:
    active_model = report["active_model"]
    rows = []
    for item in report["ranked_results"]:
        test_metrics = report["model_test_results"][item["model_key"]]["metrics"]
        active_badge = " <span class='badge good'>ACTIVE</span>" if item["model_key"] == active_model else ""
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['model_key'])}{active_badge}</td>"
            f"<td>{html.escape(item['feature_set'])}</td>"
            f"<td>{metric_cell(item['validate']['balanced_accuracy'])}</td>"
            f"<td>{metric_cell(item['test']['balanced_accuracy'])}</td>"
            f"<td>{optional_metric_cell(test_metrics['auc_no_pregnant'])}</td>"
            f"<td>{metric_cell(test_metrics['recall_no_pregnant'])}</td>"
            f"<td>{metric_cell(test_metrics['precision_no_pregnant'])}</td>"
            f"<td>{metric_cell(item['test']['f1_no_pregnant'])}</td>"
            f"<td>{html.escape(str(item['test']['confusion_matrix']))}</td>"
            f"<td>{item['threshold']:.6g}</td>"
            f"<td class='actions'><a class='button secondary' href='models/{html.escape(model_page_name(item['model_key']))}'>ดูรูปทั้งหมด</a></td>"
            "</tr>"
        )

    method_sections = []
    for method in report["methods"]:
        active_class = " active" if method["model_key"] == active_model else ""
        active_badge = "<span class='badge good'>ACTIVE</span>" if method["model_key"] == active_model else ""
        figures = []
        for sample in method["samples"]:
            image_rel = Path(sample["heatmap"]).relative_to(output_dir).as_posix()
            actual = html.escape(sample["actual_label"])
            prediction = html.escape(sample["prediction"])
            score = sample["score_no_pregnant"]
            figures.append(
                "<figure>"
                f"<img src='{html.escape(image_rel)}' alt='Heatmap {html.escape(method['model_key'])} {actual}'>"
                f"<figcaption>จริง: {actual}<br>ทำนาย: <b>{prediction}</b>, score: {score:.4f}</figcaption>"
                "</figure>"
            )
        method_sections.append(
            f"<article class='method{active_class}'>"
            "<div class='method-head'>"
            f"<div><h3>{html.escape(method['model_key'])}</h3>"
            f"<p class='small'>{html.escape(method['score_meaning'])}</p></div>{active_badge}"
            "</div>"
            f"<p class='small'>validate balanced: {metric_cell(method['validate_balanced_accuracy'])} | "
            f"test balanced: {metric_cell(method['test_balanced_accuracy'])} | threshold: {method['threshold']:.6g}</p>"
            f"<div class='figures'>{''.join(figures)}</div>"
            "</article>"
        )

    dataset_bits = []
    for split, labels in report["dataset_summary"].items():
        label_text = ", ".join(f"{label}: {count}" for label, count in labels.items())
        dataset_bits.append(f"{split}: {label_text}")

    cleaned_inputs = []
    for sample in report["cleaned_inputs"]:
        image_rel = Path(sample["image"]).relative_to(output_dir).as_posix()
        cleaned_inputs.append(
            "<figure>"
            f"<img src='{html.escape(image_rel)}' alt='Cleaned input {html.escape(sample['label'])}'>"
            f"<figcaption>{html.escape(sample['label'])}<br>ภาพหลังลบข้อความ/ตัวเลข overlay</figcaption>"
            "</figure>"
        )

    active_test = report["active_test_results"]
    active_metrics = active_test["metrics"]
    active_rows = []
    for row in active_test["rows"]:
        status_class = "pass" if row["passed"] else "fail"
        status_label = "ถูก" if row["passed"] else "ผิด"
        active_rows.append(
            "<tr>"
            f"<td>{html.escape(row['filename'])}</td>"
            f"<td>{html.escape(row['actual'])}</td>"
            f"<td>{html.escape(row['prediction'])}</td>"
            f"<td>{row['score_no_pregnant']:.6g}</td>"
            f"<td>{row['threshold']:.6g}</td>"
            f"<td class='{status_class}'>{status_label}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="th">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>รายงานโมเดลตรวจท้องสุกร</title>
  <style>{css()}</style>
</head>
<body>
  <header>
    <h1>รายงานโมเดลตรวจท้องสุกรจากภาพ Ultrasound</h1>
    <p>อัปเดตเมื่อ {html.escape(report['generated_at'])}. สีบน heatmap คือบริเวณที่พอปิดทับแล้วคะแนนของโมเดลเปลี่ยนมาก จึงเป็นจุดที่โมเดลน่าจะใช้ตัดสินใจ ไม่ใช่คำวินิจฉัยทางการแพทย์โดยตรง</p>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><span class="muted">โมเดลที่แนะนำ</span><strong>{html.escape(active_model)}</strong></div>
      <div class="metric"><span class="muted">คะแนน validate</span><strong>{metric_cell(report['best']['validate']['balanced_accuracy'])}</strong></div>
      <div class="metric"><span class="muted">คะแนน test</span><strong>{metric_cell(report['best']['test']['balanced_accuracy'])}</strong></div>
      <div class="metric"><span class="muted">AUC ไม่ท้อง</span><strong>{optional_metric_cell(active_metrics['auc_no_pregnant'])}</strong></div>
      <div class="metric"><span class="muted">Recall ไม่ท้อง</span><strong>{metric_cell(active_metrics['recall_no_pregnant'])}</strong></div>
      <div class="metric"><span class="muted">Precision ไม่ท้อง</span><strong>{metric_cell(active_metrics['precision_no_pregnant'])}</strong></div>
      <div class="metric"><span class="muted">F1 ไม่ท้อง</span><strong>{metric_cell(report['best']['test']['f1_no_pregnant'])}</strong></div>
    </section>
    <p class="small">{html.escape(' | '.join(dataset_bits))}</p>
    <p class="note">อ่านแบบเร็ว: ดูโมเดลที่แนะนำด้านบนก่อน ถ้าต้องการตรวจรูปทั้งหมดของวิธีใด ให้กดปุ่ม “ดูรูปทั้งหมด” ในตารางเปรียบเทียบ</p>

    <h2>เปรียบเทียบทุกวิธี</h2>
    <table>
      <thead>
        <tr>
          <th>วิธี</th>
          <th>ตัวเข้ารหัสภาพ</th>
          <th>Validate</th>
          <th>Test</th>
          <th>AUC ไม่ท้อง</th>
          <th>Recall ไม่ท้อง</th>
          <th>Precision ไม่ท้อง</th>
          <th>F1 ไม่ท้อง</th>
          <th>Confusion Matrix</th>
          <th>Threshold</th>
          <th>ดูภาพ</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>

    <h2>ผลทดสอบของโมเดลที่เลือก</h2>
    <section class="metric">
      <p><b>{html.escape(active_test['model_key'])}</b> ทดสอบกับรูปในชุด test ที่มี label ทั้งหมด {active_test['total']} รูป: ทายถูก {active_test['correct']} รูป, ทายผิด {active_test['incorrect']} รูป</p>
      <p class="small" style="margin-top:8px;">AUC ไม่ท้อง: <b>{optional_metric_cell(active_metrics['auc_no_pregnant'])}</b> | Recall ไม่ท้อง: <b>{metric_cell(active_metrics['recall_no_pregnant'])}</b> | Precision ไม่ท้อง: <b>{metric_cell(active_metrics['precision_no_pregnant'])}</b> | Recall ท้อง: <b>{metric_cell(active_metrics['recall_pregnant'])}</b></p>
      <table style="margin-top:12px;">
        <thead>
          <tr>
            <th>ไฟล์</th>
            <th>ค่าจริง</th>
            <th>โมเดลทำนาย</th>
            <th>Score No Pregnant</th>
            <th>Threshold</th>
            <th>ผล</th>
          </tr>
        </thead>
        <tbody>{''.join(active_rows)}</tbody>
      </table>
    </section>

    <h2>ภาพที่โมเดลเห็นจริง</h2>
    <section class="figures">{''.join(cleaned_inputs)}</section>

    <h2>จุดที่แต่ละวิธีใช้ตัดสินใจ</h2>
    <section class="methods">{''.join(method_sections)}</section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    registry, experiment, run_dir = load_active_run(args.registry)
    rows = discover_dataset(Path(experiment["asset_dir"]))
    samples = choose_sample_images(rows)
    if len(samples) < 2:
        raise SystemExit("Need at least one pregnant and one no-pregnant test image for report samples.")

    run_name, active_model = registry["active_model"].split("/", 1)
    output_dir = args.output_dir
    report_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    heatmap_dir = output_dir / f"heatmaps_{run_name}_{report_stamp}"
    detail_heatmap_dir = output_dir / f"detail_heatmaps_{run_name}_{report_stamp}"
    input_dir = output_dir / f"cleaned_inputs_{run_name}_{report_stamp}"
    thumbnail_dir = output_dir / f"test_thumbnails_{run_name}_{report_stamp}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    model_keys = selected_model_keys(args.model_keys)
    ranked_results = method_rank(filter_results(experiment["results"], model_keys))
    active_result = next(item for item in ranked_results if item["model_key"] == active_model)
    model_tests = {item["model_key"]: build_model_test_results(item) for item in ranked_results}
    thumbnails = build_test_thumbnails(model_tests[active_model]["rows"], thumbnail_dir)
    model_refs = registry["runs"][run_name]["models"]
    bundles = {item["model_key"]: load_model_bundle(Path(model_refs[item["model_key"]]["model_file"])) for item in ranked_results}
    detail_heatmaps: dict[str, dict[str, str]] = {}
    if args.detail_heatmaps != "none":
        detail_items = [active_result] if args.detail_heatmaps == "active" else ranked_results
        for item in detail_items:
            detail_heatmaps[item["model_key"]] = build_detail_heatmaps(
                item,
                bundles[item["model_key"]],
                model_tests[item["model_key"]]["rows"],
                detail_heatmap_dir / model_page_name(item["model_key"]).removesuffix(".html"),
                args.grid_size,
                args.top_heatmap_cells,
                args.batch_size,
            )
    write_model_detail_pages(ranked_results, model_tests, thumbnails, detail_heatmaps, output_dir)

    heatmap_feature_cache: dict[tuple[str, str], tuple[np.ndarray, list[tuple[int, int]], np.ndarray]] = {}
    methods: list[dict[str, Any]] = []
    cleaned_inputs: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as temp_name:
        temp_dir = Path(temp_name)
        for sample in samples:
            cleaned_path = input_dir / f"{sample.label_name}.png"
            write_image(cleaned_path, clean_ultrasound_image(sample.path))
            cleaned_inputs.append({"label": sample.label_name, "image": str(cleaned_path)})
            for feature_set in sorted({item["feature_set"] for item in ranked_results}):
                rows_for_heatmap, cells, original = make_occlusion_rows(sample, temp_dir / feature_set, args.grid_size)
                if feature_set == "patch_handcrafted":
                    features = extract_patch_handcrafted(rows_for_heatmap)
                else:
                    features = get_feature_matrix(rows_for_heatmap, feature_set, batch_size=args.batch_size)
                heatmap_feature_cache[(feature_set, sample.label_name)] = (features, cells, original)

        for item in ranked_results:
            bundle = bundles[item["model_key"]]
            sample_results = []
            for sample in samples:
                features, cells, original = heatmap_feature_cache[(item["feature_set"], sample.label_name)]
                score, prediction, heatmap_grid = score_sample_with_heatmap(bundle, features, cells, args.grid_size)
                out_path = heatmap_dir / f"{item['model_key']}__{sample.label_name}.png"
                build_overlay(original, heatmap_grid, out_path, args.top_heatmap_cells)
                sample_results.append(
                    {
                        "actual_label": sample.label_name,
                        "source_image": str(sample.path),
                        "heatmap": str(out_path),
                        "prediction": prediction,
                        "score_no_pregnant": score,
                    }
                )

            methods.append(
                {
                    "model_key": item["model_key"],
                    "feature_set": item["feature_set"],
                    "threshold": item["threshold"],
                    "validate_balanced_accuracy": item["validate"]["balanced_accuracy"],
                    "test_balanced_accuracy": item["test"]["balanced_accuracy"],
                    "score_meaning": bundle["score_meaning"],
                    "samples": sample_results,
                }
            )

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active_model": active_model,
        "dataset_summary": experiment["dataset_summary"],
        "best": ranked_results[0],
        "active_test_results": model_tests[active_result["model_key"]],
        "model_test_results": model_tests,
        "detail_heatmaps": detail_heatmaps,
        "cleaned_inputs": cleaned_inputs,
        "ranked_results": ranked_results,
        "methods": methods,
    }
    write_json(output_dir / "report_data.json", report)
    (output_dir / "index.html").write_text(render_html(report, output_dir), encoding="utf-8")
    print(f"[DONE] Report: {output_dir / 'index.html'}")
    print(f"[DONE] Heatmaps: {heatmap_dir}")


if __name__ == "__main__":
    main()
