from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
VALIDATE_ROOT = REPO_ROOT / "AnomalyDetection" / "asset" / "validate"
ANOMALY_SCRIPTS = REPO_ROOT / "AnomalyDetection" / "scripts"
REPORT_DIR = REPO_ROOT / "AnomalyDetection" / "outputs" / "report"
REPORT_HTML = REPORT_DIR / "index.html"
REPORT_JSON = REPORT_DIR / "report_data.json"

sys.path.insert(0, str(ANOMALY_SCRIPTS))

from anomaly_lib import (  # noqa: E402
    ImageRow,
    extract_patch_handcrafted,
    get_feature_matrix,
    label_prediction,
    load_json,
    load_model_bundle,
    predict_from_scores,
    score_bundle,
    score_patch_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def folder_truth(folder_name: str) -> str:
    mapping = {
        "1_Pregnant": "pregnant",
        "2_NoPregnant": "no_pregnant",
        "3_NotSure": "no_pregnant",
    }
    return mapping.get(folder_name, folder_name)


def resolve_active_model(registry_path: Path) -> Path:
    registry = load_json(registry_path)
    active = registry["active_model"]
    run_name, model_key = active.split("/", 1)
    model_ref = registry["runs"][run_name]["models"][model_key]
    model_file = str(model_ref["model_file"])
    model_path = Path(model_file)
    if model_path.exists():
        return model_path
    model_filename = PureWindowsPath(model_file).name if "\\" in model_file else model_path.name
    return registry_path.parent / run_name / model_filename


def run_anomaly_rows() -> list[dict]:
    registry_path = REPO_ROOT / "AnomalyDetection" / "artifacts" / "models" / "model_registry.json"
    model_path = resolve_active_model(registry_path)
    bundle = load_model_bundle(model_path)
    rows: list[dict] = []
    for folder in sorted([path for path in VALIDATE_ROOT.iterdir() if path.is_dir()], key=lambda item: item.name):
        for image_path in sorted([path for path in folder.iterdir() if path.is_file()], key=lambda item: item.name):
            row = ImageRow(path=image_path, split="validate", label_name=folder.name, target=None)
            if bundle["feature_set"] == "patch_handcrafted":
                score = float(score_patch_bundle(bundle, extract_patch_handcrafted([row]))[0])
            else:
                features = get_feature_matrix([row], bundle["feature_set"])
                score = float(score_bundle(bundle, features)[0])
            prediction_target = int(predict_from_scores(np.asarray([score]), float(bundle["threshold"]))[0])
            rows.append(
                {
                    "model": "anomaly",
                    "folder": folder.name,
                    "truth": folder_truth(folder.name),
                    "file": image_path.name,
                    "raw_prediction": label_prediction(prediction_target),
                    "prediction": label_prediction(prediction_target),
                    "confidence": score,
                }
            )
    return rows


def run_yolo_rows(weight: str) -> list[dict]:
    result = subprocess.run(
        [str(PYTHON_EXE), "tests/run_yolo_validate.py", "--weight", weight],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    return json.loads(stdout_lines[-1])


def build_agree_pregnant_ensemble(rows: list[dict], left_model: str, right_model: str, ensemble_name: str) -> list[dict]:
    left_rows = {(row["folder"], row["file"]): row for row in rows if row["model"] == left_model}
    right_rows = {(row["folder"], row["file"]): row for row in rows if row["model"] == right_model}
    combined_rows: list[dict] = []
    shared_keys = sorted(set(left_rows.keys()) & set(right_rows.keys()))
    for key in shared_keys:
        left_row = left_rows[key]
        right_row = right_rows[key]
        both_pregnant = left_row["prediction"] == "pregnant" and right_row["prediction"] == "pregnant"
        combined_rows.append(
            {
                "model": ensemble_name,
                "folder": left_row["folder"],
                "truth": left_row["truth"],
                "file": left_row["file"],
                "raw_prediction": f"{left_row['prediction']} | {right_row['prediction']}",
                "prediction": "pregnant" if both_pregnant else "no_pregnant",
                "confidence": None,
            }
        )
    return combined_rows


def summarize(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["model"], row["folder"])
        if key not in buckets:
            buckets[key] = {
                "model": row["model"],
                "folder": row["folder"],
                "truth": row["truth"],
                "samples": 0,
                "pred_pregnant": 0,
                "pred_no_pregnant": 0,
                "pred_not_sure": 0,
                "pred_unknown": 0,
                "pred_other": 0,
                "correct": 0,
            }
        bucket = buckets[key]
        bucket["samples"] += 1
        prediction = row["prediction"]
        if prediction == "pregnant":
            bucket["pred_pregnant"] += 1
        elif prediction == "no_pregnant":
            bucket["pred_no_pregnant"] += 1
        elif prediction == "not_sure":
            bucket["pred_not_sure"] += 1
        elif prediction == "unknown":
            bucket["pred_unknown"] += 1
        else:
            bucket["pred_other"] += 1
        if prediction == row["truth"]:
            bucket["correct"] += 1
    summary_rows = []
    for bucket in buckets.values():
        bucket["accuracy"] = round(bucket["correct"] / bucket["samples"], 2) if bucket["samples"] else None
        summary_rows.append(bucket)
    return sorted(summary_rows, key=lambda item: (item["model"], item["folder"]))


def compute_binary_metrics(rows: list[dict]) -> dict:
    binary_rows = [row for row in rows if row["truth"] in {"pregnant", "no_pregnant"}]
    total = len(binary_rows)
    correct = sum(1 for row in binary_rows if row["prediction"] == row["truth"])

    def per_label(label: str) -> dict:
        tp = sum(1 for row in binary_rows if row["truth"] == label and row["prediction"] == label)
        fp = sum(1 for row in binary_rows if row["truth"] != label and row["prediction"] == label)
        fn = sum(1 for row in binary_rows if row["truth"] == label and row["prediction"] != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return {
        "binary_total": total,
        "binary_correct": correct,
        "binary_accuracy": round(correct / total, 4) if total else 0.0,
        "pregnant": per_label("pregnant"),
        "no_pregnant": per_label("no_pregnant"),
    }


def choose_recommendations(model_metrics: list[dict]) -> dict:
    pregnant_pick = max(
        model_metrics,
        key=lambda item: (
            item["metrics"]["pregnant"]["recall"],
            item["metrics"]["pregnant"]["f1"],
            item["metrics"]["binary_accuracy"],
        ),
    )
    no_pregnant_recall_pick = max(
        model_metrics,
        key=lambda item: (
            item["metrics"]["no_pregnant"]["recall"],
            item["metrics"]["no_pregnant"]["f1"],
            item["metrics"]["binary_accuracy"],
        ),
    )
    no_pregnant_f1_pick = max(
        model_metrics,
        key=lambda item: (
            item["metrics"]["no_pregnant"]["f1"],
            item["metrics"]["no_pregnant"]["recall"],
            item["metrics"]["binary_accuracy"],
        ),
    )
    return {
        "pregnant_priority": pregnant_pick["model"],
        "no_pregnant_recall_priority": no_pregnant_recall_pick["model"],
        "no_pregnant_f1_priority": no_pregnant_f1_pick["model"],
    }


def build_html(report: dict) -> str:
    generated_at = html.escape(report["generated_at"])
    comparison_rows = []
    for item in report["models"]:
        metrics = item["metrics"]
        comparison_rows.append(
            f"""
            <tr>
              <td>{html.escape(item["model"])}</td>
              <td>{metrics["binary_accuracy"]:.2f}</td>
              <td>{metrics["pregnant"]["precision"]:.2f}</td>
              <td>{metrics["pregnant"]["recall"]:.2f}</td>
              <td>{metrics["pregnant"]["f1"]:.2f}</td>
              <td>{metrics["no_pregnant"]["precision"]:.2f}</td>
              <td>{metrics["no_pregnant"]["recall"]:.2f}</td>
              <td>{metrics["no_pregnant"]["f1"]:.2f}</td>
              <td>{metrics["binary_total"]}</td>
            </tr>
            """
        )
    model_cards = []
    for item in report["models"]:
        metrics = item["metrics"]
        model_cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(item["model"])}</h2>
              <div class="grid">
                <div><strong>Binary Accuracy</strong><span>{metrics["binary_accuracy"]:.2f}</span></div>
                <div><strong>Pregnant Recall</strong><span>{metrics["pregnant"]["recall"]:.2f}</span></div>
                <div><strong>Pregnant F1</strong><span>{metrics["pregnant"]["f1"]:.2f}</span></div>
                <div><strong>NoPregnant Recall</strong><span>{metrics["no_pregnant"]["recall"]:.2f}</span></div>
                <div><strong>NoPregnant F1</strong><span>{metrics["no_pregnant"]["f1"]:.2f}</span></div>
                <div><strong>Binary Samples</strong><span>{metrics["binary_total"]}</span></div>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Class</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td>pregnant</td><td>{metrics["pregnant"]["tp"]}</td><td>{metrics["pregnant"]["fp"]}</td><td>{metrics["pregnant"]["fn"]}</td><td>{metrics["pregnant"]["precision"]:.2f}</td><td>{metrics["pregnant"]["recall"]:.2f}</td><td>{metrics["pregnant"]["f1"]:.2f}</td></tr>
                  <tr><td>no_pregnant</td><td>{metrics["no_pregnant"]["tp"]}</td><td>{metrics["no_pregnant"]["fp"]}</td><td>{metrics["no_pregnant"]["fn"]}</td><td>{metrics["no_pregnant"]["precision"]:.2f}</td><td>{metrics["no_pregnant"]["recall"]:.2f}</td><td>{metrics["no_pregnant"]["f1"]:.2f}</td></tr>
                </tbody>
              </table>
            </section>
            """
        )

    highlight_rows = []
    for row in report["highlight_rows"]:
        highlight_rows.append(
            f"<tr><td>{html.escape(row['model'])}</td><td>{html.escape(row['file'])}</td><td>{html.escape(row['truth'])}</td><td>{html.escape(row['prediction'])}</td><td>{html.escape(str(row['confidence']))}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>รายงานเทียบโมเดลบนชุด validate</title>
  <style>
    :root {{
      --bg: #f5f3ef;
      --panel: #ffffff;
      --text: #202124;
      --muted: #65605a;
      --line: #d8d2c8;
      --accent: #0f766e;
      --good: #15803d;
      --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: var(--bg); color: var(--text); }}
    header {{ padding: 30px 32px 22px; border-bottom: 1px solid var(--line); background: #fffaf2; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; line-height: 1.15; }}
    h2 {{ margin: 0 0 12px; font-size: 21px; }}
    p {{ margin: 8px 0; color: var(--muted); line-height: 1.55; }}
    main {{ padding: 22px 32px 42px; max-width: 1480px; margin: 0 auto; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 18px 0 8px; }}
    .metric, .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .metric strong {{ display: block; font-size: 22px; margin-top: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 8px; margin: 12px 0 16px; }}
    .grid div {{ background: #f3f4f6; border-radius: 6px; padding: 10px; display: flex; justify-content: space-between; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; font-size: 14px; }}
    th {{ background: #ece7dd; color: #26221f; }}
    tr:last-child td {{ border-bottom: 0; }}
    .note {{ margin-top: 10px; padding: 10px 12px; border-left: 4px solid var(--accent); background: #f8fbf8; }}
    .rec {{ background: #ecfeff; border-left: 4px solid #0891b2; padding: 12px 16px; margin: 16px 0; }}
    .ok {{ color: var(--good); font-weight: 700; }}
    .warn {{ color: var(--warn); font-weight: 700; }}
    code {{ background: #e5e7eb; padding: 2px 6px; border-radius: 4px; }}
    @media (max-width: 760px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .summary {{ grid-template-columns: 1fr 1fr; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>รายงานเทียบโมเดลบนชุด validate</h1>
    <p>อัปเดตเมื่อ {generated_at}. รายงานนี้เทียบ anomaly, yolo 2 weight, และ ensemble แบบไม่ผ่าน Gemini โดยใช้ภาพทุกไฟล์ใน <code>{html.escape(str(REPO_ROOT / "AnomalyDetection" / "asset" / "validate"))}</code></p>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><span>เน้นจับท้อง (Recall-first)</span><strong>{html.escape(report["recommendations"]["pregnant_priority"])}</strong></div>
      <div class="metric"><span>เน้นจับไม่ท้อง (Recall-first)</span><strong>{html.escape(report["recommendations"]["no_pregnant_recall_priority"])}</strong></div>
      <div class="metric"><span>ไม่ท้องแบบสมดุล (F1-first)</span><strong>{html.escape(report["recommendations"]["no_pregnant_f1_priority"])}</strong></div>
      <div class="metric"><span>หมายเหตุ</span><strong>3_NotSure นับเป็นไม่ท้อง</strong></div>
    </section>

    <p class="note">การคำนวณ Precision / Recall / F1 ใช้กติกา binary เดียวกันทั้งชุด validate: <code>1_Pregnant = pregnant</code> และ <code>2_NoPregnant + 3_NotSure = no_pregnant</code></p>

    <div class="rec">
      <div><strong>ถ้าเน้นไม่พลาดเคสท้อง:</strong> <span class="ok">{html.escape(report["recommendations"]["pregnant_priority"])}</span></div>
      <div><strong>ถ้าเน้นไม่พลาดเคสไม่ท้องแบบ recall ล้วน:</strong> <span class="warn">{html.escape(report["recommendations"]["no_pregnant_recall_priority"])}</span></div>
      <div><strong>ถ้าจะใช้เคสไม่ท้องแบบดูสมดุล precision/recall ด้วย:</strong> <span class="ok">{html.escape(report["recommendations"]["no_pregnant_f1_priority"])}</span></div>
    </div>

    <section class="card">
      <h2>ตารางเทียบรวม</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Binary Acc</th>
            <th>Preg P</th>
            <th>Preg R</th>
            <th>Preg F1</th>
            <th>NoPreg P</th>
            <th>NoPreg R</th>
            <th>NoPreg F1</th>
            <th>Binary Samples</th>
          </tr>
        </thead>
        <tbody>{''.join(comparison_rows)}</tbody>
      </table>
    </section>

    {''.join(model_cards)}

    <section class="card">
      <h2>Highlighted file: validate01.png</h2>
      <table>
        <thead><tr><th>Model</th><th>File</th><th>Truth Folder</th><th>Prediction</th><th>Confidence</th></tr></thead>
        <tbody>{''.join(highlight_rows)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def build_report() -> dict:
    all_rows = []
    all_rows.extend(run_anomaly_rows())
    all_rows.extend(run_yolo_rows("best.pt"))
    all_rows.extend(run_yolo_rows("best_finetune_YOLO26-cls_Ver2_20260424.pt"))
    all_rows.extend(
        build_agree_pregnant_ensemble(
            all_rows,
            "anomaly",
            "yolo:best_finetune_YOLO26-cls_Ver2_20260424.pt",
            "ensemble:anomaly+yolo_finetune",
        )
    )

    models = []
    for model_name in [
        "anomaly",
        "yolo:best.pt",
        "yolo:best_finetune_YOLO26-cls_Ver2_20260424.pt",
        "ensemble:anomaly+yolo_finetune",
    ]:
        model_rows = [row for row in all_rows if row["model"] == model_name]
        models.append({"model": model_name, "metrics": compute_binary_metrics(model_rows)})

    recommendations = choose_recommendations(models)
    highlight_rows = [row for row in all_rows if row["file"] == "validate01.png"]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": models,
        "recommendations": recommendations,
        "highlight_rows": highlight_rows,
    }


def write_report_files(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_HTML.write_text(build_html(report), encoding="utf-8")


def print_table(title: str, rows: list[dict], columns: list[str]) -> None:
    print(title)
    widths = {column: max(len(column), *(len(str(row.get(column, ""))) for row in rows)) for column in columns}
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))
    print()


def main() -> int:
    args = parse_args()

    all_rows = []
    all_rows.extend(run_anomaly_rows())
    all_rows.extend(run_yolo_rows("best.pt"))
    all_rows.extend(run_yolo_rows("best_finetune_YOLO26-cls_Ver2_20260424.pt"))
    all_rows.extend(
        build_agree_pregnant_ensemble(
            all_rows,
            "anomaly",
            "yolo:best_finetune_YOLO26-cls_Ver2_20260424.pt",
            "ensemble:anomaly+yolo_finetune",
        )
    )

    summary_rows = summarize(all_rows)
    print_table(
        "SUMMARY",
        summary_rows,
        [
            "model",
            "folder",
            "truth",
            "samples",
            "pred_pregnant",
            "pred_no_pregnant",
            "pred_not_sure",
            "pred_unknown",
            "pred_other",
            "correct",
            "accuracy",
        ],
    )
    print_table(
        "DETAIL",
        all_rows,
        ["model", "folder", "file", "prediction", "raw_prediction", "confidence"],
    )
    print("JSON_SUMMARY=" + json.dumps(summary_rows, ensure_ascii=False))
    print("JSON_DETAIL=" + json.dumps(all_rows, ensure_ascii=False))

    if args.write_report:
        report = build_report()
        write_report_files(report)
        print(str(REPORT_HTML))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
