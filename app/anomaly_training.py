import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


pathInitial = Path(__file__).resolve().parent.parent
ANOMALY_ROOT = pathInitial / "AnomalyDetection"
TRAIN_SCRIPT = ANOMALY_ROOT / "scripts" / "train_anomaly_models.py"
REPORT_SCRIPT = ANOMALY_ROOT / "scripts" / "generate_report.py"
INDEX_SCRIPT = ANOMALY_ROOT / "scripts" / "build_artifact_index.py"
REGISTRY_PATH = ANOMALY_ROOT / "artifacts" / "models" / "model_registry.json"
REPORT_INDEX = ANOMALY_ROOT / "outputs" / "report" / "index.html"
JOB_DIR = pathInitial / "logs" / "anomaly_training"

JobStatus = Literal["idle", "queued", "running", "succeeded", "failed"]


class AnomalyTrainingAlreadyRunning(RuntimeError):
    """raise เมื่อมี job retrain เดิมกำลัง queued/running อยู่."""
    pass


@dataclass
class AnomalyTrainingJob:
    """state object ของ retrain job หนึ่งตัวที่ API ใช้รายงานกลับไปยัง client."""
    job_id: str
    status: JobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    active_model: str | None = None
    report_path: str | None = None
    registry_path: str = str(REGISTRY_PATH)
    log_file: str | None = None
    commands: list[list[str]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_lock = threading.Lock()
_current_job: AnomalyTrainingJob | None = None


def utc_now() -> str:
    """คืน timestamp UTC แบบ ISO เพื่อให้ log/job status เทียบกันข้ามเครื่องได้ง่าย."""
    return datetime.now(timezone.utc).isoformat()


def current_anomaly_training_job() -> dict:
    """คืน snapshot ของ job ปัจจุบัน หรือสถานะ idle ถ้ายังไม่เคย start."""
    with _lock:
        if _current_job is None:
            return {
                "status": "idle",
                "registry_path": str(REGISTRY_PATH),
                "report_path": str(REPORT_INDEX) if REPORT_INDEX.exists() else None,
            }
        return _current_job.to_dict()


def build_training_commands(
    feature_sets: str | None,
    model_keys: str | None,
    batch_size: int,
    generate_report: bool,
    detail_heatmaps: str,
    rebuild_index: bool,
) -> list[list[str]]:
    """สร้าง command list ของ retrain pipeline ตาม config กลาง.

    ลำดับ intentionally คงที่:
    1. train anomaly models
    2. rebuild artifact index ถ้าถูกเปิดไว้
    3. generate report ถ้าถูกเปิดไว้
    """
    commands = [[sys.executable, str(TRAIN_SCRIPT), "--batch-size", str(batch_size)]]
    if feature_sets:
        commands[0].extend(["--feature-sets", feature_sets])
    if model_keys:
        commands[0].extend(["--model-keys", model_keys])
    if rebuild_index:
        commands.append([sys.executable, str(INDEX_SCRIPT)])
    if generate_report:
        commands.append([sys.executable, str(REPORT_SCRIPT), "--detail-heatmaps", detail_heatmaps])
    return commands


def load_active_model() -> str | None:
    """อ่าน active anomaly model ล่าสุดจาก registry หลัง train จบ."""
    if not REGISTRY_PATH.exists():
        return None
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8-sig"))
    return registry.get("active_model")


def run_job(job: AnomalyTrainingJob) -> None:
    """worker จริงของ background retrain thread.

    ฟังก์ชันนี้รับผิดชอบ:
    - mark สถานะ queued -> running -> succeeded/failed
    - รัน command ตาม pipeline ที่ build ไว้
    - เขียน stdout/stderr ลง log file
    - snapshot active_model/report_path หลังงานจบ
    """
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    with _lock:
        job.status = "running"
        job.started_at = utc_now()

    try:
        log_path = Path(job.log_file or "")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            for command in job.commands:
                log.write(f"\n$ {' '.join(command)}\n")
                log.flush()
                completed = subprocess.run(
                    command,
                    cwd=str(pathInitial),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")

        with _lock:
            job.status = "succeeded"
            job.return_code = 0
            job.active_model = load_active_model()
            job.report_path = str(REPORT_INDEX) if REPORT_INDEX.exists() else None
            job.finished_at = utc_now()
    except Exception as exc:
        with _lock:
            job.status = "failed"
            job.return_code = 1
            job.error = str(exc)
            job.active_model = load_active_model()
            job.report_path = str(REPORT_INDEX) if REPORT_INDEX.exists() else None
            job.finished_at = utc_now()


def start_anomaly_training(
    feature_sets: str | None = None,
    model_keys: str | None = None,
    batch_size: int = 16,
    generate_report: bool = True,
    detail_heatmaps: str = "active",
    rebuild_index: bool = True,
    force: bool = False,
) -> dict:
    """entrypoint กลางของ retrain API.

    ฟังก์ชันนี้ validate input, กัน job ซ้อน, สร้าง job metadata, แล้วปล่อย
    background thread ไปทำงานจริง โดย route จะได้ response กลับเร็วพร้อม job_id
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if detail_heatmaps not in {"none", "active", "all"}:
        raise ValueError("detail_heatmaps must be one of: none, active, all")
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing train script: {TRAIN_SCRIPT}")

    JOB_DIR.mkdir(parents=True, exist_ok=True)
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    job = AnomalyTrainingJob(
        job_id=job_id,
        status="queued",
        created_at=utc_now(),
        log_file=str(JOB_DIR / f"{job_id}.log"),
        commands=build_training_commands(feature_sets, model_keys, batch_size, generate_report, detail_heatmaps, rebuild_index),
    )

    with _lock:
        global _current_job
        if _current_job and _current_job.status in {"queued", "running"}:
            if force:
                raise ValueError(
                    f"force=true ยังไม่รองรับการทับงาน train ที่กำลังรันอยู่: {_current_job.job_id}"
                )
            raise AnomalyTrainingAlreadyRunning(_current_job.job_id)
        _current_job = job

    thread = threading.Thread(target=run_job, args=(job,), daemon=True)
    thread.start()
    return job.to_dict()
