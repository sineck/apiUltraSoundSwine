from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import joblib
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_TO_TARGET = {
    "1_Pregnant": 1,
    "2_NoPregnant": 0,
}
TARGET_TO_LABEL = {
    1: "pregnant",
    0: "no_pregnant",
}


@dataclass(frozen=True)
class ImageRow:
    path: Path
    split: str
    label_name: str
    target: int | None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_dataset(asset_dir: Path) -> list[ImageRow]:
    rows: list[ImageRow] = []
    for split in ("train", "validate", "test"):
        split_dir = asset_dir / split
        if not split_dir.exists():
            continue
        for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            target = LABEL_TO_TARGET.get(label_dir.name)
            for image_path in sorted(label_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append(ImageRow(image_path, split, label_dir.name, target))
    return rows


def summarize_dataset(rows: Iterable[ImageRow]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        split_summary = summary.setdefault(row.split, {})
        split_summary[row.label_name] = split_summary.get(row.label_name, 0) + 1
    return summary


def read_bgr_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def ultrasound_sector_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rough_mask = (gray > 14).astype(np.uint8)
    rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_CLOSE, np.ones((17, 17), dtype=np.uint8))
    rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(rough_mask, connectivity=8)
    if count <= 1:
        return np.ones(gray.shape, dtype=np.uint8)

    largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
    sector_mask = (labels == largest_label).astype(np.uint8)
    sector_mask = cv2.morphologyEx(sector_mask, cv2.MORPH_CLOSE, np.ones((31, 31), dtype=np.uint8))
    return cv2.dilate(sector_mask, np.ones((9, 9), dtype=np.uint8), iterations=1)


def remove_common_overlay_zones(image: np.ndarray) -> np.ndarray:
    cleaned = image.copy()
    height, width = cleaned.shape[:2]
    blackout_zones = [
        (0.00, 0.00, 1.00, 0.235),
        (0.00, 0.88, 0.18, 1.00),
    ]
    for x1_ratio, y1_ratio, x2_ratio, y2_ratio in blackout_zones:
        x1, x2 = int(x1_ratio * width), int(x2_ratio * width)
        y1, y2 = int(y1_ratio * height), int(y2_ratio * height)
        cleaned[y1:y2, x1:x2] = 0
    return cleaned


def clean_ultrasound_image(path: Path) -> np.ndarray:
    image = read_bgr_image(path)
    sector_mask = ultrasound_sector_mask(image)
    cleaned = image.copy()
    cleaned[sector_mask == 0] = 0
    return remove_common_overlay_zones(cleaned)


def read_gray_image(path: Path, image_size: int = 224) -> np.ndarray:
    image = clean_ultrasound_image(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def handcrafted_features(path: Path, image_size: int = 224) -> np.ndarray:
    gray = read_gray_image(path, image_size=image_size)
    gray_float = gray.astype(np.float32) / 255.0

    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten().astype(np.float32)
    hist = hist / max(float(hist.sum()), 1.0)

    percentiles = np.percentile(gray_float, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    stats = np.array(
        [
            gray_float.mean(),
            gray_float.std(),
            gray_float.min(),
            gray_float.max(),
            np.mean(gray_float < 0.08),
            np.mean(gray_float < 0.16),
            np.mean(gray_float > 0.75),
            np.mean(gray_float > 0.90),
        ],
        dtype=np.float32,
    )

    blur = cv2.GaussianBlur(gray_float, (9, 9), 0)
    local_contrast = np.abs(gray_float - blur)
    edges = cv2.Canny(gray, 60, 140)
    sobel_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    laplacian = cv2.Laplacian(gray_float, cv2.CV_32F)

    texture = np.array(
        [
            np.mean(edges > 0),
            gradient.mean(),
            gradient.std(),
            local_contrast.mean(),
            local_contrast.std(),
            laplacian.var(),
        ],
        dtype=np.float32,
    )

    grid_features: list[float] = []
    for cell in np.array_split(gray_float, 4, axis=0):
        for patch in np.array_split(cell, 4, axis=1):
            grid_features.append(float(patch.mean()))
            grid_features.append(float(patch.std()))

    dark_mask = (gray < 35).astype(np.uint8)
    bright_mask = (gray > 210).astype(np.uint8)
    component_features = []
    for mask in (dark_mask, bright_mask):
        count, labels, stats_cc, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = stats_cc[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([], dtype=np.float32)
        component_features.extend(
            [
                float(len(areas)),
                float(areas.max()) if len(areas) else 0.0,
                float(areas.mean()) if len(areas) else 0.0,
                float(np.sum(areas > 25)) if len(areas) else 0.0,
            ]
        )

    return np.concatenate(
        [
            hist,
            stats,
            percentiles.astype(np.float32),
            texture,
            np.asarray(grid_features, dtype=np.float32),
            np.asarray(component_features, dtype=np.float32),
        ]
    ).astype(np.float32)


def extract_handcrafted(rows: list[ImageRow]) -> np.ndarray:
    return np.vstack([handcrafted_features(row.path) for row in rows]).astype(np.float32)


def patch_handcrafted_features(path: Path, image_size: int = 224, grid_size: int = 8) -> np.ndarray:
    gray = read_gray_image(path, image_size=image_size)
    gray_float = gray.astype(np.float32) / 255.0
    patch_features: list[list[float]] = []
    for row_parts in np.array_split(gray_float, grid_size, axis=0):
        for patch in np.array_split(row_parts, grid_size, axis=1):
            patch_u8 = np.uint8(np.clip(patch * 255.0, 0, 255))
            sobel_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            gradient = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
            hist = cv2.calcHist([patch_u8], [0], None, [8], [0, 256]).flatten().astype(np.float32)
            hist = hist / max(float(hist.sum()), 1.0)
            patch_features.append(
                [
                    float(patch.mean()),
                    float(patch.std()),
                    float(np.percentile(patch, 10)),
                    float(np.percentile(patch, 50)),
                    float(np.percentile(patch, 90)),
                    float(np.mean(patch < 0.08)),
                    float(np.mean(patch > 0.75)),
                    float(gradient.mean()),
                    float(gradient.std()),
                    *[float(v) for v in hist],
                ]
            )
    return np.asarray(patch_features, dtype=np.float32)


def extract_patch_handcrafted(rows: list[ImageRow]) -> list[np.ndarray]:
    return [patch_handcrafted_features(row.path) for row in rows]


def extract_resnet18(rows: list[ImageRow], batch_size: int = 16, device: str = "auto") -> np.ndarray:
    import torch
    from PIL import Image
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    actual_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if actual_device == "auto":
        actual_device = "cpu"
    model.to(actual_device)
    model.eval()

    features: list[np.ndarray] = []
    batch_tensors = []
    with torch.no_grad():
        for row in rows:
            clean_bgr = clean_ultrasound_image(row.path)
            clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(clean_rgb)
            tensor = transform(image)
            batch_tensors.append(tensor)
            if len(batch_tensors) == batch_size:
                batch = torch.stack(batch_tensors).to(actual_device)
                features.append(model(batch).cpu().numpy())
                batch_tensors = []
        if batch_tensors:
            batch = torch.stack(batch_tensors).to(actual_device)
            features.append(model(batch).cpu().numpy())
    return np.vstack(features).astype(np.float32)


def extract_dinov2(rows: list[ImageRow], batch_size: int = 8, device: str = "auto") -> np.ndarray:
    import torch
    from PIL import Image
    from torchvision import transforms

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    actual_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if actual_device == "auto":
        actual_device = "cpu"
    model.to(actual_device)
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    features: list[np.ndarray] = []
    batch_tensors = []
    with torch.no_grad():
        for row in rows:
            clean_bgr = clean_ultrasound_image(row.path)
            clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(clean_rgb)
            batch_tensors.append(transform(image))
            if len(batch_tensors) == batch_size:
                batch = torch.stack(batch_tensors).to(actual_device)
                features.append(model(batch).cpu().numpy())
                batch_tensors = []
        if batch_tensors:
            batch = torch.stack(batch_tensors).to(actual_device)
            features.append(model(batch).cpu().numpy())
    return np.vstack(features).astype(np.float32)


def get_feature_matrix(rows: list[ImageRow], feature_set: str, batch_size: int = 16) -> np.ndarray:
    if feature_set == "handcrafted":
        return extract_handcrafted(rows)
    if feature_set == "resnet18":
        return extract_resnet18(rows, batch_size=batch_size)
    if feature_set == "dinov2":
        return extract_dinov2(rows, batch_size=max(1, min(batch_size, 8)))
    raise ValueError(f"Unknown feature set: {feature_set}")


def save_model_bundle(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model_bundle(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def score_bundle(bundle: dict[str, Any], features: np.ndarray) -> np.ndarray:
    estimator_type = bundle["estimator_type"]
    if estimator_type == "mahalanobis":
        mean = np.asarray(bundle["weights"]["mean"], dtype=np.float32)
        precision = np.asarray(bundle["weights"]["precision"], dtype=np.float32)
        centered = features - mean
        return np.einsum("ij,jk,ik->i", centered, precision, centered)
    if estimator_type == "normal_quantile":
        mean = np.asarray(bundle["weights"]["mean"], dtype=np.float32)
        std = np.asarray(bundle["weights"]["std"], dtype=np.float32)
        z = (features - mean) / np.maximum(std, 1.0e-6)
        return np.sqrt(np.mean(z * z, axis=1))
    if estimator_type == "sklearn_anomaly":
        scaled = bundle["scaler"].transform(features)
        return -bundle["estimator"].decision_function(scaled)
    if estimator_type == "sklearn_supervised":
        scaled = bundle["scaler"].transform(features)
        estimator = bundle["estimator"]
        if hasattr(estimator, "predict_proba"):
            no_pregnant_index = list(estimator.classes_).index(0)
            return estimator.predict_proba(scaled)[:, no_pregnant_index]
        decision = estimator.decision_function(scaled)
        return 1.0 / (1.0 + np.exp(decision))
    raise ValueError(f"Unknown estimator type: {estimator_type}")


def score_patch_bundle(bundle: dict[str, Any], patch_features: list[np.ndarray]) -> np.ndarray:
    estimator_type = bundle["estimator_type"]
    if estimator_type == "patchcore":
        scaler = bundle["scaler"]
        memory_bank = np.asarray(bundle["memory_bank"], dtype=np.float32)
        scores = []
        for patches in patch_features:
            scaled = scaler.transform(patches).astype(np.float32)
            distances = np.sqrt(((scaled[:, None, :] - memory_bank[None, :, :]) ** 2).sum(axis=2))
            nearest = distances.min(axis=1)
            scores.append(float(np.mean(np.sort(nearest)[-5:])))
        return np.asarray(scores, dtype=np.float32)
    if estimator_type == "padim_diag":
        means = np.asarray(bundle["means"], dtype=np.float32)
        variances = np.asarray(bundle["variances"], dtype=np.float32)
        scores = []
        for patches in patch_features:
            limit = min(len(patches), len(means))
            diff = patches[:limit] - means[:limit]
            distance = np.sqrt(np.mean((diff * diff) / np.maximum(variances[:limit], 1.0e-6), axis=1))
            scores.append(float(np.mean(np.sort(distance)[-5:])))
        return np.asarray(scores, dtype=np.float32)
    raise ValueError(f"Unknown patch estimator type: {estimator_type}")


def predict_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(scores >= threshold, 0, 1)


def label_prediction(target: int) -> str:
    return TARGET_TO_LABEL[int(target)]
