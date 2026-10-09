"""Local research dashboard for the trained melanoma classifiers."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models, transforms

from classifiers.multiple_classifier.cnn import CustomClassifier
from classifiers.multiple_classifier.cnn_attention import AttentionClassifier


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
OUT = ROOT / "out"
APP_DATA = ROOT / "app_data"
UPLOADS = APP_DATA / "uploads"
HEATMAPS = APP_DATA / "heatmaps"
DB_PATH = APP_DATA / "history.sqlite3"
CLASSES = ["BCC", "MEL", "SCC"]  # torchvision ImageFolder alphabetical order
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
IMAGE_TRANSFORM = transforms.Compose(
    [transforms.Resize((299, 299)), transforms.ToTensor(), NORMALIZE]
)
MODEL_SPECS = {
    "efficientnet": {
        "label": "EfficientNet-B0",
        "checkpoint": OUT / "baseline_mult" / "EfficientNet_best.pth",
        "kind": "efficientnet",
        "attention": False,
    },
    "inception": {
        "label": "InceptionV3",
        "checkpoint": OUT / "baseline_mult" / "Inception3_best.pth",
        "kind": "inception",
        "attention": False,
    },
    "attention": {
        "label": "EfficientNet-B0 + Feature Attention",
        "checkpoint": OUT / "proposed_attention" / "EfficientNet_Attention_best.pth",
        "kind": "efficientnet",
        "attention": True,
    },
    "attention_focal": {
        "label": "EfficientNet-B0 + Attention + Focal Loss",
        "checkpoint": OUT / "research_ablation" / "attention_focal" / "best.pth",
        "kind": "efficientnet",
        "attention": True,
    },
}


def db_connect():
    APP_DATA.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with db_connect() as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                original_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                primary_model TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities_json TEXT NOT NULL,
                results_json TEXT NOT NULL,
                heatmap_path TEXT NOT NULL
            )"""
        )


class ModelRegistry:
    """Load checkpoint-backed models lazily and keep them on the selected device."""

    def __init__(self):
        self.models = {}
        self.lock = threading.RLock()

    def available(self):
        return [
            {
                "id": model_id,
                "name": spec["label"],
                "available": spec["checkpoint"].is_file(),
                "checkpoint": str(spec["checkpoint"].relative_to(ROOT)).replace("\\", "/"),
            }
            for model_id, spec in MODEL_SPECS.items()
        ]

    def get(self, model_id):
        if model_id not in MODEL_SPECS:
            raise ValueError("Unknown model")
        with self.lock:
            if model_id in self.models:
                return self.models[model_id]
            spec = MODEL_SPECS[model_id]
            path = spec["checkpoint"]
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {path.relative_to(ROOT)}")
            if spec["kind"] == "efficientnet":
                model = models.efficientnet_b0(weights=None)
                input_size = model.classifier[1].in_features
                if spec["attention"]:
                    model.classifier = AttentionClassifier(
                        input_size=input_size,
                        hidden_size1=1024,
                        hidden_size2=512,
                        num_classes=len(CLASSES),
                    )
                else:
                    model.classifier[1] = CustomClassifier(
                        input_size=input_size,
                        hidden_size1=1024,
                        hidden_size2=512,
                        num_classes=len(CLASSES),
                    )
                target_layer = model.features[-1]
            else:
                model = models.inception_v3(
                    weights=None, aux_logits=True, init_weights=False
                )
                model.fc = CustomClassifier(
                    input_size=model.fc.in_features,
                    hidden_size1=1024,
                    hidden_size2=512,
                    num_classes=len(CLASSES),
                )
                target_layer = model.Mixed_7c
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            model.to(DEVICE).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model._dashboard_target_layer = target_layer
            self.models[model_id] = model
            return model


REGISTRY = ModelRegistry()


def make_heatmap(model, tensor, class_index):
    activations = []
    gradients = []
    layer = model._dashboard_target_layer

    def capture(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    hook = layer.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        input_tensor = tensor.to(DEVICE).requires_grad_(True)
        with torch.enable_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            score = output[0, class_index]
            score.backward()
        activation = activations[0].detach()
        gradient = gradients[0].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(299, 299), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.cpu().numpy()
    finally:
        hook.remove()


def colorize_heatmap(image, cam):
    import numpy as np

    source = np.asarray(image.resize((299, 299))).astype("float32")
    # Lightweight blue-to-yellow-to-red color map, no plotting dependency.
    x = np.clip(cam, 0, 1)
    red = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    colors = np.stack([red, green, blue], axis=-1) * 255
    overlay = (0.55 * source + 0.45 * colors).clip(0, 255).astype("uint8")
    return Image.fromarray(overlay, "RGB")


def predict_all(image):
    tensor = NORMALIZE(transforms.ToTensor()(image.resize((299, 299)))).unsqueeze(0)
    predictions = {}
    heatmap = None
    with REGISTRY.lock:
        for model_id, spec in MODEL_SPECS.items():
            if not spec["checkpoint"].is_file():
                continue
            model = REGISTRY.get(model_id)
            model.zero_grad(set_to_none=True)
            x = tensor.to(DEVICE).requires_grad_(True)
            with torch.enable_grad():
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probabilities = torch.softmax(logits[0].detach(), dim=0)
                pred_index = int(probabilities.argmax().item())
                cam = make_heatmap(model, tensor, pred_index)
            predictions[model_id] = {
                "name": spec["label"],
                "predicted_class": CLASSES[pred_index],
                "confidence": float(probabilities[pred_index].item()),
                "probabilities": {
                    name: float(probabilities[i].item())
                    for i, name in enumerate(CLASSES)
                },
            }
            if model_id == "attention":
                heatmap = colorize_heatmap(image, cam)
        if not predictions:
            raise FileNotFoundError("No trained multiclass checkpoints were found in out/.")
    # Keep the established baseline as the default prediction. The research
    # ablation remains available for side-by-side comparison even when its
    # validation results do not justify making it the default model.
    primary = "efficientnet" if "efficientnet" in predictions else (
        "attention" if "attention" in predictions else next(iter(predictions))
    )
    return predictions, primary, heatmap


def metric_summary(matrix):
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(matrix)))
    accuracy = correct / total if total else 0.0
    actual = [sum(row) for row in matrix]
    predicted = [sum(matrix[r][c] for r in range(len(matrix))) for c in range(len(matrix))]
    precision, recall, f1 = [], [], []
    for i in range(len(matrix)):
        p = matrix[i][i] / predicted[i] if predicted[i] else 0.0
        r = matrix[i][i] / actual[i] if actual[i] else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    expected = sum(actual[i] * predicted[i] for i in range(len(matrix))) / (total * total) if total else 0.0
    kappa = (accuracy - expected) / (1 - expected) if total and expected < 1 else 0.0
    return {
        "accuracy": accuracy,
        "precision_macro": sum(precision) / len(precision),
        "recall_macro": sum(recall) / len(recall),
        "f1_macro": sum(f1) / len(f1),
        "kappa": kappa,
        "confusion_matrix": matrix,
        "classes": CLASSES,
        "support": actual,
    }


def roc_summary(labels, class_scores, class_index):
    """One-vs-rest ROC points and trapezoid AUC from held-out probabilities."""
    binary_labels = [1 if label == class_index else 0 for label in labels]
    positives = sum(binary_labels)
    negatives = len(binary_labels) - positives
    if not positives or not negatives:
        return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": 0.0}
    ordered = sorted(zip(class_scores, binary_labels), key=lambda pair: pair[0], reverse=True)
    true_positive = false_positive = 0
    fpr = [0.0]
    tpr = [0.0]
    cursor = 0
    while cursor < len(ordered):
        threshold = ordered[cursor][0]
        while cursor < len(ordered) and ordered[cursor][0] == threshold:
            if ordered[cursor][1]:
                true_positive += 1
            else:
                false_positive += 1
            cursor += 1
        tpr.append(true_positive / positives)
        fpr.append(false_positive / negatives)
    auc = sum(
        (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
        for i in range(1, len(fpr))
    )
    return {"fpr": fpr, "tpr": tpr, "auc": auc}


def evaluate_models():
    cache = APP_DATA / "metrics.json"
    fingerprints = {
        model_id: spec["checkpoint"].stat().st_mtime_ns
        for model_id, spec in MODEL_SPECS.items()
        if spec["checkpoint"].is_file()
    }
    if cache.exists():
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved.get("schema_version") == 2 and saved.get("fingerprints") == fingerprints:
                return saved
        except (OSError, ValueError):
            pass

    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    validation_dir = ROOT / "data_mult" / "val"
    if not validation_dir.is_dir():
        raise FileNotFoundError("Validation images not found at data_mult/val")
    dataset = ImageFolder(str(validation_dir), transform=IMAGE_TRANSFORM)
    if dataset.classes != CLASSES:
        raise ValueError(f"Expected validation classes {CLASSES}, found {dataset.classes}")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    result = {
        "schema_version": 2,
        "source": "Recomputed from data_mult/val using the saved project checkpoints",
        "validation_images": len(dataset),
        "fingerprints": fingerprints,
        "models": {},
    }
    for model_id, spec in MODEL_SPECS.items():
        if not spec["checkpoint"].is_file():
            continue
        model = REGISTRY.get(model_id)
        matrix = [[0 for _ in CLASSES] for _ in CLASSES]
        labels_seen = []
        scores_seen = []
        with torch.no_grad():
            for images, labels in loader:
                logits = model(images.to(DEVICE))
                if isinstance(logits, tuple):
                    logits = logits[0]
                probabilities = torch.softmax(logits, dim=1).cpu().tolist()
                truths = labels.tolist()
                preds = [max(range(len(CLASSES)), key=lambda index: row[index]) for row in probabilities]
                labels_seen.extend(truths)
                scores_seen.extend(probabilities)
                for truth, pred in zip(truths, preds):
                    matrix[truth][pred] += 1
        result["models"][model_id] = {
            "name": spec["label"],
            **metric_summary(matrix),
            "roc": {
                class_name: roc_summary(
                    labels_seen,
                    [row[index] for row in scores_seen],
                    index,
                )
                for index, class_name in enumerate(CLASSES)
            },
        }
    cache.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def pdf_escape(value):
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(analysis):
    """Create a small one-page PDF with the uploaded image and Grad-CAM output."""
    image_files = [Path(analysis["image_path"]), Path(analysis["heatmap_path"])]
    objects = [None]

    def add_obj(data):
        objects.append(data)
        return len(objects) - 1

    catalog_id = add_obj(b"")
    pages_id = add_obj(b"")
    font_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    image_ids = []
    for path in image_files:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=88)
            jpeg = buffer.getvalue()
            width, height = rgb.size
        header = (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(jpeg)} >>\nstream\n"
        ).encode("ascii")
        image_ids.append(add_obj(header + jpeg + b"\nendstream"))

    lines = [
        "MelanoVexa | Skin Lesion Research Studio | Image Analysis",
        f"Case ID: {analysis['id']}",
        f"Created: {analysis['created_at']}",
        f"Image: {analysis['original_name'][:70]}",
        f"Primary model: {MODEL_SPECS[analysis['primary_model']]['label']}",
        f"Predicted category: {analysis['predicted_class']}",
        f"Model confidence: {analysis['confidence'] * 100:.1f}%",
        "Class probabilities:",
    ]
    probabilities = json.loads(analysis["probabilities_json"])
    lines.extend([f"  {name}: {value * 100:.1f}%" for name, value in probabilities.items()])
    lines.extend(["Model comparison:"])
    predictions = json.loads(analysis["results_json"])
    lines.extend([
        f"  {item['name']}: {item['predicted_class']} ({item['confidence'] * 100:.1f}%)"
        for item in predictions.values()
    ])
    lines.extend([
        "Grad-CAM shows image regions that influenced the selected model's score.",
        "Research demonstration only. Not a medical diagnosis or a substitute for clinical review.",
    ])
    commands = ["BT", "/F1 16 Tf", "50 746 Td", f"({pdf_escape(lines[0])}) Tj", "/F1 10 Tf"]
    for line in lines[1:]:
        commands.extend(["0 -22 Td", f"({pdf_escape(line)}) Tj"])
    commands.extend([
        "ET", "q", "230 0 0 230 55 120 cm", "/Im1 Do", "Q",
        "q", "230 0 0 230 325 120 cm", "/Im2 Do", "Q",
        "BT", "/F1 9 Tf", "120 100 Td", "(Uploaded image) Tj", "260 0 Td", "(Grad-CAM overlay) Tj", "ET",
    ])
    content = "\n".join(commands).encode("latin-1", errors="replace")
    content_id = add_obj(f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")
    resources_id = add_obj(
        f"<< /Font << /F1 {font_id} 0 R >> /XObject << /Im1 {image_ids[0]} 0 R /Im2 {image_ids[1]} 0 R >> >>".encode()
    )
    page_id = add_obj(
        f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] /Resources {resources_id} 0 R /Contents {content_id} 0 R >>".encode()
    )
    objects[catalog_id] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode()
    objects[pages_id] = f"<< /Type /Pages /Kids [{page_id} 0 R] /Count 1 >>".encode()
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id in range(1, len(objects)):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode())
        output.extend(objects[object_id])
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(objects)} /Root {catalog_id} 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()
    )
    return bytes(output)


def json_bytes(data):
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "MelanomaResearchDashboard/1.0"

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_bytes(self, body, content_type, status=200, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data, status=200):
        self.send_bytes(json_bytes(data), "application/json; charset=utf-8", status)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/health":
                return self.send_json({"ok": True, "device": str(DEVICE), "models": REGISTRY.available()})
            if path == "/api/models":
                return self.send_json({"device": str(DEVICE), "models": REGISTRY.available()})
            if path == "/api/experiments":
                metrics_path = ROOT / "classifiers" / "multiple_classifier" / "results.json"
                recorded = json.loads(metrics_path.read_text(encoding="utf-8"))
                research_runs = []
                for manifest_path in sorted((OUT / "research_ablation").glob("*/manifest.json")):
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        research_runs.append({
                            "experiment": manifest.get("experiment", manifest_path.parent.name),
                            "best_epoch": manifest.get("best_epoch"),
                            "metrics": manifest.get("validation_metrics", {}),
                            "checkpoint": str((manifest_path.parent / "best.pth").relative_to(ROOT)).replace("\\", "/"),
                        })
                    except (OSError, ValueError):
                        continue
                return self.send_json({"recorded": recorded, "research_runs": research_runs})
            if path == "/api/metrics":
                return self.send_json(evaluate_models())
            if path == "/api/history":
                query = parse_qs(parsed.query).get("q", [""])[0].strip()
                with db_connect() as db:
                    rows = db.execute(
                        "SELECT id, created_at, original_name, primary_model, predicted_class, confidence FROM analyses WHERE original_name LIKE ? OR predicted_class LIKE ? ORDER BY created_at DESC LIMIT 100",
                        (f"%{query}%", f"%{query}%"),
                    ).fetchall()
                return self.send_json({"cases": [dict(row) for row in rows]})
            if path.startswith("/api/cases/"):
                case_id = path.rsplit("/", 1)[-1]
                with db_connect() as db:
                    row = db.execute("SELECT * FROM analyses WHERE id = ?", (case_id,)).fetchone()
                if not row:
                    return self.send_json({"error": "Case not found"}, 404)
                result = dict(row)
                result["probabilities"] = json.loads(result.pop("probabilities_json"))
                result["results"] = json.loads(result.pop("results_json"))
                result["image_url"] = f"/media/{case_id}/image"
                result["heatmap_url"] = f"/media/{case_id}/heatmap"
                result["report_url"] = f"/api/report/{case_id}.pdf"
                return self.send_json(result)
            if path.startswith("/api/report/") and path.endswith(".pdf"):
                case_id = path.rsplit("/", 1)[-1][:-4]
                with db_connect() as db:
                    row = db.execute("SELECT * FROM analyses WHERE id = ?", (case_id,)).fetchone()
                if not row:
                    return self.send_json({"error": "Case not found"}, 404)
                self.send_bytes(
                    build_pdf(dict(row)),
                    "application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="melanoma-analysis-{case_id[:8]}.pdf"'},
                )
                return
            if path.startswith("/media/"):
                parts = path.strip("/").split("/")
                if len(parts) == 3 and parts[2] in {"image", "heatmap"}:
                    column = "image_path" if parts[2] == "image" else "heatmap_path"
                    with db_connect() as db:
                        row = db.execute(f"SELECT {column} FROM analyses WHERE id = ?", (parts[1],)).fetchone()
                    if row and Path(row[column]).is_file():
                        data = Path(row[column]).read_bytes()
                        return self.send_bytes(data, mimetypes.guess_type(row[column])[0] or "image/jpeg")
                return self.send_json({"error": "Image not found"}, 404)
            if path in {"/", "/index.html"}:
                return self.send_bytes((WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            static_path = (WEB / path.lstrip("/")).resolve()
            if WEB.resolve() not in static_path.parents or not static_path.is_file():
                return self.send_json({"error": "Not found"}, 404)
            return self.send_bytes(
                static_path.read_bytes(), mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
            )
        except Exception as exc:
            print(f"GET {path} failed: {exc}")
            return self.send_json({"error": str(exc)}, 500)

    def do_POST(self):
        if urlparse(self.path).path != "/api/analyze":
            return self.send_json({"error": "Not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES * 2:
                return self.send_json({"error": "Image upload is empty or exceeds the 12 MB limit."}, 413)
            payload = json.loads(self.rfile.read(length))
            data_url = payload.get("image", "")
            match = re.fullmatch(r"data:image/(png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=\r\n]+)", data_url)
            if not match:
                return self.send_json({"error": "Upload a PNG, JPG, or WebP image."}, 400)
            raw = base64.b64decode(match.group(2), validate=True)
            if len(raw) > MAX_UPLOAD_BYTES:
                return self.send_json({"error": "Image exceeds the 12 MB limit."}, 413)
            with Image.open(io.BytesIO(raw)) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                if image.width < 32 or image.height < 32:
                    return self.send_json({"error": "Image is too small. Use an image at least 32 × 32 pixels."}, 400)
                image.thumbnail((2048, 2048))
                image.load()
            APP_DATA.mkdir(parents=True, exist_ok=True)
            UPLOADS.mkdir(parents=True, exist_ok=True)
            HEATMAPS.mkdir(parents=True, exist_ok=True)
            predictions, primary, heatmap = predict_all(image)
            if heatmap is None:
                raise RuntimeError("Attention checkpoint is not available, so no Grad-CAM was produced.")
            case_id = str(uuid.uuid4())
            image_path = UPLOADS / f"{case_id}.jpg"
            heatmap_path = HEATMAPS / f"{case_id}.jpg"
            image.save(image_path, format="JPEG", quality=92)
            heatmap.save(heatmap_path, format="JPEG", quality=92)
            now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
            selected = predictions[primary]
            with db_connect() as db:
                db.execute(
                    "INSERT INTO analyses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        case_id,
                        now,
                        Path(payload.get("name") or "uploaded-image.jpg").name[:180],
                        str(image_path),
                        primary,
                        selected["predicted_class"],
                        selected["confidence"],
                        json.dumps(selected["probabilities"]),
                        json.dumps(predictions),
                        str(heatmap_path),
                    ),
                )
            return self.send_json(
                {
                    "id": case_id,
                    "created_at": now,
                    "original_name": Path(payload.get("name") or "uploaded-image.jpg").name,
                    "primary_model": primary,
                    "predicted_class": selected["predicted_class"],
                    "confidence": selected["confidence"],
                    "probabilities": selected["probabilities"],
                    "results": predictions,
                    "image_url": f"/media/{case_id}/image",
                    "heatmap_url": f"/media/{case_id}/heatmap",
                    "report_url": f"/api/report/{case_id}.pdf",
                },
            )
        except (json.JSONDecodeError, ValueError) as exc:
            return self.send_json({"error": str(exc) or "Invalid request."}, 400)
        except Exception as exc:
            print(f"POST /api/analyze failed: {exc}")
            return self.send_json({"error": str(exc)}, 500)


def main():
    initialize_database()
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("DASHBOARD_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"MelanoVexa Skin Lesion Research Studio running at http://{host}:{port}")
    print(f"Inference device: {DEVICE}; checkpoint files are loaded from out/ on demand.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
