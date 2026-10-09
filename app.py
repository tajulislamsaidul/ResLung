import os
import glob
import json
import csv
import sqlite3
import threading
import traceback
import webbrowser
import math
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tensorflow as tf

# Optional PDF/XLSX dependencies
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as PDFImage, PageBreak, KeepTogether
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from openpyxl import Workbook
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# ============================================================
# LUNGCANET 4.0 - EXPLAINABLE LUNG CT RESEARCH DASHBOARD
# ============================================================

APP_NAME = "LungCANet"
APP_VERSION = "4.1.0"
IMG_SIZE = (224, 224)
CLASS_LABELS = ["Benign", "Malignant", "Normal"]
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
LOW_CONFIDENCE_THRESHOLD = 0.70
HIGH_CONFIDENCE_THRESHOLD = 0.85

GAMMA_VALUE = 0.9
CLAHE_CLIP = 3.0
CLAHE_TILE = (8, 8)
DEFAULT_GRADCAM_ALPHA = 0.45

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
HISTORY_DIR = os.path.join(OUTPUT_DIR, "history")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
BATCH_DIR = os.path.join(OUTPUT_DIR, "batches")
DB_PATH = os.path.join(OUTPUT_DIR, "lungcanet_history.db")

# ============================================================
# LIGHT / DARK THEMES
# ============================================================

THEMES = {
    "light": {
        "BG": "#EEF4FB", "PANEL": "#FFFFFF", "PANEL_SOFT": "#F6FAFF",
        "NAVY": "#071A2F", "NAVY_2": "#0E355A", "BLUE": "#0B84FF",
        "BLUE_DARK": "#0666D8", "TEAL": "#10B6B3", "GREEN": "#19A974",
        "RED": "#E5485D", "ORANGE": "#F59E0B", "PURPLE": "#7C4DFF",
        "TEXT": "#102A43", "TEXT_2": "#35516F", "MUTED": "#6E8298",
        "BORDER": "#D7E3F0", "WHITE": "#FFFFFF", "BLACK": "#091523",
        "CANVAS": "#F4F8FC", "ORANGE_LIGHT": "#FFF7E6", "GREEN_LIGHT": "#E9FAF1",
        "RED_LIGHT": "#FFF0F2", "BLUE_LIGHT": "#E8F3FF", "TEAL_LIGHT": "#E5FBFA",
        "PURPLE_LIGHT": "#F1EDFF", "SHADOW": "#C8D6E5", "INK_SOFT": "#DDE8F4",
    },
    "dark": {
        "BG": "#07111E", "PANEL": "#0F1D2C", "PANEL_SOFT": "#13263A",
        "NAVY": "#04101D", "NAVY_2": "#0B2742", "BLUE": "#34A4FF",
        "BLUE_DARK": "#1377C7", "TEAL": "#24D2CB", "GREEN": "#42D98A",
        "RED": "#FF6E79", "ORANGE": "#F8B24A", "PURPLE": "#A98BFF",
        "TEXT": "#EDF6FF", "TEXT_2": "#C6D8EA", "MUTED": "#91A8BD",
        "BORDER": "#284056", "WHITE": "#EAF3FC", "BLACK": "#FFFFFF",
        "CANVAS": "#0C1A29", "ORANGE_LIGHT": "#392D18", "GREEN_LIGHT": "#133B29",
        "RED_LIGHT": "#3A1E25", "BLUE_LIGHT": "#12314C", "TEAL_LIGHT": "#103A3A",
        "PURPLE_LIGHT": "#2B214B", "SHADOW": "#06101B", "INK_SOFT": "#193148",
    },
}

# ============================================================
# MODEL CONFIGURATION
# ============================================================

_ARCH_CONFIG = {
    "lungcanet": {"preprocess_fn": None, "use_raw_255": False},
    "mobilenet": {"preprocess_fn": tf.keras.applications.mobilenet_v2.preprocess_input, "use_raw_255": False},
    "efficientnetb3": {"preprocess_fn": None, "use_raw_255": True},
    "efficientnet": {"preprocess_fn": None, "use_raw_255": True},
    "densenet": {"preprocess_fn": tf.keras.applications.densenet.preprocess_input, "use_raw_255": False},
    "resnet": {"preprocess_fn": tf.keras.applications.resnet50.preprocess_input, "use_raw_255": False},
}
_DEFAULT_CFG = {"preprocess_fn": None, "use_raw_255": False}
MODEL_CACHE = {}
MODEL_LOCK = threading.Lock()


def resolve_arch_config(filename):
    name = os.path.basename(filename).lower()
    keys = ["efficientnetb3", "mobilenet", "densenet", "resnet", "lungcanet", "efficientnet"]
    for key in keys:
        if key in name:
            return _ARCH_CONFIG[key]
    return _DEFAULT_CFG


def discover_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = glob.glob(os.path.join(MODELS_DIR, "*.keras")) + glob.glob(os.path.join(MODELS_DIR, "*.h5"))
    return sorted(os.path.basename(path) for path in paths)


def load_model_cached(filename):
    with MODEL_LOCK:
        if filename in MODEL_CACHE:
            return MODEL_CACHE[filename]
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model not found:\n{path}")
        model = tf.keras.models.load_model(path, compile=False)
        MODEL_CACHE[filename] = model
        return model


def describe_model(model, filename):
    total_params = int(model.count_params()) if hasattr(model, "count_params") else None
    trainable = None
    non_trainable = None
    try:
        trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
        non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
    except Exception:
        pass
    return {
        "name": filename,
        "input_shape": str(getattr(model, "input_shape", "Unknown")),
        "output_shape": str(getattr(model, "output_shape", "Unknown")),
        "layers": len(getattr(model, "layers", [])),
        "parameters": total_params,
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
        "architecture_config": resolve_arch_config(filename),
    }

# ============================================================
# IMAGE PROCESSING / PREDICTION
# ============================================================


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Unable to read image.")
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def gamma_correction(image, gamma):
    if gamma <= 0:
        raise ValueError("Gamma must be greater than zero.")
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def preprocess_image(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    img_clahe = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    img_gamma = gamma_correction(img_clahe, GAMMA_VALUE)
    img_norm = cv2.normalize(img_gamma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.resize(img_norm, IMG_SIZE, interpolation=cv2.INTER_AREA)


def apply_model_scaling(image, cfg):
    image = image.astype(np.float32)
    if cfg["use_raw_255"]:
        return image
    if cfg["preprocess_fn"] is not None:
        return cfg["preprocess_fn"](image.copy())
    return image / 255.0


def normalize_predictions(predictions):
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1)
    if predictions.size != 3:
        raise ValueError("This model must produce exactly 3 outputs: Benign / Malignant / Normal")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Model returned invalid values.")
    total = float(np.sum(predictions))
    if np.all(predictions >= 0) and np.isclose(total, 1.0, atol=0.05):
        predictions = predictions / total
    else:
        predictions = predictions - np.max(predictions)
        exp_values = np.exp(predictions)
        predictions = exp_values / np.sum(exp_values)
    return predictions.astype(np.float32)


def find_last_conv_layer(model):
    found = None

    def search(current_model):
        nonlocal found
        if not hasattr(current_model, "layers"):
            return
        for layer in current_model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                found = layer
            if hasattr(layer, "layers"):
                search(layer)

    search(model)
    return found


def compute_gradcam(model, batch, class_idx):
    target_layer = find_last_conv_layer(model)
    if target_layer is None:
        raise RuntimeError("No convolutional layer found. Grad-CAM cannot be generated.")
    try:
        grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
    except Exception as error:
        raise RuntimeError(f"Grad-CAM is not compatible with this model architecture.\n\n{error}")
    batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch_tensor, training=False)
        predictions = tf.reshape(predictions, [tf.shape(predictions)[0], -1])
        class_score = predictions[0, class_idx]
    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        raise RuntimeError("Grad-CAM gradients are unavailable.")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    maximum = tf.reduce_max(heatmap)
    max_value = float(maximum.numpy())
    heatmap = heatmap / maximum if max_value > 1e-8 else tf.zeros_like(heatmap)
    return heatmap.numpy(), target_layer.name


def create_gradcam_overlay(image_bgr, heatmap, alpha=DEFAULT_GRADCAM_ALPHA, colormap=cv2.COLORMAP_JET):
    heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.resize(heatmap_color, (image_bgr.shape[1], image_bgr.shape[0]))
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def confidence_level(confidence):
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH CONFIDENCE", "high"
    if confidence >= LOW_CONFIDENCE_THRESHOLD:
        return "MODERATE CONFIDENCE", "moderate"
    return "LOW CONFIDENCE - REVIEW RECOMMENDED", "low"


def generate_report_id():
    stamp = datetime.now()
    return stamp.strftime("LCN-%Y%m%d-%H%M%S-%f")[:-3]

# ============================================================
# DATABASE / HISTORY
# ============================================================


def db_connect():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = db_connect()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT UNIQUE,
            created_at TEXT,
            case_id TEXT,
            patient_initials TEXT,
            age TEXT,
            sex TEXT,
            scan_date TEXT,
            referring_center TEXT,
            reviewer TEXT,
            notes TEXT,
            image_path TEXT,
            model TEXT,
            prediction TEXT,
            confidence REAL,
            benign REAL,
            malignant REAL,
            normal REAL,
            confidence_level TEXT,
            gradcam_layer TEXT,
            pdf_path TEXT,
            json_path TEXT,
            preprocessed_path TEXT,
            gradcam_path TEXT
        )"""
    )
    conn.commit()
    conn.close()


def save_history(record):
    conn = db_connect()
    columns = [
        "report_id", "created_at", "case_id", "patient_initials", "age", "sex",
        "scan_date", "referring_center", "reviewer", "notes", "image_path",
        "model", "prediction", "confidence", "benign", "malignant", "normal",
        "confidence_level", "gradcam_layer", "pdf_path", "json_path",
        "preprocessed_path", "gradcam_path",
    ]
    values = (
            record.get("report_id"), record.get("created_at"), record.get("case_id"),
            record.get("patient_initials"), record.get("age"), record.get("sex"),
            record.get("scan_date"), record.get("referring_center"), record.get("reviewer"),
            record.get("notes"), record.get("image_path"), record.get("model"),
            record.get("prediction"), record.get("confidence"),
            record.get("probabilities", {}).get("Benign", 0.0),
            record.get("probabilities", {}).get("Malignant", 0.0),
            record.get("probabilities", {}).get("Normal", 0.0),
            record.get("confidence_level"), record.get("gradcam_layer"), record.get("pdf_path"),
            record.get("json_path"), record.get("preprocessed_path"), record.get("gradcam_path"),
    )
    if len(values) != len(columns):
        raise RuntimeError(f"History record mismatch: {len(values)} values for {len(columns)} columns.")
    placeholders = ",".join("?" for _ in columns)
    conn.execute(
        f"""INSERT OR REPLACE INTO analyses
        ({",".join(columns)})
        VALUES ({placeholders})""",
        values,
    )
    conn.commit()
    conn.close()


def fetch_history(limit=200):
    conn = db_connect()
    rows = conn.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def fetch_case(report_id):
    conn = db_connect()
    row = conn.execute("SELECT * FROM analyses WHERE report_id = ?", (report_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

# ============================================================
# PDF / JSON / EXPORT
# ============================================================


def safe_pdf_text(value):
    if value is None or value == "":
        return "-"
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pdf_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    navy = colors.HexColor("#102A43")
    muted = colors.HexColor("#718096")
    border = colors.HexColor("#D9E2EC")
    canvas.setStrokeColor(border)
    canvas.setLineWidth(0.7)
    canvas.line(18 * mm, height - 16 * mm, width - 18 * mm, height - 16 * mm)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(navy)
    canvas.drawString(18 * mm, height - 13 * mm, "LungCANet")
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(muted)
    canvas.drawRightString(width - 18 * mm, height - 13 * mm, "Explainable AI - Lung CT Research Dashboard")
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(18 * mm, 9 * mm, "Research / Educational Use Only")
    canvas.drawRightString(width - 18 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def pdf_image(path, max_width, max_height):
    if not path or not os.path.isfile(path):
        return None
    img = Image.open(path)
    width, height = img.size
    if width <= 0 or height <= 0:
        return None
    scale = min(max_width / width, max_height / height)
    return PDFImage(path, width=width * scale, height=height * scale)


def generate_probability_chart(path, predictions, title="Class Probability"):
    width, height = 1100, 460
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font_big = ImageFont.truetype("arial.ttf", 34)
        font = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 23)
    except Exception:
        font_big = font = font_small = ImageFont.load_default()
    draw.text((35, 25), title, fill="#102A43", font=font_big)
    left, top, right = 260, 110, 1020
    max_bar = right - left
    for i, label in enumerate(CLASS_LABELS):
        y = top + i * 105
        pct = float(predictions[i])
        draw.text((35, y + 12), label, fill="#334E68", font=font)
        draw.rounded_rectangle((left, y, right, y + 48), radius=10, fill="#E7F1FB", outline="#D9E2EC")
        draw.rounded_rectangle((left, y, left + int(max_bar * pct), y + 48), radius=10, fill="#1976D2")
        draw.text((right - 95, y + 12), f"{pct * 100:.1f}%", fill="#172B4D", font=font_small)
    img.save(path)
    return path


def generate_professional_pdf_report(output_path, record, model_info=None, comparison_rows=None, ensemble_info=None):
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("PDF reporting requires ReportLab. Install it with: pip install reportlab")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#102A43")))
    styles.add(ParagraphStyle(name="Sub", parent=styles["Normal"], fontSize=9, leading=13, textColor=colors.HexColor("#718096")))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#102A43"), spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=8.4, leading=11, textColor=colors.HexColor("#334E68")))
    styles.add(ParagraphStyle(name="Metric", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=17, leading=20, alignment=TA_CENTER, textColor=colors.HexColor("#102A43")))
    styles.add(ParagraphStyle(name="Label", parent=styles["BodyText"], fontSize=7.5, leading=9, alignment=TA_CENTER, textColor=colors.HexColor("#718096")))
    styles.add(ParagraphStyle(name="Warn", parent=styles["BodyText"], fontSize=7.5, leading=10, textColor=colors.HexColor("#79551D")))

    probabilities = record["probabilities"]
    confidence = float(record["confidence"])
    pred_class = record["prediction"]
    story = [
        Paragraph("LungCANet Analysis Report", styles["ReportTitle"]),
        Paragraph("Explainable AI assessment of a lung CT image", styles["Sub"]),
        Spacer(1, 3 * mm),
    ]

    summary = Table([
        [Paragraph("PREDICTED CLASS", styles["Label"]), Paragraph("CONFIDENCE", styles["Label"]), Paragraph("REVIEW STATUS", styles["Label"])],
        [Paragraph(safe_pdf_text(pred_class), styles["Metric"]), Paragraph(f"{confidence * 100:.2f}%", styles["Metric"]), Paragraph(safe_pdf_text(confidence_level(confidence)[0]), styles["BodySmall"])],
    ], colWidths=[57 * mm, 46 * mm, 72 * mm])
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F9FC")), ("BOX", (0, 0), (-1, -1), .7, colors.HexColor("#D9E2EC")),
        ("INNERGRID", (0, 0), (-1, -1), .4, colors.HexColor("#E5ECF3")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [summary, Spacer(1, 4 * mm)]

    story.append(Paragraph("Case & Analysis Information", styles["Section"]))
    metadata = [
        ["Report ID", record.get("report_id")], ["Case ID", record.get("case_id")], ["Patient initials", record.get("patient_initials")],
        ["Age / Sex", f"{record.get('age') or '-'} / {record.get('sex') or '-'}"], ["Scan date", record.get("scan_date")],
        ["Referring center", record.get("referring_center")], ["Reviewer", record.get("reviewer")],
        ["Generated", record.get("created_at")], ["Input image", os.path.basename(record.get("image_path", ""))],
        ["Model", record.get("model")], ["Preprocessing", "CLAHE + Gamma + normalization + resize to 224 x 224"],
        ["Grad-CAM layer", record.get("gradcam_layer") or "Unavailable"],
    ]
    meta_table = Table([[safe_pdf_text(a), safe_pdf_text(b)] for a, b in metadata], colWidths=[42 * mm, 133 * mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F7F9FC")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#D9E2EC")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#334E68")), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)

    chart_path = os.path.join(os.path.dirname(output_path), f"{record['report_id']}_probability_chart.png")
    generate_probability_chart(chart_path, [probabilities[x] for x in CLASS_LABELS])

    story.append(Paragraph("Class Probabilities", styles["Section"]))
    prob_rows = [["Class", "Probability", "Level"]]
    for label in CLASS_LABELS:
        pct = float(probabilities[label]) * 100
        prob_rows.append([label, f"{pct:.2f}%", "High" if pct >= 70 else "Moderate" if pct >= 40 else "Low"])
    prob_table = Table(prob_rows, colWidths=[70 * mm, 45 * mm, 60 * mm])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102A43")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#D9E2EC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
        ("ALIGN", (1, 1), (2, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(prob_table)
    story += [Spacer(1, 3 * mm), pdf_image(chart_path, 170 * mm, 70 * mm) or Spacer(1, 1 * mm)]

    if comparison_rows:
        story.append(PageBreak())
        story.append(Paragraph("Multi-Model Comparison", styles["Section"]))
        rows = [["Model", "Prediction", "Confidence"]]
        for row in comparison_rows:
            rows.append([safe_pdf_text(row["model"]), safe_pdf_text(row["prediction"]), f"{row['confidence'] * 100:.2f}%"])
        table = Table(rows, colWidths=[88 * mm, 47 * mm, 40 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163B5C")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#D9E2EC")), ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ]))
        story.append(table)
        if ensemble_info:
            story.append(Spacer(1, 4 * mm))
            story.append(Paragraph(f"Ensemble prediction: <b>{safe_pdf_text(ensemble_info['prediction'])}</b> ({ensemble_info['confidence'] * 100:.2f}%), using {ensemble_info['model_count']} model(s).", styles["BodySmall"]))

    story.append(PageBreak())
    story.append(Paragraph("Visual Analysis", styles["Section"]))
    visual_data = []
    input_img = pdf_image(record.get("image_path"), 82 * mm, 72 * mm)
    proc_img = pdf_image(record.get("preprocessed_path"), 82 * mm, 72 * mm)
    visual_data.append([input_img or Paragraph("Original image unavailable", styles["BodySmall"]), proc_img or Paragraph("Preprocessed image unavailable", styles["BodySmall"])])
    visual_data.append([Paragraph("Original CT image", styles["Label"]), Paragraph("Preprocessed CT image", styles["Label"])])
    visual_table = Table(visual_data, colWidths=[87 * mm, 87 * mm])
    visual_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(visual_table)
    story.append(Spacer(1, 4 * mm))
    grad = pdf_image(record.get("gradcam_path"), 165 * mm, 90 * mm)
    story.append(grad or Paragraph("Grad-CAM could not be generated for this model.", styles["BodySmall"]))
    if grad:
        story.append(Paragraph("Grad-CAM visualization for the predicted class.", styles["BodySmall"]))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Research Notes", styles["Section"]))
    story.append(Paragraph(safe_pdf_text(record.get("notes")), styles["BodySmall"]))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("Reviewer / Supervisor Sign-off", styles["Section"]))
    signature_rows = [["Prepared by", safe_pdf_text(record.get("reviewer") or "____________________________")],
                      ["Reviewer signature", "_______________________________________________"],
                      ["Date", safe_pdf_text(record.get("created_at") or "")]]
    sig_table = Table(signature_rows, colWidths=[42 * mm, 133 * mm])
    sig_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#D9E2EC")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story.append(sig_table)
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("IMPORTANT: LungCANet is a research and educational prototype. This report is not a medical diagnosis, does not replace a radiologist or physician, and should not be used as the sole basis for clinical decisions.", styles["Warn"]))

    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=22 * mm, bottomMargin=19 * mm, title="LungCANet Analysis Report", author="LungCANet")
    doc.build(story, onFirstPage=pdf_header_footer, onLaterPages=pdf_header_footer)
    return output_path


def save_analysis_json(path, record, model_info=None, comparison_rows=None, ensemble_info=None):
    payload = dict(record)
    payload["model_info"] = model_info
    payload["model_comparison"] = comparison_rows or []
    payload["ensemble"] = ensemble_info
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def export_history_csv(rows, path):
    fields = ["report_id", "created_at", "case_id", "patient_initials", "age", "sex", "scan_date", "referring_center", "reviewer", "model", "prediction", "confidence", "benign", "malignant", "normal", "confidence_level", "pdf_path"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def export_history_xlsx(rows, path):
    if not OPENPYXL_AVAILABLE:
        raise RuntimeError("Excel export requires openpyxl. Install it with: pip install openpyxl")
    wb = Workbook()
    ws = wb.active
    ws.title = "LungCANet Results"
    fields = ["report_id", "created_at", "case_id", "patient_initials", "age", "sex", "scan_date", "referring_center", "reviewer", "model", "prediction", "confidence", "benign", "malignant", "normal", "confidence_level", "pdf_path"]
    ws.append(fields)
    for cell in ws[1]:
        cell.font = cell.font.copy(bold=True)
    for row in rows:
        ws.append([row.get(field, "") for field in fields])
    ws.freeze_panes = "A2"
    for col in ws.columns:
        max_len = min(45, max(len(str(cell.value or "")) for cell in col) + 2)
        ws.column_dimensions[col[0].column_letter].width = max_len
    wb.save(path)
    return path

# ============================================================
# MAIN APPLICATION
# ============================================================

class LungCANetApp:
    def __init__(self, root):
        self.root = root
        self.theme_name = "light"
        self.root.title(f"{APP_NAME} | Explainable Lung CT AI")
        self.root.geometry("1580x980")
        self.root.minsize(1280, 820)
        self.busy = False

        self.selected_image_path = None
        self.original_image = None
        self.processed_image = None
        self.gradcam_image = None
        self.last_heatmap = None
        self.current_record = None
        self.current_report_path = None
        self.history_rows = []
        self.zoom_factor = 1.0
        self.gradcam_alpha = DEFAULT_GRADCAM_ALPHA
        self.gradcam_colormap = "JET"

        self.models = discover_models()
        init_database()

        self.setup_styles()
        self.build_header()
        self.build_main()
        self.build_statusbar()
        self.refresh_models()
        self.refresh_history()
        self.update_model_info()

    @property
    def C(self):
        return THEMES[self.theme_name]

    # ---------- styles ----------
    def setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Professional.TCombobox", padding=9, font=("Segoe UI", 9),
            fieldbackground=self.C["CANVAS"], background=self.C["CANVAS"],
            foreground=self.C["TEXT"], bordercolor=self.C["BORDER"], lightcolor=self.C["BORDER"],
            arrowcolor=self.C["BLUE"]
        )
        for sty, color in [("Confidence.Horizontal.TProgressbar",self.C["PURPLE"]),("Benign.Horizontal.TProgressbar",self.C["GREEN"]),("Malignant.Horizontal.TProgressbar",self.C["RED"]),("Normal.Horizontal.TProgressbar",self.C["BLUE"]),("Blue.Horizontal.TProgressbar",self.C["BLUE"])]:
            style.configure(sty,troughcolor=self.C["BORDER"],background=color,bordercolor=self.C["BORDER"],lightcolor=color,darkcolor=color)
        style.configure(
            "Treeview", rowheight=29, font=("Segoe UI", 8),
            fieldbackground=self.C["PANEL"], background=self.C["PANEL"], foreground=self.C["TEXT"],
            bordercolor=self.C["BORDER"]
        )
        style.configure(
            "Treeview.Heading", font=("Segoe UI", 8, "bold"),
            background=self.C["BLUE_LIGHT"], foreground=self.C["NAVY"], relief="flat"
        )
        style.map("Treeview", background=[("selected", self.C["TEAL_LIGHT"])], foreground=[("selected", self.C["NAVY"])])
        style.configure("TNotebook", background=self.C["PANEL"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(15, 8), font=("Segoe UI", 8, "bold"))

    def style_button(self, parent, text, command, bg=None, fg=None, state="normal"):
        base = bg or self.C["BLUE"]
        hover = self.C["BLUE_DARK"] if base not in (self.C["PANEL"], self.C["CANVAS"]) else self.C["BLUE_LIGHT"]
        button = tk.Button(
            parent, text=text, command=command,
            font=("Segoe UI", 9, "bold"),
            fg=fg or self.C["WHITE"], bg=base,
            activebackground=hover, activeforeground=self.C["WHITE"],
            relief="flat", bd=0, cursor="hand2", pady=9, state=state
        )
        button.bind("<Enter>", lambda e: button.config(bg=hover) if button["state"] != "disabled" else None)
        button.bind("<Leave>", lambda e: button.config(bg=base) if button["state"] != "disabled" else None)
        return button

    def make_gradient(self, width, height, left, right):
        width=max(2,int(width)); height=max(2,int(height))
        a=np.array(tuple(int(left[i:i+2],16) for i in (1,3,5)),dtype=float)
        b=np.array(tuple(int(right[i:i+2],16) for i in (1,3,5)),dtype=float)
        arr=np.zeros((height,width,3),dtype=np.uint8)
        for x in range(width):
            t=x/max(1,width-1); arr[:,x,:]=np.clip(a*(1-t)+b*t,0,255).astype(np.uint8)
        return Image.fromarray(arr,"RGB")

    def draw_lung_logo(self, parent, size=54, bg=None):
        bg=bg or self.C["NAVY_2"]
        c=tk.Canvas(parent,width=size,height=size,bg=bg,highlightthickness=0,bd=0)
        s=size/54.0
        c.create_line(27*s,8*s,27*s,16*s,fill=self.C["TEAL"],width=max(2,int(2*s)))
        c.create_arc(10*s,12*s,27*s,40*s,start=95,extent=170,style="arc",outline=self.C["BLUE"],width=max(2,int(3*s)))
        c.create_arc(27*s,12*s,44*s,40*s,start= -85,extent=170,style="arc",outline=self.C["BLUE"],width=max(2,int(3*s)))
        c.create_line(27*s,16*s,20*s,23*s,fill=self.C["TEAL"],width=max(1,int(2*s)))
        c.create_line(27*s,16*s,34*s,23*s,fill=self.C["TEAL"],width=max(1,int(2*s)))
        for px,py in [(17,25),(37,25),(14,33),(40,33),(27,21)]:
            r=2.2*s; c.create_oval((px*s-r,py*s-r,px*s+r,py*s+r),fill=self.C["PURPLE"],outline="")
        c.create_arc(19*s,34*s,35*s,50*s,start=200,extent=140,style="arc",outline=self.C["TEAL"],width=max(1,int(2*s)))
        return c

    def section_title(self, parent, title, subtitle=None, accent=None):
        row=tk.Frame(parent,bg=self.C["PANEL"]); row.pack(fill="x",padx=14,pady=(12,6))
        strip=tk.Frame(row,bg=accent or self.C["BLUE"],width=4,height=30); strip.pack(side="left",fill="y",padx=(0,9)); strip.pack_propagate(False)
        body=tk.Frame(row,bg=self.C["PANEL"]); body.pack(side="left",fill="x",expand=True)
        tk.Label(body,text=title,font=("Segoe UI",9,"bold"),fg=self.C["TEXT"],bg=self.C["PANEL"]).pack(anchor="w")
        if subtitle:
            tk.Label(body,text=subtitle,font=("Segoe UI",7),fg=self.C["MUTED"],bg=self.C["PANEL"]).pack(anchor="w",pady=(2,0))
        return row

    # ---------- header ----------
    def build_header(self):
        self.header=tk.Frame(self.root,bg=self.C["NAVY"],height=92)
        self.header.pack(fill="x"); self.header.pack_propagate(False)
        grad=self.make_gradient(1600,92,self.C["NAVY"],"#154B76")
        self.header_bg=ImageTk.PhotoImage(grad)
        self.header_bg_label=tk.Label(self.header,image=self.header_bg,bd=0); self.header_bg_label.place(relx=0,rely=0,relwidth=1,relheight=1)
        brand=tk.Frame(self.header,bg=self.C["NAVY"]); brand.place(x=20,y=12)
        mark=self.draw_lung_logo(brand,58,self.C["NAVY"]); mark.pack(side="left")
        txt=tk.Frame(brand,bg=self.C["NAVY"]); txt.pack(side="left",padx=12)
        tk.Label(txt,text="LungCANet",font=("Segoe UI",23,"bold"),fg=self.C["WHITE"],bg=self.C["NAVY"]).pack(anchor="w")
        tk.Label(txt,text="Explainable Lung CT AI • Research & Analysis Suite",font=("Segoe UI",9),fg="#B8D2EA",bg=self.C["NAVY"]).pack(anchor="w",pady=(2,0))

        self.theme_button=self.style_button(self.header,"Dark Mode",self.toggle_theme,bg=self.C["BLUE"],fg=self.C["WHITE"])
        self.theme_button.pack(side="right",padx=16,pady=25)
        pill=tk.Label(self.header,text="RESEARCH MODE",font=("Segoe UI",8,"bold"),fg=self.C["WHITE"],bg=self.C["PURPLE"],padx=13,pady=7)
        pill.pack(side="right",padx=(8,4),pady=25)
        tk.Label(self.header,text=f"v{APP_VERSION}",font=("Segoe UI",8,"bold"),fg="#D4E7F7",bg=self.C["NAVY_2"],padx=10,pady=7).pack(side="right",pady=25)

    # ---------- main ----------
    def build_main(self):
        self.container = tk.Frame(self.root, bg=self.C["BG"])
        self.container.pack(fill="both", expand=True, padx=16, pady=14)
        self.build_sidebar()
        self.build_workspace_area()

    def build_sidebar(self):
        self.sidebar=tk.Frame(self.container,bg=self.C["PANEL"],width=304,highlightbackground=self.C["BORDER"],highlightthickness=1)
        self.sidebar.pack(side="left",fill="y",padx=(0,14)); self.sidebar.pack_propagate(False)
        head=tk.Frame(self.sidebar,bg=self.C["PANEL"]); head.pack(fill="x",padx=16,pady=(15,7))
        mark=self.draw_lung_logo(head,42,self.C["BLUE_LIGHT"]); mark.pack(side="left")
        tk.Label(head,text="ANALYSIS CONTROL",font=("Segoe UI",9,"bold"),fg=self.C["TEXT"],bg=self.C["PANEL"]).pack(side="left",padx=10)
        tk.Label(self.sidebar,text="Lung CT Assessment",font=("Segoe UI",17,"bold"),fg=self.C["TEXT"],bg=self.C["PANEL"]).pack(anchor="w",padx=18)
        tk.Label(self.sidebar,text="Prepare a case, select an AI model, and generate an explainable research result.",font=("Segoe UI",8),fg=self.C["MUTED"],bg=self.C["PANEL"],wraplength=260,justify="left").pack(anchor="w",padx=18,pady=(3,11))

        case_frame=tk.LabelFrame(self.sidebar,text=" 01 • Case Information ",font=("Segoe UI",8,"bold"),fg=self.C["BLUE"],bg=self.C["PANEL"],bd=1,relief="solid",labelanchor="nw")
        case_frame.pack(fill="x",padx=14,pady=6)
        self.case_vars={}
        fields=[("Case ID","case_id"),("Patient initials","patient_initials"),("Age","age"),("Scan date","scan_date"),("Referring center","referring_center")]
        for label,key in fields:
            row=tk.Frame(case_frame,bg=self.C["PANEL"]); row.pack(fill="x",padx=7,pady=2)
            tk.Label(row,text=label,width=15,anchor="w",font=("Segoe UI",8),fg=self.C["TEXT_2"],bg=self.C["PANEL"]).pack(side="left")
            var=tk.StringVar(); self.case_vars[key]=var
            tk.Entry(row,textvariable=var,font=("Segoe UI",8),bg=self.C["CANVAS"],fg=self.C["TEXT"],insertbackground=self.C["TEXT"],relief="flat",highlightthickness=1,highlightbackground=self.C["BORDER"]).pack(side="right",fill="x",expand=True,padx=(4,0),ipady=4)
        row=tk.Frame(case_frame,bg=self.C["PANEL"]); row.pack(fill="x",padx=7,pady=2)
        tk.Label(row,text="Sex",width=15,anchor="w",font=("Segoe UI",8),fg=self.C["TEXT_2"],bg=self.C["PANEL"]).pack(side="left")
        self.case_vars["sex"]=tk.StringVar(value="Not specified")
        ttk.Combobox(row,textvariable=self.case_vars["sex"],values=["Not specified","Female","Male","Other"],state="readonly",style="Professional.TCombobox").pack(side="right",fill="x",expand=True,padx=(4,0))
        tk.Label(case_frame,text="Research / reviewer notes",font=("Segoe UI",8),fg=self.C["TEXT_2"],bg=self.C["PANEL"]).pack(anchor="w",padx=7,pady=(6,2))
        self.notes_text=tk.Text(case_frame,height=3,font=("Segoe UI",8),bg=self.C["CANVAS"],fg=self.C["TEXT"],insertbackground=self.C["TEXT"],relief="flat",highlightthickness=1,highlightbackground=self.C["BORDER"])
        self.notes_text.pack(fill="x",padx=7,pady=(0,7))
        self.reviewer_var=tk.StringVar(); row=tk.Frame(case_frame,bg=self.C["PANEL"]); row.pack(fill="x",padx=7,pady=(0,7))
        tk.Label(row,text="Reviewer",width=15,anchor="w",font=("Segoe UI",8),fg=self.C["TEXT_2"],bg=self.C["PANEL"]).pack(side="left")
        tk.Entry(row,textvariable=self.reviewer_var,font=("Segoe UI",8),bg=self.C["CANVAS"],fg=self.C["TEXT"],insertbackground=self.C["TEXT"],relief="flat",highlightthickness=1,highlightbackground=self.C["BORDER"]).pack(side="right",fill="x",expand=True,padx=(4,0),ipady=4)

        model_frame=tk.LabelFrame(self.sidebar,text=" 02 • AI Model ",font=("Segoe UI",8,"bold"),fg=self.C["PURPLE"],bg=self.C["PANEL"],bd=1,relief="solid",labelanchor="nw")
        model_frame.pack(fill="x",padx=14,pady=6)
        self.model_var=tk.StringVar(); self.model_dropdown=ttk.Combobox(model_frame,textvariable=self.model_var,values=self.models,state="readonly",style="Professional.TCombobox")
        self.model_dropdown.pack(fill="x",padx=9,pady=(7,4)); self.model_dropdown.bind("<<ComboboxSelected>>",lambda _:self.update_model_info())
        self.model_info_var=tk.StringVar(value="No model selected")
        tk.Label(model_frame,textvariable=self.model_info_var,font=("Segoe UI",7),fg=self.C["MUTED"],bg=self.C["PANEL"],justify="left",wraplength=255).pack(anchor="w",padx=10,pady=(0,8))

        action_frame=tk.LabelFrame(self.sidebar,text=" 03 • Analysis Actions ",font=("Segoe UI",8,"bold"),fg=self.C["TEAL"],bg=self.C["PANEL"],bd=1,relief="solid",labelanchor="nw")
        action_frame.pack(fill="x",padx=14,pady=6)
        self.upload_button=self.style_button(action_frame,"＋  Open CT Image",self.upload_image,bg=self.C["BLUE"]); self.upload_button.pack(fill="x",padx=9,pady=(7,3))
        self.analyze_button=self.style_button(action_frame,"▶  Analyze CT Image",self.start_prediction,bg=self.C["PURPLE"]); self.analyze_button.pack(fill="x",padx=9,pady=3)
        self.batch_button=self.style_button(action_frame,"▦  Batch Analysis",self.start_batch_analysis,bg=self.C["TEAL"]); self.batch_button.pack(fill="x",padx=9,pady=3)
        self.compare_models_button=self.style_button(action_frame,"◈  Compare Models",self.start_model_comparison,bg=self.C["NAVY_2"]); self.compare_models_button.pack(fill="x",padx=9,pady=3)
        self.ensemble_button=self.style_button(action_frame,"✦  Ensemble Prediction",self.start_ensemble_prediction,bg=self.C["GREEN"]); self.ensemble_button.pack(fill="x",padx=9,pady=3)
        self.reset_button=self.style_button(action_frame,"Reset Workspace",self.reset_all,bg=self.C["CANVAS"],fg=self.C["TEXT_2"]); self.reset_button.pack(fill="x",padx=9,pady=(3,8))

        output_card=tk.Frame(self.sidebar,bg=self.C["PANEL_SOFT"],highlightbackground=self.C["BORDER"],highlightthickness=1); output_card.pack(fill="x",padx=14,pady=(7,5))
        tk.Label(output_card,text="CURRENT OUTPUT",font=("Segoe UI",7,"bold"),fg=self.C["MUTED"],bg=self.C["PANEL_SOFT"]).pack(anchor="w",padx=11,pady=(8,2))
        self.file_var=tk.StringVar(value="No CT image selected"); tk.Label(output_card,textvariable=self.file_var,font=("Segoe UI",8,"bold"),fg=self.C["TEXT"],bg=self.C["PANEL_SOFT"],wraplength=260,justify="left").pack(anchor="w",padx=11)
        self.report_var=tk.StringVar(value="No PDF report generated"); tk.Label(output_card,textvariable=self.report_var,font=("Segoe UI",7),fg=self.C["MUTED"],bg=self.C["PANEL_SOFT"],wraplength=260,justify="left").pack(anchor="w",padx=11,pady=(1,9))

        warning=tk.Frame(self.sidebar,bg=self.C["ORANGE_LIGHT"],highlightbackground=self.C["ORANGE"],highlightthickness=1); warning.pack(fill="x",padx=14,pady=8)
        tk.Label(warning,text="⚠  RESEARCH / EDUCATIONAL USE",font=("Segoe UI",7,"bold"),fg=self.C["ORANGE"],bg=self.C["ORANGE_LIGHT"]).pack(anchor="w",padx=10,pady=(8,3))
        tk.Label(warning,text="This prototype supports research and education. Results are not a clinical diagnosis.",font=("Segoe UI",7),fg=self.C["TEXT_2"],bg=self.C["ORANGE_LIGHT"],wraplength=255,justify="left").pack(anchor="w",padx=10,pady=(0,8))

    def build_workspace_area(self):
        self.workspace=tk.Frame(self.container,bg=self.C["BG"]); self.workspace.pack(side="right",fill="both",expand=True)
        titlebar=tk.Frame(self.workspace,bg=self.C["BG" ]); titlebar.pack(fill="x",pady=(0,9))
        icon=self.draw_lung_logo(titlebar,44,self.C["BLUE"]); icon.pack(side="left",padx=(2,10))
        title=tk.Frame(titlebar,bg=self.C["BG"]); title.pack(side="left")
        tk.Label(title,text="Research Analysis Workspace",font=("Segoe UI",21,"bold"),fg=self.C["TEXT"],bg=self.C["BG"]).pack(anchor="w")
        tk.Label(title,text="Unified CT viewer • Explainable AI • Model insights • Reporting",font=("Segoe UI",8),fg=self.C["MUTED"],bg=self.C["BG"]).pack(anchor="w",pady=(1,0))
        self.system_status=tk.Label(titlebar,text="● SYSTEM READY",font=("Segoe UI",8,"bold"),fg=self.C["GREEN"],bg=self.C["GREEN_LIGHT"],padx=13,pady=7); self.system_status.pack(side="right",pady=7)

        toolbar=tk.Frame(self.workspace,bg=self.C["PANEL"],highlightbackground=self.C["BORDER"],highlightthickness=1); toolbar.pack(fill="x",pady=(0,9))
        self.open_pdf_button=self.style_button(toolbar,"↗ Open PDF",self.open_latest_report,bg=self.C["BLUE"],fg=self.C["WHITE"]); self.open_pdf_button.pack(side="left",padx=7,pady=6)
        self.save_pdf_button=self.style_button(toolbar,"▣ Generate PDF",self.generate_report_from_current_result,bg=self.C["GREEN"],fg=self.C["WHITE"]); self.save_pdf_button.pack(side="left",padx=3,pady=6)
        self.zoom_out_button=self.style_button(toolbar,"−",lambda:self.change_zoom(-0.1),bg=self.C["CANVAS"],fg=self.C["TEXT_2"]); self.zoom_out_button.pack(side="left",padx=(14,2),pady=6)
        self.zoom_reset_button=self.style_button(toolbar,"100%",self.reset_zoom,bg=self.C["CANVAS"],fg=self.C["TEXT_2"]); self.zoom_reset_button.pack(side="left",padx=2,pady=6)
        self.zoom_in_button=self.style_button(toolbar,"+",lambda:self.change_zoom(0.1),bg=self.C["CANVAS"],fg=self.C["TEXT_2"]); self.zoom_in_button.pack(side="left",padx=2,pady=6)
        tk.Frame(toolbar,bg=self.C["BORDER"],width=1,height=24).pack(side="left",padx=10)
        tk.Label(toolbar,text="Grad-CAM",font=("Segoe UI",8,"bold"),fg=self.C["TEXT_2"],bg=self.C["PANEL"]).pack(side="left",padx=(2,4))
        self.alpha_scale=tk.Scale(toolbar,from_=0.1,to=0.9,resolution=0.05,orient="horizontal",length=120,showvalue=False,command=self.on_alpha_change,bg=self.C["PANEL"],fg=self.C["TEXT"],highlightthickness=0,bd=0,activebackground=self.C["PURPLE"]); self.alpha_scale.set(DEFAULT_GRADCAM_ALPHA); self.alpha_scale.pack(side="left")
        self.colormap_var=tk.StringVar(value="JET"); cmap_combo=ttk.Combobox(toolbar,textvariable=self.colormap_var,values=["JET","TURBO","HOT","VIRIDIS"],state="readonly",width=9,style="Professional.TCombobox"); cmap_combo.pack(side="left",padx=7,pady=6); cmap_combo.bind("<<ComboboxSelected>>",lambda _:self.on_colormap_change())

        paned=tk.PanedWindow(self.workspace,orient="vertical",sashrelief="flat",bg=self.C["BG"],bd=0); paned.pack(fill="both",expand=True)
        upper=tk.Frame(paned,bg=self.C["BG"]); lower=tk.Frame(paned,bg=self.C["BG"]); paned.add(upper,minsize=460); paned.add(lower,minsize=260)
        viewers=tk.Frame(upper,bg=self.C["BG"]); viewers.pack(fill="both",expand=True)

        image_card=tk.Frame(viewers,bg=self.C["PANEL"],highlightbackground=self.C["BORDER"],highlightthickness=1); image_card.pack(side="left",fill="both",expand=True,padx=(0,6))
        self.before_after=ttk.Notebook(image_card); self.before_after.pack(fill="both",expand=True,padx=8,pady=8)
        self.tab_original=tk.Frame(self.before_after,bg=self.C["CANVAS"]); self.tab_processed=tk.Frame(self.before_after,bg=self.C["CANVAS"]); self.before_after.add(self.tab_original,text="Original CT"); self.before_after.add(self.tab_processed,text="Preprocessed")
        self.original_canvas=tk.Canvas(self.tab_original,bg=self.C["CANVAS"],highlightthickness=0); self.original_canvas.pack(fill="both",expand=True)
        self.processed_canvas=tk.Canvas(self.tab_processed,bg=self.C["CANVAS"],highlightthickness=0); self.processed_canvas.pack(fill="both",expand=True)
        for canvas in (self.original_canvas,self.processed_canvas):
            canvas.bind("<Button-4>",lambda e:self.change_zoom(0.1)); canvas.bind("<Button-5>",lambda e:self.change_zoom(-0.1)); canvas.bind("<MouseWheel>",lambda e:self.change_zoom(0.1 if e.delta>0 else -0.1)); canvas.bind("<ButtonPress-1>",lambda e,c=canvas:c.scan_mark(e.x,e.y)); canvas.bind("<B1-Motion>",lambda e,c=canvas:c.scan_dragto(e.x,e.y,gain=1))

        grad_frame=tk.Frame(viewers,bg=self.C["PANEL"],highlightbackground=self.C["PURPLE"],highlightthickness=1,width=430); grad_frame.pack(side="right",fill="both",padx=(6,0)); grad_frame.pack_propagate(False)
        top=tk.Frame(grad_frame,bg=self.C["PANEL"]); top.pack(fill="x",padx=13,pady=(11,5))
        mini=self.draw_lung_logo(top,32,self.C["PURPLE_LIGHT"]); mini.pack(side="left",padx=(0,8))
        cap=tk.Frame(top,bg=self.C["PANEL"]); cap.pack(side="left")
        tk.Label(cap,text="GRAD-CAM EXPLAINABILITY",font=("Segoe UI",9,"bold"),fg=self.C["TEXT"],bg=self.C["PANEL"]).pack(anchor="w")
        tk.Label(cap,text="Attention regions influencing the predicted class",font=("Segoe UI",7),fg=self.C["MUTED"],bg=self.C["PANEL"]).pack(anchor="w",pady=(2,0))
        self.grad_canvas=tk.Canvas(grad_frame,bg=self.C["CANVAS"],highlightthickness=0); self.grad_canvas.pack(fill="both",expand=True,padx=9,pady=8)
        self.grad_canvas.bind("<Button-4>",lambda e:self.change_zoom(0.1)); self.grad_canvas.bind("<Button-5>",lambda e:self.change_zoom(-0.1)); self.grad_canvas.bind("<MouseWheel>",lambda e:self.change_zoom(0.1 if e.delta>0 else -0.1)); self.grad_canvas.bind("<ButtonPress-1>",lambda e:self.grad_canvas.scan_mark(e.x,e.y)); self.grad_canvas.bind("<B1-Motion>",lambda e:self.grad_canvas.scan_dragto(e.x,e.y,gain=1))
        self.grad_layer_var=tk.StringVar(value="Layer: -"); tk.Label(grad_frame,textvariable=self.grad_layer_var,font=("Segoe UI",7),fg=self.C["MUTED"],bg=self.C["PANEL"]).pack(anchor="w",padx=14,pady=(0,10))
        self.build_results_panel(lower)

    def build_results_panel(self, parent):
        left=tk.Frame(parent,bg=self.C["BG"]); left.pack(side="left",fill="both",expand=True,padx=(0,6))
        right=tk.Frame(parent,bg=self.C["BG"]); right.pack(side="right",fill="both",expand=True,padx=(6,0))

        card=tk.Frame(left,bg=self.C["PANEL"],highlightbackground=self.C["BORDER"],highlightthickness=1); card.pack(fill="both",expand=True)
        hdr=tk.Frame(card,bg=self.C["PANEL"]); hdr.pack(fill="x",padx=15,pady=(11,0))
        tk.Label(hdr,text="AI ANALYSIS RESULT",font=("Segoe UI",8,"bold"),fg=self.C["MUTED"],bg=self.C["PANEL"]).pack(side="left")
        self.review_var=tk.StringVar(value="Review status -"); self.review_label=tk.Label(hdr,textvariable=self.review_var,font=("Segoe UI",8,"bold"),fg=self.C["MUTED"],bg=self.C["PANEL_SOFT"],padx=10,pady=5); self.review_label.pack(side="right")

        self.prediction_var=tk.StringVar(value="-")
        self.prediction_label=tk.Label(card,textvariable=self.prediction_var,font=("Segoe UI",25,"bold"),fg=self.C["TEXT"],bg=self.C["PANEL"]); self.prediction_label.pack(anchor="w",padx=15,pady=(5,0))
        self.confidence_var=tk.StringVar(value="Confidence -"); tk.Label(card,textvariable=self.confidence_var,font=("Segoe UI",9),fg=self.C["TEXT_2"],bg=self.C["PANEL"]).pack(anchor="w",padx=15)
        self.confidence_bar=ttk.Progressbar(card,maximum=100,style="Confidence.Horizontal.TProgressbar"); self.confidence_bar.pack(fill="x",padx=15,pady=(7,10))

        self.probability_bars={}
        styles={"Benign":"Benign.Horizontal.TProgressbar","Malignant":"Malignant.Horizontal.TProgressbar","Normal":"Normal.Horizontal.TProgressbar"}
        for label in CLASS_LABELS:
            row=tk.Frame(card,bg=self.C["PANEL"]); row.pack(fill="x",padx=15,pady=2)
            tk.Label(row,text=label,width=11,anchor="w",font=("Segoe UI",8,"bold"),fg=self.C["TEXT"],bg=self.C["PANEL"]).pack(side="left")
            bar=ttk.Progressbar(row,maximum=100,style=styles[label]); bar.pack(side="left",fill="x",expand=True,padx=6)
            val=tk.Label(row,text="0.0%",width=7,anchor="e",font=("Segoe UI",8,"bold"),fg=self.C["TEXT"],bg=self.C["PANEL"]); val.pack(side="right")
            self.probability_bars[label]=(bar,val)
        self.ensemble_status_var=tk.StringVar(value="Ensemble: not run"); tk.Label(card,textvariable=self.ensemble_status_var,font=("Segoe UI",7),fg=self.C["MUTED"],bg=self.C["PANEL"]).pack(anchor="w",padx=15,pady=(7,9))

        history_card=tk.Frame(right,bg=self.C["PANEL"],highlightbackground=self.C["BORDER"],highlightthickness=1); history_card.pack(fill="both",expand=True)
        header=tk.Frame(history_card,bg=self.C["PANEL"]); header.pack(fill="x",padx=10,pady=8)
        tk.Label(header,text="HISTORY & CASE COMPARISON",font=("Segoe UI",8,"bold"),fg=self.C["TEXT"],bg=self.C["PANEL"]).pack(side="left")
        self.export_csv_btn=self.style_button(header,"CSV",self.export_history_csv_action,bg=self.C["CANVAS"],fg=self.C["TEXT_2"]); self.export_csv_btn.pack(side="right",padx=2)
        self.export_xlsx_btn=self.style_button(header,"Excel",self.export_history_xlsx_action,bg=self.C["CANVAS"],fg=self.C["TEXT_2"]); self.export_xlsx_btn.pack(side="right",padx=2)
        self.compare_case_btn=self.style_button(header,"Compare",self.compare_selected_cases,bg=self.C["BLUE"],fg=self.C["WHITE"]); self.compare_case_btn.pack(side="right",padx=2)
        cols=("report_id","case_id","prediction","confidence","model","created_at"); self.history_tree=ttk.Treeview(history_card,columns=cols,show="headings",selectmode="extended")
        headings={"report_id":"Report ID","case_id":"Case","prediction":"Prediction","confidence":"Confidence","model":"Model","created_at":"Date"}; widths={"report_id":135,"case_id":100,"prediction":90,"confidence":85,"model":135,"created_at":125}
        for col in cols: self.history_tree.heading(col,text=headings[col]); self.history_tree.column(col,width=widths[col],anchor="center")
        self.history_tree.pack(fill="both",expand=True,padx=8,pady=(0,8)); self.history_tree.bind("<Double-1>",self.load_selected_history_case)

    # ---------- status ----------
    def build_statusbar(self):
        self.statusbar = tk.Frame(self.root, bg=self.C["NAVY"], height=30); self.statusbar.pack(fill="x"); self.statusbar.pack_propagate(False)
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.statusbar, textvariable=self.status_var, font=("Segoe UI", 8), fg=self.C["WHITE"], bg=self.C["NAVY"], anchor="w").pack(side="left", padx=14)
        tk.Label(self.statusbar, text=f"{APP_NAME} {APP_VERSION}", font=("Segoe UI", 8), fg=self.C["MUTED"], bg=self.C["NAVY"]).pack(side="right", padx=14)

    def set_status(self, text, state="ready"):
        self.status_var.set(text)
        if state == "working":
            self.system_status.config(text="● ANALYZING", fg=self.C["ORANGE"], bg=self.C["ORANGE_LIGHT"])
        elif state == "error":
            self.system_status.config(text="● ERROR", fg=self.C["RED"], bg=self.C["RED_LIGHT"])
        else:
            self.system_status.config(text="● SYSTEM READY", fg=self.C["GREEN"], bg=self.C["GREEN_LIGHT"])

    # ---------- model ----------
    def refresh_models(self):
        self.models = discover_models()
        self.model_dropdown["values"] = self.models
        if self.models:
            self.model_dropdown.current(0)
            self.update_model_info()
        else:
            self.model_var.set("")
            self.model_info_var.set("No trained models found. Place .keras or .h5 files in the models folder.")
            self.set_status("No trained models found in the models folder.", "error")

    def update_model_info(self):
        name = self.model_var.get()
        if not name:
            return
        try:
            model = load_model_cached(name)
            info = describe_model(model, name)
            params = info["parameters"] or 0
            self.model_info_var.set(f"Layers: {info['layers']}  •  Parameters: {params:,}\nInput: {info['input_shape']}  •  Output: {info['output_shape']}")
        except Exception as error:
            self.model_info_var.set(f"Model info unavailable: {error}")

    # ---------- upload / analysis ----------
    def upload_image(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(title="Select Lung CT Image", filetypes=[("Medical image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if not path:
            return
        try:
            image = read_image(path)
        except Exception as error:
            messagebox.showerror("Image Error", str(error)); return
        self.selected_image_path = path
        self.original_image = image
        self.file_var.set(os.path.basename(path))
        self.reset_results_only()
        self.show_array_on_canvas(self.original_image, self.original_canvas, rgb=False)
        self.set_status("CT image loaded. Ready for analysis.", "ready")

    def collect_case_info(self):
        notes = self.notes_text.get("1.0", "end").strip()
        case_id = self.case_vars["case_id"].get().strip()
        if not case_id:
            case_id = "CASE-" + datetime.now().strftime("%Y%m%d-%H%M%S")
            self.case_vars["case_id"].set(case_id)
        return {
            "case_id": case_id, "patient_initials": self.case_vars["patient_initials"].get().strip(),
            "age": self.case_vars["age"].get().strip(), "sex": self.case_vars["sex"].get().strip(),
            "scan_date": self.case_vars["scan_date"].get().strip(), "referring_center": self.case_vars["referring_center"].get().strip(),
            "reviewer": self.reviewer_var.get().strip(), "notes": notes,
        }

    def start_prediction(self):
        if self.busy: return
        if not self.selected_image_path:
            messagebox.showwarning("No CT Image", "Please upload a CT image first."); return
        if not self.model_var.get():
            messagebox.showwarning("No Model", "Please select a trained model."); return
        self.lock_analysis_buttons(True)
        self.set_status("Analyzing CT image...", "working")
        args = (self.model_var.get(), self.selected_image_path, self.collect_case_info())
        threading.Thread(target=self.run_single_prediction, args=args, daemon=True).start()

    def run_single_prediction(self, model_name, image_path, case_info):
        try:
            model = load_model_cached(model_name)
            image = read_image(image_path)
            processed = preprocess_image(image)
            batch = np.expand_dims(apply_model_scaling(processed, resolve_arch_config(model_name)), 0).astype(np.float32)
            predictions = normalize_predictions(model.predict(batch, verbose=0))
            idx = int(np.argmax(predictions)); predicted = CLASS_LABELS[idx]; confidence = float(predictions[idx])
            gradcam_rgb = None; gradcam_path = None; gradcam_layer = None
            gradcam_error = None
            try:
                heatmap, gradcam_layer = compute_gradcam(model, batch, idx)
                self.root.after(0, self.set_last_heatmap, heatmap)
                gradcam_rgb = create_gradcam_overlay(processed, heatmap, self.gradcam_alpha, self.get_colormap())
            except Exception as error:
                gradcam_error = str(error)
            record, paths = self.create_and_save_result(image_path, processed, gradcam_rgb, model_name, predicted, confidence, predictions, gradcam_layer, case_info)
            record["gradcam_error"] = gradcam_error
            self.root.after(0, self.display_results, record, processed, gradcam_rgb)
        except Exception as error:
            self.root.after(0, self.show_error, str(error))

    def create_and_save_result(self, image_path, processed, gradcam_rgb, model_name, predicted, confidence, predictions, gradcam_layer, case_info, model_comparison=None, ensemble_info=None):
        os.makedirs(REPORTS_DIR, exist_ok=True); os.makedirs(HISTORY_DIR, exist_ok=True)
        report_id = generate_report_id()
        stem = Path(image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        preprocessed_path = os.path.join(HISTORY_DIR, f"{report_id}_preprocessed.png")
        gradcam_path = None
        cv2.imwrite(preprocessed_path, processed)
        if gradcam_rgb is not None:
            gradcam_path = os.path.join(HISTORY_DIR, f"{report_id}_gradcam.png")
            cv2.imwrite(gradcam_path, cv2.cvtColor(gradcam_rgb, cv2.COLOR_RGB2BGR))
        record = {
            "report_id": report_id, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **case_info,
            "image_path": os.path.abspath(image_path), "model": model_name, "prediction": predicted,
            "confidence": confidence, "probabilities": {label: float(predictions[i]) for i, label in enumerate(CLASS_LABELS)},
            "confidence_level": confidence_level(confidence)[0], "gradcam_layer": gradcam_layer,
            "preprocessed_path": preprocessed_path, "gradcam_path": gradcam_path,
            "pdf_path": None, "json_path": None,
        }
        pdf_path = os.path.join(REPORTS_DIR, f"{report_id}_{stem}_LungCANet_Report.pdf")
        json_path = os.path.join(HISTORY_DIR, f"{report_id}_analysis.json")
        try:
            model_info = describe_model(load_model_cached(model_name), model_name)
            generate_professional_pdf_report(pdf_path, record, model_info, model_comparison, ensemble_info)
            record["pdf_path"] = pdf_path
            save_analysis_json(json_path, record, model_info, model_comparison, ensemble_info)
            record["json_path"] = json_path
        except Exception:
            record["json_path"] = json_path
            try:
                save_analysis_json(json_path, record, None, model_comparison, ensemble_info)
            except Exception:
                pass
        save_history(record)
        return record, {"pdf": pdf_path if os.path.isfile(pdf_path) else None, "json": json_path}

    # ---------- model comparison / ensemble ----------
    def choose_models_dialog(self, title):
        available = discover_models()
        if not available:
            messagebox.showerror("No Models", "No .keras or .h5 models were found in the models folder."); return None
        dialog = tk.Toplevel(self.root); dialog.title(title); dialog.geometry("430x430"); dialog.transient(self.root); dialog.grab_set(); dialog.configure(bg=self.C["PANEL"])
        tk.Label(dialog, text=title, font=("Segoe UI", 12, "bold"), fg=self.C["TEXT"], bg=self.C["PANEL"]).pack(anchor="w", padx=15, pady=12)
        tk.Label(dialog, text="Select one or more models", font=("Segoe UI", 8), fg=self.C["MUTED"], bg=self.C["PANEL"]).pack(anchor="w", padx=15)
        lb = tk.Listbox(dialog, selectmode="multiple", font=("Segoe UI", 9), bg=self.C["CANVAS"], fg=self.C["TEXT"], height=13)
        lb.pack(fill="both", expand=True, padx=15, pady=10)
        for name in available: lb.insert("end", name)
        if self.model_var.get() in available: lb.selection_set(available.index(self.model_var.get()))
        result = {"selection": None}
        def accept():
            result["selection"] = [available[i] for i in lb.curselection()]
            if not result["selection"]: messagebox.showwarning("Select Models", "Select at least one model.", parent=dialog); return
            dialog.destroy()
        self.style_button(dialog, "Continue", accept, bg=self.C["TEAL"]).pack(pady=12)
        dialog.wait_window(); return result["selection"]

    def start_model_comparison(self):
        if self.busy: return
        if not self.selected_image_path:
            messagebox.showwarning("No CT Image", "Upload a CT image first."); return
        selected = self.choose_models_dialog("Compare Multiple Models")
        if not selected: return
        self.lock_analysis_buttons(True); self.set_status("Comparing selected models...", "working")
        threading.Thread(target=self.run_model_comparison, args=(selected, self.selected_image_path), daemon=True).start()

    def run_model_comparison(self, model_names, image_path):
        try:
            image = read_image(image_path); processed = preprocess_image(image); results = []
            for name in model_names:
                model = load_model_cached(name); batch = np.expand_dims(apply_model_scaling(processed, resolve_arch_config(name)), 0).astype(np.float32)
                probs = normalize_predictions(model.predict(batch, verbose=0)); idx = int(np.argmax(probs))
                results.append({"model": name, "prediction": CLASS_LABELS[idx], "confidence": float(probs[idx]), "probabilities": {label: float(probs[i]) for i, label in enumerate(CLASS_LABELS)}})
            self.root.after(0, self.show_model_comparison, results)
        except Exception as error:
            self.root.after(0, self.show_error, str(error))

    def show_model_comparison(self, results):
        self.busy = False; self.lock_analysis_buttons(False)
        dlg = tk.Toplevel(self.root); dlg.title("Model Comparison"); dlg.geometry("650x420"); dlg.configure(bg=self.C["PANEL"])
        tk.Label(dlg, text="Model Comparison Results", font=("Segoe UI", 14, "bold"), fg=self.C["TEXT"], bg=self.C["PANEL"]).pack(anchor="w", padx=15, pady=12)
        cols = ("model", "prediction", "confidence", "benign", "malignant", "normal")
        tree = ttk.Treeview(dlg, columns=cols, show="headings")
        for col, heading, width in [("model","Model",180),("prediction","Prediction",100),("confidence","Confidence",90),("benign","Benign",80),("malignant","Malignant",90),("normal","Normal",80)]:
            tree.heading(col, text=heading); tree.column(col, width=width, anchor="center")
        tree.pack(fill="both", expand=True, padx=15, pady=10)
        for r in results:
            p=r["probabilities"]; tree.insert("", "end", values=(r["model"], r["prediction"], f"{r['confidence']*100:.1f}%", f"{p['Benign']*100:.1f}%", f"{p['Malignant']*100:.1f}%", f"{p['Normal']*100:.1f}%"))
        best=max(results, key=lambda x:x["confidence"]); tk.Label(dlg, text=f"Highest-confidence model: {best['model']} - {best['prediction']} ({best['confidence']*100:.2f}%)", font=("Segoe UI",9,"bold"), fg=self.C["TEAL"], bg=self.C["PANEL"]).pack(anchor="w", padx=15, pady=6)
        self.set_status("Model comparison complete.", "ready")

    def start_ensemble_prediction(self):
        if self.busy: return
        if not self.selected_image_path:
            messagebox.showwarning("No CT Image", "Upload a CT image first."); return
        selected = self.choose_models_dialog("Create Ensemble Prediction")
        if not selected: return
        self.lock_analysis_buttons(True); self.set_status("Calculating ensemble prediction...", "working")
        threading.Thread(target=self.run_ensemble, args=(selected, self.selected_image_path), daemon=True).start()

    def run_ensemble(self, model_names, image_path):
        try:
            image = read_image(image_path); processed = preprocess_image(image); all_probs=[]; details=[]
            for name in model_names:
                model=load_model_cached(name); batch=np.expand_dims(apply_model_scaling(processed, resolve_arch_config(name)),0).astype(np.float32)
                probs=normalize_predictions(model.predict(batch, verbose=0)); all_probs.append(probs); details.append(name)
            ensemble=np.mean(np.stack(all_probs),axis=0); idx=int(np.argmax(ensemble))
            result={"prediction":CLASS_LABELS[idx],"confidence":float(ensemble[idx]),"probabilities":{label:float(ensemble[i]) for i,label in enumerate(CLASS_LABELS)},"model_count":len(model_names),"models":model_names}
            self.root.after(0, self.show_ensemble_result, result)
        except Exception as error:
            self.root.after(0, self.show_error, str(error))

    def show_ensemble_result(self, result):
        self.busy=False; self.lock_analysis_buttons(False)
        self.ensemble_status_var.set(f"Ensemble: {result['prediction']} ({result['confidence']*100:.2f}%) using {result['model_count']} model(s)")
        messagebox.showinfo("Ensemble Prediction", f"Prediction: {result['prediction']}\nConfidence: {result['confidence']*100:.2f}%\nModels: {result['model_count']}\n\nMean-probability ensemble across selected models.")
        self.set_status("Ensemble prediction complete.", "ready")

    # ---------- batch ----------
    def start_batch_analysis(self):
        if self.busy: return
        if not self.model_var.get():
            messagebox.showwarning("No Model", "Select a model before batch analysis."); return
        folder = filedialog.askdirectory(title="Select CT Image Folder")
        if not folder: return
        files = sorted([os.path.join(folder,f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS])
        if not files:
            messagebox.showwarning("No Images", "No supported image files were found in the selected folder."); return
        if not messagebox.askyesno("Batch Analysis", f"Analyze {len(files)} image(s) with model:\n{self.model_var.get()}?\n\nEach case will receive an automatic report ID and JSON record."):
            return
        self.lock_analysis_buttons(True); self.set_status(f"Batch analysis started: 0/{len(files)}", "working")
        threading.Thread(target=self.run_batch, args=(files, self.model_var.get()), daemon=True).start()

    def run_batch(self, files, model_name):
        results=[]; failed=[]; model=load_model_cached(model_name)
        batch_id=datetime.now().strftime("BATCH-%Y%m%d-%H%M%S")
        batch_output=os.path.join(BATCH_DIR,batch_id); os.makedirs(batch_output,exist_ok=True)
        for i,path in enumerate(files,1):
            try:
                image=read_image(path); processed=preprocess_image(image)
                batch=np.expand_dims(apply_model_scaling(processed,resolve_arch_config(model_name)),0).astype(np.float32)
                probs=normalize_predictions(model.predict(batch,verbose=0)); idx=int(np.argmax(probs))
                predicted=CLASS_LABELS[idx]; confidence=float(probs[idx])
                case_id=f"{Path(path).stem}-{datetime.now().strftime('%H%M%S%f')[:-3]}"
                gradcam_rgb=None; gradcam_path=None; layer=None
                try:
                    heatmap,layer=compute_gradcam(model,batch,idx)
                    gradcam_rgb=create_gradcam_overlay(processed,heatmap,self.gradcam_alpha,self.get_colormap())
                except Exception:
                    pass
                record,_=self.create_and_save_result(
                    path, processed, gradcam_rgb, model_name, predicted, confidence, probs, layer,
                    {"case_id":case_id,"patient_initials":"","age":"","sex":"Not specified","scan_date":"",
                     "referring_center":"","reviewer":"","notes":f"Batch analysis: {batch_id}"}
                )
                results.append({"file":os.path.basename(path),"path":path,"prediction":predicted,"confidence":confidence,
                                "benign":float(probs[0]),"malignant":float(probs[1]),"normal":float(probs[2]),"report_id":record["report_id"],"pdf_path":record.get("pdf_path")})
            except Exception as error:
                failed.append({"file":os.path.basename(path),"error":str(error)})
            self.root.after(0, lambda i=i,n=len(files): self.status_var.set(f"Batch analysis: {i}/{n}"))
        export_rows=[self.batch_row_to_history(r,model_name,batch_id) for r in results]
        csv_path=os.path.join(batch_output,f"{batch_id}_results.csv"); export_history_csv(export_rows,csv_path)
        xlsx_path=None
        if OPENPYXL_AVAILABLE:
            xlsx_path=os.path.join(batch_output,f"{batch_id}_results.xlsx"); export_history_xlsx(export_rows,xlsx_path)
        summary_path=os.path.join(batch_output,f"{batch_id}_summary.json")
        with open(summary_path,"w",encoding="utf-8") as fh: json.dump({"batch_id":batch_id,"model":model_name,"processed":len(results),"failed":failed,"results":results},fh,indent=2)
        self.root.after(0,self.finish_batch,batch_id,len(results),len(failed),csv_path,xlsx_path)

    def batch_row_to_history(self,r,model_name,batch_id):
        return {"report_id":r.get("report_id", batch_id),"created_at":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"case_id":Path(r["file"]).stem,"patient_initials":"","age":"","sex":"","scan_date":"","referring_center":"","reviewer":"","model":model_name,"prediction":r["prediction"],"confidence":r["confidence"],"benign":r["benign"],"malignant":r["malignant"],"normal":r["normal"],"confidence_level":confidence_level(r["confidence"])[0],"pdf_path":r.get("pdf_path","")}

    def finish_batch(self,batch_id,processed,failed,csv_path,xlsx_path):
        self.busy=False; self.lock_analysis_buttons(False); self.refresh_history(); self.set_status("Batch analysis complete.","ready")
        text=f"Batch: {batch_id}\nProcessed: {processed}\nFailed: {failed}\n\nCSV: {csv_path}\n"
        if xlsx_path: text += f"Excel: {xlsx_path}\n"
        messagebox.showinfo("Batch Analysis Complete", text)

    # ---------- display ----------
    def display_results(self, record, processed, gradcam_rgb):
        self.busy=False; self.lock_analysis_buttons(False)
        self.current_record=record; self.current_report_path=record.get("pdf_path")
        self.processed_image=processed.copy(); self.gradcam_image=gradcam_rgb.copy() if gradcam_rgb is not None else None
        self.last_heatmap = getattr(self, "last_heatmap", None)
        self.prediction_var.set(record["prediction"]); self.confidence_var.set(f"Confidence  •  {record['confidence']*100:.2f}%")
        level, key=confidence_level(record["confidence"]); self.review_var.set(level)
        self.review_label.config(fg=self.C["GREEN"] if key=="high" else self.C["ORANGE"] if key=="moderate" else self.C["RED"], bg=self.C["GREEN_LIGHT"] if key=="high" else self.C["ORANGE_LIGHT"] if key=="moderate" else self.C["RED_LIGHT"])
        self.prediction_label.config(fg=self.C["RED"] if record["prediction"]=="Malignant" else self.C["GREEN"] if record["prediction"]=="Benign" else self.C["BLUE"]); self.confidence_bar["value"]=record["confidence"]*100
        for label in CLASS_LABELS:
            pct=record["probabilities"][label]*100; bar,val=self.probability_bars[label]; bar["value"]=pct; val.config(text=f"{pct:.1f}%")
        self.show_array_on_canvas(record["image_path"], self.original_canvas, rgb=False)
        self.show_array_on_canvas(processed, self.processed_canvas, rgb=False)
        if gradcam_rgb is not None:
            self.show_array_on_canvas(gradcam_rgb, self.grad_canvas, rgb=True)
            self.grad_layer_var.set(f"Layer: {record.get('gradcam_layer') or '-'}")
        else:
            self.grad_canvas.delete("all"); self.grad_canvas.create_text(180, 120, text="Grad-CAM unavailable", fill=self.C["MUTED"], font=("Segoe UI", 11)); self.grad_layer_var.set("Layer: unavailable")
        if self.current_report_path and os.path.isfile(self.current_report_path): self.report_var.set(f"PDF ready: {os.path.basename(self.current_report_path)}")
        else: self.report_var.set("PDF not available - recreate report")
        self.refresh_history()
        status="Analysis complete. " + ("PDF report created automatically." if self.current_report_path and os.path.isfile(self.current_report_path) else "Prediction saved; PDF can be recreated.")
        if record.get("gradcam_error"): status += " Grad-CAM unavailable."
        self.set_status(status,"ready")

    def show_array_on_canvas(self, data, canvas, rgb=False):
        try:
            if isinstance(data,str): image=Image.open(data).convert("RGB")
            else:
                arr=data.copy();
                if arr.ndim==2: image=Image.fromarray(arr.astype(np.uint8)).convert("RGB")
                else: image=Image.fromarray(arr.astype(np.uint8) if rgb else cv2.cvtColor(arr.astype(np.uint8),cv2.COLOR_BGR2RGB))
            cw=max(canvas.winfo_width(),300); ch=max(canvas.winfo_height(),260)
            iw,ih=image.size; scale=min(cw/iw,ch/ih)*self.zoom_factor
            scale=max(.2,min(scale,4.0)); resized=image.resize((max(1,int(iw*scale)),max(1,int(ih*scale))),Image.Resampling.LANCZOS)
            photo=ImageTk.PhotoImage(resized); canvas.delete("all"); canvas.create_image(cw//2,ch//2,image=photo,anchor="center",tags="image"); canvas.configure(scrollregion=canvas.bbox("all")); canvas.xview_moveto(0.5); canvas.yview_moveto(0.5); canvas.image=photo
        except Exception as error:
            canvas.delete("all"); canvas.create_text(200,100,text=f"Display error: {error}",fill=self.C["RED"])

    def change_zoom(self, delta):
        self.zoom_factor=max(.4,min(3.0,self.zoom_factor+delta));
        if self.original_image is not None: self.show_array_on_canvas(self.original_image,self.original_canvas,rgb=False)
        if self.processed_image is not None: self.show_array_on_canvas(self.processed_image,self.processed_canvas,rgb=False)
        if self.gradcam_image is not None: self.show_array_on_canvas(self.gradcam_image,self.grad_canvas,rgb=True)
        self.zoom_reset_button.config(text=f"{int(self.zoom_factor*100)}%")

    def reset_zoom(self): self.zoom_factor=1.0; self.change_zoom(0)

    def on_alpha_change(self, value):
        self.gradcam_alpha=float(value); self.refresh_gradcam_visual()

    def get_colormap(self):
        return {"JET":cv2.COLORMAP_JET,"TURBO":cv2.COLORMAP_TURBO,"HOT":cv2.COLORMAP_HOT,"VIRIDIS":cv2.COLORMAP_VIRIDIS}.get(self.colormap_var.get(),cv2.COLORMAP_JET)

    def on_colormap_change(self): self.gradcam_colormap=self.colormap_var.get(); self.refresh_gradcam_visual()

    def set_last_heatmap(self, heatmap):
        self.last_heatmap = np.asarray(heatmap).copy()

    def refresh_gradcam_visual(self):
        if self.last_heatmap is None or self.processed_image is None: return
        self.gradcam_image=create_gradcam_overlay(self.processed_image,self.last_heatmap,self.gradcam_alpha,self.get_colormap()); self.show_array_on_canvas(self.gradcam_image,self.grad_canvas,rgb=True)
        if self.current_record:
            path = self.current_record.get("gradcam_path")
            if not path:
                path = os.path.join(HISTORY_DIR, f"{self.current_record['report_id']}_gradcam.png")
                self.current_record["gradcam_path"] = path
            cv2.imwrite(path, cv2.cvtColor(self.gradcam_image, cv2.COLOR_RGB2BGR))

    # ---------- reporting ----------
    def generate_report_from_current_result(self):
        if not self.current_record:
            messagebox.showwarning("No Analysis", "Run an analysis before creating a PDF report."); return
        record=dict(self.current_record); report_id=generate_report_id(); record["report_id"]=report_id; record["created_at"]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf_path=os.path.join(REPORTS_DIR,f"{report_id}_{Path(record['image_path']).stem}_LungCANet_Report.pdf")
        json_path=os.path.join(HISTORY_DIR,f"{report_id}_analysis.json")
        try:
            model_info=describe_model(load_model_cached(record["model"]),record["model"]); generate_professional_pdf_report(pdf_path,record,model_info); record["pdf_path"]=pdf_path; save_analysis_json(json_path,record,model_info); record["json_path"]=json_path; save_history(record); self.current_record=record; self.current_report_path=pdf_path; self.report_var.set(f"PDF ready: {os.path.basename(pdf_path)}"); self.refresh_history(); self.set_status("Professional PDF report created successfully.","ready")
            messagebox.showinfo("Report Created", f"Professional PDF report created.\n\n{pdf_path}")
        except Exception as error:
            messagebox.showerror("PDF Error", str(error)); self.set_status("PDF generation failed.","error")

    def open_latest_report(self):
        if not self.current_report_path or not os.path.isfile(self.current_report_path):
            messagebox.showwarning("No PDF", "No current PDF report is available."); return
        webbrowser.open(os.path.abspath(self.current_report_path))

    # ---------- history ----------
    def refresh_history(self):
        self.history_rows=fetch_history(200)
        if not hasattr(self,"history_tree"): return
        for item in self.history_tree.get_children(): self.history_tree.delete(item)
        for row in self.history_rows:
            self.history_tree.insert("", "end", iid=row["report_id"], values=(row["report_id"], row.get("case_id", ""), row.get("prediction", ""), f"{(row.get('confidence') or 0)*100:.1f}%", row.get("model", ""), row.get("created_at", "")))

    def selected_history_rows(self):
        ids=self.history_tree.selection(); return [fetch_case(i) for i in ids]

    def compare_selected_cases(self):
        rows=self.selected_history_rows()
        if len(rows)!=2:
            messagebox.showwarning("Select Two Cases","Select exactly two history rows to compare."); return
        dlg=tk.Toplevel(self.root); dlg.title("Case Comparison"); dlg.geometry("760x500"); dlg.configure(bg=self.C["PANEL"])
        tk.Label(dlg,text="Case Comparison",font=("Segoe UI",14,"bold"),fg=self.C["TEXT"],bg=self.C["PANEL"]).pack(anchor="w",padx=15,pady=12)
        left,right=rows; data=[["Field", "Case 1", "Case 2"],["Case ID",left.get("case_id"),right.get("case_id")],["Report ID",left.get("report_id"),right.get("report_id")],["Prediction",left.get("prediction"),right.get("prediction")],["Confidence",f"{left.get('confidence',0)*100:.2f}%",f"{right.get('confidence',0)*100:.2f}%"],["Model",left.get("model"),right.get("model")],["Benign",f"{left.get('benign',0)*100:.2f}%",f"{right.get('benign',0)*100:.2f}%"],["Malignant",f"{left.get('malignant',0)*100:.2f}%",f"{right.get('malignant',0)*100:.2f}%"],["Normal",f"{left.get('normal',0)*100:.2f}%",f"{right.get('normal',0)*100:.2f}%"]]
        tree=ttk.Treeview(dlg,columns=("field","one","two"),show="headings",height=10)
        for col,h in zip(("field","one","two"),("Field","Case 1","Case 2")): tree.heading(col,text=h); tree.column(col,width=240 if col=="field" else 230,anchor="center")
        tree.pack(fill="both",expand=True,padx=15,pady=10)
        for row in data[1:]: tree.insert("","end",values=row)
        tk.Label(dlg,text="Use the comparison to review changes between saved analyses; it is not a clinical conclusion.",font=("Segoe UI",8),fg=self.C["MUTED"],bg=self.C["PANEL"]).pack(anchor="w",padx=15,pady=10)

    def load_selected_history_case(self,event=None):
        sel=self.history_tree.selection()
        if not sel: return
        row=fetch_case(sel[0])
        if not row: return
        for key in ["case_id","patient_initials","age","scan_date","referring_center","sex"]:
            if key in self.case_vars: self.case_vars[key].set(row.get(key) or "")
        self.reviewer_var.set(row.get("reviewer") or "")
        self.notes_text.delete("1.0","end"); self.notes_text.insert("1.0",row.get("notes") or "")
        self.model_var.set(row.get("model") or "")
        if row.get("image_path") and os.path.isfile(row["image_path"]):
            self.selected_image_path=row["image_path"]; self.file_var.set(os.path.basename(row["image_path"]))
        self.current_report_path=row.get("pdf_path"); self.report_var.set(os.path.basename(row.get("pdf_path")) if row.get("pdf_path") else "No PDF")
        self.set_status(f"Loaded history record {row['report_id']}","ready")

    def export_history_csv_action(self):
        path=filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV","*.csv")],initialfile="LungCANet_History.csv")
        if not path:return
        export_history_csv(self.history_rows,path); messagebox.showinfo("Export Complete",f"CSV exported to:\n{path}")

    def export_history_xlsx_action(self):
        if not OPENPYXL_AVAILABLE:
            messagebox.showerror("Excel Export", "openpyxl is not installed. Run: pip install openpyxl"); return
        path=filedialog.asksaveasfilename(defaultextension=".xlsx",filetypes=[("Excel workbook","*.xlsx")],initialfile="LungCANet_History.xlsx")
        if not path:return
        export_history_xlsx(self.history_rows,path); messagebox.showinfo("Export Complete",f"Excel exported to:\n{path}")

    # ---------- theme ----------
    def toggle_theme(self):
        self.theme_name="dark" if self.theme_name=="light" else "light"
        # Recolor existing widgets conservatively; notebook/canvas colors are updated directly.
        self.apply_theme_recursive(self.root)
        self.setup_styles()
        self.theme_button.config(text="Light Mode" if self.theme_name=="dark" else "Dark Mode", bg=self.C["BLUE"])
        try:
            self.header_bg=ImageTk.PhotoImage(self.make_gradient(1600,92,self.C["NAVY"],"#154B76"))
            self.header_bg_label.config(image=self.header_bg)
        except Exception: pass
        for canvas in [self.original_canvas,self.processed_canvas,self.grad_canvas]: canvas.config(bg=self.C["CANVAS"])
        self.confidence_bar.configure(style="Confidence.Horizontal.TProgressbar")
        self.set_status(f"Switched to {self.theme_name} mode.","ready")

    def apply_theme_recursive(self, widget):
        c=self.C
        try:
            cls=widget.winfo_class()
            if cls in ("Frame", "Labelframe"):
                widget.config(bg=c["PANEL"] if widget is not self.container else c["BG"])
            elif cls=="Label":
                widget.config(bg=c["PANEL"], fg=c["TEXT"])
            elif cls=="Button":
                # Preserve dark/light readability; action buttons are re-painted below where needed.
                widget.config(fg=c["TEXT"], activebackground=c["BLUE_DARK"], activeforeground=c["WHITE"])
            elif cls=="Entry":
                widget.config(bg=c["CANVAS"], fg=c["TEXT"], insertbackground=c["TEXT"])
            elif cls=="Text":
                widget.config(bg=c["CANVAS"], fg=c["TEXT"], insertbackground=c["TEXT"])
        except Exception:
            pass
        for child in widget.winfo_children(): self.apply_theme_recursive(child)
        try:
            self.header.config(bg=c["NAVY"]); self.statusbar.config(bg=c["NAVY"])
        except Exception: pass

    # ---------- reset / errors ----------
    def lock_analysis_buttons(self, locked):
        state="disabled" if locked else "normal"
        for button in [self.upload_button,self.analyze_button,self.batch_button,self.compare_models_button,self.ensemble_button,self.reset_button]: button.config(state=state)
        self.busy=locked

    def reset_results_only(self):
        self.processed_image=None; self.gradcam_image=None; self.last_heatmap=None; self.current_record=None; self.current_report_path=None
        self.prediction_var.set("-"); self.confidence_var.set("Confidence -"); self.confidence_bar["value"]=0; self.review_var.set("Review status -"); self.review_label.config(fg=self.C["MUTED"]); self.prediction_label.config(fg=self.C["TEXT"]); self.ensemble_status_var.set("Ensemble: not run")
        for label in CLASS_LABELS: bar,val=self.probability_bars[label]; bar["value"]=0; val.config(text="0.0%")
        self.grad_canvas.delete("all"); self.grad_canvas.create_text(180,120,text="Grad-CAM unavailable\nRun classification to generate",fill=self.C["MUTED"],font=("Segoe UI",10),justify="center")
        self.grad_layer_var.set("Layer: -"); self.report_var.set("No PDF report generated")
        self.processed_canvas.delete("all")

    def reset_all(self):
        if self.busy:return
        self.selected_image_path=None; self.original_image=None; self.file_var.set("No CT image selected")
        for var in self.case_vars.values():
            if var is not self.case_vars["sex"]: var.set("")
        self.case_vars["sex"].set("Not specified"); self.reviewer_var.set(""); self.notes_text.delete("1.0","end"); self.original_canvas.delete("all"); self.reset_results_only(); self.set_status("Workspace reset.","ready")

    def show_error(self,error_message):
        self.busy=False; self.lock_analysis_buttons(False); self.set_status("Analysis failed.","error"); messagebox.showerror("Analysis Error",error_message)


def main():
    for folder in [MODELS_DIR, OUTPUT_DIR, HISTORY_DIR, REPORTS_DIR, BATCH_DIR]: os.makedirs(folder,exist_ok=True)
    root=tk.Tk(); app=LungCANetApp(root); root.mainloop()

if __name__=="__main__": main()
