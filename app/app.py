import io
import os
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from ultralytics import YOLO

# 1) Determine the directory of this file (app folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2) Configure Flask to look for templates/static under app/
template_dir = os.path.join(BASE_DIR, "templates")
static_dir   = os.path.join(BASE_DIR, "static")
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# 3) Emotion labels (must match your data.yaml order)
labels = [
    "angry",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "natural",
    "sad",
    "sleepy",
    "surprised"
]

# 4) Pick device: MPS if available (macOS), else CPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = "mps"
    print("⎯⎯⎯ Using MPS for inference")
else:
    DEVICE = "cpu"
    print("⎯⎯⎯ Using CPU for inference")

# 5) Load YOLOv8 checkpoint from ../models/best.pt
weights_path = os.path.join(BASE_DIR, "..", "models", "best.pt")
model = YOLO(weights_path)
model.to(DEVICE)

@app.route("/")
def index():
    """
    Serve the main HTML page (index.html in app/templates).
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects form-data field "image" (JPEG/PNG).
    Returns JSON:
      {
        "detections": [
          {"x1":…, "y1":…, "x2":…, "y2":…, "label": "...", "conf": …},
          …
        ],
        "top_emotion": "<label>",
        "top_conf": 0.87
      }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # 1) Load image via PIL, convert to RGB
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # 2) Convert PIL → NumPy (RGB → BGR)
    img_np = np.array(pil_img)[:, :, ::-1].copy()

    # 3) Run YOLOv8 inference
    results = model(img_np, device=DEVICE)[0]  # returns a Results object

    detections = []
    top_emotion = "natural"
    top_conf = 0.0

    # 4) Collect boxes with confidence ≥ 0.25
    for box in results.boxes:
        conf = float(box.conf.cpu().numpy()[0])
        if conf < 0.25:
            continue
        cls_idx = int(box.cls.cpu().numpy()[0])
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].tolist()
        label = labels[cls_idx]
        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "label": label, "conf": conf
        })
        if conf > top_conf:
            top_conf = conf
            top_emotion = label

    # If no valid detection, default to "natural"
    if len(detections) == 0:
        top_emotion = "natural"
        top_conf = 0.0

    return jsonify({
        "detections": detections,
        "top_emotion": top_emotion,
        "top_conf": top_conf
    })

if __name__ == "__main__":
    # Bind explicitly to localhost:5000
    app.run(debug=True, host="127.0.0.1", port=5000)
