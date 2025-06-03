import io
import os
import requests
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# 1) Determine the directory of this file (the app folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2) Paths for model download and local storage
MODELS_DIR   = os.path.join(BASE_DIR, "..", "models")
MODEL_FILENAME = "best.pt"
MODEL_PATH   = os.path.join(MODELS_DIR, MODEL_FILENAME)

# 3) Google Drive “export=download” URL using your file’s ID
GDRIVE_FILE_ID = "1Hyfo-AXQjZQ8tRunf8xHV-XWbaWP0UFJ"
GDRIVE_URL     = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

# 4) If the model isn’t already on disk, download it from Google Drive
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Downloading best.pt from Google Drive...")

    session = requests.Session()
    # First request to get any confirmation token (for large files)
    response = session.get(GDRIVE_URL, stream=True)
    token = None

    # Google Drive may return a confirmation token in cookies if the file is large
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        # Append &confirm= token to force the download
        download_url = f"{GDRIVE_URL}&confirm={token}"
        response = session.get(download_url, stream=True)

    # Stream the response to the local file
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Download complete: saved to", MODEL_PATH)
else:
    print("Model already exists at", MODEL_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# 5) Configure Flask to look for templates/static under app/
template_dir = os.path.join(BASE_DIR, "templates")
static_dir   = os.path.join(BASE_DIR, "static")
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# 6) Emotion labels (must match your data.yaml order)
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

# 7) Pick device: MPS if available (macOS), else CPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = "mps"
    print("⎯⎯⎯ Using MPS for inference")
else:
    DEVICE = "cpu"
    print("⎯⎯⎯ Using CPU for inference")

# 8) Load YOLOv8 checkpoint from the local models folder
model = YOLO(MODEL_PATH)
model.to(DEVICE)

# ──────────────────────────────────────────────────────────────────────────────
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
