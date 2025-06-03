// ------------------------------------
// static/script.js
// ------------------------------------

// 1) DOM references
const video       = document.getElementById("webcam");
const overlay     = document.getElementById("overlay");
const capture     = document.getElementById("capture");
const emotionDisp = document.getElementById("emotion-display");

// Poll interval in milliseconds (~3 frames/sec)
const POLL_INTERVAL = 300;
let pollingTimer = null;

// 2) Map emotion labels to emojis
const emojiMap = {
  angry:      "😠",
  contempt:   "😒",
  disgust:    "🤢",
  fear:       "😨",
  happy:      "😄",
  natural:    "😐",
  sad:        "😢",
  sleepy:     "😴",
  surprised:  "😲"
};

// 3) Setup webcam feed
async function setupWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
      video.onloadedmetadata = () => resolve();
    });
  } catch (err) {
    alert("Error accessing webcam: " + err);
  }
}

// 4) Draw bounding boxes on overlay canvas
function drawBoxes(detections) {
  const ctx = overlay.getContext("2d");
  // Clear previous
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  // Box style
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#00FF00";
  ctx.fillStyle = "#00FF00";
  ctx.font = "16px sans-serif";

  detections.forEach(det => {
    const { x1, y1, x2, y2, label, conf } = det;
    // Draw rect
    ctx.beginPath();
    ctx.rect(x1, y1, x2 - x1, y2 - y1);
    ctx.stroke();

    // Draw label background
    const text = `${label} ${(conf * 100).toFixed(0)}%`;
    const textWidth = ctx.measureText(text).width;
    const textHeight = 18;
    ctx.fillRect(x1, y1 - textHeight, textWidth + 6, textHeight);

    // Label text in black
    ctx.fillStyle = "#000000";
    ctx.fillText(text, x1 + 3, y1 - 4);
    ctx.fillStyle = "#00FF00"; // restore for next box
  });
}

// 5) Capture a frame, send to /predict, update overlay and emotion text
async function processFrame() {
  if (video.readyState !== 4) return; // not ready

  // Match overlay + capture to video dims
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  capture.width = video.videoWidth;
  capture.height = video.videoHeight;

  // Draw current frame onto hidden capture canvas
  const ctxCap = capture.getContext("2d");
  ctxCap.drawImage(video, 0, 0, capture.width, capture.height);

  // Convert to JPEG blob
  capture.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("image", blob, "frame.jpg");

    try {
      // POST to /predict
      const response = await fetch("/predict", {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        console.error("Server error:", await response.text());
        return;
      }
      const result = await response.json();
      // result = { detections: [...], top_emotion: "happy", top_conf: 0.87 }

      // 6) Draw all boxes
      drawBoxes(result.detections);

      // 7) Update “You are <Label> <Emoji> (<Confidence>%)”
      const { top_emotion, top_conf } = result;
      const emoji = emojiMap[top_emotion] || "";
      const label = top_emotion.charAt(0).toUpperCase() + top_emotion.slice(1);
      emotionDisp.innerHTML = 
        " <strong>" + label + "</strong> " +
        emoji +
        " <span class=\"confidence\">(" + (top_conf * 100).toFixed(0) + "%)</span>";
    } catch (err) {
      console.error("Request failed:", err);
    }
  }, "image/jpeg");
}

// 8) Start polling loop
function startLiveDetection() {
  if (pollingTimer !== null) return;
  pollingTimer = setInterval(processFrame, POLL_INTERVAL);
}

// 9) Initialize on page load
window.addEventListener("DOMContentLoaded", async () => {
  await setupWebcam();
  startLiveDetection();
});
