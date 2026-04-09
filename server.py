from flask import Flask, Response, jsonify, request, render_template
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import time
import threading
import os
import requests

app = Flask(__name__)
CORS(app)

# ================== CONFIG ==================
MODEL_PATH = "best.pt"
CAPTURE_FOLDER = "static/captures"
NODE_API_URL = "https://maui-server-1.onrender.com/api/detections"

CONF_THRESHOLD = 0.6
COOLDOWN_SECONDS = 30
CAMERA_ID = "Camera 1"

os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# ================== LOAD MODEL ==================
model = YOLO(MODEL_PATH)

# ================== GLOBAL STATE ==================
lock = threading.Lock()
camera = None
last_detection_text = "No animal detected"
LAST_SENT = {}

# ================== CAMERA ==================
def open_camera_source(source):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

# ================== YOLO STREAM ==================
def generate_frames():
    global camera, last_detection_text

    while True:
        with lock:
            if camera is None:
                time.sleep(0.2)
                continue
            success, frame = camera.read()

        if not success:
            continue

        results = model(frame)
        detections = {}
        boxed_frame = frame.copy()

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue

                cls = int(box.cls[0])
                label = model.names[cls]
                detections[label] = detections.get(label, 0) + 1

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cv2.rectangle(boxed_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    boxed_frame,
                    label,
                    (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # ================== SEND TO NODE ==================
        if detections:
            species = list(detections.keys())[0]
            count = detections[species]
            now = time.time()

            if CAMERA_ID not in LAST_SENT or now - LAST_SENT[CAMERA_ID] > COOLDOWN_SECONDS:
                LAST_SENT[CAMERA_ID] = now

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                file_name = f"capture_{int(now)}.jpg"
                image_path = os.path.join(CAPTURE_FOLDER, file_name)

                cv2.imwrite(image_path, boxed_frame)

                payload = {
                    "species": species,
                    "count": count,
                    "cameraId": CAMERA_ID,
                    "image": f"/static/captures/{file_name}",
                    "timestamp": timestamp
                }

                try:
                    requests.post(NODE_API_URL, json=payload, timeout=2)
                    print("✔ Detection sent to Node.js")
                except Exception as e:
                    print("❌ Failed to send detection:", e)

            last_detection_text = f"{species} ({count})"
        else:
            last_detection_text = "No animal detected"

        ret, buffer = cv2.imencode(".jpg", boxed_frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

        time.sleep(0.3)

# ================== ROUTES ==================
@app.route("/")
def home():
    return render_template("camera-monitoring.html")

@app.route("/start_camera", methods=["POST"])
def start_camera():
    global camera
    source = request.json.get("source", 0)

    with lock:
        if camera is not None:
            camera.release()
        camera = open_camera_source(source)

    return jsonify({"status": "camera started"})

@app.route("/stop_camera")
def stop_camera():
    global camera
    with lock:
        if camera:
            camera.release()
            camera = None
    return jsonify({"status": "camera stopped"})

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_detection")
def get_detection():
    return jsonify({
        "detection": last_detection_text,
        "alert": last_detection_text != "No animal detected"
    })

# ================== RUN ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


