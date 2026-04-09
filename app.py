from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import datetime
import csv
import os

app = Flask(__name__)

# Load your trained YOLOv8 model
model = YOLO("best.pt")

# Create detections.csv if it doesn't exist
if not os.path.exists("detections.csv"):
    with open("detections.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Animal", "Count"])

# Open webcam (change 0 → RTSP link if using CCTV)
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO detection
        results = model(frame)
        boxes = results[0].boxes
        detected_animals = []

        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detected_animals.append(label)

            # Draw boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Log detections
        if detected_animals:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            counts = {}
            for animal in detected_animals:
                counts[animal] = counts.get(animal, 0) + 1

            with open("detections.csv", "a", newline="") as f:
                writer = csv.writer(f)
                for animal, count in counts.items():
                    writer.writerow([timestamp, animal, count])

        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_detections')
def latest_detections():
    data = []
    if os.path.exists("detections.csv"):
        with open("detections.csv", "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)[-5:]  # last 5 detections
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
